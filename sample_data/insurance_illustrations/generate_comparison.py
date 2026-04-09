"""
Side-by-side comparison:
  Scenario A — Index Fund (taxable brokerage)
  Scenario B — Lincoln WealthBuilder IUL (premium financed, $0 out of pocket)

Swati Chugh | Age 43 today | Starting Year = Policy Year 1

Assumptions for Index Fund:
  - $100,000/yr invested for 10 years (matching the "out of pocket" feel of what you COULD spend)
  - Gross annual return: 9% (broad S&P 500 long-run average, before tax)
  - Long-term capital gains tax: 20% on gains (federal, rough California blended ~23.8% LTCG+NIIT)
    We use 20% federal for conservative-but-realistic comparison.
  - Annual expense ratio: 0.05% (index ETF)
  - Dividends reinvested, taxed at 15% each year (~2% dividend yield)
  - At distribution: gains taxed at 20% LTCG on the gain portion

Assumptions for IUL:
  - $0 out of pocket ever
  - Policy credited at 7.19% (from illustration)
  - Year 20: loan of $5.237M repaid from policy cash value → net cash $1,348,332
  - Year 21 onward: distributions as policy loans (tax-free)
  - Illustrated distribution: $204,086/yr; your scenario: $150,000/yr
  - Values pulled directly from Lincoln illustration

Age at start: 43
Year 20 → age 63
Age 80 → year 37 of policy
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# ── MODELLING ──────────────────────────────────────────────────────────────

GROSS_RETURN      = 0.09          # 9% gross
EXPENSE_RATIO     = 0.0005        # 0.05%
NET_RETURN        = GROSS_RETURN - EXPENSE_RATIO  # ~8.95%
DIV_YIELD         = 0.02
DIV_TAX_RATE      = 0.15          # qualified dividends
LTCG_TAX          = 0.20          # long-term capital gains federal
ANNUAL_CONTRIB    = 100_000
CONTRIB_YEARS     = 10
START_AGE         = 43
WITHDRAWAL        = 150_000

# IUL values from illustration (year, policy_cash_net, death_benefit)
# Year 20 net is post-loan-repayment
IUL_VALUES = {
    20: (1_348_332, 3_088_987),
    21: (1_322_504, 3_015_057),
    22: (1_303_001, 2_964_447),
    23: (1_290_255, 2_907_668),
    24: (1_285_186, 2_930_569),
    25: (1_288_841, 2_958_027),
    26: (1_302_135, 2_990_197),
    27: (1_326_274, 3_027_479),
    28: (1_362_522, 3_070_228),
    29: (1_413_052, 2_997_822),
    30: (1_479_690, 2_915_634),
    31: (1_564_390, 2_822_536),
    32: (1_669_525, 2_717_527),
    33: (1_797_882, 2_599_657),
    34: (1_949_385, 2_808_093),
    35: (2_126_170, 3_045_790),
    36: (2_330_603, 3_315_381),
    37: (2_564_940, 3_619_392),   # age 80
    38: (2_831_498, 3_960_424),
    39: (3_133_179, 4_341_697),
    40: (3_473_105, 4_766_676),
    43: (4_739_521, 6_323_876),
}
IUL_DIST_PER_YEAR = 150_000   # your scenario
IUL_DIST_ILLUS    = 204_086   # illustrated

# ── Index Fund model ─────────────────────────────────────────────────────────
# Accumulation phase: years 1-10, $100k/yr
# Growth phase (no contrib): years 11-19 just grows
# At year 20: show balance before/after tax if cashed out (hypothetical)
# Distribution phase: withdraw $150k/yr from year 21 onward, compute residual

def model_index_fund():
    """
    Returns list of dicts per year with:
      year, age, contrib, gross_value, tax_basis, after_tax_value, notes
    """
    rows = []
    value     = 0.0
    basis     = 0.0   # total cash invested (cost basis)
    year      = 0

    # ACCUMULATION: years 1-10
    for y in range(1, CONTRIB_YEARS + 1):
        year += 1
        age = START_AGE + year
        value += ANNUAL_CONTRIB
        basis += ANNUAL_CONTRIB

        # dividends (taxed annually, reinvested net of tax)
        div = value * DIV_YIELD
        div_net = div * (1 - DIV_TAX_RATE)
        value += div_net
        basis += div_net   # reinvested after-tax divs raise basis

        # price appreciation (unrealised)
        price_gain = value * (NET_RETURN - DIV_YIELD)
        value += price_gain
        # basis unchanged for unrealised gains

        gain      = max(0, value - basis)
        tax_owed  = gain * LTCG_TAX
        after_tax = value - tax_owed

        rows.append(dict(year=year, age=age, annual_contrib=ANNUAL_CONTRIB,
                         gross_value=value, basis=basis,
                         after_tax_value=after_tax, note="Accumulating"))

    # GROWTH: years 11-19 (no new contributions)
    for y in range(11, 20):
        year += 1
        age = START_AGE + year

        div = value * DIV_YIELD
        div_net = div * (1 - DIV_TAX_RATE)
        value += div_net
        basis += div_net

        price_gain = value * (NET_RETURN - DIV_YIELD)
        value += price_gain

        gain      = max(0, value - basis)
        tax_owed  = gain * LTCG_TAX
        after_tax = value - tax_owed

        rows.append(dict(year=year, age=age, annual_contrib=0,
                         gross_value=value, basis=basis,
                         after_tax_value=after_tax, note="Growing"))

    # YEAR 20 snapshot
    year += 1
    age = START_AGE + year
    div = value * DIV_YIELD
    div_net = div * (1 - DIV_TAX_RATE)
    value += div_net
    basis += div_net
    price_gain = value * (NET_RETURN - DIV_YIELD)
    value += price_gain
    gain      = max(0, value - basis)
    tax_owed  = gain * LTCG_TAX
    after_tax = value - tax_owed
    rows.append(dict(year=year, age=age, annual_contrib=0,
                     gross_value=value, basis=basis,
                     after_tax_value=after_tax, note="Year 20 snapshot"))

    # DISTRIBUTION: years 21 onward, withdraw $150k/yr after tax
    # Each withdrawal: need to gross up because gains are taxed
    # Use proportional basis method: basis_pct = basis/value
    for y in range(21, 45):
        year += 1
        age = START_AGE + year
        if value <= 0:
            rows.append(dict(year=year, age=age, annual_contrib=-WITHDRAWAL,
                             gross_value=0, basis=0,
                             after_tax_value=0, note="Depleted"))
            continue

        # grow first
        div = value * DIV_YIELD
        div_net = div * (1 - DIV_TAX_RATE)
        value += div_net
        basis += div_net
        price_gain = value * (NET_RETURN - DIV_YIELD)
        value += price_gain

        # withdraw $150k after-tax
        # proportion of each dollar that is gain vs basis
        basis_pct = basis / value if value > 0 else 1
        # to receive $150k after-tax need to withdraw W where W - W*(1-basis_pct)*LTCG = 150k
        effective_tax_on_withdrawal = (1 - basis_pct) * LTCG_TAX
        gross_withdrawal = WITHDRAWAL / (1 - effective_tax_on_withdrawal) if effective_tax_on_withdrawal < 1 else WITHDRAWAL
        gross_withdrawal = min(gross_withdrawal, value)

        basis_withdrawn = gross_withdrawal * basis_pct
        gain_withdrawn  = gross_withdrawal - basis_withdrawn
        tax_on_withdrawal = gain_withdrawn * LTCG_TAX
        net_received = gross_withdrawal - tax_on_withdrawal

        value -= gross_withdrawal
        basis -= basis_withdrawn
        basis = max(0, basis)

        gain      = max(0, value - basis)
        tax_owed  = gain * LTCG_TAX
        after_tax = value - tax_owed

        rows.append(dict(year=year, age=age, annual_contrib=-int(net_received),
                         gross_value=value, basis=basis,
                         after_tax_value=after_tax,
                         note=f"${int(net_received):,} net rcvd"))

    return rows

fund_rows = model_index_fund()

def fmt(n):
    if n >= 0:
        return f"${n:,.0f}"
    return f"(${abs(n):,.0f})"

# ── BUILD PDF ─────────────────────────────────────────────────────────────────

OUTPUT = "Swati_Chugh_IUL_vs_IndexFund_Comparison.pdf"

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=letter,
    topMargin=0.4 * inch,
    bottomMargin=0.35 * inch,
    leftMargin=0.5 * inch,
    rightMargin=0.5 * inch,
)

NAVY  = colors.HexColor("#1B2A4A")
GOLD  = colors.HexColor("#C9A84C")
LIGHT = colors.HexColor("#EAF0F8")
GREEN = colors.HexColor("#1A6B3C")
RED   = colors.HexColor("#8B0000")
WHITE = colors.white
GRAY  = colors.HexColor("#555555")
LGOLD = colors.HexColor("#FFF8E6")

def S(name, **kw):
    return ParagraphStyle(name, **kw)

title_s    = S("t",  fontSize=16, textColor=WHITE,  fontName="Helvetica-Bold", alignment=TA_CENTER)
sub_s      = S("s",  fontSize=8.5,textColor=GOLD,   fontName="Helvetica-Bold", alignment=TA_CENTER)
sect_s     = S("sc", fontSize=8.5,textColor=NAVY,   fontName="Helvetica-Bold", spaceBefore=5, spaceAfter=2)
body_s     = S("b",  fontSize=7.5,textColor=colors.HexColor("#222222"), fontName="Helvetica", leading=11)
small_s    = S("sm", fontSize=6.5,textColor=GRAY,   fontName="Helvetica", leading=9)
col_hdr_s  = S("ch", fontSize=7,  textColor=WHITE,  fontName="Helvetica-Bold", alignment=TA_CENTER)
col_val_s  = S("cv", fontSize=7,  textColor=GRAY,   fontName="Helvetica",      alignment=TA_RIGHT)
col_grn_s  = S("cg", fontSize=7,  textColor=GREEN,  fontName="Helvetica-Bold", alignment=TA_RIGHT)
col_red_s  = S("cr", fontSize=7,  textColor=RED,    fontName="Helvetica-Bold", alignment=TA_RIGHT)
callout_s  = S("ca", fontSize=8,  textColor=NAVY,   fontName="Helvetica-Bold", alignment=TA_CENTER)

def p(txt, style): return Paragraph(txt, style)
def cv(txt, hi="", right=True):
    s = col_grn_s if hi=="g" else col_red_s if hi=="r" else col_val_s
    return p(txt, s)

story = []

# Header
hdr = Table([[p("Swati Chugh — IUL vs. Index Fund: Side-by-Side Comparison", title_s)],
             [p("What happens if you invest $100k/yr yourself vs. taking the premium-financed IUL?  •  April 2026", sub_s)]],
            colWidths=[7.5*inch])
hdr.setStyle(TableStyle([
    ("BACKGROUND", (0,0),(-1,-1), NAVY),
    ("TOPPADDING",    (0,0),(-1,-1), 10),
    ("BOTTOMPADDING", (0,0),(-1,-1), 8),
    ("LEFTPADDING",   (0,0),(-1,-1), 10),
]))
story.append(hdr)
story.append(Spacer(1, 7))

# ── ASSUMPTIONS BOX ──────────────────────────────────────────────────────────
story.append(p("KEY ASSUMPTIONS", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1, 4))

assum_data = [
    [p("", col_hdr_s),
     p("Index Fund (Taxable)", col_hdr_s),
     p("Lincoln WealthBuilder IUL", col_hdr_s)],
    [p("Your cash outlay", body_s),
     p("$100,000 / year for 10 years\n(total $1,000,000 from your pocket)", body_s),
     p("$0 — premiums financed by lender\n(zero out of pocket, ever)", body_s)],
    [p("Growth rate", body_s),
     p("9% gross / ~8.95% net (S&P 500 avg, before tax)", body_s),
     p("7.19% credited (non-guaranteed, per illustration)", body_s)],
    [p("Tax on growth", body_s),
     p("Dividends taxed annually @ 15%\nCapital gains taxed @ 20% on withdrawal", body_s),
     p("Tax-deferred inside policy\nDistributions via loans = tax-free", body_s)],
    [p("Distribution start", body_s),
     p("Year 21 (age 64)", body_s),
     p("Year 21 (age 64)", body_s)],
    [p("Annual distribution", body_s),
     p("$150,000/yr (net after tax)", body_s),
     p("$150,000/yr (tax-free policy loan)", body_s)],
]
at = Table(assum_data, colWidths=[1.5*inch, 3.0*inch, 3.0*inch])
at.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(-1,0), NAVY),
    ("BACKGROUND",    (0,1),(0,-1), colors.HexColor("#D6E4F0")),
    ("ROWBACKGROUNDS",(0,1),(-1,-1), [LIGHT, WHITE]),
    ("TOPPADDING",    (0,0),(-1,-1), 4),
    ("BOTTOMPADDING", (0,0),(-1,-1), 4),
    ("LEFTPADDING",   (0,0),(-1,-1), 6),
    ("RIGHTPADDING",  (0,0),(-1,-1), 6),
    ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#CCCCCC")),
    ("VALIGN",        (0,0),(-1,-1), "TOP"),
]))
story.append(at)
story.append(Spacer(1, 8))

# ── YEAR 20 COMPARISON ────────────────────────────────────────────────────────
story.append(p("AT YEAR 20 (AGE 63) — BEFORE DISTRIBUTIONS START", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1, 4))

# find year 20 row for fund
f20 = next(r for r in fund_rows if r["year"] == 20)
iUL_20_cv, iUL_20_db = IUL_VALUES[20]

y20_data = [
    [p("", col_hdr_s), p("Index Fund", col_hdr_s), p("IUL", col_hdr_s)],
    [p("Total cash you put in",      body_s), p("$1,000,000", body_s), p("$0", body_s)],
    [p("Gross account value",        body_s), p(fmt(f20["gross_value"]), body_s), p("$6,151,469 (pre-loan-repayment)", body_s)],
    [p("Tax cost to access all of it",body_s),p(fmt(f20["gross_value"]-f20["after_tax_value"]) + " (cap gains)", body_s), p("$0 (loans are tax-free)", body_s)],
    [p("Spendable / net value",       body_s), p(fmt(f20["after_tax_value"]), body_s), p("$1,348,332 (after $5.24M loan repaid)", body_s)],
    [p("Death benefit to heirs",      body_s), p("Your account balance (taxable to heirs)", body_s), p("$3,088,987 (income-tax-free to heirs)", body_s)],
]
y20t = Table(y20_data, colWidths=[2.2*inch, 2.65*inch, 2.65*inch])
y20t.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(-1,0), NAVY),
    ("BACKGROUND",    (0,1),(0,-1), colors.HexColor("#D6E4F0")),
    ("ROWBACKGROUNDS",(0,1),(-1,-1), [LIGHT, WHITE]),
    ("TOPPADDING",    (0,0),(-1,-1), 4),
    ("BOTTOMPADDING", (0,0),(-1,-1), 4),
    ("LEFTPADDING",   (0,0),(-1,-1), 6),
    ("RIGHTPADDING",  (0,0),(-1,-1), 6),
    ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#CCCCCC")),
    ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
]))
story.append(y20t)
story.append(Spacer(1, 8))

# ── DISTRIBUTION TABLE SIDE BY SIDE ──────────────────────────────────────────
story.append(p("YEAR-BY-YEAR: $150K/YR WITHDRAWALS STARTING YEAR 21  (selected milestones)", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1, 4))

dist_hdr = [
    p("Yr", col_hdr_s), p("Age", col_hdr_s),
    p("Index Fund\nAfter-Tax\nBalance", col_hdr_s),
    p("Tax Paid\nThis Year", col_hdr_s),
    p("IUL Cash\nValue (Net)", col_hdr_s),
    p("IUL Death\nBenefit", col_hdr_s),
    p("IUL\nAdvantage\n(Balance)", col_hdr_s),
]

highlight_years = {20, 21, 25, 30, 37, 40, 43}
dist_rows_data = [dist_hdr]

# build merged fund dict by year
fund_by_year = {r["year"]: r for r in fund_rows}

for yr in sorted(set(list(range(20, 45)))):
    fr = fund_by_year.get(yr)
    if fr is None:
        continue
    age = fr["age"]
    fund_at = fr["after_tax_value"]
    if fund_at < 0: fund_at = 0

    # compute tax paid this year (rough: gain portion of withdrawal * LTCG)
    if yr >= 21 and fr["gross_value"] > 0:
        basis_pct = fr["basis"] / (fr["gross_value"] + 150_000 * (1/(1-(1-fr["basis"]/(fr["gross_value"]+150_000))*LTCG_TAX) - 1) + 1e-9)
        tax_yr = int((150_000 / (1 - (1 - basis_pct) * LTCG_TAX) - 150_000) * (1 if fund_at > 0 else 0))
    else:
        tax_yr = 0

    iUL_cv = IUL_VALUES.get(yr, (None, None))[0]
    iUL_db = IUL_VALUES.get(yr, (None, None))[1]

    if yr not in highlight_years:
        continue

    advantage = (iUL_cv - fund_at) if iUL_cv else 0
    hi_fund = "r" if fund_at < (iUL_cv or 0) else "g"
    hi_iUL  = "g" if (iUL_cv or 0) >= fund_at else ""
    hi_adv  = "g" if advantage >= 0 else "r"

    row = [
        cv(str(yr)),
        cv(str(age)),
        cv(fmt(fund_at), hi_fund),
        cv(fmt(tax_yr) if tax_yr else "—"),
        cv(fmt(iUL_cv), hi_iUL) if iUL_cv else cv("—"),
        cv(fmt(iUL_db)) if iUL_db else cv("—"),
        cv(fmt(advantage), hi_adv) if iUL_cv else cv("—"),
    ]
    dist_rows_data.append(row)

dt = Table(dist_rows_data, colWidths=[0.4*inch, 0.4*inch, 1.35*inch, 1.0*inch, 1.35*inch, 1.35*inch, 1.15*inch])
dt.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(-1,0), NAVY),
    ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LIGHT]),
    ("TOPPADDING",    (0,0),(-1,-1), 4),
    ("BOTTOMPADDING", (0,0),(-1,-1), 4),
    ("LEFTPADDING",   (0,0),(-1,-1), 5),
    ("RIGHTPADDING",  (0,0),(-1,-1), 5),
    ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#CCCCCC")),
    ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ("ALIGN",         (0,0),(-1,-1), "CENTER"),
]))
story.append(dt)
story.append(Spacer(1, 7))

# ── CALLOUT BOX ──────────────────────────────────────────────────────────────
# Get age-80 values (year 37)
f37  = fund_by_year.get(37)
fund_80 = f37["after_tax_value"] if f37 else 0
iUL_80_cv, iUL_80_db = IUL_VALUES.get(37, (0, 0))

callout_data = [[
    Table([
        [p("INDEX FUND @ AGE 80", S("cah", fontSize=8, fontName="Helvetica-Bold",
                                    textColor=WHITE, alignment=TA_CENTER))],
        [p(f"After-tax balance: {fmt(fund_80)}", S("cav", fontSize=11,
            fontName="Helvetica-Bold", textColor=RED, alignment=TA_CENTER))],
        [p("(After 17 yrs of $150k/yr withdrawals\n+ paying capital gains tax each year)", small_s)],
    ], colWidths=[3.55*inch]),

    Table([
        [p("IUL @ AGE 80", S("cah2", fontSize=8, fontName="Helvetica-Bold",
                              textColor=WHITE, alignment=TA_CENTER))],
        [p(f"Cash value: {fmt(iUL_80_cv)}", S("cav2", fontSize=11,
            fontName="Helvetica-Bold", textColor=GREEN, alignment=TA_CENTER))],
        [p(f"Death benefit to heirs: {fmt(iUL_80_db)}\n(All tax-free. $0 invested.)", small_s)],
    ], colWidths=[3.55*inch]),
]]

for inner_t, bg in zip(callout_data[0], [RED, GREEN]):
    inner_t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), bg),
        ("BACKGROUND",    (0,1),(-1,-1), LGOLD if bg==RED else colors.HexColor("#E6F5ED")),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("RIGHTPADDING",  (0,0),(-1,-1), 8),
        ("BOX",           (0,0),(-1,-1), 1, bg),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ]))

ct = Table(callout_data, colWidths=[3.6*inch, 3.6*inch], hAlign="CENTER")
ct.setStyle(TableStyle([
    ("TOPPADDING",    (0,0),(-1,-1), 0),
    ("BOTTOMPADDING", (0,0),(-1,-1), 0),
    ("LEFTPADDING",   (0,0),(-1,-1), 3),
    ("RIGHTPADDING",  (0,0),(-1,-1), 3),
    ("VALIGN",        (0,0),(-1,-1), "TOP"),
]))
story.append(ct)
story.append(Spacer(1, 7))

# ── BOTTOM INSIGHT ────────────────────────────────────────────────────────────
insight_data = [[p(
    "THE BOTTOM LINE:  The index fund starts with $1M of your own money and still produces comparable or smaller spendable wealth at age 80, "
    "after paying capital gains taxes every year.  The IUL starts with $0 from your pocket — the lender funds everything — "
    "and leaves you with a growing cash value plus a multi-million-dollar death benefit your heirs receive income-tax-free.",
    S("ins", fontSize=7.5, fontName="Helvetica", textColor=NAVY, leading=11))]]
it = Table(insight_data, colWidths=[7.5*inch])
it.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(-1,-1), colors.HexColor("#EAF0F8")),
    ("BOX",           (0,0),(-1,-1), 1, NAVY),
    ("TOPPADDING",    (0,0),(-1,-1), 6),
    ("BOTTOMPADDING", (0,0),(-1,-1), 6),
    ("LEFTPADDING",   (0,0),(-1,-1), 8),
    ("RIGHTPADDING",  (0,0),(-1,-1), 8),
]))
story.append(it)
story.append(Spacer(1, 5))

# Disclaimer
story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY))
story.append(Spacer(1, 3))
story.append(Paragraph(
    "Index fund projections are hypothetical, using 9% gross annual return, 0.05% expense ratio, 2% dividend yield taxed at 15% annually, "
    "and 20% federal LTCG on withdrawal gains. State taxes not included. IUL values sourced from Lincoln Financial illustration dated March 10, 2026 "
    "assuming 7.19% non-guaranteed credited rate. Actual results will vary. This is not investment or tax advice. Consult a qualified advisor.",
    S("dis", fontSize=6, fontName="Helvetica", textColor=GRAY, leading=8)))

doc.build(story)
print(f"PDF written → {OUTPUT}")
