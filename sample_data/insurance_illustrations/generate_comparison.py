"""
Simple gross comparison — NO tax assumptions.
Index Fund: $100k/yr years 1-10 only, grows at 9%/yr gross, $150k/yr withdrawn from year 21.
IUL: values from Lincoln Financial illustration (7.19% credited, $0 out of pocket).
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT

OUTPUT = "Swati_Chugh_IUL_vs_IndexFund_Comparison.pdf"

# ── colours ──────────────────────────────────────────────────────────────────
NAVY  = colors.HexColor("#1B2A4A")
GOLD  = colors.HexColor("#C9A84C")
LIGHT = colors.HexColor("#EAF0F8")
GREEN = colors.HexColor("#1A6B3C")
WHITE = colors.white
GRAY  = colors.HexColor("#555555")
LGRN  = colors.HexColor("#E6F5ED")
LGOLD = colors.HexColor("#FFF8E6")

def S(name, **kw): return ParagraphStyle(name, **kw)

title_s   = S("t",  fontSize=16, textColor=WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER)
sub_s     = S("s",  fontSize=9,  textColor=GOLD,  fontName="Helvetica-Bold", alignment=TA_CENTER)
sect_s    = S("sc", fontSize=9,  textColor=NAVY,  fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=3)
body_s    = S("b",  fontSize=8,  textColor=colors.HexColor("#222222"), fontName="Helvetica", leading=12)
small_s   = S("sm", fontSize=6.5,textColor=GRAY,  fontName="Helvetica", leading=9)
ch_s      = S("ch", fontSize=8,  textColor=WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER)
cv_s      = S("cv", fontSize=8,  textColor=GRAY,  fontName="Helvetica",      alignment=TA_RIGHT)
cg_s      = S("cg", fontSize=8,  textColor=GREEN, fontName="Helvetica-Bold", alignment=TA_RIGHT)
cn_s      = S("cn", fontSize=8,  textColor=NAVY,  fontName="Helvetica-Bold", alignment=TA_RIGHT)
cc_s      = S("cc", fontSize=8,  textColor=GRAY,  fontName="Helvetica",      alignment=TA_CENTER)

def p(txt, st): return Paragraph(str(txt), st)
def cv(txt, style=None): return p(txt, style or cv_s)
def fmt(n): return f"${n:,.0f}" if n >= 0 else f"(${abs(n):,.0f})"

# ── INDEX FUND MODEL (no tax) ─────────────────────────────────────────────────
RATE        = 0.09
CONTRIB     = 100_000
START_AGE   = 43
WITHDRAWAL  = 150_000

fund_snapshots = {}
value = 0.0
for y in range(1, 84):
    contrib = CONTRIB if y <= 10 else 0
    value = (value + contrib) * (1 + RATE)
    if y >= 21:
        value = max(0, value - WITHDRAWAL)
    fund_snapshots[y] = value

# IUL values from illustration (year: (cash_value_net, death_benefit))
IUL = {
    20: (1_348_332,  3_088_987),
    21: (1_322_504,  3_015_057),
    25: (1_288_841,  2_958_027),
    30: (1_479_690,  2_915_634),
    35: (2_126_170,  3_045_790),
    37: (2_564_940,  3_619_392),
    40: (3_473_105,  4_766_676),
    43: (4_739_521,  6_323_876),
    50: (11_069_442, 12_577_529),
}

# ── BUILD PDF ─────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(OUTPUT, pagesize=letter,
    topMargin=0.45*inch, bottomMargin=0.4*inch,
    leftMargin=0.55*inch, rightMargin=0.55*inch)

story = []

# Header
hdr = Table([
    [p("Swati Chugh — Index Fund vs. IUL: Simple Comparison", title_s)],
    [p("$100k/yr for 10 years only  •  $150k/yr withdrawals from Age 64  •  No tax assumptions", sub_s)],
], colWidths=[7.4*inch])
hdr.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(-1,-1), NAVY),
    ("TOPPADDING",    (0,0),(-1,-1), 10),
    ("BOTTOMPADDING", (0,0),(-1,-1), 8),
    ("LEFTPADDING",   (0,0),(-1,-1), 10),
]))
story.append(hdr)
story.append(Spacer(1, 8))

# ── ASSUMPTIONS ───────────────────────────────────────────────────────────────
story.append(p("ASSUMPTIONS", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1, 4))

assum = Table([
    [p("", ch_s), p("Index Fund (S&P 500 ETF)", ch_s), p("Lincoln WealthBuilder IUL", ch_s)],
    [p("Your cash outlay", body_s),
     p("$100,000/yr for Years 1–10 only\nTotal invested: $1,000,000. Then STOPS.", body_s),
     p("$0 — lender funds all premiums\nYou never write a check.", body_s)],
    [p("Growth rate", body_s),
     p("9%/yr gross (S&P 500 long-run avg)\nNo taxes modelled", body_s),
     p("7.19%/yr credited (non-guaranteed)\nPer Lincoln Financial illustration", body_s)],
    [p("Withdrawals", body_s),
     p("$150,000/yr gross starting Year 21 (age 64)", body_s),
     p("$150,000/yr as tax-free policy loans from Year 21", body_s)],
], colWidths=[1.6*inch, 2.9*inch, 2.9*inch])
assum.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(-1,0), NAVY),
    ("BACKGROUND",    (0,1),(0,-1), colors.HexColor("#D6E4F0")),
    ("ROWBACKGROUNDS",(0,1),(-1,-1), [LIGHT, WHITE]),
    ("TOPPADDING",    (0,0),(-1,-1), 5),
    ("BOTTOMPADDING", (0,0),(-1,-1), 5),
    ("LEFTPADDING",   (0,0),(-1,-1), 7),
    ("RIGHTPADDING",  (0,0),(-1,-1), 7),
    ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#CCCCCC")),
    ("VALIGN",        (0,0),(-1,-1), "TOP"),
]))
story.append(assum)
story.append(Spacer(1, 9))

# ── ACCUMULATION (years 1-20, no withdrawals yet) ────────────────────────────
story.append(p("ACCUMULATION PHASE — YEARS 1 TO 20  (before any withdrawals)", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1, 4))

acc_hdr = [p(h, ch_s) for h in ["Year", "Age", "Your Annual\nContrib", "Index Fund\nBalance (gross)", "IUL Net\nCash Value", "IUL Death\nBenefit"]]
acc_rows = [acc_hdr]
IUL_ACC = {
    1:  (None, 5_750_000),
    5:  (None, 6_381_029),
    10: (3_326_633, 8_326_633),
    15: (4_680_134, 8_326_633),
    20: (1_348_332, 3_088_987),
}
for yr, age_contrib in [(1,"+$100k"),(5,"+$100k"),(10,"+$100k"),(15,"$0"),(20,"$0")]:
    age = START_AGE + yr
    fv  = fund_snapshots.get(yr, 0)
    iUL_cv, iUL_db = IUL_ACC.get(yr, (None, None))
    acc_rows.append([
        cv(str(yr), cc_s),
        cv(str(age), cc_s),
        p(age_contrib, S("ac", fontSize=8, textColor=GREEN if "100" in age_contrib else GRAY,
                          fontName="Helvetica-Bold" if "100" in age_contrib else "Helvetica",
                          alignment=TA_CENTER)),
        cv(fmt(fv), cg_s if fv > 0 else cv_s),
        cv(fmt(iUL_cv) if iUL_cv else "—", cv_s),
        cv(fmt(iUL_db) if iUL_db else "—", cv_s),
    ])

acc_t = Table(acc_rows, colWidths=[0.55*inch, 0.55*inch, 1.2*inch, 1.7*inch, 1.5*inch, 1.5*inch])
acc_t.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(-1,0), NAVY),
    ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LIGHT]),
    ("TOPPADDING",    (0,0),(-1,-1), 5),
    ("BOTTOMPADDING", (0,0),(-1,-1), 5),
    ("LEFTPADDING",   (0,0),(-1,-1), 5),
    ("RIGHTPADDING",  (0,0),(-1,-1), 5),
    ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#CCCCCC")),
    ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
]))
story.append(acc_t)

note1 = Table([[p(
    "Note: IUL cash value at Year 20 is $1,348,332 NET after repaying the $5.24M financing loan from the policy's own value. "
    "Gross policy value before repayment was $6,151,469.", small_s)]], colWidths=[7.4*inch])
note1.setStyle(TableStyle([
    ("BACKGROUND", (0,0),(-1,-1), LGOLD),
    ("BOX",        (0,0),(-1,-1), 0.5, GOLD),
    ("TOPPADDING", (0,0),(-1,-1), 4), ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ("LEFTPADDING",(0,0),(-1,-1), 7), ("RIGHTPADDING",  (0,0),(-1,-1), 7),
]))
story.append(Spacer(1, 4))
story.append(note1)
story.append(Spacer(1, 9))

# ── WITHDRAWAL PHASE TABLE ────────────────────────────────────────────────────
story.append(p("WITHDRAWAL PHASE — $150K/YR STARTING YEAR 21  (key milestones)", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1, 4))

wdraw_hdr = [p(h, ch_s) for h in [
    "Year", "Age",
    "Cumulative\n$150k Rcvd",
    "Index Fund\nBalance",
    "IUL Cash\nValue (Net)",
    "IUL Death\nBenefit",
]]
wdraw_rows = [wdraw_hdr]

milestones = [21, 25, 30, 35, 37, 40, 43, 50]
for yr in milestones:
    age = START_AGE + yr
    fv  = fund_snapshots.get(yr, 0)
    cum = (yr - 20) * 150_000
    iUL_cv, iUL_db = IUL.get(yr, (None, None))
    hi = yr == 37  # age 80 highlight

    row_style = cg_s if not hi else S("hi", fontSize=8, textColor=NAVY,
                                       fontName="Helvetica-Bold", alignment=TA_RIGHT)
    wdraw_rows.append([
        cv(str(yr), S("yrc", fontSize=8, textColor=NAVY if hi else GRAY,
                       fontName="Helvetica-Bold" if hi else "Helvetica", alignment=TA_CENTER)),
        cv(str(age) + (" ←AGE 80" if hi else ""),
           S("agc", fontSize=8, textColor=NAVY if hi else GRAY,
             fontName="Helvetica-Bold" if hi else "Helvetica", alignment=TA_CENTER)),
        cv(fmt(cum), cv_s),
        cv(fmt(fv), cg_s),
        cv(fmt(iUL_cv) if iUL_cv else "—", cg_s if iUL_cv else cv_s),
        cv(fmt(iUL_db) if iUL_db else "—", cv_s),
    ])

wd_t = Table(wdraw_rows, colWidths=[0.55*inch, 1.05*inch, 1.35*inch, 1.55*inch, 1.45*inch, 1.45*inch])
wd_t.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(-1,0), NAVY),
    ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LIGHT]),
    ("BACKGROUND",    (0,4),(-1,4), colors.HexColor("#D6E4F0")),  # age 80 row (index 4 = yr 37)
    ("TOPPADDING",    (0,0),(-1,-1), 5),
    ("BOTTOMPADDING", (0,0),(-1,-1), 5),
    ("LEFTPADDING",   (0,0),(-1,-1), 5),
    ("RIGHTPADDING",  (0,0),(-1,-1), 5),
    ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#CCCCCC")),
    ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
]))
story.append(wd_t)
story.append(Spacer(1, 8))

# ── AGE 80 CALLOUT ────────────────────────────────────────────────────────────
fund_80 = fund_snapshots.get(37, 0)
iUL_80_cv, iUL_80_db = IUL.get(37, (0, 0))

box_left = Table([
    [p("INDEX FUND  @  AGE 80", S("bhl", fontSize=9, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER))],
    [p(fmt(fund_80), S("bvl", fontSize=18, fontName="Helvetica-Bold", textColor=GREEN, alignment=TA_CENTER))],
    [p("Remaining balance (gross, no tax assumed)\nAfter $150k/yr for 17 years\nYou invested $1,000,000 (years 1–10 only)", small_s)],
], colWidths=[3.55*inch])
box_left.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(-1,0), NAVY),
    ("BACKGROUND",    (0,1),(-1,-1), LGRN),
    ("BOX",           (0,0),(-1,-1), 1.2, NAVY),
    ("TOPPADDING",    (0,0),(-1,-1), 7),
    ("BOTTOMPADDING", (0,0),(-1,-1), 7),
    ("LEFTPADDING",   (0,0),(-1,-1), 8),
    ("RIGHTPADDING",  (0,0),(-1,-1), 8),
    ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
]))

box_right = Table([
    [p("IUL  @  AGE 80", S("bhr", fontSize=9, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER))],
    [p(fmt(iUL_80_cv), S("bvr", fontSize=18, fontName="Helvetica-Bold", textColor=GREEN, alignment=TA_CENTER))],
    [p(f"Cash value  +  Death benefit to heirs: {fmt(iUL_80_db)} (tax-free)\nYou invested $0 — lender paid everything", small_s)],
], colWidths=[3.55*inch])
box_right.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(-1,0), GREEN),
    ("BACKGROUND",    (0,1),(-1,-1), LGRN),
    ("BOX",           (0,0),(-1,-1), 1.2, GREEN),
    ("TOPPADDING",    (0,0),(-1,-1), 7),
    ("BOTTOMPADDING", (0,0),(-1,-1), 7),
    ("LEFTPADDING",   (0,0),(-1,-1), 8),
    ("RIGHTPADDING",  (0,0),(-1,-1), 8),
    ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
]))

callout = Table([[box_left, Spacer(0.3*inch, 1), box_right]], colWidths=[3.55*inch, 0.3*inch, 3.55*inch])
callout.setStyle(TableStyle([
    ("VALIGN",        (0,0),(-1,-1), "TOP"),
    ("TOPPADDING",    (0,0),(-1,-1), 0),
    ("BOTTOMPADDING", (0,0),(-1,-1), 0),
    ("LEFTPADDING",   (0,0),(-1,-1), 0),
    ("RIGHTPADDING",  (0,0),(-1,-1), 0),
]))
story.append(callout)
story.append(Spacer(1, 8))

# ── BOTTOM LINE ───────────────────────────────────────────────────────────────
bl = Table([[p(
    "THE BOTTOM LINE:  Putting $100k/yr for 10 years into an S&P 500 index fund at 9%/yr and withdrawing $150k/yr from age 64 "
    "leaves you with ~$11.4M at age 80 — the fund grows faster than you draw.  "
    "The IUL leaves ~$2.6M cash value at age 80 (plus $3.6M death benefit tax-free to heirs) while costing you $0.  "
    "Trade-off: index fund builds more liquid wealth if you can invest $1M of your own money; "
    "IUL wins on zero cash outlay, tax-free income, and a guaranteed death benefit.",
    S("bl", fontSize=7.5, fontName="Helvetica", textColor=NAVY, leading=11))]], colWidths=[7.4*inch])
bl.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(-1,-1), LIGHT),
    ("BOX",           (0,0),(-1,-1), 1, NAVY),
    ("TOPPADDING",    (0,0),(-1,-1), 7),
    ("BOTTOMPADDING", (0,0),(-1,-1), 7),
    ("LEFTPADDING",   (0,0),(-1,-1), 8),
    ("RIGHTPADDING",  (0,0),(-1,-1), 8),
]))
story.append(bl)
story.append(Spacer(1, 8))

# ── HONEST SUMMARY ────────────────────────────────────────────────────────────
story.append(p("THE HONEST ANSWER", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1, 5))

honest_rows = [
    [p("", ch_s), p("Index Fund", ch_s), p("IUL", ch_s)],
    [p("Your cash invested", body_s),
     p("$1,000,000  (real money, from your pocket, yrs 1–10)", body_s),
     p("$0  (lender pays everything — you never write a check)", body_s)],
    [p("Balance at age 80", body_s),
     p("~$11.4M  (gross, before any tax)", body_s),
     p("~$2.6M cash value", body_s)],
    [p("Tax on withdrawals", body_s),
     p("Yes — capital gains tax owed every year you withdraw", body_s),
     p("No — policy loans are tax-free income", body_s)],
    [p("Death benefit\nto heirs", body_s),
     p("Whatever is in your account\n(taxable as part of estate)", body_s),
     p("$3.6M at age 80, growing —\nincome-tax-free to your heirs", body_s)],
]
ht = Table(honest_rows, colWidths=[1.6*inch, 2.9*inch, 2.9*inch])
ht.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(-1,0), NAVY),
    ("BACKGROUND",    (0,1),(0,-1), colors.HexColor("#D6E4F0")),
    ("ROWBACKGROUNDS",(0,1),(-1,-1), [LIGHT, WHITE]),
    ("TOPPADDING",    (0,0),(-1,-1), 5),
    ("BOTTOMPADDING", (0,0),(-1,-1), 5),
    ("LEFTPADDING",   (0,0),(-1,-1), 7),
    ("RIGHTPADDING",  (0,0),(-1,-1), 7),
    ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#CCCCCC")),
    ("VALIGN",        (0,0),(-1,-1), "TOP"),
]))
story.append(ht)
story.append(Spacer(1, 7))

# Key insight boxes
insight_left = Table([
    [p("INDEX FUND WINS IF...", S("ihl", fontSize=8, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER))],
    [p(
        "• You have $1M available to invest\n"
        "• You are comfortable with market volatility\n"
        "• You want maximum liquid wealth\n"
        "• You don't need a death benefit\n"
        "• You can manage the annual tax drag on withdrawals",
        S("ibl", fontSize=7.5, fontName="Helvetica", textColor=colors.HexColor("#222222"), leading=12))],
], colWidths=[3.55*inch])
insight_left.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(-1,0), NAVY),
    ("BACKGROUND",    (0,1),(-1,-1), LIGHT),
    ("BOX",           (0,0),(-1,-1), 1, NAVY),
    ("TOPPADDING",    (0,0),(-1,-1), 6),
    ("BOTTOMPADDING", (0,0),(-1,-1), 6),
    ("LEFTPADDING",   (0,0),(-1,-1), 8),
    ("RIGHTPADDING",  (0,0),(-1,-1), 8),
    ("VALIGN",        (0,0),(-1,-1), "TOP"),
]))

insight_right = Table([
    [p("IUL WINS IF...", S("ihr", fontSize=8, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER))],
    [p(
        "• You want to invest $0 of your own money\n"
        "• You want tax-free retirement income\n"
        "• You want a large tax-free death benefit for heirs\n"
        "• You have better uses for that $1M\n"
        "• You want protection from market downturns (0% floor)",
        S("ibr", fontSize=7.5, fontName="Helvetica", textColor=colors.HexColor("#222222"), leading=12))],
], colWidths=[3.55*inch])
insight_right.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(-1,0), GREEN),
    ("BACKGROUND",    (0,1),(-1,-1), LGRN),
    ("BOX",           (0,0),(-1,-1), 1, GREEN),
    ("TOPPADDING",    (0,0),(-1,-1), 6),
    ("BOTTOMPADDING", (0,0),(-1,-1), 6),
    ("LEFTPADDING",   (0,0),(-1,-1), 8),
    ("RIGHTPADDING",  (0,0),(-1,-1), 8),
    ("VALIGN",        (0,0),(-1,-1), "TOP"),
]))

insight_tbl = Table([[insight_left, Spacer(0.3*inch,1), insight_right]],
                    colWidths=[3.55*inch, 0.3*inch, 3.55*inch])
insight_tbl.setStyle(TableStyle([
    ("VALIGN",        (0,0),(-1,-1), "TOP"),
    ("TOPPADDING",    (0,0),(-1,-1), 0),
    ("BOTTOMPADDING", (0,0),(-1,-1), 0),
    ("LEFTPADDING",   (0,0),(-1,-1), 0),
    ("RIGHTPADDING",  (0,0),(-1,-1), 0),
]))
story.append(insight_tbl)
story.append(Spacer(1, 7))

# Final one-liner
final = Table([[p(
    "THE REAL QUESTION IS NOT 'which grows more' — it is: "
    "would you rather put $1,000,000 of your own money to work in the market, "
    "or put $0 in and use the bank's money while keeping your $1M free for other opportunities?",
    S("fin", fontSize=8, fontName="Helvetica-Bold", textColor=NAVY, leading=12, alignment=TA_CENTER))
]], colWidths=[7.4*inch])
final.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(-1,-1), LGOLD),
    ("BOX",           (0,0),(-1,-1), 1.2, GOLD),
    ("TOPPADDING",    (0,0),(-1,-1), 8),
    ("BOTTOMPADDING", (0,0),(-1,-1), 8),
    ("LEFTPADDING",   (0,0),(-1,-1), 10),
    ("RIGHTPADDING",  (0,0),(-1,-1), 10),
]))
story.append(final)
story.append(Spacer(1, 5))

story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY))
story.append(Spacer(1, 3))
story.append(p(
    "Index fund projections are hypothetical using 9% gross annual return with no tax or fee assumptions. "
    "IUL values sourced from Lincoln Financial illustration dated March 10, 2026 at a 7.19% non-guaranteed credited rate. "
    "Actual results will vary. Not investment or tax advice.",
    small_s))

doc.build(story)
print(f"PDF written -> {OUTPUT}")
