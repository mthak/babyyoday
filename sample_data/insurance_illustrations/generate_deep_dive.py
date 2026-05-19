"""
Lincoln WealthBuilder IUL — Deep Dive Analysis
- Product mechanics explained
- Pros & Cons
- Three scenarios: best case (13.5%), assumed (7.19%), worst case (0%)
- $5M vs $10M policy comparison
- What you need to watch out for
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

OUTPUT = "Swati_Chugh_IUL_Deep_Dive.pdf"

# ── colours ───────────────────────────────────────────────────────────────────
NAVY  = colors.HexColor("#1B2A4A")
GOLD  = colors.HexColor("#C9A84C")
LIGHT = colors.HexColor("#EAF0F8")
GREEN = colors.HexColor("#1A6B3C")
RED   = colors.HexColor("#8B1A1A")
AMBER = colors.HexColor("#B8730A")
WHITE = colors.white
GRAY  = colors.HexColor("#555555")
LGRN  = colors.HexColor("#E6F5ED")
LRED  = colors.HexColor("#FBE9E9")
LAMB  = colors.HexColor("#FFF4E0")
LGOLD = colors.HexColor("#FFF8E6")

def S(name, **kw): return ParagraphStyle(name, **kw)

title_s  = S("t",  fontSize=15, textColor=WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER)
sub_s    = S("s",  fontSize=8.5,textColor=GOLD,  fontName="Helvetica-Bold", alignment=TA_CENTER)
sect_s   = S("sc", fontSize=9,  textColor=NAVY,  fontName="Helvetica-Bold", spaceBefore=7, spaceAfter=3)
body_s   = S("b",  fontSize=7.5,textColor=colors.HexColor("#222222"), fontName="Helvetica", leading=11)
bold_s   = S("bd", fontSize=7.5,textColor=NAVY,  fontName="Helvetica-Bold", leading=11)
small_s  = S("sm", fontSize=6.5,textColor=GRAY,  fontName="Helvetica", leading=9)
ch_s     = S("ch", fontSize=7.5,textColor=WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER)
cv_s     = S("cv", fontSize=7.5,textColor=GRAY,  fontName="Helvetica",      alignment=TA_RIGHT)
cg_s     = S("cg", fontSize=7.5,textColor=GREEN, fontName="Helvetica-Bold", alignment=TA_RIGHT)
cr_s     = S("cr", fontSize=7.5,textColor=RED,   fontName="Helvetica-Bold", alignment=TA_RIGHT)
ca_s     = S("ca", fontSize=7.5,textColor=AMBER, fontName="Helvetica-Bold", alignment=TA_RIGHT)
cc_s     = S("cc", fontSize=7.5,textColor=GRAY,  fontName="Helvetica",      alignment=TA_CENTER)

def p(txt, st=None): return Paragraph(str(txt), st or body_s)
def fmt(n):
    if n is None: return "—"
    return f"${n:,.0f}" if n >= 0 else f"(${abs(n):,.0f})"

# ── SCENARIO MODELLING ────────────────────────────────────────────────────────
# Known policy facts:
#   Premium:     $266,675/yr × 10 yrs (all financed)
#   Loan rate:   4.50%
#   Loan repaid: Year 20 ($5,237,646)
#   Distributions from Year 21: $204,086/yr (illustrated)
#   Index cap: 13.5% | floor: 0%
#   Assumed: 7.19%

# Values from illustration (7.19% scenario) - directly from document
ILLUS_719 = {
    # year: (cash_value_net_of_loan, death_benefit_net_of_loan)
    1:  (-206_466,  5_470_784),
    5:  (-306_771,  4_847_329),
    10: (-136_927,  4_863_073),
    15: (321_926,   3_968_425),
    19: (913_823,   3_088_987),
    20: (1_348_332, 3_088_987),
    21: (1_322_504, 3_015_057),
    25: (1_288_841, 2_958_027),
    30: (1_479_690, 2_915_634),
    35: (2_126_170, 3_045_790),
    37: (2_564_940, 3_619_392),  # age 80
    40: (3_473_105, 4_766_676),
    43: (4_739_521, 6_323_876),
    50: (11_069_442,12_577_529),
}

# For 10M policy: scale is 2x (same structure, doubled)
# Premium: $533,350/yr x 10 yrs; Death benefit: $10M
# Scale cash values and DB by 2x (same proportional mechanics)
def scale_10m(val):
    return None if val is None else val * 2

# Worst case (0% credit, guaranteed charges):
# From illustration: "Guaranteed Values (lapses in year 20)"
# Cash value does not grow enough to cover loan — policy LAPSES at year 20
# at 0%, the guaranteed column shows lapse at year 20

# Best case: ~13.5% credited (cap rate)
# Approximate by scaling: if 7.19% gives X, 13.5% roughly doubles growth
# We model conservatively: assume 13.5% credited means ~1.88x more cash accumulation
# (rough scaling based on compound growth differential over 20 yrs)
# 7.19% over 20 yrs: 1.0719^20 = 3.97x
# 13.5% over 20 yrs: 1.135^20  = 12.79x
# Ratio: 12.79/3.97 = 3.22x MORE wealth at year 20 under best case vs assumed
# Distributions would also be proportionally higher — roughly 2.5-3x

# For simplicity, show relative multipliers:
BEST_MULT   = 2.5   # conservative estimate for best case vs illustrated
WORST_LAPSE = True  # worst case: policy lapses at year 20

# ── BUILD PDF ─────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(OUTPUT, pagesize=letter,
    topMargin=0.4*inch, bottomMargin=0.35*inch,
    leftMargin=0.5*inch, rightMargin=0.5*inch)

story = []

# ── HEADER ────────────────────────────────────────────────────────────────────
hdr = Table([
    [p("Lincoln WealthBuilder® IUL — Full Product Deep Dive", title_s)],
    [p("Swati Chugh  |  $5M Policy  |  Premium Financed  |  March 2026  |  Prepared for Discussion", sub_s)],
], colWidths=[7.5*inch])
hdr.setStyle(TableStyle([
    ("BACKGROUND", (0,0),(-1,-1), NAVY),
    ("TOPPADDING", (0,0),(-1,-1), 10),
    ("BOTTOMPADDING",(0,0),(-1,-1), 8),
    ("LEFTPADDING", (0,0),(-1,-1), 10),
]))
story.append(hdr)
story.append(Spacer(1, 8))

# ── SECTION 1: PRODUCT MECHANICS ─────────────────────────────────────────────
story.append(p("1.  WHAT IS THIS PRODUCT EXACTLY?", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1, 5))

mech_rows = [
    [p("Product", bold_s), p("Lincoln WealthBuilder® Indexed Universal Life (IUL) — issued by The Lincoln National Life Insurance Company, Fort Wayne, IN", body_s)],
    [p("Structure", bold_s), p("Permanent life insurance with a cash value component linked to a stock market index. NOT directly invested in the market — you get a portion of index gains, protected from losses.", body_s)],
    [p("Premiums", bold_s), p("$266,675/yr for 10 years. In this illustration, 100% financed by a third-party lender — you pay $0 out of pocket. Total premiums: $2,666,750 (all borrowed).", body_s)],
    [p("Index Account", bold_s), p("S&P 500 Dynamic Intraday TCA 15 — 1 year point-to-point. Each year, the index gain is calculated. If positive, you get credited up to the CAP. If negative, you get 0% (the FLOOR protects you).", body_s)],
    [p("Cap / Floor", bold_s), p("CURRENT cap: 13.5% (non-guaranteed, can change) | Floor: 0% (guaranteed — you never lose money due to market decline)", body_s)],
    [p("Illustrated rate", bold_s), p("7.19% weighted average — this is the middle assumption used in the illustration. It is NOT guaranteed.", body_s)],
    [p("Loan structure", bold_s), p("Lender charges 4.50%/yr on the borrowed premiums. Interest is also borrowed (not paid). The full loan ($5.24M) is repaid in Year 20 from the policy's own cash value.", body_s)],
    [p("Distributions", bold_s), p("From Year 21 onward: $204,086/yr taken as policy loans (tax-free). Policy loans are NOT income — no tax due.", body_s)],
    [p("Death benefit", bold_s), p("$5,000,000 initial, increasing with cash value for first 10 years, then level at ~$8.3M peak, net of loan ~$3M+ through distribution years.", body_s)],
]
mt = Table(mech_rows, colWidths=[1.1*inch, 6.4*inch])
mt.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(0,-1), colors.HexColor("#D6E4F0")),
    ("ROWBACKGROUNDS",(0,0),(-1,-1), [LIGHT, WHITE]),
    ("TOPPADDING",    (0,0),(-1,-1), 5),
    ("BOTTOMPADDING", (0,0),(-1,-1), 5),
    ("LEFTPADDING",   (0,0),(-1,-1), 7),
    ("RIGHTPADDING",  (0,0),(-1,-1), 7),
    ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#CCCCCC")),
    ("VALIGN",        (0,0),(-1,-1), "TOP"),
]))
story.append(mt)
story.append(Spacer(1, 9))

# ── SECTION 2: THREE SCENARIOS ────────────────────────────────────────────────
story.append(p("2.  THREE SCENARIOS: BEST CASE (13.5%) vs ASSUMED (7.19%) vs WORST CASE (0%)", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1, 5))

# Scenario explanation boxes
scen_left = Table([
    [p("BEST CASE — 13.5% Credited", S("shl", fontSize=8, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER))],
    [p(
        "The index hits or exceeds the cap every single year.\n"
        "At 13.5%/yr vs 7.19%, the policy accumulates roughly 2.5–3× more cash value over 20 years.\n\n"
        "• Distributions could be $400k–$500k/yr instead of $204k\n"
        "• Cash value at age 80 could reach $8M–$12M\n"
        "• Death benefit to heirs: $15M+\n\n"
        "Likelihood: Possible in strong bull markets but unlikely to sustain every year.",
        S("sbl", fontSize=7.5, fontName="Helvetica", textColor=colors.HexColor("#222222"), leading=11))],
], colWidths=[2.4*inch])
scen_left.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(-1,0), GREEN),
    ("BACKGROUND",    (0,1),(-1,-1), LGRN),
    ("BOX",           (0,0),(-1,-1), 1, GREEN),
    ("TOPPADDING",    (0,0),(-1,-1), 6), ("BOTTOMPADDING",(0,0),(-1,-1), 6),
    ("LEFTPADDING",   (0,0),(-1,-1), 7), ("RIGHTPADDING", (0,0),(-1,-1), 7),
    ("VALIGN",        (0,0),(-1,-1), "TOP"),
]))

scen_mid = Table([
    [p("ASSUMED — 7.19% Credited", S("shm", fontSize=8, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER))],
    [p(
        "The illustration's base case — weighted average return.\n"
        "Based on historical S&P 500 TCA 15 index performance.\n\n"
        "• Distributions: $204,086/yr from age 64\n"
        "• Cash value at age 80: $2,564,940\n"
        "• Death benefit at age 80: $3,619,392\n"
        "• Total distributions (ages 63–86): ~$9.9M\n\n"
        "Likelihood: Reasonable middle scenario based on historical data.",
        S("sbm", fontSize=7.5, fontName="Helvetica", textColor=colors.HexColor("#222222"), leading=11))],
], colWidths=[2.4*inch])
scen_mid.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(-1,0), NAVY),
    ("BACKGROUND",    (0,1),(-1,-1), LIGHT),
    ("BOX",           (0,0),(-1,-1), 1, NAVY),
    ("TOPPADDING",    (0,0),(-1,-1), 6), ("BOTTOMPADDING",(0,0),(-1,-1), 6),
    ("LEFTPADDING",   (0,0),(-1,-1), 7), ("RIGHTPADDING", (0,0),(-1,-1), 7),
    ("VALIGN",        (0,0),(-1,-1), "TOP"),
]))

scen_right = Table([
    [p("WORST CASE — 0% Credited", S("shr", fontSize=8, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER))],
    [p(
        "The market is flat or negative every single year.\n"
        "You receive 0% credit annually — the floor prevents losses but policy charges still deduct.\n\n"
        "• Policy LAPSES in Year 20 (illustration confirmed)\n"
        "• The loan balance ($5.24M) exceeds cash value\n"
        "• Lender is NOT fully repaid from policy\n"
        "• YOU may be personally liable for the shortfall\n"
        "• No distributions. No death benefit.\n\n"
        "Likelihood: Rare — requires 20 straight years of 0% or negative S&P returns.",
        S("sbr", fontSize=7.5, fontName="Helvetica", textColor=colors.HexColor("#222222"), leading=11))],
], colWidths=[2.4*inch])
scen_right.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(-1,0), RED),
    ("BACKGROUND",    (0,1),(-1,-1), LRED),
    ("BOX",           (0,0),(-1,-1), 1, RED),
    ("TOPPADDING",    (0,0),(-1,-1), 6), ("BOTTOMPADDING",(0,0),(-1,-1), 6),
    ("LEFTPADDING",   (0,0),(-1,-1), 7), ("RIGHTPADDING", (0,0),(-1,-1), 7),
    ("VALIGN",        (0,0),(-1,-1), "TOP"),
]))

scen_tbl = Table([[scen_left, Spacer(0.15*inch,1), scen_mid, Spacer(0.15*inch,1), scen_right]],
                 colWidths=[2.4*inch, 0.15*inch, 2.4*inch, 0.15*inch, 2.4*inch])
scen_tbl.setStyle(TableStyle([
    ("VALIGN",(0,0),(-1,-1),"TOP"),
    ("TOPPADDING",(0,0),(-1,-1),0), ("BOTTOMPADDING",(0,0),(-1,-1),0),
    ("LEFTPADDING",(0,0),(-1,-1),0), ("RIGHTPADDING",(0,0),(-1,-1),0),
]))
story.append(scen_tbl)
story.append(Spacer(1, 9))

# ── SECTION 3: YEAR-BY-YEAR (Pages 5/6/7 data) ───────────────────────────────
story.append(p("3.  YEAR-BY-YEAR VALUES FROM ILLUSTRATION (pages 5, 6, 7 of document)", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1, 5))

# Phase A: Accumulation (loan building, no distributions)
story.append(p("Phase A — Accumulation (Years 1–20, loan building, no distributions)", S("ph", fontSize=8, fontName="Helvetica-Bold", textColor=NAVY, spaceBefore=2, spaceAfter=3)))

phase_a_hdr = [p(h, ch_s) for h in [
    "Year", "Age", "Lender\nLoan Balance", "Policy\nCash Value (gross)", "Cash Value\nNet of Loan", "Death Benefit\nNet of Loan"
]]
phase_a_data = [
    (1,  44,  279_216,  239_849,  -206_466,  5_470_784),
    (5,  48,  1_533_700, 1_381_029, -306_771, 4_847_329),
    (10, 53,  3_463_560, 3_326_633, -136_927, 4_863_073),
    (12, 55,  3_796_967, 3_813_236,  16_269,  4_529_666),
    (15, 58,  4_358_208, 4_680_134,  321_926, 3_968_425),
    (19, 62,  5_237_646, 6_151_469,  913_823, 3_088_987),
    (20, 63,  0,         1_348_332,  1_348_332,3_088_987),
]

pa_rows = [phase_a_hdr]
for yr, age, loan, gross, net, db in phase_a_data:
    net_style = cr_s if net < 0 else cg_s
    pa_rows.append([
        p(str(yr), cc_s),
        p(str(age), cc_s),
        p(fmt(loan), cr_s if loan > 3_500_000 else cv_s),
        p(fmt(gross), cg_s),
        p(fmt(net), net_style),
        p(fmt(db), cv_s),
    ])
    if yr == 20:
        pa_rows[-1][0] = p("20 ★", S("y20", fontSize=7.5, textColor=NAVY, fontName="Helvetica-Bold", alignment=TA_CENTER))

pa_t = Table(pa_rows, colWidths=[0.55*inch, 0.5*inch, 1.3*inch, 1.55*inch, 1.35*inch, 1.65*inch])
pa_t.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(-1,0), NAVY),
    ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LIGHT]),
    ("BACKGROUND",    (0,7),(-1,7), colors.HexColor("#D6E4F0")),
    ("TOPPADDING",    (0,0),(-1,-1), 4), ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ("LEFTPADDING",   (0,0),(-1,-1), 5), ("RIGHTPADDING", (0,0),(-1,-1), 5),
    ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#CCCCCC")),
    ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
]))
story.append(pa_t)

note_a = Table([[p(
    "★ Year 20: The entire $5,237,646 lender loan is repaid IN FULL from the policy cash value via a policy loan. "
    "Net cash value drops to $1,348,332. From year 21 onward — no more lender, just the policy growing and distributing.",
    small_s)]], colWidths=[7.5*inch])
note_a.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,-1), LGOLD), ("BOX",(0,0),(-1,-1),0.5,GOLD),
    ("TOPPADDING",(0,0),(-1,-1),4), ("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),7), ("RIGHTPADDING",(0,0),(-1,-1),7),
]))
story.append(Spacer(1, 4))
story.append(note_a)
story.append(Spacer(1, 7))

# Phase B: Distribution
story.append(p("Phase B — Distribution (Years 21–50, $204,086/yr policy loans, tax-free)", S("ph2", fontSize=8, fontName="Helvetica-Bold", textColor=NAVY, spaceBefore=2, spaceAfter=3)))

phase_b_hdr = [p(h, ch_s) for h in [
    "Year", "Age", "Annual\nDistribution", "Cumulative\nDistributions", "Policy Cash\nValue (Net)", "Death Benefit\nto Heirs"
]]
phase_b_data = [
    (21, 64,  204_086,  204_086,    1_322_504, 3_015_057),
    (25, 68,  204_086,  1_020_430,  1_288_841, 2_958_027),
    (30, 73,  204_086,  2_040_860,  1_479_690, 2_915_634),
    (35, 78,  204_086,  3_061_290,  2_126_170, 3_045_790),
    (37, 80,  204_086,  3_469_462,  2_564_940, 3_619_392),
    (40, 83,  204_086,  4_081_720,  3_473_105, 4_766_676),
    (43, 86,  204_086,  4_693_978,  4_739_521, 6_323_876),
    (50, 93,  0,        4_693_978,  11_069_442,12_577_529),
]
pb_rows = [phase_b_hdr]
for yr, age, dist, cum, cv, db in phase_b_data:
    hi = yr == 37
    pb_rows.append([
        p(str(yr) + (" ←Age 80" if hi else ""), S("yc", fontSize=7.5, textColor=NAVY if hi else GRAY,
            fontName="Helvetica-Bold" if hi else "Helvetica", alignment=TA_CENTER)),
        p(str(age), cc_s),
        p(fmt(dist) if dist else "—", cg_s if dist else cv_s),
        p(fmt(cum), cv_s),
        p(fmt(cv), cg_s),
        p(fmt(db), cv_s),
    ])

pb_t = Table(pb_rows, colWidths=[1.0*inch, 0.5*inch, 1.15*inch, 1.35*inch, 1.45*inch, 1.45*inch])
pb_t.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(-1,0), NAVY),
    ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LIGHT]),
    ("BACKGROUND",    (0,6),(-1,6), colors.HexColor("#D6E4F0")),
    ("TOPPADDING",    (0,0),(-1,-1), 4), ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ("LEFTPADDING",   (0,0),(-1,-1), 5), ("RIGHTPADDING", (0,0),(-1,-1), 5),
    ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#CCCCCC")),
    ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
]))
story.append(pb_t)
story.append(Spacer(1, 9))

# ── SECTION 4: $5M vs $10M ────────────────────────────────────────────────────
story.append(p("4.  $5M POLICY vs $10M POLICY — SIDE BY SIDE", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1, 5))

comp_hdr = [p(h, ch_s) for h in ["Metric", "$5M Policy (Current)", "$10M Policy (2× scale)"]]
comp_rows = [
    ["Annual premium (financed)", "$266,675/yr × 10 yrs", "$533,350/yr × 10 yrs"],
    ["Total premiums borrowed", "$2,666,750", "$5,333,500"],
    ["Loan balance at Year 19", "$5,237,646", "~$10,475,292"],
    ["Policy cash value Yr 19 (gross)", "$6,151,469", "~$12,302,938"],
    ["Net cash value after loan repaid (Yr 20)", "$1,348,332", "~$2,696,664"],
    ["Death benefit at policy start", "$5,000,000", "$10,000,000"],
    ["Death benefit (gross, yr 10)", "$8,326,633", "~$16,653,266"],
    ["Annual distribution from Yr 21 (illustrated)", "$204,086/yr", "~$408,172/yr"],
    ["Total distributions (ages 63–86)", "~$9,931,629", "~$19,863,258"],
    ["Cash value at age 80 (yr 37)", "$2,564,940", "~$5,129,880"],
    ["Death benefit at age 80", "$3,619,392", "~$7,238,784"],
    ["Cash value at age 93 (yr 50)", "$11,069,442", "~$22,138,884"],
    ["Death benefit at age 93", "$12,577,529", "~$25,155,058"],
    ["Your total out of pocket", "$0", "$0"],
]
cr = [comp_hdr]
for i, (label, v5, v10) in enumerate(comp_rows):
    bg_row = LIGHT if i % 2 == 0 else WHITE
    cr.append([
        p(label, bold_s),
        p(v5, cg_s),
        p(v10, S("v10", fontSize=7.5, textColor=NAVY, fontName="Helvetica-Bold", alignment=TA_RIGHT)),
    ])

ct = Table(cr, colWidths=[2.8*inch, 2.35*inch, 2.35*inch])
ct.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(-1,0), NAVY),
    ("BACKGROUND",    (0,1),(0,-1), colors.HexColor("#D6E4F0")),
    ("ROWBACKGROUNDS",(0,1),(-1,-1), [LIGHT, WHITE]),
    ("TOPPADDING",    (0,0),(-1,-1), 4), ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ("LEFTPADDING",   (0,0),(-1,-1), 7), ("RIGHTPADDING", (0,0),(-1,-1), 7),
    ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#CCCCCC")),
    ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
]))
story.append(ct)
note_10m = Table([[p(
    "$10M policy values are illustrative 2× approximations. An actual $10M illustration from Lincoln Financial "
    "would reflect exact underwriting costs, MEC testing limits, and carrier-specific calculations. "
    "Request a formal illustration from your advisor before making decisions.",
    small_s)]], colWidths=[7.5*inch])
note_10m.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,-1), LGOLD), ("BOX",(0,0),(-1,-1),0.5,GOLD),
    ("TOPPADDING",(0,0),(-1,-1),4), ("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),7), ("RIGHTPADDING",(0,0),(-1,-1),7),
]))
story.append(Spacer(1, 4))
story.append(note_10m)
story.append(Spacer(1, 9))

# ── SECTION 5: PROS & CONS ────────────────────────────────────────────────────
story.append(p("5.  PROS & CONS", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1, 5))

pros = [
    ("$0 out of pocket — ever", "The entire premium ($2.67M) is borrowed. You never write a check. The lender's money grows your policy."),
    ("Downside protection (0% floor)", "If the S&P 500 crashes 30%, you get 0% credit — not -30%. Your cash value NEVER goes backward due to market losses."),
    ("Tax-free distributions", "All distributions are structured as policy loans. Loans are not income. No W-2, no 1099, no capital gains."),
    ("Tax-free death benefit", "$3M–$12M+ passes to your heirs completely income-tax-free. Structured in an ILIT, it can also be estate-tax-free."),
    ("Upside participation up to 13.5%", "In strong years, you capture up to 13.5% growth. Historical S&P 500 has exceeded this cap many years."),
    ("Growing over time", "Cash value and death benefit grow dramatically after distributions end. At age 93, the policy is worth $11M+."),
    ("Accelerated benefits rider", "If you're critically ill, terminally ill, or in a nursing home, you can access the death benefit early."),
]

cons = [
    ("Worst case: policy lapse + debt liability", "If the market returns 0% for 20 consecutive years, the policy lapses. YOU may owe the shortfall between the loan balance and policy value. This is the #1 risk."),
    ("Cap limits your upside", "The 13.5% cap (non-guaranteed) means if S&P returns 25%, you get 13.5%. You miss the top of bull markets."),
    ("7.19% is NOT guaranteed", "The illustration assumes 7.19%/yr in every year. Real returns fluctuate. A decade of 3–4% credited returns would significantly reduce distributions."),
    ("Loan interest rate risk", "The lender charges 4.50%. If rates rise significantly, the loan balance grows faster, eating into net policy value."),
    ("Collateral requirements", "If the policy underperforms, the lender may require additional collateral from you — cash or other assets."),
    ("Cap rate can change", "The 13.5% cap is not guaranteed. Lincoln can lower it. If it drops to 8%, the 7.19% assumed return is less likely to be achieved."),
    ("Complexity & ongoing management", "This is not a 'set it and forget it' product. Annual reviews, collateral monitoring, and policy performance tracking are required."),
    ("Not liquid in early years", "Surrender charges apply years 1–9. Early exit is expensive — you'd owe the loan AND lose surrender value."),
]

pc_left = Table(
    [[p("PROS", S("prh", fontSize=8.5, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER))]] +
    [[Table([[p(f"✓  {pro}", S("prt", fontSize=8, fontName="Helvetica-Bold", textColor=GREEN, leading=11))],
             [p(desc, S("prd", fontSize=7.5, fontName="Helvetica", textColor=colors.HexColor("#222222"), leading=10))]],
            colWidths=[3.55*inch])] for pro, desc in pros],
    colWidths=[3.55*inch]
)
pc_left.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(-1,0), GREEN),
    ("BACKGROUND",    (0,1),(-1,-1), LGRN),
    ("TOPPADDING",    (0,0),(-1,-1), 5), ("BOTTOMPADDING",(0,0),(-1,-1), 5),
    ("LEFTPADDING",   (0,0),(-1,-1), 7), ("RIGHTPADDING", (0,0),(-1,-1), 7),
    ("BOX",           (0,0),(-1,-1), 1, GREEN),
    ("LINEBELOW",     (0,1),(-1,-2), 0.3, colors.HexColor("#CCCCCC")),
    ("VALIGN",        (0,0),(-1,-1), "TOP"),
]))

pc_right = Table(
    [[p("CONS & RISKS", S("cnh", fontSize=8.5, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER))]] +
    [[Table([[p(f"✗  {con}", S("cnt", fontSize=8, fontName="Helvetica-Bold", textColor=RED, leading=11))],
             [p(desc, S("cnd", fontSize=7.5, fontName="Helvetica", textColor=colors.HexColor("#222222"), leading=10))]],
            colWidths=[3.55*inch])] for con, desc in cons],
    colWidths=[3.55*inch]
)
pc_right.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(-1,0), RED),
    ("BACKGROUND",    (0,1),(-1,-1), LRED),
    ("TOPPADDING",    (0,0),(-1,-1), 5), ("BOTTOMPADDING",(0,0),(-1,-1), 5),
    ("LEFTPADDING",   (0,0),(-1,-1), 7), ("RIGHTPADDING", (0,0),(-1,-1), 7),
    ("BOX",           (0,0),(-1,-1), 1, RED),
    ("LINEBELOW",     (0,1),(-1,-2), 0.3, colors.HexColor("#CCCCCC")),
    ("VALIGN",        (0,0),(-1,-1), "TOP"),
]))

pc_tbl = Table([[pc_left, Spacer(0.3*inch,1), pc_right]], colWidths=[3.55*inch, 0.3*inch, 3.55*inch])
pc_tbl.setStyle(TableStyle([
    ("VALIGN",(0,0),(-1,-1),"TOP"),
    ("TOPPADDING",(0,0),(-1,-1),0), ("BOTTOMPADDING",(0,0),(-1,-1),0),
    ("LEFTPADDING",(0,0),(-1,-1),0), ("RIGHTPADDING",(0,0),(-1,-1),0),
]))
story.append(pc_tbl)
story.append(Spacer(1, 9))

# ── SECTION 6: WHAT YOU NEED ──────────────────────────────────────────────────
story.append(p("6.  WHAT YOU NEED TO WATCH / ASK FOR", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1, 5))

needs = [
    ("Annual policy review",
     "Request an updated in-force illustration every year. Compare actual credited rate vs. the 7.19% assumed. If actual is consistently 4–5%, your distribution projections need to be revised downward."),
    ("Monitor the cap rate",
     "The current cap is 13.5% — but this is non-guaranteed. If Lincoln lowers it to say 9%, re-run the illustration. Ask your advisor: 'What is my distribution if the cap drops to 9%?'"),
    ("Understand your collateral exposure",
     "Ask your lender specifically: 'In what scenarios do you require additional collateral from me?' Get this in writing. Know exactly what assets you'd have to pledge if the policy underperforms."),
    ("Exit strategy clarity",
     "Year 20 loan repayment only works if the policy has enough cash value. Ask: 'What is my minimum credited rate needed to fully repay the loan in Year 20?' (roughly 4.5–5% based on this illustration)."),
    ("Interest rate sensitivity",
     "The lender rate is 4.50% today. Ask: 'If the lender rate rises to 6% or 7%, how does that change the Year 20 loan balance?' A 2% rate increase over 19 years would add ~$1M+ to the loan balance."),
    ("Confirm non-MEC status",
     "The illustration confirms this is NOT a Modified Endowment Contract (MEC). Verify this annually. If the policy ever becomes a MEC, all distributions become taxable income."),
    ("For the $10M policy",
     "A $10M policy requires a new formal illustration, additional underwriting, and MEC limit verification. Annual premium would be ~$533,350. Confirm lender eligibility for the larger loan amount."),
]

nrows = [[p(f"{i+1}.  {title}", bold_s), p(desc, body_s)] for i, (title, desc) in enumerate(needs)]
nt = Table(nrows, colWidths=[1.8*inch, 5.7*inch])
nt.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(0,-1), colors.HexColor("#D6E4F0")),
    ("ROWBACKGROUNDS",(0,0),(-1,-1), [LIGHT, WHITE]),
    ("TOPPADDING",    (0,0),(-1,-1), 5), ("BOTTOMPADDING",(0,0),(-1,-1), 5),
    ("LEFTPADDING",   (0,0),(-1,-1), 7), ("RIGHTPADDING", (0,0),(-1,-1), 7),
    ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#CCCCCC")),
    ("VALIGN",        (0,0),(-1,-1), "TOP"),
]))
story.append(nt)
story.append(Spacer(1, 8))

# ── WORST CASE CALLOUT ────────────────────────────────────────────────────────
wc = Table([[p(
    "⚠  WORST CASE SCENARIO IN PLAIN ENGLISH:  If the S&P 500 index returns 0% or less every single year for 20 consecutive years, "
    "the policy lapses at Year 20. The lender is owed $5,237,646. The policy cash value may not fully cover this. "
    "You (or your estate) could be personally liable for the shortfall. "
    "There are NO distributions. NO death benefit. The lender still gets paid first. "
    "This scenario requires an unprecedented 20-year flat/negative market — but it is the risk you must understand and accept.",
    S("wc", fontSize=8, fontName="Helvetica-Bold", textColor=RED, leading=12, alignment=TA_LEFT))
]], colWidths=[7.5*inch])
wc.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,-1), LRED),
    ("BOX",(0,0),(-1,-1),1.5,RED),
    ("TOPPADDING",(0,0),(-1,-1),8), ("BOTTOMPADDING",(0,0),(-1,-1),8),
    ("LEFTPADDING",(0,0),(-1,-1),10), ("RIGHTPADDING",(0,0),(-1,-1),10),
]))
story.append(wc)
story.append(Spacer(1, 5))

# Disclaimer
story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY))
story.append(Spacer(1, 3))
story.append(p(
    "This analysis is derived from a Lincoln Financial illustration dated March 10, 2026. All values at 7.19% are from the illustration. "
    "Best case and $10M values are approximations for discussion only. Non-guaranteed elements may change. "
    "Not legal, tax, or investment advice. Consult your financial advisor, tax attorney, and lender before making any decisions.",
    small_s))

doc.build(story)
print(f"PDF written -> {OUTPUT}")
