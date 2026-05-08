"""
John Hancock Accumulation IUL - $10M
One-page summary with cost analysis and worth-it verdict
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT

OUTPUT = "Swati_Chugh_JohnHancock_IUL_10M_Summary.pdf"

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
LBLUE = colors.HexColor("#D6E4F0")
JH    = colors.HexColor("#00704A")   # JH brand green

def S(n, **kw): return ParagraphStyle(n, **kw)
title_s = S("t",  fontSize=15, textColor=WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER)
sub_s   = S("s",  fontSize=8,  textColor=GOLD,  fontName="Helvetica-Bold", alignment=TA_CENTER)
sect_s  = S("sc", fontSize=8.5,textColor=NAVY,  fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=3)
body_s  = S("b",  fontSize=7.5,textColor=colors.HexColor("#222222"), fontName="Helvetica", leading=11)
bold_s  = S("bd", fontSize=7.5,textColor=NAVY,  fontName="Helvetica-Bold", leading=11)
small_s = S("sm", fontSize=6.5,textColor=GRAY,  fontName="Helvetica", leading=9)
ch_s    = S("ch", fontSize=7.5,textColor=WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER)
cv_s    = S("cv", fontSize=7.5,textColor=GRAY,  fontName="Helvetica",      alignment=TA_RIGHT)
cg_s    = S("cg", fontSize=7.5,textColor=GREEN, fontName="Helvetica-Bold", alignment=TA_RIGHT)
cr_s    = S("cr", fontSize=7.5,textColor=RED,   fontName="Helvetica-Bold", alignment=TA_RIGHT)
ca_s    = S("ca", fontSize=7.5,textColor=AMBER, fontName="Helvetica-Bold", alignment=TA_RIGHT)
cc_s    = S("cc", fontSize=7.5,textColor=GRAY,  fontName="Helvetica",      alignment=TA_CENTER)

def p(txt, st=None): return Paragraph(str(txt), st or body_s)
def fmt(n):
    if n is None: return "—"
    return f"${n:,.0f}" if n >= 0 else f"(${abs(n):,.0f})"

doc = SimpleDocTemplate(OUTPUT, pagesize=letter,
    topMargin=0.38*inch, bottomMargin=0.32*inch,
    leftMargin=0.48*inch, rightMargin=0.48*inch)
story = []

# ── HEADER ────────────────────────────────────────────────────────────────────
hdr = Table([
    [p("Swati Chugh — John Hancock Accumulation IUL  |  $10M Death Benefit", title_s)],
    [p("Premium Financed  |  6% Assumed Return  |  Presented April 30, 2026  |  Broker: Nick Burgess, The Burgess Group", sub_s)],
], colWidths=[7.54*inch])
hdr.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),NAVY),
    ("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),7),("LEFTPADDING",(0,0),(-1,-1),10)]))
story.append(hdr)
story.append(Spacer(1,6))

# ── SECTION 1: POLICY AT A GLANCE ─────────────────────────────────────────────
story.append(p("1.  POLICY AT A GLANCE", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,4))

basics_data = [
    [p("Carrier / Product", bold_s), p("John Hancock Life Insurance Co. (USA)  |  Accumulation IUL  |  Form 26AIUL", body_s)],
    [p("Insured", bold_s), p("Swati Chugh  |  Female  |  Age 44  |  California  |  Preferred Non-Smoker  |  Initial Status: Bronze", body_s)],
    [p("Death Benefit", bold_s), p("$10,000,000  |  Option 2 (increasing by cash value) Years 1–20, then Option 1 (level) from Year 21", body_s)],
    [p("Annual Premium", bold_s), p("$689,782/yr (financed by lender)  —  varies by year due to financing structure  |  Total: $6,347,244", body_s)],
    [p("Your out of pocket", bold_s), p("$0  —  Premiums fully financed by third-party lender", S("op",fontSize=7.5,fontName="Helvetica-Bold",textColor=GREEN,leading=11))],
    [p("Assumed return", bold_s), p("6.00% (more conservative than typical — Lincoln used 7.19%)  |  Floor: 0%  |  Guaranteed rate: 0%", body_s)],
    [p("Index accounts", bold_s), p("Base Capped 11.65%  |  Nasdaq 14%  |  Enhanced High Capped 12.65% (+80% multiplier)  |  Select Capped 9.75% (+11% multiplier)  |  Barclays Global MA (uncapped, 165% participation)", body_s)],
    [p("Loan payoff year", bold_s), p("Year 21 (Age 65)  —  policy uses its own cash value to repay the entire financing loan", body_s)],
    [p("Annual distributions", bold_s), p("$200,000/yr starting Year 22 (Age 66)  —  taken as policy loans (tax-free)", body_s)],
    [p("Key riders", bold_s), p("Overloan Protection Rider (prevents lapse)  |  Healthy Engagement / Vitality PLUS (health-linked bonus credits)", body_s)],
    [p("Non-MEC status", bold_s), p("Confirmed — distributions remain tax-free policy loans  |  7-Pay limit: $689,782/yr", body_s)],
]
bt = Table(basics_data, colWidths=[1.2*inch, 6.3*inch])
bt.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(0,-1), LBLUE),
    ("ROWBACKGROUNDS",(0,0),(-1,-1),[LIGHT,WHITE]),
    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"TOP"),
]))
story.append(bt)
story.append(Spacer(1,7))

# ── SECTION 2: COST BREAKDOWN ─────────────────────────────────────────────────
story.append(p("2.  WHAT IT ACTUALLY COSTS (Directly from Annual Account Summary)", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,4))

cost_hdr = [p(h, ch_s) for h in [
    "Policy Yr", "Age", "Annual\nPremium", "Premium\nLoad", "Admin /\nIssue", "COI\n(Insurance)", "TOTAL\nCharges", "Interest\nCredited", "Policy\nValue"
]]
cost_data_rows = [
    (1,45,689782,48285,42026,947,91282,0,598500),
    (2,46,689782,41387,45975,2482,89868,81541,1279955),
    (5,49,118973,7138,53714,4832,65708,189920,3010254),
    (10,54,528939,31736,65991,9652,107403,407432,6484576),
    (11,55,528939,10579,68783,10899,90285,463926,7387155),
    (12,56,528939,10579,71574,12170,94347,523912,8345659),
    (15,59,0,0,68524,16486,85034,617717,9847276),
    (20,64,0,0,25604,26762,52390,832296,13301220),
]
cost_rows = [cost_hdr]
for yr,age,prem,pl,admin,coi,total,interest,pv in cost_data_rows:
    hi20 = yr==20
    cost_rows.append([
        p(str(yr), S("yc",fontSize=7.5,fontName="Helvetica-Bold",textColor=NAVY,alignment=TA_CENTER) if hi20 else cc_s),
        p(str(age), cc_s),
        p(fmt(prem), cv_s),
        p(fmt(pl), cr_s if pl>30000 else ca_s if pl>0 else cv_s),
        p(fmt(admin), cv_s),
        p(fmt(coi), cv_s),
        p(fmt(total), cr_s if total>90000 else ca_s),
        p(fmt(interest), cg_s if interest>0 else cv_s),
        p(fmt(pv), cg_s),
    ])

# Totals
cost_rows.append([
    p("TOTALS", S("tot",fontSize=7.5,fontName="Helvetica-Bold",textColor=NAVY,alignment=TA_CENTER)),
    p("Yrs 1-20", S("ta",fontSize=7,fontName="Helvetica",textColor=GRAY,alignment=TA_CENTER)),
    p(fmt(6347244), S("tv",fontSize=7.5,fontName="Helvetica-Bold",textColor=NAVY,alignment=TA_RIGHT)),
    p(fmt(345417), S("tr",fontSize=7.5,fontName="Helvetica-Bold",textColor=RED,alignment=TA_RIGHT)),
    p(fmt(1034724), S("tr2",fontSize=7.5,fontName="Helvetica-Bold",textColor=AMBER,alignment=TA_RIGHT)),
    p(fmt(231480), S("tr3",fontSize=7.5,fontName="Helvetica-Bold",textColor=AMBER,alignment=TA_RIGHT)),
    p(fmt(1611621), S("tr4",fontSize=7.5,fontName="Helvetica-Bold",textColor=RED,alignment=TA_RIGHT)),
    p(""),
    p(""),
])
ct = Table(cost_rows, colWidths=[0.5*inch,0.42*inch,0.85*inch,0.78*inch,0.78*inch,0.72*inch,0.78*inch,0.88*inch,0.85*inch])
style_cmds = [
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("BACKGROUND",(0,-1),(-1,-1),colors.HexColor("#F0F0F0")),
    ("LINEABOVE",(0,-1),(-1,-1),1,NAVY),
    ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
    ("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]
for i in range(1, len(cost_rows)-1):
    style_cmds.append(("BACKGROUND",(0,i),(-1,i), LIGHT if i%2==0 else WHITE))
ct.setStyle(TableStyle(style_cmds))
story.append(ct)
story.append(Spacer(1,3))

cost_note = Table([[p(
    "TOTAL charges (Yrs 1–20): $1,611,621 = Premium load $345,417 + Admin/Issue $1,034,724 + COI $231,480. "
    "For comparison: 20-yr term $10M, female 44, preferred ≈ $10,000–12,000/yr ($200–240k total). "
    "IUL costs ~6–8× more than term but includes permanent DB, 0% floor, and tax-free income for life.",
    small_s)]], colWidths=[7.54*inch])
cost_note.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LGOLD),("BOX",(0,0),(-1,-1),0.5,GOLD),
    ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
    ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6)]))
story.append(cost_note)
story.append(Spacer(1,7))

# ── SECTION 3: THREE SCENARIOS ────────────────────────────────────────────────
story.append(p("3.  THREE SCENARIOS — WHAT DOES THE ILLUSTRATION ACTUALLY SHOW?", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,5))

scen_hdr = [p(h, ch_s) for h in ["Scenario","Rate","Year 20\nPolicy Value","After Loan\nRepaid Yr 21","Distributions","Death Benefit\nAge 70","Status"]]
scen_rows = [scen_hdr,
    [p("Worst Case", S("wc",fontSize=7.5,fontName="Helvetica-Bold",textColor=RED,alignment=TA_LEFT)),
     p("0% (guaranteed)",cv_s), p("$2,391,573",cv_s), p("LAPSES",cr_s),
     p("None",cr_s), p("NONE",cr_s),
     p("✗ LAPSED Yr 21",S("ls",fontSize=7.5,fontName="Helvetica-Bold",textColor=RED,alignment=TA_CENTER))],
    [p("Alternate / Low", S("al",fontSize=7.5,fontName="Helvetica-Bold",textColor=AMBER,alignment=TA_LEFT)),
     p("5.20%",cv_s), p("$11,727,228",ca_s), p("~$2.9M net",ca_s),
     p("$200k/yr → lapses Yr 38 (Age 82)",ca_s), p("~$8.8M at 80",ca_s),
     p("⚠ Runs out Age 82",S("wr",fontSize=7.5,fontName="Helvetica-Bold",textColor=AMBER,alignment=TA_CENTER))],
    [p("Assumed / Base", S("ab",fontSize=7.5,fontName="Helvetica-Bold",textColor=GREEN,alignment=TA_LEFT)),
     p("~6% (base capped)",cv_s), p("$13,301,220",cg_s), p("$4,659,450 net",cg_s),
     p("$200k/yr — never lapses",cg_s), p("$12,551,771",cg_s),
     p("✓ Sustainable",S("ok",fontSize=7.5,fontName="Helvetica-Bold",textColor=GREEN,alignment=TA_CENTER))],
]
st = Table(scen_rows, colWidths=[0.9*inch,0.85*inch,1.1*inch,1.0*inch,1.85*inch,1.1*inch,0.84*inch])
st.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LRED,LAMB,LGRN]),
    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]))
story.append(st)
story.append(Spacer(1,7))

# ── SECTION 4: JH vs LINCOLN COMPARISON ──────────────────────────────────────
story.append(p("4.  JOHN HANCOCK vs LINCOLN ($5M) — WHAT CHANGED?", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,5))

comp_hdr = [p(h, ch_s) for h in ["Factor", "Lincoln WealthBuilder IUL\n($5M — previous)", "John Hancock AIUL\n($10M — new)", "Better?"]]
comp_rows = [comp_hdr,
    [p("Carrier",bold_s), p("Lincoln National",cv_s), p("John Hancock",cv_s), p("Both strong",cc_s)],
    [p("Death benefit",bold_s), p("$5,000,000",cv_s), p("$10,000,000",cg_s), p("✓ JH (2×)",S("jw",fontSize=7.5,fontName="Helvetica-Bold",textColor=GREEN,alignment=TA_CENTER))],
    [p("Assumed return",bold_s), p("7.19% (aggressive)",cr_s), p("~6.00% (conservative)",cg_s), p("✓ JH (safer proj.)",S("jw2",fontSize=7.5,fontName="Helvetica-Bold",textColor=GREEN,alignment=TA_CENTER))],
    [p("Index cap (main)",bold_s), p("13.5% (S&P TCA)",cv_s), p("11.65% base / 14% Nasdaq",cv_s), p("Similar",cc_s)],
    [p("Premium load",bold_s), p("~$460k/10yrs (15%)",cr_s), p("~$345k/12yrs (6–7%)",cg_s), p("✓ JH (lower %)",S("jw3",fontSize=7.5,fontName="Helvetica-Bold",textColor=GREEN,alignment=TA_CENTER))],
    [p("Annual distributions",bold_s), p("$204,086/yr from Yr 21",cv_s), p("$200,000/yr from Yr 22",cv_s), p("Similar",cc_s)],
    [p("Loan rate (standard)",bold_s), p("4.50%/yr fixed",cv_s), p("3.25% yrs 1-10 / 3.00% yr 11+",cg_s), p("✓ JH (lower)",S("jw4",fontSize=7.5,fontName="Helvetica-Bold",textColor=GREEN,alignment=TA_CENTER))],
    [p("Vitality/health bonus",bold_s), p("None",cr_s), p("Yes — earn credits for healthy habits",cg_s), p("✓ JH unique",S("jw5",fontSize=7.5,fontName="Helvetica-Bold",textColor=GREEN,alignment=TA_CENTER))],
    [p("Worst case",bold_s), p("Lapses yr 20",cr_s), p("Lapses yr 21 (one year better)",ca_s), p("⚠ Similar",cc_s)],
    [p("Alternate (5-6%)",bold_s), p("Not shown",cv_s), p("Policy runs to Age 82 then lapses",ca_s), p("⚠ Watch this",S("wt",fontSize=7.5,fontName="Helvetica-Bold",textColor=AMBER,alignment=TA_CENTER))],
]
comp_t = Table(comp_rows, colWidths=[1.1*inch, 2.15*inch, 2.35*inch, 1.04*inch])
comp_t.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("BACKGROUND",(0,1),(0,-1),LBLUE),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LIGHT,WHITE]),
    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]))
story.append(comp_t)
story.append(Spacer(1,7))

# ── SECTION 5: VERDICT ────────────────────────────────────────────────────────
story.append(p("5.  IS IT WORTH IT? — HONEST VERDICT", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,5))

pros_cons = Table([[
    Table([
        [p("WHAT'S BETTER in this JH policy", S("ph",fontSize=8,fontName="Helvetica-Bold",textColor=WHITE,alignment=TA_CENTER))],
        [p("✓  Double the death benefit ($10M vs $5M)\n"
           "✓  More conservative 6% assumed return (less likely to disappoint)\n"
           "✓  Lower standard loan rate: 3.25% → 3.00% (vs 4.50% Lincoln)\n"
           "✓  Lower premium load % (~6-7% vs ~15% at Lincoln)\n"
           "✓  Vitality PLUS: get paid to stay healthy — unique to JH\n"
           "✓  More index options including Nasdaq (14% cap)",
           S("pb",fontSize=7.5,fontName="Helvetica",textColor=colors.HexColor("#222222"),leading=12))],
    ], colWidths=[3.6*inch]),
    Table([
        [p("WHAT TO WATCH OUT FOR", S("ch2",fontSize=8,fontName="Helvetica-Bold",textColor=WHITE,alignment=TA_CENTER))],
        [p("⚠  At 5.2% return, policy LAPSES at Age 82 — you need ~6%+ sustained\n"
           "⚠  Total charges $1.6M over 20 years on a $10M policy — significant\n"
           "⚠  Premium structure is complex (varies years 1-12) — confirm with broker\n"
           "⚠  Still NOT PPLI — premium loads exist (though lower than Lincoln)\n"
           "⚠  Base Capped cap is 11.65% vs Lincoln's 13.5% — slightly lower ceiling\n"
           "⚠  0% guaranteed rate — if credits consistently below 5%, policy at risk",
           S("cb",fontSize=7.5,fontName="Helvetica",textColor=colors.HexColor("#222222"),leading=12))],
    ], colWidths=[3.6*inch]),
]], colWidths=[3.64*inch, 3.64*inch])

for i, (inner_t, col) in enumerate(zip(pros_cons._cellvalues[0], [GREEN, RED])):
    inner_t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),col),
        ("BACKGROUND",(0,1),(-1,-1),LGRN if col==GREEN else LRED),
        ("BOX",(0,0),(-1,-1),1,col),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
    ]))
pros_cons.setStyle(TableStyle([
    ("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0),
    ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
    ("VALIGN",(0,0),(-1,-1),"TOP"),
]))
story.append(pros_cons)
story.append(Spacer(1,6))

verdict = Table([[p(
    "BOTTOM LINE:  The John Hancock policy is a material upgrade from the Lincoln $5M illustration — "
    "double the death benefit, lower loan rate, lower premium load %, and a more conservative assumed return that is "
    "less likely to disappoint.  At 6% credited, the policy is self-sustaining for life.  "
    "The key risk is if returns average 5% or below — the alternate scenario shows the policy depletes at age 82, "
    "not catastrophic but no longer a lifetime vehicle.  "
    "At $0 out of pocket, a $10M death benefit from day one, $200k/yr tax-free income from age 66, "
    "and $12.5M death benefit still in place at age 70 — this is a strong structure IF the market credits "
    "an average of 6%/yr.  Based on 25 years of S&P 500 history, that is historically achievable.  "
    "Ask your broker: (1) What index/allocation is assumed to get 6%?  (2) Can you run a 4% stressed scenario?  "
    "(3) Is this eligible for PPLI to eliminate the $1.6M in charges?",
    S("fv",fontSize=8,fontName="Helvetica-Bold",textColor=NAVY,leading=12))
]], colWidths=[7.54*inch])
verdict.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LIGHT),("BOX",(0,0),(-1,-1),1.5,NAVY),
    ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
    ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10)]))
story.append(verdict)
story.append(Spacer(1,4))
story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY))
story.append(Spacer(1,2))
story.append(p("Values sourced directly from John Hancock illustration dated April 30, 2026. "
               "Assumed return ~6% per Select Capped/Base Capped account as illustrated. "
               "Not financial, tax or legal advice. Consult your advisor before any decision.", small_s))

doc.build(story)
print(f"PDF written -> {OUTPUT}")
