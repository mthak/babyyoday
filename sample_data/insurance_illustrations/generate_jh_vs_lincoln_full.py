"""
Full comparison: JH $10M vs Lincoln $5M — all conversations summarised in one PDF
Covers: cost, charges, cap rates, loan rates, output impact, honest verdict
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT

OUTPUT = "Swati_Chugh_JH_vs_Lincoln_Full_Comparison.pdf"

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

def S(n, **kw): return ParagraphStyle(n, **kw)
title_s = S("t",  fontSize=15, textColor=WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER)
sub_s   = S("s",  fontSize=8,  textColor=GOLD,  fontName="Helvetica-Bold", alignment=TA_CENTER)
sect_s  = S("sc", fontSize=9,  textColor=NAVY,  fontName="Helvetica-Bold", spaceBefore=7, spaceAfter=3)
body_s  = S("b",  fontSize=7.5,textColor=colors.HexColor("#222222"), fontName="Helvetica", leading=11)
bold_s  = S("bd", fontSize=7.5,textColor=NAVY,  fontName="Helvetica-Bold", leading=11)
small_s = S("sm", fontSize=6.5,textColor=GRAY,  fontName="Helvetica", leading=9)
ch_s    = S("ch", fontSize=7.5,textColor=WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER)
cv_s    = S("cv", fontSize=7.5,textColor=GRAY,  fontName="Helvetica",      alignment=TA_RIGHT)
cg_s    = S("cg", fontSize=7.5,textColor=GREEN, fontName="Helvetica-Bold", alignment=TA_RIGHT)
cr_s    = S("cr", fontSize=7.5,textColor=RED,   fontName="Helvetica-Bold", alignment=TA_RIGHT)
ca_s    = S("ca", fontSize=7.5,textColor=AMBER, fontName="Helvetica-Bold", alignment=TA_RIGHT)
cc_s    = S("cc", fontSize=7.5,textColor=GRAY,  fontName="Helvetica",      alignment=TA_CENTER)
cg2_s   = S("cg2",fontSize=7.5,textColor=GREEN, fontName="Helvetica-Bold", alignment=TA_CENTER)
cr2_s   = S("cr2",fontSize=7.5,textColor=RED,   fontName="Helvetica-Bold", alignment=TA_CENTER)
ca2_s   = S("ca2",fontSize=7.5,textColor=AMBER, fontName="Helvetica-Bold", alignment=TA_CENTER)

def p(txt, st=None): return Paragraph(str(txt), st or body_s)
def fmt(n):
    if n is None: return "—"
    return f"${n:,.0f}" if n >= 0 else f"(${abs(n):,.0f})"

doc = SimpleDocTemplate(OUTPUT, pagesize=letter,
    topMargin=0.4*inch, bottomMargin=0.35*inch,
    leftMargin=0.5*inch, rightMargin=0.5*inch)
story = []

# ── PAGE 1: HEADER + POLICY BASICS + COST ────────────────────────────────────
hdr = Table([
    [p("Swati Chugh — IUL Policy Comparison: JH $10M vs Lincoln $5M", title_s)],
    [p("Full cost analysis  |  Charges breakdown  |  Cap rates  |  Output impact  |  Honest verdict  |  May 2026", sub_s)],
], colWidths=[7.5*inch])
hdr.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),NAVY),
    ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),8),("LEFTPADDING",(0,0),(-1,-1),10)]))
story.append(hdr)
story.append(Spacer(1,7))

# ── SECTION 1: POLICY AT A GLANCE ─────────────────────────────────────────────
story.append(p("1.  POLICY AT A GLANCE", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,4))

glance_hdr = [p(h, ch_s) for h in ["Factor", "Lincoln WealthBuilder IUL\n($5M — April 2026)", "John Hancock AIUL\n($10M — April 2026)", "Which is Better?"]]
glance_rows = [glance_hdr,
    [p("Carrier",bold_s), p("Lincoln National Life",cv_s), p("John Hancock Life",cv_s), p("Both A-rated",cc_s)],
    [p("Death benefit",bold_s), p("$5,000,000",cv_s), p("$10,000,000",cg_s), p("✓ JH — 2×",cg2_s)],
    [p("Insured age / class",bold_s), p("Age 43, Female, Preferred NS",cv_s), p("Age 44, Female, Preferred NS",cv_s), p("Same",cc_s)],
    [p("Annual premium (financed)",bold_s), p("$266,675/yr × 10 yrs",cv_s), p("$689,782/yr × 12 yrs (variable)",cv_s), p("Lincoln simpler",ca2_s)],
    [p("Total premiums borrowed",bold_s), p("$2,666,750",cv_s), p("$6,347,244",cv_s), p("JH larger loan",ca2_s)],
    [p("Your out-of-pocket cost",bold_s), p("$0",cg_s), p("$0",cg_s), p("Same — $0",cg2_s)],
    [p("Assumed return",bold_s), p("7.19% (aggressive)",cr_s), p("~6.00% (conservative)",cg_s), p("✓ JH safer",cg2_s)],
    [p("Main index cap",bold_s), p("13.50% (S&P 500 TCA)",cg_s), p("11.65% base / 14% Nasdaq",cv_s), p("⚠ JH base lower",ca2_s)],
    [p("Floor (downside protection)",bold_s), p("0% (guaranteed)",cg_s), p("0% (guaranteed)",cg_s), p("Same",cc_s)],
    [p("Loan repayment year",bold_s), p("Year 20 (Age 63)",cv_s), p("Year 21 (Age 65)",cv_s), p("JH 1yr later",cc_s)],
    [p("Annual distributions",bold_s), p("$204,086/yr from Year 21",cv_s), p("$200,000/yr from Year 22",cv_s), p("Lincoln slightly more",ca2_s)],
    [p("Distribution loan type",bold_s), p("Policy loans (tax-free)",cg_s), p("Policy loans (tax-free)",cg_s), p("Same",cc_s)],
    [p("Dist. policy loan rate",bold_s), p("Fixed rate ~4%",cv_s), p("3.25% yrs 1-10 / 3.00% yr 11+",cg_s), p("✓ JH lower",cg2_s)],
    [p("Health/bonus rider",bold_s), p("None",cr_s), p("Vitality PLUS (earn credits for healthy habits)",cg_s), p("✓ JH unique",cg2_s)],
    [p("Premium paying years",bold_s), p("10 years (flat $266k/yr)",cv_s), p("12 years (variable, front-loaded)",cv_s), p("Lincoln simpler",ca2_s)],
]
gt = Table(glance_rows, colWidths=[1.5*inch, 2.2*inch, 2.3*inch, 1.1*inch])
gt.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("BACKGROUND",(0,1),(0,-1),LBLUE),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LIGHT,WHITE]),
    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]))
story.append(gt)
story.append(Spacer(1,8))

# ── SECTION 2: TOTAL COST BREAKDOWN ───────────────────────────────────────────
story.append(p("2.  TOTAL COST BREAKDOWN — WHAT DOES EACH POLICY ACTUALLY COST?", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,4))

cost_hdr = [p(h, ch_s) for h in ["Cost Component", "Lincoln $5M", "JH $10M", "JH per $5M\nequivalent", "JH vs Lincoln\n(per $5M)"]]
cost_rows_data = [
    ("A) POLICY CHARGES (reduce your policy value silently)", "", "", "", ""),
    ("Premium load", "$474,012", "$345,417", "$172,709", "Lincoln 2.7× worse"),
    ("Admin / Issue charges", "~$48,000", "$1,035,923", "$517,962", "JH 10.8× worse ⚠"),
    ("COI (Cost of Insurance)", "~$7,782", "$417,002", "$208,501", "JH 26.8× worse ⚠"),
    ("Rider charges", "$0", "$600", "$300", "Same"),
    ("TOTAL JH CHARGES", "~$529,751", "$1,798,942", "$899,471", "JH 1.70× more expensive"),
    ("", "", "", "", ""),
    ("B) BANK LOAN INTEREST (external lender — repaid from policy)", "", "", "", ""),
    ("Total premiums borrowed", "$2,666,750", "$6,347,244", "$3,173,622", ""),
    ("Total repaid to lender", "$5,237,646", "$9,196,929", "$4,598,465", ""),
    ("Bank profit (interest)", "$2,570,896", "$2,849,685", "$1,424,843", "JH slightly higher"),
    ("", "", "", "", ""),
    ("C) GRAND TOTAL ECONOMIC COST", "~$3,100,647", "$4,648,627", "$2,324,314", "JH 1.50× more"),
    ("Your cash out of pocket", "$0", "$0", "$0", "SAME"),
]

cost_rows = [cost_hdr]
for label, lnc, jh, jh5, verdict in cost_rows_data:
    is_section = label.startswith("A)") or label.startswith("B)") or label.startswith("C)")
    is_total = "TOTAL" in label or "GRAND" in label
    is_empty = label == ""
    if is_empty:
        cost_rows.append([p(""),p(""),p(""),p(""),p("")])
        continue
    ls = S("lh",fontSize=7.5,fontName="Helvetica-Bold",textColor=NAVY,alignment=TA_LEFT) if (is_section or is_total) else bold_s if is_total else body_s
    vs = cr_s if "worse" in verdict and "JH" in verdict else (cg_s if "worse" in verdict and "Lincoln" in verdict else (cr_s if "1.7" in verdict or "1.5" in verdict else cv_s))
    cost_rows.append([
        p(label, ls),
        p(lnc or " ", cv_s),
        p(jh or " ", cr_s if jh and ("$1,035" in jh or "$417" in jh or "$1,798" in jh) else cv_s),
        p(jh5 or " ", cv_s),
        p(verdict or " ", vs if verdict else cv_s),
    ])

cost_t = Table(cost_rows, colWidths=[2.1*inch, 1.15*inch, 1.15*inch, 1.2*inch, 1.85*inch])
style_cmds = [
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
    ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]
section_rows = [1,7,12]
total_rows   = [5,12]
for i, (label, *_) in enumerate(cost_rows_data):
    ri = i+1
    if label.startswith("A)") or label.startswith("B)") or label.startswith("C)"):
        style_cmds.append(("BACKGROUND",(0,ri),(-1,ri),LBLUE))
    elif "TOTAL" in label or "GRAND" in label:
        style_cmds.append(("BACKGROUND",(0,ri),(-1,ri),colors.HexColor("#F0F0F0")))
        style_cmds.append(("LINEABOVE",(0,ri),(-1,ri),0.8,NAVY))
    elif label == "":
        style_cmds.append(("BACKGROUND",(0,ri),(-1,ri),WHITE))
    else:
        style_cmds.append(("BACKGROUND",(0,ri),(-1,ri),LIGHT if i%2==0 else WHITE))
cost_t.setStyle(TableStyle(style_cmds))
story.append(cost_t)
story.append(Spacer(1,4))

cost_note = Table([[p(
    "KEY INSIGHT: The JH admin/issue charge ($1,035,923) is 21× larger than Lincoln's (~$48,000). "
    "This is the main driver of JH being 1.7× more expensive. The premium load is actually LOWER at JH (6-7%) vs Lincoln (~15%). "
    "Ask your broker: Why is the admin charge $1M+ and what exactly does it cover?",
    small_s)]], colWidths=[7.5*inch])
cost_note.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LGOLD),("BOX",(0,0),(-1,-1),0.5,GOLD),
    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7)]))
story.append(cost_note)
story.append(Spacer(1,8))

# ── SECTION 3: CHARGES IMPACT ON OUTPUT ───────────────────────────────────────
story.append(p("3.  HOW HIGHER CHARGES DIRECTLY HIT YOUR DISTRIBUTIONS", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,5))

impact_rows = [
    [p("", ch_s), p("Actual JH $10M\n(current charges)", ch_s), p("JH with Lincoln-level\ncharges (hypothetical)", ch_s), p("What You Lose\nwith JH as-is", ch_s)],
    [p("Extra charges vs Lincoln (20 yrs)", bold_s), p("—", cv_s), p("—", cv_s), p("$553,940 more", cr_s)],
    [p("Lost compounding (6%/yr × 20yrs)", bold_s), p("—", cv_s), p("—", cv_s), p("$739,504 in yr 20 value", cr_s)],
    [p("Policy value at Year 20", bold_s), p("$13,301,220", cv_s), p("$14,040,724", cg_s), p("$739,504 less", cr_s)],
    [p("Loan repaid to bank (same)", bold_s), p("$9,196,929", cv_s), p("$9,196,929", cv_s), p("No difference", cc_s)],
    [p("Net cash after loan repaid", bold_s), p("$4,104,291", cv_s), p("$4,843,795", cg_s), p("$739,504 less", cr_s)],
    [p("Annual tax-free distributions", bold_s), p("$200,000/yr", cv_s), p("$236,035/yr", cg_s), p("$36,035/yr less — FOREVER", cr_s)],
    [p("Death benefit at age 70 (yr 26)", bold_s), p("$12,551,771", cv_s), p("~$13,291,275", cg_s), p("~$740k less", cr_s)],
]
it = Table(impact_rows, colWidths=[2.2*inch, 1.9*inch, 1.9*inch, 1.6*inch])
it.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("BACKGROUND",(0,1),(0,-1),LBLUE),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LIGHT,WHITE]),
    ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]))
story.append(it)
story.append(Spacer(1,8))

# ── SECTION 4: CAP RATE REALITY ───────────────────────────────────────────────
story.append(p("4.  CAP RATE REALITY — 25 YEARS OF S&P 500 HISTORY", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,5))

cap_hdr = [p(h, ch_s) for h in ["Year","S&P Return","Lincoln\n13.5% cap","JH Base\n11.65% cap","JH Nasdaq\n14.0% cap","Diff: Lincoln\nvs JH Base"]]
spy_data = [
    (2000,-0.1014,0,0,0,0),(2001,-0.1304,0,0,0,0),(2002,-0.2337,0,0,0,0),
    (2003,0.2638,0.135,0.1165,0.14,0.0185),(2004,0.0899,0.0899,0.0899,0.0899,0),
    (2005,0.030,0.030,0.030,0.030,0),(2006,0.1362,0.135,0.1165,0.1362,0.0185),
    (2007,0.0353,0.0353,0.0353,0.0353,0),(2008,-0.3849,0,0,0,0),
    (2009,0.2345,0.135,0.1165,0.14,0.0185),(2010,0.1278,0.1278,0.1165,0.1278,0.0113),
    (2011,0,0,0,0,0),(2012,0.1341,0.1341,0.1165,0.1341,0.0176),
    (2013,0.2960,0.135,0.1165,0.14,0.0185),(2014,0.1139,0.1139,0.1139,0.1139,0),
    (2015,-0.0073,0,0,0,0),(2016,0.0954,0.0954,0.0954,0.0954,0),
    (2017,0.1942,0.135,0.1165,0.14,0.0185),(2018,-0.0624,0,0,0,0),
    (2019,0.2888,0.135,0.1165,0.14,0.0185),(2020,0.1626,0.135,0.1165,0.14,0.0185),
    (2021,0.2689,0.135,0.1165,0.14,0.0185),(2022,-0.1944,0,0,0,0),
    (2023,0.2423,0.135,0.1165,0.14,0.0185),(2024,0.2331,0.135,0.1165,0.14,0.0185),
]
avg_spy=sum(r[1] for r in spy_data)/25
avg_lnc=sum(r[2] for r in spy_data)/25
avg_jh =sum(r[3] for r in spy_data)/25
avg_nasd=sum(r[4] for r in spy_data)/25

cap_rows = [cap_hdr]
for yr, spy, lnc, jh, nasd, diff in spy_data:
    neg = spy < 0
    jh_worse = diff > 0
    cap_rows.append([
        p(str(yr), cc_s),
        p(f"{spy:.2%}", cr_s if neg else (cg_s if spy>=0.20 else cv_s)),
        p(f"{lnc:.2%}", cr_s if neg else cg_s if lnc>0 else cv_s),
        p(f"{jh:.2%}", cr_s if neg else (ca_s if diff>0 else (cg_s if jh>0 else cv_s))),
        p(f"{nasd:.2%}", cg_s if nasd>0 else (cr_s if neg else cv_s)),
        p(f"+{diff:.2%}" if diff>0 else "—", cr_s if diff>0 else cv_s),
    ])
cap_rows.append([
    p("AVG", S("avg",fontSize=7.5,fontName="Helvetica-Bold",textColor=NAVY,alignment=TA_CENTER)),
    p(f"{avg_spy:.2%}", cv_s),
    p(f"{avg_lnc:.2%}", cg_s),
    p(f"{avg_jh:.2%}", ca_s),
    p(f"{avg_nasd:.2%}", cg_s),
    p(f"+{avg_lnc-avg_jh:.2%}", cr_s),
])
cap_t = Table(cap_rows, colWidths=[0.52*inch,0.82*inch,0.92*inch,0.92*inch,0.92*inch,1.05*inch],
              repeatRows=1)
cap_style = [
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("BACKGROUND",(0,-1),(-1,-1),colors.HexColor("#F0F0F0")),
    ("LINEABOVE",(0,-1),(-1,-1),0.8,NAVY),
    ("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2),
    ("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]
for i, (yr,spy,lnc,jh,nasd,diff) in enumerate(spy_data):
    ri = i+1
    if spy < 0: cap_style.append(("BACKGROUND",(0,ri),(-1,ri),LRED))
    elif diff > 0: cap_style.append(("BACKGROUND",(0,ri),(-1,ri),LAMB))
    else: cap_style.append(("BACKGROUND",(0,ri),(-1,ri),LIGHT if i%2==0 else WHITE))
cap_t.setStyle(TableStyle(cap_style))

# Two column layout for cap table + cap analysis
cap_analysis = Table([
    [p("CAP RATE ANALYSIS", S("ca_h",fontSize=8,fontName="Helvetica-Bold",textColor=WHITE,alignment=TA_CENTER))],
    [p(
        "Lincoln 13.5% cap averaged 7.91%/yr\n"
        "JH Base 11.65% cap averaged 7.05%/yr\n"
        "JH Nasdaq 14.0% cap averaged 8.09%/yr\n\n"
        "JH Base cap left money on table in\n"
        "12 out of 25 years (48% of all years)\n\n"
        "Annual gap: Lincoln beats JH Base\n"
        "by 0.86%/yr on average\n\n"
        "FIX: Switch to JH Nasdaq 14% cap —\n"
        "averages 8.09%, BEATS Lincoln's 7.91%\n"
        "and beats both by ~0.18%/yr\n\n"
        "Key: 7 years had negative S&P, both\n"
        "policies credited 0% (floor protected).\n"
        "Floor is the same regardless of cap.",
        S("ca_b",fontSize=7.5,fontName="Helvetica",textColor=colors.HexColor("#222222"),leading=12))],
], colWidths=[2.2*inch])
cap_analysis.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),AMBER),
    ("BACKGROUND",(0,1),(-1,-1),LAMB),
    ("BOX",(0,0),(-1,-1),1,AMBER),
    ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
    ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
    ("VALIGN",(0,0),(-1,-1),"TOP"),
]))

cap_layout = Table([[cap_t, Spacer(0.2*inch,1), cap_analysis]],
                   colWidths=[5.08*inch, 0.2*inch, 2.22*inch])
cap_layout.setStyle(TableStyle([
    ("VALIGN",(0,0),(-1,-1),"TOP"),
    ("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0),
    ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
]))
story.append(cap_layout)
story.append(Spacer(1,8))

# ── SECTION 5: THREE SCENARIOS ────────────────────────────────────────────────
story.append(p("5.  JH THREE SCENARIOS — WHAT DOES THE ILLUSTRATION ACTUALLY SHOW?", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,5))

scen_hdr = [p(h, ch_s) for h in ["Scenario","Rate Assumed","Year 20 Value","Net after\nLoan Yr 21","$200k/yr Dists","Age 70\nDeath Benefit","Verdict"]]
scen_rows = [scen_hdr,
    [p("Worst Case\n(guaranteed 0%)",S("wc",fontSize=7.5,fontName="Helvetica-Bold",textColor=RED,alignment=TA_LEFT)),
     p("0%",cr_s), p("$2,391,573",cr_s), p("LAPSES",cr_s), p("None",cr_s), p("NONE",cr_s),
     p("✗ LAPSED",cr2_s)],
    [p("Midpoint / Low\n(~5.2%)",S("mc",fontSize=7.5,fontName="Helvetica-Bold",textColor=AMBER,alignment=TA_LEFT)),
     p("~5.2%",ca_s), p("$11,727,228",ca_s), p("~$2.9M",ca_s), p("$200k/yr until\nAge 82, then lapses",ca_s),
     p("~$8.8M at 80",ca_s), p("⚠ Runs out\nAge 82",ca2_s)],
    [p("Assumed / Base\n(~6%)",S("ac",fontSize=7.5,fontName="Helvetica-Bold",textColor=GREEN,alignment=TA_LEFT)),
     p("~6%",cg_s), p("$13,301,220",cg_s), p("$4,104,291",cg_s), p("$200k/yr — FOREVER\nnever lapses",cg_s),
     p("$12,551,771",cg_s), p("✓ Sustainable\nfor life",cg2_s)],
]
st = Table(scen_rows, colWidths=[1.05*inch,0.75*inch,1.05*inch,0.9*inch,1.45*inch,1.05*inch,0.75*inch])
st.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LRED,LAMB,LGRN]),
    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]))
story.append(st)
story.append(Spacer(1,8))

# ── SECTION 6: TWO CLARIFICATIONS ─────────────────────────────────────────────
story.append(p("6.  IMPORTANT CLARIFICATIONS FROM OUR ANALYSIS", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,5))

clarif_rows = [
    ("Where is the 3.25% loan rate?",
     "The 3.25% → 3.00% rate is the INTERNAL POLICY LOAN RATE — what JH charges when you take distributions (the $200k/yr). "
     "It is NOT the external bank's premium financing rate. "
     "The bank's rate (SOFR+1%) is set in a SEPARATE loan agreement, NOT in this illustration. "
     "Both Lincoln and JH use the same external lender market for premium financing.",
     AMBER),
    ("Are you paying for 20 years of premiums?",
     "No — the JH policy pays premiums for 12 years (not 20). Years 1-4: $689,782/yr. Year 5: $118,973. "
     "Year 6: $295,509. Years 7-12: $528,939. Year 13 onwards: $0. The illustration confirms 'Years Premium Paid: 12'. "
     "It is 2 more years than Lincoln's 10 — but the lower bank loan rate compensates.",
     NAVY),
    ("Why is your out-of-pocket $0 if there are still charges?",
     "The $0 means no cash ever leaves your bank account. But the $1.8M in JH charges still EXIST — "
     "they reduce the policy's cash value silently each year. You pay it through reduced investment performance, "
     "not through writing checks. This is why the policy must earn 6%+ to sustain the distributions — "
     "it needs to overcome all the embedded charges.",
     RED),
]
for title, text, col in clarif_rows:
    row = Table([[
        p("!", S("ic",fontSize=14,fontName="Helvetica-Bold",textColor=col,alignment=TA_CENTER)),
        Table([
            [p(title, S("ct",fontSize=8,fontName="Helvetica-Bold",textColor=col,leading=11))],
            [p(text, S("cb",fontSize=7.5,fontName="Helvetica",textColor=colors.HexColor("#222222"),leading=11))],
        ], colWidths=[7.1*inch])
    ]], colWidths=[0.25*inch, 7.1*inch])
    row.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,-1), LAMB if col==AMBER else (LRED if col==RED else LBLUE)),
        ("BACKGROUND",(1,0),(1,-1), LIGHT),
        ("BOX",(0,0),(-1,-1),0.5,col),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LINEBEFORE",(1,0),(1,-1),0.5,col),
    ]))
    story.append(row)
    story.append(Spacer(1,4))

story.append(Spacer(1,5))

# ── SECTION 7: FINAL VERDICT ──────────────────────────────────────────────────
story.append(p("7.  FINAL VERDICT — IS THE JH POLICY WORTH IT vs LINCOLN?", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,5))

verdict_boxes = Table([[
    Table([
        [p("JH WINS ON", S("jw",fontSize=8,fontName="Helvetica-Bold",textColor=WHITE,alignment=TA_CENTER))],
        [p(
            "✓  Double death benefit ($10M vs $5M)\n"
            "✓  Conservative 6% assumed rate (safer projection)\n"
            "✓  Lower premium load % (6-7% vs ~15%)\n"
            "✓  Lower distribution loan rate (3.00%)\n"
            "✓  Vitality PLUS health bonus (unique)\n"
            "✓  Nasdaq 14% cap option (beats Lincoln)\n"
            "✓  1 extra year before loan repayment",
            S("jb",fontSize=7.5,fontName="Helvetica",textColor=colors.HexColor("#222222"),leading=12))],
    ], colWidths=[3.6*inch]),
    Table([
        [p("LINCOLN WINS ON", S("lw",fontSize=8,fontName="Helvetica-Bold",textColor=WHITE,alignment=TA_CENTER))],
        [p(
            "✓  1.7× cheaper per $5M of coverage\n"
            "✓  $739,504 more in Year 20 policy value\n"
            "✓  ~$36k/yr more in distributions\n"
            "✓  Higher base cap (13.5% vs 11.65%)\n"
            "✓  Much lower admin/issue charges\n"
            "✓  Simpler premium structure (flat 10 yrs)\n"
            "✓  COI costs are minimal vs JH",
            S("lb",fontSize=7.5,fontName="Helvetica",textColor=colors.HexColor("#222222"),leading=12))],
    ], colWidths=[3.6*inch]),
]], colWidths=[3.64*inch, 3.64*inch])

for inner_t, col, bg in zip(verdict_boxes._cellvalues[0], [GREEN, NAVY], [LGRN, LBLUE]):
    inner_t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),col),
        ("BACKGROUND",(0,1),(-1,-1),bg),
        ("BOX",(0,0),(-1,-1),1,col),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
    ]))
verdict_boxes.setStyle(TableStyle([
    ("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0),
    ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
    ("VALIGN",(0,0),(-1,-1),"TOP"),
]))
story.append(verdict_boxes)
story.append(Spacer(1,7))

final = Table([[p(
    "THE BOTTOM LINE:  The JH policy gives you 2× the death benefit but charges 1.7× more per dollar of coverage. "
    "The $739k in extra charges directly reduces your Year 20 net cash and costs you ~$36k/yr in lifetime distributions. "
    "If the only priority is maximising distributions and minimising costs, Lincoln is the better deal. "
    "If the priority is a $10M permanent death benefit, JH is the only option here — but push back on the admin charges. "
    "The most important question: switch the JH index allocation to Nasdaq 14% cap (from Base Capped 11.65%) — "
    "this flips the cap disadvantage to a cap advantage at no extra cost. "
    "Also ask: is PPLI available? It would eliminate most of the $1.8M in charges entirely.",
    S("fv",fontSize=8,fontName="Helvetica-Bold",textColor=NAVY,leading=12))
]], colWidths=[7.5*inch])
final.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LIGHT),("BOX",(0,0),(-1,-1),1.5,NAVY),
    ("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),9),
    ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10)]))
story.append(final)
story.append(Spacer(1,5))
story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY))
story.append(Spacer(1,3))
story.append(p(
    "Lincoln illustration: Lincoln Financial, March 10, 2026, $5M WealthBuilder IUL, 7.19% assumed. "
    "JH illustration: John Hancock Life, April 30, 2026, $10M Accumulation IUL, ~6% assumed. "
    "Charges derived from Annual Account Summary. Lincoln charges partially derived/estimated. "
    "Not financial, tax or legal advice.", small_s))

doc.build(story)
print(f"PDF written -> {OUTPUT}")
