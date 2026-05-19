"""
4-Carrier IUL Comparison: Pacific Life vs John Hancock vs Securian vs Nationwide
$5M policy, Monaj Thakkar, Male Age 47, California, Premium Financed
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT

OUTPUT = "Monaj_Thakkar_4Carrier_IUL_Comparison.pdf"

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
PURP  = colors.HexColor("#5B2C8D")
LPUR  = colors.HexColor("#F0E8FF")
LPAC  = colors.HexColor("#E6F0F8")   # Pacific Life blue
LNW   = colors.HexColor("#E8F5E9")   # Nationwide green
LSEC  = colors.HexColor("#FFF3E0")   # Securian orange
LJH   = colors.HexColor("#F3E5F5")   # JH purple

def S(n, **kw): return ParagraphStyle(n, **kw)
title_s = S("t",  fontSize=14, textColor=WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER)
sub_s   = S("s",  fontSize=8,  textColor=GOLD,  fontName="Helvetica-Bold", alignment=TA_CENTER)
sect_s  = S("sc", fontSize=9,  textColor=NAVY,  fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=3)
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

# Carrier colors
PAC_COL = colors.HexColor("#003087")   # Pacific Life deep blue
JH_COL  = colors.HexColor("#5B2C8D")   # JH purple
SEC_COL = colors.HexColor("#E65100")   # Securian orange
NW_COL  = colors.HexColor("#1B5E20")   # Nationwide green

doc = SimpleDocTemplate(OUTPUT, pagesize=letter,
    topMargin=0.38*inch, bottomMargin=0.32*inch,
    leftMargin=0.45*inch, rightMargin=0.45*inch)
story = []

# ── HEADER ─────────────────────────────────────────────────────────────────
hdr = Table([
    [p("4-Carrier IUL Comparison — $5M Premium Financed Policy", title_s)],
    [p("Monaj Thakkar  |  Male, Age 47  |  California  |  May 18, 2026  |  Broker: Nick Burgess, The Burgess Group", sub_s)],
], colWidths=[7.6*inch])
hdr.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),NAVY),
    ("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),7),("LEFTPADDING",(0,0),(-1,-1),10)]))
story.append(hdr)
story.append(Spacer(1,6))

# ── SECTION 1: QUICK FACTS ─────────────────────────────────────────────────
story.append(p("1.  POLICY QUICK FACTS — SIDE BY SIDE", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,4))

qf_hdr = [p("", ch_s),
    p("Pacific Life\nHorizon IUL 2 LTP", S("ph",fontSize=8,fontName="Helvetica-Bold",textColor=WHITE,alignment=TA_CENTER)),
    p("John Hancock\nAccumulation IUL", S("jh",fontSize=8,fontName="Helvetica-Bold",textColor=WHITE,alignment=TA_CENTER)),
    p("Securian / MN Life\nEclipse Accumulator II", S("sc",fontSize=8,fontName="Helvetica-Bold",textColor=WHITE,alignment=TA_CENTER)),
    p("Nationwide\nIUL", S("nw",fontSize=8,fontName="Helvetica-Bold",textColor=WHITE,alignment=TA_CENTER)),
]
qf_rows = [
    ["Carrier / Rating",        "Pacific Life — A+ (AM Best)",      "John Hancock — A+ (AM Best)",        "Securian / MN Life — A+ (AM Best)", "Nationwide — A+ (AM Best)"],
    ["Product",                  "Horizon IUL 2 Long-Term Perf.",    "Accumulation IUL (26AIUL)",          "Eclipse Accumulator II IUL",         "North American IUL"],
    ["Underwriting class",       "Preferred Tobacco ★★",             "⚠ Standard Smoker (worse class)",    "Standard, Tobacco",                  "Standard Tobacco"],
    ["Annual premium (Yr1)",     "$387,566",                         "$381,790 (LOWEST)",                  "$391,870",                           "$392,134 (HIGHEST)"],
    ["Premium pattern",          "Variable: $387k/351k/205k",        "~$381k/yr years 1-25",               "Variable: $391k/351k",               "Variable: $392k/325k"],
    ["Total premiums (25 yrs)",  "~$8,780,838",                      "~$8,701,376",                        "~$8,765,285 est.",                   "~$8,765,285 est."],
    ["Death benefit",            "$5,000,000 (increasing opt. B)",   "$5,000,000 (increasing opt. 2)",     "$5,000,000 (increasing)",            "$5,000,000 (increasing)"],
    ["Illustrated return",       "6.00%",                            "~6.00%",                             "6.62% (max illustrated)",            "6.59% (max illustrated)"],
    ["Policy value — Year 20",   "$11,316,416 ★ BEST",               "$11,182,884",                        "$10,204,334",                        "$10,604,989"],
    ["No-lapse guarantee",       "To AGE 90 (free) ★★",              "15 years only",                      "Not specified",                      "Not specified"],
    ["Non-MEC confirmed",        "Yes",                              "Yes",                                "Yes ($391,870 max)",                 "Yes"],
    ["Key unique feature",       "Age 90 NLG, EPFR rider, 5-yr uncapped", "Vitality PLUS health bonus, Nasdaq 14%", "Hindsight multi-index, Bonus credit yr11", "Multi-Index 25% cap, BNPP uncapped 300% par"],
]

qf_data = [qf_hdr]
col_bgs = [LPAC, LJH, LSEC, LNW]
for i, row in enumerate(qf_rows):
    label = row[0]; vals = row[1:]
    formatted = [p(label, bold_s)]
    for j, v in enumerate(vals):
        # Style based on content
        if "★" in v or "BEST" in v or "LOWEST" in v or "Age 90" in v:
            st = cg_s
        elif "⚠" in v or "HIGHEST" in v or "worse" in v or "15 years" in v:
            st = cr_s
        elif "Standard" in v and j == 1:  # JH standard smoker
            st = cr_s
        else:
            st = cv_s
        formatted.append(p(v, st))
    qf_data.append(formatted)

qf_t = Table(qf_data, colWidths=[1.45*inch, 1.55*inch, 1.55*inch, 1.55*inch, 1.5*inch])
qf_style = [
    ("BACKGROUND",(0,0),(0,0), NAVY),
    ("BACKGROUND",(1,0),(1,0), PAC_COL),
    ("BACKGROUND",(2,0),(2,0), JH_COL),
    ("BACKGROUND",(3,0),(3,0), SEC_COL),
    ("BACKGROUND",(4,0),(4,0), NW_COL),
    ("BACKGROUND",(0,1),(0,-1), LBLUE),
    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"TOP"),
]
for i in range(1, len(qf_data)):
    bg = LIGHT if i % 2 == 0 else WHITE
    if "11,316" in str(qf_rows[i-1]) or "Age 90" in str(qf_rows[i-1]):
        bg = LGOLD
    qf_style.append(("BACKGROUND",(0,i),(-1,i), bg))
qf_t.setStyle(TableStyle(qf_style))
story.append(qf_t)
story.append(Spacer(1,7))

# ── SECTION 2: CHARGES BREAKDOWN ──────────────────────────────────────────
story.append(p("2.  ANNUAL CHARGES BREAKDOWN — WHAT EACH CARRIER DEDUCTS FROM YOUR POLICY", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,4))

ch_hdr2 = [p(h, ch_s) for h in ["Year / Age", "Charge Type",
    "Pacific Life", "John Hancock", "Securian (MN Life)", "Nationwide"]]
ch_data = [ch_hdr2,
    # Year 1
    [p("Year 1\nAge 47", bold_s), p("Premium load", body_s), p("$22,866 (5.9%)", cg_s), p("$26,725 (7.0%)", ca_s), p("~$23,501 est. (6%)", cv_s), p("$31,371 (8.0%) ⚠", cr_s)],
    [p("", cv_s), p("Admin / Issue", body_s), p("$120", cg_s), p("$24,870", cr_s), p("~$24,870 est.", cr_s), p("$120", cg_s)],
    [p("", cv_s), p("COI (Insurance)", body_s), p("$68,094", cr_s), p("$1,091", cg_s), p("~est.", cv_s), p("$21,351", ca_s)],
    [p("", cv_s), p("Coverage / Index chg", body_s), p("$4,766", cv_s), p("$24 rider", cg_s), p("—", cv_s), p("$4,216", cv_s)],
    [p("", cv_s), p("TOTAL Year 1", S("tot",fontSize=7.5,fontName="Helvetica-Bold",textColor=NAVY,alignment=TA_LEFT)), p("$95,846", cr_s), p("$52,710", cg_s), p("~est.", cv_s), p("$57,058", ca_s)],
    # Year 10
    [p("Year 10\nAge 57", bold_s), p("Premium load", body_s), p("$20,723", cv_s), p("$20,883", cv_s), p("~est.", cv_s), p("$19,527", cg_s)],
    [p("", cv_s), p("Admin / Issue", body_s), p("$120", cg_s), p("$64,425", cr_s), p("~est.", cv_s), p("$120", cg_s)],
    [p("", cv_s), p("COI (Insurance)", body_s), p("$19,825", cv_s), p("$20,881", cv_s), p("~$20k est.", cv_s), p("$26,427", ca_s)],
    [p("", cv_s), p("TOTAL Year 10", S("tot2",fontSize=7.5,fontName="Helvetica-Bold",textColor=NAVY,alignment=TA_LEFT)), p("$40,668", cg_s), p("$106,213", cr_s), p("~est.", cv_s), p("$46,074", cv_s)],
    # Year 20
    [p("Year 20\nAge 67", bold_s), p("Premium load", body_s), p("$20,723", cv_s), p("$6,961", cg_s), p("~est.", cv_s), p("$19,527", cv_s)],
    [p("", cv_s), p("Admin / Issue", body_s), p("$120", cg_s), p("$35,107", cr_s), p("~est.", cv_s), p("$120", cg_s)],
    [p("", cv_s), p("COI (Insurance)", body_s), p("$52,782", ca_s), p("$55,976", ca_s), p("~est.", cv_s), p("$75,056", cr_s)],
    [p("", cv_s), p("TOTAL Year 20", S("tot3",fontSize=7.5,fontName="Helvetica-Bold",textColor=NAVY,alignment=TA_LEFT)), p("$73,625", cg_s), p("$98,044", cr_s), p("~est.", cv_s), p("$94,703", ca_s)],
    # 20-yr totals
    [p("20-Year\nTOTALS", S("totyr",fontSize=7.5,fontName="Helvetica-Bold",textColor=NAVY,alignment=TA_LEFT)),
     p("ALL CHARGES", S("tota",fontSize=7.5,fontName="Helvetica-Bold",textColor=NAVY,alignment=TA_LEFT)),
     p("~$1,220,005 ★", cg_s), p("~$1,781,539 ⚠", cr_s), p("~est.", cv_s), p("~$1,200,000 est.", cg_s)],
]
ct = Table(ch_data, colWidths=[0.75*inch, 1.15*inch, 1.35*inch, 1.35*inch, 1.35*inch, 1.35*inch])
ct.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("BACKGROUND",(0,1),(1,1),LPAC),("BACKGROUND",(0,6),(1,6),LPAC),("BACKGROUND",(0,10),(1,10),LPAC),
    ("BACKGROUND",(0,14),(1,14),LIGHT),
    ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
    ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LPAC,LIGHT,LIGHT,LIGHT,colors.HexColor("#FFFDE7"),
                                       LPAC,LIGHT,LIGHT,colors.HexColor("#FFFDE7"),
                                       LPAC,LIGHT,LIGHT,colors.HexColor("#FFFDE7"),LIGHT]),
]))
story.append(ct)
story.append(Spacer(1,4))
charges_note = Table([[p(
    "⚠  KEY FLAGS:  (1) JH Year 1 admin charge ($24,870) and Year 10 admin ($64,425) are very high — JH embeds most charges in admin/issue vs other carriers. "
    "(2) Nationwide has the HIGHEST premium load at 8.0% in Year 1. "
    "(3) Pacific Life front-loads the COI in early years (LTP design) but it drops significantly after Year 5 — this is the 'Long-Term Performance' design feature. "
    "(4) JH's underwriting class is STANDARD SMOKER while PAC, SEC, NW are STANDARD TOBACCO — "
    "if Monaj qualifies for PREFERRED TOBACCO, JH's charges would drop materially.",
    small_s)]], colWidths=[7.6*inch])
charges_note.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LGOLD),("BOX",(0,0),(-1,-1),0.5,GOLD),
    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7)]))
story.append(charges_note)
story.append(Spacer(1,7))

# ── SECTION 3: INDEX ACCOUNTS & CAPS ─────────────────────────────────────
story.append(p("3.  INDEX ACCOUNTS — CAPS, FLOORS, PARTICIPATION RATES & HISTORICAL RETURNS", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,4))

idx_hdr = [p(h, ch_s) for h in ["Carrier", "Account Name", "Index", "Cap Rate", "Participation", "Floor",
    "25-yr Avg\nor Max Illus.", "≥ 7.5%?", "Account\nCharge"]]
idx_rows = [idx_hdr,
    # Pacific Life
    [p("Pacific Life", S("pl",fontSize=7.5,fontName="Helvetica-Bold",textColor=PAC_COL,alignment=TA_LEFT)),
     p("1-Yr Indexed (main)", body_s), p("S&P 500", cc_s), p("10.00%", ca_s), p("100%", cv_s), p("0%", cg_s),
     p("6.72% (25-yr avg)", cr_s), p("✗ NO", cr_s), p("None", cg_s)],
    [p("", cv_s), p("1-Yr High Cap ★", body_s), p("S&P 500", cc_s), p("12.00%", cg_s), p("100%", cv_s), p("0%", cg_s),
     p("7.89% (25-yr avg) ✓", cg_s), p("✓ YES", cg_s), p("0.80%/yr", ca_s)],
    [p("", cv_s), p("1-Yr QQQ", body_s), p("Invesco QQQ", cc_s), p("10.50%", cv_s), p("100%", cv_s), p("0%", cg_s),
     p("7.46% (25-yr avg) ✓", cg_s), p("✓ YES", cg_s), p("None", cg_s)],
    [p("", cv_s), p("1-Yr No Cap Dynamic Par", body_s), p("S&P 500", cc_s), p("Uncapped", cg_s), p("45% illus / 5% guar", ca_s), p("0%", cg_s),
     p("Uncapped — variable", ca_s), p("Varies", ca_s), p("None", cg_s)],
    [p("", cv_s), p("1-Yr High Par (Volatility Ctrl)", body_s), p("BlackRock Endura", cc_s), p("Uncapped", cg_s), p("200% cur / 25% guar", cv_s), p("0%", cg_s),
     p("Data < 10 yrs", ca_s), p("?", ca_s), p("None", cg_s)],
    [p("", cv_s), p("5-Yr High Par ★★", body_s), p("S&P 500", cc_s), p("Uncapped", cg_s), p("110% cur / 105% guar", cg_s), p("0%", cg_s),
     p("7.13% annualized avg", ca_s), p("~Marginal", ca_s), p("None", cg_s)],
    # John Hancock
    [p("John Hancock", S("jh2",fontSize=7.5,fontName="Helvetica-Bold",textColor=JH_COL,alignment=TA_LEFT)),
     p("Base Capped S&P", body_s), p("S&P 500", cc_s), p("11.65%", cg_s), p("100%", cv_s), p("0%", cg_s),
     p("7.39% (25-yr avg)", ca_s), p("~Marginal", ca_s), p("None", cg_s)],
    [p("", cv_s), p("Nasdaq Capped ★★", body_s), p("Nasdaq-100", cc_s), p("14.00%", cg_s), p("100%", cv_s), p("0%", cg_s),
     p("9.11% (25-yr avg) ★", cg_s), p("✓ YES ★", cg_s), p("0.96%/yr", ca_s)],
    [p("", cv_s), p("High Capped + 30% mult ★", body_s), p("S&P 500", cc_s), p("12.25% + 30%×", cg_s), p("100%", cv_s), p("0%", cg_s),
     p("9.94% (25-yr avg) ★", cg_s), p("✓ YES ★★", cg_s), p("1.98%/yr", ca_s)],
    [p("", cv_s), p("Enh. High Cap + 80% mult ★★", body_s), p("S&P 500", cc_s), p("12.65% + 80%×", cg_s), p("100%", cv_s), p("0%", cg_s),
     p("14.00% (25-yr avg) ★★", cg_s), p("✓ YES ★★★", cg_s), p("4.98%/yr ⚠", cr_s)],
    [p("", cv_s), p("Barclays Global MA Classic", body_s), p("Barclays Global MA", cc_s), p("Uncapped", cg_s), p("165% par (20% guar)", cv_s), p("0%", cg_s),
     p("Index < 10 yrs", ca_s), p("?", ca_s), p("None", cg_s)],
    # Securian
    [p("Securian\n(MN Life)", S("sc2",fontSize=7.5,fontName="Helvetica-Bold",textColor=SEC_COL,alignment=TA_LEFT)),
     p("S&P 500 1-yr (main)", body_s), p("S&P 500", cc_s), p("10.50%", cv_s), p("100%", cv_s), p("0%", cg_s),
     p("6.93% compound avg", cr_s), p("✗ NO", cr_s), p("None", cg_s)],
    [p("", cv_s), p("S&P 500 Uncapped", body_s), p("S&P 500", cc_s), p("Uncapped", cg_s), p("100%", cv_s), p("0%", cg_s),
     p("Max illus 6.62%", cr_s), p("✗ NO", cr_s), p("5.50% spread", ca_s)],
    [p("", cv_s), p("Hindsight Multi-Index ★", body_s), p("S&P/Russell/Nasdaq blend", cc_s), p("8.25% cap", cv_s), p("60/40/0 best-of", cg_s), p("0%", cg_s),
     p("Max illus 8.25% ✓", cg_s), p("✓ YES", cg_s), p("None", cg_s)],
    [p("", cv_s), p("Performance Trigger", body_s), p("S&P 500", cc_s), p("Trigger-based", cv_s), p("N/A", cv_s), p("0%", cg_s),
     p("Max illus 6.62%", cr_s), p("✗ NO", cr_s), p("None", cg_s)],
    # Nationwide
    [p("Nationwide", S("nw2",fontSize=7.5,fontName="Helvetica-Bold",textColor=NW_COL,alignment=TA_LEFT)),
     p("S&P 500 PtP (main)", body_s), p("S&P 500", cc_s), p("10.50%", cv_s), p("100%", cv_s), p("0%", cg_s),
     p("Max illus 6.59% ⚠", cr_s), p("✗ NO ⚠", cr_s), p("None", cg_s)],
    [p("", cv_s), p("Multi-Index 14.5% cap", body_s), p("Multi-Index blend", cc_s), p("14.50%", cg_s), p("100%", cv_s), p("0%", cg_s),
     p("Max illus 6.59% ⚠", cr_s), p("✗ NO ⚠", cr_s), p("None", cg_s)],
    [p("", cv_s), p("Multi-Index High-Cap ★", body_s), p("Multi-Index blend", cc_s), p("25.00%", cg_s), p("100%", cv_s), p("0%", cg_s),
     p("Max illus 7.18%", ca_s), p("~Marginal", ca_s), p("0.55%/yr", cv_s)],
    [p("", cv_s), p("Nasdaq Monthly Avg", body_s), p("Nasdaq-100", cc_s), p("15.50%", cg_s), p("100%", cv_s), p("0%", cg_s),
     p("Max illus 6.59% ⚠", cr_s), p("✗ NO ⚠", cr_s), p("None", cg_s)],
    [p("", cv_s), p("BNPP Global H-Factor 300% ★", body_s), p("BNPP H-Factor", cc_s), p("Uncapped", cg_s), p("300% par (65% guar)", cg_s), p("0%", cg_s),
     p("Max illus 7.40%", ca_s), p("~Marginal", ca_s), p("0.75%/yr", cv_s)],
]

it = Table(idx_rows, colWidths=[0.75*inch,1.55*inch,0.95*inch,0.75*inch,0.9*inch,0.45*inch,1.1*inch,0.6*inch,0.6*inch])
it_style = [
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
    ("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]
pac_rows = [1,2,3,4,5,6]; jh_rows=[7,8,9,10,11]; sec_rows=[12,13,14,15]; nw_rows=[16,17,18,19,20]
for r in pac_rows: it_style.append(("BACKGROUND",(0,r),(-1,r), LPAC if r%2==0 else colors.HexColor("#F0F4F8")))
for r in jh_rows:  it_style.append(("BACKGROUND",(0,r),(-1,r), LJH if r%2==0 else colors.HexColor("#F5F0FA")))
for r in sec_rows: it_style.append(("BACKGROUND",(0,r),(-1,r), LSEC if r%2==0 else colors.HexColor("#FFF8F0")))
for r in nw_rows:  it_style.append(("BACKGROUND",(0,r),(-1,r), LNW if r%2==0 else colors.HexColor("#F0F5F0")))
it.setStyle(TableStyle(it_style))
story.append(it)
story.append(Spacer(1,4))

idx_flag = Table([[p(
    "⚠  NATIONWIDE RED FLAG:  ALL Nationwide index accounts have a maximum illustrated rate of 6.59% or below — "
    "NONE can illustrate above 7.5%. The highest is 7.40% (BNPP Select with 0.75% charge). "
    "With SOFR+1% loan rate of ~5.3-5.5%, the maximum arbitrage at Nationwide is only ~1.0-1.9%. "
    "This is dangerously thin. Any rate increase or underperformance eliminates the spread entirely.",
    S("fl",fontSize=7.5,fontName="Helvetica-Bold",textColor=RED,leading=11))
]], colWidths=[7.6*inch])
idx_flag.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LRED),("BOX",(0,0),(-1,-1),1,RED),
    ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8)]))
story.append(idx_flag)
story.append(Spacer(1,7))

# ── SECTION 4: ARBITRAGE ANALYSIS ─────────────────────────────────────────
story.append(p("4.  ARBITRAGE ANALYSIS — POLICY RETURN vs LOAN COST (SOFR+1%)", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,4))

# Assume SOFR = 4.35% as of May 2026
SOFR = 0.0435; LOAN = SOFR + 0.01

arb_hdr = [p(h, ch_s) for h in ["Carrier", "Account Used", "Expected\nReturn", "Loan Rate\n(SOFR+1%)", "Arbitrage\n(Return - Loan)", "≥ 7.5%\nReturn?", "Status"]]
arb_rows = [arb_hdr,
    # Illustrated scenario (6%)
    [p("ALL — Illustrated", S("ai",fontSize=7.5,fontName="Helvetica-Bold",textColor=NAVY,alignment=TA_LEFT)),
     p("6% base illustration", body_s), p("6.00%", ca_s), p(f"{LOAN:.2%}", cv_s),
     p(f"{0.06-LOAN:+.2%} ← VERY THIN ⚠", cr_s), p("✗ NO", cr_s), p("Insufficient", cr_s)],
    # PAC scenarios
    [p("Pacific Life", S("pl2",fontSize=7.5,fontName="Helvetica-Bold",textColor=PAC_COL,alignment=TA_LEFT)),
     p("1-Yr S&P 10% cap", body_s), p("6.72% (25-yr avg)", ca_s), p(f"{LOAN:.2%}", cv_s),
     p(f"{0.0672-LOAN:+.2%}", ca_s), p("✗ NO", cr_s), p("Thin", ca_s)],
    [p("", cv_s), p("High Cap 12% ★", body_s), p("7.89% (25-yr avg)", cg_s), p(f"{LOAN:.2%}", cv_s),
     p(f"{0.0789-LOAN:+.2%}", cg_s), p("✓ YES", cg_s), p("Acceptable", cg_s)],
    [p("", cv_s), p("QQQ 10.5%", body_s), p("7.46% (25-yr avg)", ca_s), p(f"{LOAN:.2%}", cv_s),
     p(f"{0.0746-LOAN:+.2%}", ca_s), p("~Marginal", ca_s), p("Borderline", ca_s)],
    # JH scenarios
    [p("John Hancock", S("jh3",fontSize=7.5,fontName="Helvetica-Bold",textColor=JH_COL,alignment=TA_LEFT)),
     p("Base Capped 11.65%", body_s), p("7.39% (25-yr avg)", ca_s), p(f"{LOAN:.2%}", cv_s),
     p(f"{0.0739-LOAN:+.2%}", ca_s), p("~Marginal", ca_s), p("Borderline", ca_s)],
    [p("", cv_s), p("Nasdaq 14% ★★", body_s), p("9.11% (25-yr avg)", cg_s), p(f"{LOAN:.2%}", cv_s),
     p(f"{0.0911-LOAN:+.2%} ★", cg_s), p("✓ YES ★★", cg_s), p("Strong ★", cg_s)],
    [p("", cv_s), p("High Capped +30% mult", body_s), p("9.94% (25-yr avg)", cg_s), p(f"{LOAN:.2%}", cv_s),
     p(f"{0.0994-LOAN:+.2%} ★", cg_s), p("✓ YES ★★★", cg_s), p("Very Strong ★", cg_s)],
    # SEC scenarios
    [p("Securian", S("sc3",fontSize=7.5,fontName="Helvetica-Bold",textColor=SEC_COL,alignment=TA_LEFT)),
     p("S&P 500 10.5% cap", body_s), p("6.93% (compound)", cr_s), p(f"{LOAN:.2%}", cv_s),
     p(f"{0.0693-LOAN:+.2%}", cr_s), p("✗ NO", cr_s), p("Thin", ca_s)],
    [p("", cv_s), p("Hindsight Multi-Index ★", body_s), p("8.25% (max illus)", cg_s), p(f"{LOAN:.2%}", cv_s),
     p(f"{0.0825-LOAN:+.2%}", cg_s), p("✓ YES", cg_s), p("Good", cg_s)],
    # NW scenarios
    [p("Nationwide", S("nw3",fontSize=7.5,fontName="Helvetica-Bold",textColor=NW_COL,alignment=TA_LEFT)),
     p("S&P 500 10.5% (main)", body_s), p("6.59% (max illus) ⚠", cr_s), p(f"{LOAN:.2%}", cv_s),
     p(f"{0.0659-LOAN:+.2%} ⚠", cr_s), p("✗ NO ⚠", cr_s), p("⚠ FLAGGED", cr_s)],
    [p("", cv_s), p("Multi-Index High-Cap 25%", body_s), p("7.18% (max illus)", ca_s), p(f"{LOAN:.2%}", cv_s),
     p(f"{0.0718-LOAN:+.2%}", ca_s), p("✗ NO", cr_s), p("Thin ⚠", ca_s)],
    [p("", cv_s), p("BNPP Select 300% par", body_s), p("7.40% (max illus)", ca_s), p(f"{LOAN:.2%}", cv_s),
     p(f"{0.0740-LOAN:+.2%}", ca_s), p("~Marginal", ca_s), p("Below threshold ⚠", ca_s)],
]

at = Table(arb_rows, colWidths=[0.9*inch, 1.75*inch, 1.05*inch, 0.9*inch, 1.2*inch, 0.8*inch, 1.0*inch])
at_style = [
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("BACKGROUND",(0,1),(-1,1),colors.HexColor("#FFEEE0")),  # illustrated — amber
    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]
for r in [2,3,4]: at_style.append(("BACKGROUND",(0,r),(-1,r), LPAC))
for r in [5,6,7]: at_style.append(("BACKGROUND",(0,r),(-1,r), LJH))
for r in [8,9]: at_style.append(("BACKGROUND",(0,r),(-1,r), LSEC))
for r in [10,11,12]: at_style.append(("BACKGROUND",(0,r),(-1,r), LNW))
at.setStyle(TableStyle(at_style))
story.append(at)
story.append(Spacer(1,4))

arb_note = Table([[p(
    f"SOFR assumed at {SOFR:.2%} (May 2026 approximate). Loan rate = SOFR+1% = {LOAN:.2%}. "
    "For premium financing to generate meaningful wealth, the policy needs to earn at LEAST 7.5% "
    "for a 2%+ spread. Accounts earning below 7.5% are flagged. "
    "JH Nasdaq (9.11%) and High Capped (9.94%) offer the strongest arbitrage. "
    "Nationwide's ENTIRE account lineup is below or at the 7.5% threshold — its max illustrated rate of 6.59% "
    "produces only a 1.24% spread. If SOFR rises to 5%+, Nationwide's illustrated scenario has NEGATIVE arbitrage.",
    small_s)]], colWidths=[7.6*inch])
arb_note.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LGOLD),("BOX",(0,0),(-1,-1),0.5,GOLD),
    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7)]))
story.append(arb_note)
story.append(Spacer(1,7))

# ── SECTION 5: POLICY VALUES & PERFORMANCE ────────────────────────────────
story.append(p("5.  POLICY VALUE COMPARISON AT KEY MILESTONES (All at 6% illustrated)", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,4))

pv_hdr = [p(h, ch_s) for h in ["Year / Age", "Pacific Life", "John Hancock ⚠", "Securian (MN Life)", "Nationwide"]]
pv_rows = [pv_hdr,
    [p("Year 1 / Age 48", bold_s), p("$291,720", cv_s), p("$329,079", cg_s), p("~$354,311 est.", cg_s), p("$335,711", cv_s)],
    [p("Year 5 / Age 52", bold_s), p("~$1,530,320", cv_s), p("~$1,892,360", cg_s), p("~$1,529,251 est.", cv_s), p("~$1,754,741", cv_s)],
    [p("Year 10 / Age 57", bold_s), p("$3,931,944", cv_s), p("$3,923,656", cv_s), p("~$3,980,296 est.", cg_s), p("$3,804,790", cr_s)],
    [p("Year 15 / Age 62", bold_s), p("~$7,095,996", cg_s), p("~$6,924,250", cv_s), p("~$7,081,513 est.", cv_s), p("~$6,757,432", cr_s)],
    [p("Year 20 / Age 67 ★", bold_s),
     p("$11,316,416 ★", cg_s), p("$11,182,884", cv_s), p("$10,241,142", cr_s), p("$10,604,989", ca_s)],
    [p("Year 25 / Age 72", bold_s), p("~$13,351,037", cg_s), p("~$13,075,374", cv_s), p("~$11,139,940 est.", cr_s), p("~$15,594,042", cg_s)],
]
pvt = Table(pv_rows, colWidths=[1.2*inch, 1.6*inch, 1.6*inch, 1.6*inch, 1.6*inch])
pvt.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LIGHT,WHITE]),
    ("BACKGROUND",(0,5),(-1,5),colors.HexColor("#FFF0C8")),  # Year 20 highlight
    ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]))
story.append(pvt)
story.append(Spacer(1,4))
pv_note = Table([[p(
    "⚠  CRITICAL NOTE ON JH UNDERWRITING:  JH is illustrated at STANDARD SMOKER while all others are STANDARD/PREFERRED TOBACCO. "
    "If Monaj qualifies for PREFERRED TOBACCO at JH, its Year 20 value would be significantly higher than shown above — "
    "potentially matching or exceeding Pacific Life. Request JH at Preferred Tobacco class before making a decision. "
    "Year 25 Nationwide value ($15.6M) appears highest because premiums stop after Year 25 for others; NW may continue premiums.",
    small_s)]], colWidths=[7.6*inch])
pv_note.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LGOLD),("BOX",(0,0),(-1,-1),0.5,GOLD),
    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7)]))
story.append(pv_note)
story.append(Spacer(1,7))

# ── SECTION 6: PROS & CONS ─────────────────────────────────────────────────
story.append(p("6.  PROS & CONS — EACH CARRIER AT A GLANCE", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,4))

carriers_pc = [
    ("Pacific Life", PAC_COL, LPAC,
     ["Preferred Tobacco class = lowest COI for a smoker",
      "Age 90 No-Lapse Guarantee (FREE) — longest NLG of all 4",
      "Highest Year 20 policy value at 6% illustrated",
      "5-Year uncapped account (110% par, guaranteed no cap)",
      "LTP design: front-loaded charges → much lower in years 11+",
      "BlackRock Endura uncapped (200% par) — unique"],
     ["Main 1-yr S&P cap only 10% (below 7.5% threshold at 6.72% avg)",
      "High Cap 12% has 0.80%/yr account charge",
      "Front-loaded Year 1 charges ($95k) look high vs others",
      "No Nasdaq account (uses Invesco QQQ at 10.5% instead)",
      "Max illustrated rate only 6.35% for main account"]),
    ("John Hancock", JH_COL, LJH,
     ["Nasdaq 14% cap — best cap of all 4 carriers (9.11% 25-yr avg)",
      "High Capped (12.25% + 30% mult) = 9.94% historical avg",
      "Enhanced High Capped (12.65% + 80% mult) = 14% avg (high charge)",
      "Vitality PLUS health bonus — unique, can add $100k-$500k over 20 yrs",
      "Barclays Global MA uncapped (165% par)",
      "Lowest annual premium ($381,790)"],
     ["⚠ Illustrated at STANDARD SMOKER — worse class than others",
      "If at Preferred Tobacco, values improve significantly (unknown)",
      "Admin/Issue charge very high ($24k-$65k/yr) vs peers",
      "No-lapse guarantee only 15 years (shortest of all 4)",
      "High Capped and Enhanced accounts have significant charges (1.98-4.98%/yr)"]),
    ("Securian (MN Life)", SEC_COL, LSEC,
     ["Lowest Year 20 risk of lapse (bonus interest from Year 11+)",
      "Hindsight multi-index (best of S&P/Russell/Nasdaq) — 8.25% max",
      "Strong carrier (Minnesota Life = top-tier)",
      "Simple transparent structure",
      "Fixed loan rate: 4% years 1-10, 4% years 11+ (low)"],
     ["Lowest Year 20 policy value ($10.2M vs PAC's $11.3M)",
      "Main S&P cap only 10.5% — below threshold at 6.93% avg",
      "Max illustrated rate only 6.62% — thin arbitrage",
      "Hindsight blended account is complex and less transparent",
      "No dedicated Nasdaq account"]),
    ("Nationwide", NW_COL, LNW,
     ["Highest cap options (25% Multi-Index, 15.5% Nasdaq monthly avg)",
      "300% participation on BNPP H-Factor (uncapped)",
      "Multiple index diversification options",
      "Competitive COI structure"],
     ["⚠ ALL accounts have max illustrated rate BELOW 7.5% threshold",
      "⚠ Main S&P max illustrated only 6.59% — FLAGGED",
      "⚠ Even best account (BNPP Select) max = 7.40% < 7.5%",
      "HIGHEST premium load at 8.0% Year 1 ($31,371)",
      "Thin arbitrage at current loan rates; any rate increase = near-zero spread",
      "Lowest flexibility — if SOFR rises 1%+, strategy unravels"]),
]

for carrier, col, bg, pros, cons in carriers_pc:
    pros_box = Table(
        [[p(f"✓ {pr}", S("pr",fontSize=7,fontName="Helvetica",textColor=colors.HexColor("#222222"),leading=10))] for pr in pros],
        colWidths=[3.6*inch])
    cons_box = Table(
        [[p(f"✗ {cn}", S("cn",fontSize=7,fontName="Helvetica",textColor=colors.HexColor("#222222"),leading=10))] for cn in cons],
        colWidths=[3.6*inch])
    for box, bcol, bbg in [(pros_box, GREEN, LGRN), (cons_box, RED, LRED)]:
        box.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),bbg),
            ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
            ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
        ]))

    header_row = Table([[p(f"  {carrier}", S("ch3",fontSize=8.5,fontName="Helvetica-Bold",textColor=WHITE,alignment=TA_LEFT))]],
                       colWidths=[7.6*inch])
    header_row.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),col),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8)]))
    story.append(header_row)

    pc_row = Table([[pros_box, Spacer(0.3*inch,1), cons_box]], colWidths=[3.65*inch,0.3*inch,3.65*inch])
    pc_row.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0),
        ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
    ]))
    story.append(pc_row)
    story.append(Spacer(1,5))

story.append(Spacer(1,5))

# ── FINAL VERDICT ─────────────────────────────────────────────────────────
story.append(p("7.  VERDICT & RECOMMENDED ACTIONS", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,5))

verdict_rows = [
    [p("RANKING\n(Year 20\nat 6%)", ch_s), p("Carrier", ch_s), p("Verdict", ch_s), p("Action Required", ch_s)],
    [p("#1", cg_s), p("Pacific Life (PAC)", S("v1",fontSize=8,fontName="Helvetica-Bold",textColor=PAC_COL,alignment=TA_LEFT)),
     p("Best year 20 value, Preferred Tobacco class, Age 90 NLG (free), LTP design drops charges long-term", body_s),
     p("Use High Cap 12% account (7.89%) NOT the default 10% S&P. Arbitrage threshold met.", body_s)],
    [p("#2", ca_s), p("John Hancock (JH)", S("v2",fontSize=8,fontName="Helvetica-Bold",textColor=JH_COL,alignment=TA_LEFT)),
     p("⚠ Currently at Standard Smoker — MUST get Preferred Tobacco re-quote. With Nasdaq 14% (9.11% avg), would likely rank #1 overall.", body_s),
     p("CRITICAL: Request re-illustration at Preferred Tobacco. Switch to Nasdaq 14% account. If at Preferred Tobacco, JH becomes the best choice.", body_s)],
    [p("#3", ca_s), p("Nationwide (NW)", S("v3",fontSize=8,fontName="Helvetica-Bold",textColor=NW_COL,alignment=TA_LEFT)),
     p("⚠ ALL ACCOUNTS below 7.5% threshold. Max illustrated 6.59%-7.40%. Thin arbitrage that evaporates if SOFR rises.", body_s),
     p("FLAG: Arbitrage concern. Only viable if SOFR stays below 4% for duration. Ask: why are max illustrated rates capped at 6.59%?", body_s)],
    [p("#4", cr_s), p("Securian / MN Life (SEC)", S("v4",fontSize=8,fontName="Helvetica-Bold",textColor=SEC_COL,alignment=TA_LEFT)),
     p("Lowest Year 20 value ($10.2M). Main account at 6.93% — below threshold. Hindsight account (8.25%) is viable but complex.", body_s),
     p("If choosing Securian, allocate to Hindsight Multi-Index account (8.25%), not the main S&P 10.5% cap.", body_s)],
]
vt = Table(verdict_rows, colWidths=[0.55*inch, 1.3*inch, 2.8*inch, 2.95*inch])
vt.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("BACKGROUND",(0,1),(-1,1),LGRN),
    ("BACKGROUND",(0,2),(-1,2),LJH),
    ("BACKGROUND",(0,3),(-1,3),LRED),
    ("BACKGROUND",(0,4),(-1,4),LSEC),
    ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"TOP"),
]))
story.append(vt)
story.append(Spacer(1,6))

fv = Table([[p(
    "BOTTOM LINE:  Ask your broker to re-run JH at PREFERRED TOBACCO class and allocate to the Nasdaq 14% account. "
    "If JH can get Preferred Tobacco underwriting, it becomes the clear winner on arbitrage (9.11% vs 5.35% loan = 3.76% spread). "
    "Pacific Life is the best of the 4 as currently illustrated, with the longest no-lapse guarantee (Age 90) and best smoker class. "
    "Nationwide should be questioned on why ALL its index accounts are capped at 6.59% max illustrated — this is below the minimum viable arbitrage threshold. "
    "All 4 carriers are illustrated at 6% which provides only ~0.5% arbitrage — request each carrier to show you the illustration at the MAX allowed rate, not the conservative 6%.",
    S("fv",fontSize=8,fontName="Helvetica-Bold",textColor=NAVY,leading=12))
]], colWidths=[7.6*inch])
fv.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LIGHT),("BOX",(0,0),(-1,-1),1.5,NAVY),
    ("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),9),
    ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10)]))
story.append(fv)
story.append(Spacer(1,5))
story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY))
story.append(Spacer(1,3))
story.append(p("Values derived from carrier illustrations dated May 18, 2026 provided by Nick Burgess, The Burgess Group. "
               "SOFR assumed at 4.35% (May 2026). Historical 25-year averages from carrier documents. Securian and NW estimated values based on available data. "
               "JH underwriting class noted as Standard Smoker which affects comparability. Not financial advice. Consult qualified advisors.", small_s))

doc.build(story)
print(f"PDF written -> {OUTPUT}")
