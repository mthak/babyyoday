"""
$10M Premium Financed IUL vs $10M PPLI — Complete Comparison PDF
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT

OUTPUT = "IUL_vs_PPLI_10M_Premium_Financed_Comparison.pdf"

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

doc = SimpleDocTemplate(OUTPUT, pagesize=letter,
    topMargin=0.4*inch, bottomMargin=0.35*inch,
    leftMargin=0.5*inch, rightMargin=0.5*inch)
story = []

# ── HEADER ────────────────────────────────────────────────────────────────────
hdr = Table([
    [p("$10M Premium Financed IUL vs $10M PPLI — Complete Comparison", title_s)],
    [p("Same financing  |  Same tax treatment  |  Same $0 out of pocket  |  What actually differs?", sub_s)],
], colWidths=[7.5*inch])
hdr.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),NAVY),
    ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),8),
    ("LEFTPADDING",(0,0),(-1,-1),10)]))
story.append(hdr)
story.append(Spacer(1,7))

# ── SECTION 1: WHAT IS IDENTICAL ──────────────────────────────────────────────
story.append(p("1.  WHAT IS IDENTICAL — BOTH PRODUCTS DO THIS EQUALLY WELL", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,5))

same_hdr = [p(h, ch_s) for h in ["Factor", "IUL ($10M JH)", "PPLI ($10M)", "Identical?"]]
same_rows = [same_hdr,
    [p("Legal structure", bold_s),
     p("Life insurance — IRC Section 7702", body_s),
     p("Life insurance — IRC Section 7702", body_s),
     p("✓ Identical", cg2_s)],
    [p("Premium financing", bold_s),
     p("Bank lends to ILIT at SOFR+1%", body_s),
     p("Bank lends to ILIT at SOFR+1%", body_s),
     p("✓ Identical", cg2_s)],
    [p("Your cash out of pocket", bold_s),
     p("$0 — lender pays all premiums", cg_s),
     p("$0 — lender pays all premiums", cg_s),
     p("✓ Identical", cg2_s)],
    [p("Loan repayment", bold_s),
     p("From policy cash value at Year 20/21", body_s),
     p("From policy cash value at Year 20/21", body_s),
     p("✓ Identical", cg2_s)],
    [p("Tax on distributions", bold_s),
     p("$0 — taken as policy loans (not income)", cg_s),
     p("$0 — taken as policy loans (not income)", cg_s),
     p("✓ Identical", cg2_s)],
    [p("Death benefit to heirs", bold_s),
     p("Income-tax-free under IRC 101(a)", cg_s),
     p("Income-tax-free under IRC 101(a)", cg_s),
     p("✓ Identical", cg2_s)],
    [p("Estate tax treatment (ILIT)", bold_s),
     p("Removed from taxable estate entirely", cg_s),
     p("Removed from taxable estate entirely", cg_s),
     p("✓ Identical", cg2_s)],
    [p("Tax-deferred growth inside", bold_s),
     p("Yes — grows tax-deferred inside policy", cg_s),
     p("Yes — grows tax-deferred inside policy", cg_s),
     p("✓ Identical", cg2_s)],
    [p("Non-MEC status", bold_s),
     p("Required — maintained by both structures", body_s),
     p("Required — maintained by both structures", body_s),
     p("✓ Identical", cg2_s)],
    [p("Step-up in basis issue", bold_s),
     p("Not applicable — death benefit is tax-free regardless", cg_s),
     p("Not applicable — death benefit is tax-free regardless", cg_s),
     p("✓ Identical", cg2_s)],
    [p("COI (mortality charge)", bold_s),
     p("~$417,002 over 20 years", body_s),
     p("~$150,000–200,000 over 20 years", body_s),
     p("Similar", ca2_s)],
]
st = Table(same_rows, colWidths=[1.7*inch, 2.25*inch, 2.25*inch, 1.1*inch])
st.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0), NAVY),
    ("BACKGROUND",(0,1),(0,-1), LBLUE),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LIGHT,WHITE]),
    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"TOP"),
]))
story.append(st)
story.append(Spacer(1,9))

# ── SECTION 2: WHAT IS DIFFERENT ──────────────────────────────────────────────
story.append(p("2.  WHAT IS DIFFERENT — THE FOUR KEY DISTINCTIONS", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,5))

diff_hdr = [p(h, ch_s) for h in ["Factor", "IUL ($10M JH)", "PPLI ($10M)", "Winner"]]
diff_rows = [diff_hdr,
    # FEES
    [p("Premium load", bold_s),
     p("$345,417 (broker/distributor commission)", cr_s),
     p("$0 — no retail distribution channel", cg_s),
     p("PPLI", cg2_s)],
    [p("Admin / Issue charges", bold_s),
     p("$1,035,923 over 20 years", cr_s),
     p("~$200,000–300,000 over 20 years", cg_s),
     p("PPLI", cg2_s)],
    [p("Total policy charges (20 yrs)", bold_s),
     p("$1,799,479", S("cr2b",fontSize=8,fontName="Helvetica-Bold",textColor=RED,alignment=TA_RIGHT)),
     p("~$400,000–600,000", S("cg2b",fontSize=8,fontName="Helvetica-Bold",textColor=GREEN,alignment=TA_RIGHT)),
     p("PPLI saves\n~$1.2–1.4M", cg2_s)],
    [p("Savings compounded\n@ 7% over 20 years", bold_s),
     p("—", cv_s),
     p("~$5,000,000 more in policy value", cg_s),
     p("PPLI", cg2_s)],
    # RETURNS
    [p("Return cap (upside)", bold_s),
     p("13.5% max — any S&P gain above this is lost", cr_s),
     p("None — full market participation uncapped", cg_s),
     p("PPLI", cg2_s)],
    [p("Return floor (downside)", bold_s),
     p("0% — policy NEVER loses value from market", cg_s),
     p("None — full downside in bad market years", cr_s),
     p("IUL for\nfinancing safety", ca2_s)],
    [p("Historical avg return\n(25 yrs S&P data)", bold_s),
     p("~7.05–7.91%/yr (after cap/floor applied)", body_s),
     p("~8–10%/yr depends on strategy (uncapped)", cg_s),
     p("PPLI likely\nhigher", cg2_s)],
    # INVESTMENT UNIVERSE
    [p("Investment options", bold_s),
     p("Carrier's indexed accounts only\n(S&P, Nasdaq, Barclays — 7 options)", cr_s),
     p("Any institutional manager — hedge funds,\nprivate equity, global equity, custom SMA", cg_s),
     p("PPLI\n(more choice)", cg2_s)],
    [p("Dividend capture", bold_s),
     p("No — S&P 500 index is price-only\n(no dividends)", cr_s),
     p("Yes — can invest in dividend-paying strategies", cg_s),
     p("PPLI", cg2_s)],
    # COLLATERAL RISK
    [p("Collateral call risk\n(premium financing)", bold_s),
     p("Very low — 0% floor means policy never\ngoes underwater from market losses", cg_s),
     p("Moderate — portfolio can drop in crashes,\npotential collateral call from lender", cr_s),
     p("IUL safer for\nfinancing", ca2_s)],
    # QUALIFICATION
    [p("Who can buy", bold_s),
     p("Anyone who medically qualifies", body_s),
     p("Qualified Purchaser only\n($5M+ investable assets)", body_s),
     p("IUL\n(accessible)", ca2_s)],
    [p("Investor control rule", bold_s),
     p("Not applicable — carrier controls\nindexed accounts", cg_s),
     p("IRS investor control doctrine applies —\ncannot direct specific trades", cr_s),
     p("IUL simpler", ca2_s)],
]
dt = Table(diff_rows, colWidths=[1.7*inch, 2.55*inch, 2.4*inch, 0.95*inch])
dt.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0), NAVY),
    ("BACKGROUND",(0,1),(0,-1), LBLUE),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LIGHT,WHITE]),
    # Shade fee rows differently
    ("BACKGROUND",(0,1),(-1,4), colors.HexColor("#FFF2F2")),   # fee rows — IUL loses
    ("BACKGROUND",(0,5),(-1,8), colors.HexColor("#F0FFF4")),   # return rows
    ("BACKGROUND",(0,9),(-1,10), colors.HexColor("#FFFDE7")),  # collateral rows
    ("BACKGROUND",(0,11),(-1,12), LIGHT),                      # qualification
    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"TOP"),
]))
story.append(dt)
story.append(Spacer(1,9))

# ── SECTION 3: THE FEE IMPACT ─────────────────────────────────────────────────
story.append(p("3.  THE FEE IMPACT IN DOLLAR TERMS", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,5))

fee_boxes = Table([[
    Table([
        [p("IUL — Charges You Pay", S("ih",fontSize=8.5,fontName="Helvetica-Bold",textColor=WHITE,alignment=TA_CENTER))],
        [Table([
            [p("Premium load (broker commission):", bold_s), p("$345,417", cr_s)],
            [p("Admin / Issue charges:", bold_s),           p("$1,035,923", cr_s)],
            [p("COI (insurance):", bold_s),                 p("$417,002", cr_s)],
            [p("Rider charges:", bold_s),                   p("$600", cv_s)],
            [p("TOTAL (20 years):", S("tot",fontSize=8,fontName="Helvetica-Bold",textColor=RED,leading=11)),
             p("$1,799,479", S("totv",fontSize=8,fontName="Helvetica-Bold",textColor=RED,alignment=TA_RIGHT))],
        ], colWidths=[2.1*inch, 1.2*inch])],
    ], colWidths=[3.4*inch]),
    Table([
        [p("PPLI — Charges You Pay", S("ph",fontSize=8.5,fontName="Helvetica-Bold",textColor=WHITE,alignment=TA_CENTER))],
        [Table([
            [p("Premium load (no distribution):", bold_s), p("$0", cg_s)],
            [p("Admin / Issue charges:", bold_s),          p("~$250,000", cg_s)],
            [p("COI (insurance):", bold_s),                p("~$175,000", cg_s)],
            [p("Other:", bold_s),                          p("~$75,000", cv_s)],
            [p("TOTAL (20 years):", S("tot2",fontSize=8,fontName="Helvetica-Bold",textColor=GREEN,leading=11)),
             p("~$500,000", S("tot2v",fontSize=8,fontName="Helvetica-Bold",textColor=GREEN,alignment=TA_RIGHT))],
        ], colWidths=[2.1*inch, 1.2*inch])],
    ], colWidths=[3.4*inch]),
]], colWidths=[3.5*inch, 3.5*inch])

for inner_t, col, bg in zip(fee_boxes._cellvalues[0], [RED, GREEN], [LRED, LGRN]):
    inner_t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), col),
        ("BACKGROUND",(0,1),(-1,-1), bg),
        ("BOX",(0,0),(-1,-1), 1, col),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
    ]))
    # Style inner table
    for sub_t in inner_t._cellvalues[1]:
        if hasattr(sub_t, 'setStyle'):
            sub_t.setStyle(TableStyle([
                ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
                ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
                ("LINEABOVE",(0,-1),(-1,-1),0.8,col),
            ]))

fee_boxes.setStyle(TableStyle([
    ("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0),
    ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
    ("VALIGN",(0,0),(-1,-1),"TOP"),
]))
story.append(fee_boxes)
story.append(Spacer(1,5))

savings_box = Table([[p(
    "PPLI saves ~$1,299,479 in charges.  Compounded at 7%/yr inside the policy for 20 years, "
    "that savings grows to ~$5,028,574 in additional policy value — all tax-free.",
    S("sv",fontSize=8.5,fontName="Helvetica-Bold",textColor=GREEN,leading=12,alignment=TA_CENTER))
]], colWidths=[7.5*inch])
savings_box.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,-1),LGRN),("BOX",(0,0),(-1,-1),1.5,GREEN),
    ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
    ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
]))
story.append(savings_box)
story.append(Spacer(1,9))

# ── SECTION 4: THE FLOOR TRADE-OFF ────────────────────────────────────────────
story.append(p("4.  THE ONE REAL STRUCTURAL DIFFERENCE — THE 0% FLOOR", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,5))

floor_hdr = [p(h, ch_s) for h in ["S&P 500 Return","IUL Credits\n(13.5% cap / 0% floor)","PPLI Returns\n(uncapped / no floor)","Difference"]]
spy_years = [
    (2000, -0.1014, 0.0000, -0.1014),
    (2001, -0.1304, 0.0000, -0.1304),
    (2002, -0.2337, 0.0000, -0.2337),
    (2003,  0.2638, 0.1350,  0.2638),
    (2006,  0.1362, 0.1350,  0.1362),
    (2008, -0.3849, 0.0000, -0.3849),
    (2009,  0.2345, 0.1350,  0.2345),
    (2013,  0.2960, 0.1350,  0.2960),
    (2019,  0.2888, 0.1350,  0.2888),
    (2021,  0.2689, 0.1350,  0.2689),
    (2022, -0.1944, 0.0000, -0.1944),
    (2023,  0.2423, 0.1350,  0.2423),
    (2024,  0.2331, 0.1350,  0.2331),
]
floor_rows = [floor_hdr]
for yr, spy, iul, ppli_raw in spy_years:
    iul_cred = max(0, min(0.135, spy))
    ppli_ret = spy  # uncapped, no floor (assume same index for comparison)
    diff = ppli_ret - iul_cred
    is_neg = spy < 0
    is_capped = spy > 0.135
    floor_rows.append([
        p(str(yr), cc_s),
        p(f"{iul_cred:.2%}", cg_s if iul_cred>0 else cr_s),
        p(f"{ppli_ret:.2%}", cr_s if ppli_ret<0 else (cg_s if ppli_ret>0.135 else cv_s)),
        p(f"{diff:+.2%}", cg_s if diff>0 else cr_s),
    ])
# Average row
avg_spy = sum(r[1] for r in spy_years)/len(spy_years)
avg_iul = sum(max(0,min(0.135,r[1])) for r in spy_years)/len(spy_years)
avg_ppli = sum(r[1] for r in spy_years)/len(spy_years)
floor_rows.append([
    p("AVG", S("avg",fontSize=7.5,fontName="Helvetica-Bold",textColor=NAVY,alignment=TA_CENTER)),
    p(f"{avg_iul:.2%}", cg_s),
    p(f"{avg_ppli:.2%}", ca_s),
    p(f"{avg_ppli-avg_iul:+.2%}", cr_s if avg_ppli<avg_iul else cv_s),
])

floor_t = Table(floor_rows, colWidths=[0.7*inch, 1.9*inch, 1.9*inch, 1.3*inch])
floor_style = [
    ("BACKGROUND",(0,0),(-1,0), NAVY),
    ("BACKGROUND",(0,-1),(-1,-1), colors.HexColor("#F0F0F0")),
    ("LINEABOVE",(0,-1),(-1,-1), 0.8, NAVY),
    ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
    ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]
for i, (yr, spy, *_) in enumerate(spy_years):
    ri = i+1
    if spy < 0:
        floor_style.append(("BACKGROUND",(0,ri),(-1,ri), LRED))
    elif spy > 0.135:
        floor_style.append(("BACKGROUND",(0,ri),(-1,ri), LGRN))
    else:
        floor_style.append(("BACKGROUND",(0,ri),(-1,ri), LIGHT if i%2==0 else WHITE))
floor_t.setStyle(TableStyle(floor_style))

floor_note = Table([
    [p("Green rows = S&P above 13.5% cap — PPLI captures full gain, IUL cuts off at 13.5%", small_s)],
    [p("Red rows = S&P negative — IUL credited 0% (floor saves you), PPLI takes the full loss", small_s)],
], colWidths=[5.9*inch])
floor_note.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0), LGRN),("BACKGROUND",(0,1),(-1,1), LRED),
    ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
    ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
]))

floor_layout = Table([[floor_t, Spacer(0.2*inch,1), floor_note]],
                     colWidths=[5.9*inch, 0.2*inch, 1.4*inch])
floor_layout.setStyle(TableStyle([
    ("VALIGN",(0,0),(-1,-1),"TOP"),
    ("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0),
    ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
]))
story.append(floor_layout)
story.append(Spacer(1,5))

floor_insight = Table([[p(
    "The 0% floor is the IUL's most valuable structural feature in a PREMIUM-FINANCED context. "
    "If the PPLI portfolio drops 30% in a crash year (as 2008 showed), "
    "the lender may issue a collateral call — demanding cash from you when you least want to pay. "
    "The IUL's floor completely eliminates this risk. "
    "PPLI can mitigate this by using a conservative balanced strategy (60/40), "
    "but it cannot guarantee zero downside the way the IUL floor does.",
    small_s)]], colWidths=[7.5*inch])
floor_insight.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LAMB),("BOX",(0,0),(-1,-1),0.5,AMBER),
    ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8)]))
story.append(floor_insight)
story.append(Spacer(1,9))

# ── SECTION 5: FINAL SCORECARD ────────────────────────────────────────────────
story.append(p("5.  FINAL SCORECARD — WHO WINS WHERE", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,5))

sc_hdr = [p(h, ch_s) for h in ["Category","IUL","PPLI","Winner"]]
sc_rows = [sc_hdr,
    [p("Total charges (20 years)", bold_s), p("$1,799,479", cr_s), p("~$500,000", cg_s),
     p("PPLI — saves ~$1.3M", cg2_s)],
    [p("Savings compounded (20 yrs @7%)", bold_s), p("—", cv_s), p("~$5,000,000 more", cg_s),
     p("PPLI", cg2_s)],
    [p("Return upside", bold_s), p("Capped at 13.5%", cr_s), p("Uncapped — full market", cg_s),
     p("PPLI", cg2_s)],
    [p("Return downside (floor)", bold_s), p("0% — never negative", cg_s), p("Full downside", cr_s),
     p("IUL — for financing safety", ca2_s)],
    [p("Collateral call risk (financing)", bold_s), p("None — floor prevents", cg_s), p("Possible in crashes", cr_s),
     p("IUL", ca2_s)],
    [p("Investment universe", bold_s), p("7 indexed accounts only", cr_s), p("Institutional — hedge funds, PE, custom", cg_s),
     p("PPLI", cg2_s)],
    [p("Tax treatment (all of it)", bold_s), p("Tax-free loans, tax-free DB", cg_s), p("Tax-free loans, tax-free DB", cg_s),
     p("IDENTICAL", S("id",fontSize=7.5,fontName="Helvetica-Bold",textColor=NAVY,alignment=TA_CENTER))],
    [p("Estate planning (ILIT)", bold_s), p("Removes from estate", cg_s), p("Removes from estate", cg_s),
     p("IDENTICAL", S("id2",fontSize=7.5,fontName="Helvetica-Bold",textColor=NAVY,alignment=TA_CENTER))],
    [p("Financing structure", bold_s), p("SOFR+1%, ILIT, $0 out of pocket", cg_s), p("SOFR+1%, ILIT, $0 out of pocket", cg_s),
     p("IDENTICAL", S("id3",fontSize=7.5,fontName="Helvetica-Bold",textColor=NAVY,alignment=TA_CENTER))],
    [p("Accessibility", bold_s), p("Any qualifying insured", cg_s), p("Qualified Purchaser ($5M+ investable) only", cr_s),
     p("IUL — easier access", ca2_s)],
    [p("Simplicity", bold_s), p("Carrier manages index accounts", cg_s), p("Investor control rules apply", cr_s),
     p("IUL — simpler", ca2_s)],
]
sct = Table(sc_rows, colWidths=[2.0*inch, 2.15*inch, 2.15*inch, 1.25*inch])
sct.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0), NAVY),
    ("BACKGROUND",(0,1),(0,-1), LBLUE),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LIGHT,WHITE]),
    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"TOP"),
]))
story.append(sct)
story.append(Spacer(1,7))

# ── FINAL BOTTOM LINE ─────────────────────────────────────────────────────────
fv = Table([[p(
    "THE BOTTOM LINE:  Both products are premium financed identically — "
    "same SOFR+1% lender, same ILIT structure, same $0 out of pocket, same tax-free treatment. "
    "The ONLY substantive differences are:  "
    "(1) PPLI charges ~$1.3M less in fees (worth ~$5M compounded over 20 years), "
    "(2) PPLI has no 13.5% cap so you capture full market upside, but "
    "(3) PPLI has no 0% floor — the IUL's floor is critical in a financed structure because "
    "a market crash cannot trigger a lender collateral call when the floor holds the policy value flat. "
    "If you qualify for PPLI ($5M+ investable assets), pursue it for the $1.3M in charge savings "
    "but ensure the investment strategy inside is conservative enough to protect the financing collateral.",
    S("fv",fontSize=8,fontName="Helvetica-Bold",textColor=NAVY,leading=12))
]], colWidths=[7.5*inch])
fv.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LIGHT),("BOX",(0,0),(-1,-1),1.5,NAVY),
    ("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),9),
    ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10)]))
story.append(fv)
story.append(Spacer(1,5))
story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY))
story.append(Spacer(1,3))
story.append(p("IUL charges from JH Accumulation IUL illustration April 2026. PPLI charges are estimates for $10M policy — "
               "actual charges vary by carrier (Pacific Life, Crown Global, Zurich, etc.). "
               "S&P 500 returns are historical price returns excluding dividends. "
               "Not financial, tax or legal advice. Consult qualified advisors.", small_s))

doc.build(story)
print(f"PDF written -> {OUTPUT}")
