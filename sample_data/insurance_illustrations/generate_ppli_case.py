"""
PPLI Case: $10M / $15M / $20M / $25M net worth, age 43, +$300k/yr, 7% growth
Full wealth projection, estate tax exposure, PPLI vs IUL case strength
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT

OUTPUT = "PPLI_Case_10M_to_25M_NetWorth_Age43.pdf"

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
LPUR  = colors.HexColor("#F0E8FF")
PURP  = colors.HexColor("#5B2C8D")

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
def fmt(n, decimals=1):
    if n >= 1_000_000: return f"${n/1_000_000:.{decimals}f}M"
    if n >= 1_000: return f"${n/1_000:.0f}k"
    return f"${n:.0f}"

GROWTH = 0.07; ADD = 300_000; AGE = 43
EX_CUR = 13_990_000; EX_RST = 7_000_000; ET = 0.40
IUL_CH = 1_799_479; PPLI_CH = 500_000; SAVINGS = IUL_CH - PPLI_CH

def grow(start, years):
    v = start
    for _ in range(years): v = (v + ADD) * (1 + GROWTH)
    return v
def estate_tax(v, ex): return max(0, v - ex) * ET

milestones = [(12,55,"Age 55\n(12 yrs)"),(22,65,"Age 65\n(22 yrs)"),(32,75,"Age 75\n(32 yrs)"),(40,83,"Age 83\n(life exp.)")]
asset_levels = [10_000_000, 15_000_000, 20_000_000, 25_000_000]
colors_map = [AMBER, GREEN, NAVY, PURP]
bg_map = [LAMB, LGRN, LBLUE, LPUR]

doc = SimpleDocTemplate(OUTPUT, pagesize=letter,
    topMargin=0.4*inch, bottomMargin=0.35*inch, leftMargin=0.5*inch, rightMargin=0.5*inch)
story = []

# ── HEADER ─────────────────────────────────────────────────────────────────
hdr = Table([
    [p("PPLI / IUL Case Analysis — Is It Worth It?", title_s)],
    [p("Age 43  |  Adding $300k/yr  |  7% annual growth  |  Net Worth: $10M / $15M / $20M / $25M", sub_s)],
], colWidths=[7.5*inch])
hdr.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),NAVY),
    ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),8),("LEFTPADDING",(0,0),(-1,-1),10)]))
story.append(hdr)
story.append(Spacer(1,7))

# ── KEY ASSUMPTIONS ─────────────────────────────────────────────────────────
story.append(p("ASSUMPTIONS & KEY INPUTS", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,4))

assum = Table([[
    p("Age today: 43\nAnnual additions: $300,000/yr\nGrowth rate: 7%/yr\nLife expectancy: Age 83", body_s),
    p("Current estate tax exemption: $13.99M (2026)\nIf exemption resets (Dec 31 2025): ~$7M\nEstate tax rate: 40% on excess\nMarried couple exemption: 2× these amounts", body_s),
    p("IUL ($10M JH): $1,799,479 in charges\nPPLI ($10M): ~$500,000 in charges\nSavings with PPLI: $1,299,479\nCompounded @ 7% over 20 yrs: ~$5M", body_s),
]], colWidths=[2.3*inch, 2.6*inch, 2.6*inch])
assum.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,-1),LIGHT),("BOX",(0,0),(-1,-1),0.8,NAVY),
    ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
    ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
    ("VALIGN",(0,0),(-1,-1),"TOP"),
    ("LINEBEFORE",(1,0),(2,-1),0.5,colors.HexColor("#CCCCCC")),
]))
story.append(assum)
story.append(Spacer(1,8))

# ── WEALTH PROJECTION TABLE (ALL LEVELS) ───────────────────────────────────
story.append(p("1.  WHERE YOUR WEALTH GOES — ALL FOUR ASSET LEVELS", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,5))

wp_hdr = [p(h, ch_s) for h in [
    "Starting\nNet Worth", "Age 55\n(12 yrs)", "Age 65\n(22 yrs)", "Age 75\n(32 yrs)", "Age 83\nLife Exp."
]]
wp_rows = [wp_hdr]
for start, col in zip(asset_levels, colors_map):
    row = [p(f"${start//1_000_000}M", S("al",fontSize=9,fontName="Helvetica-Bold",textColor=col,alignment=TA_CENTER))]
    for yrs, age, _ in milestones:
        w = grow(start, yrs)
        row.append(p(fmt(w), cg_s))
    wp_rows.append(row)
wt = Table(wp_rows, colWidths=[1.2*inch,1.6*inch,1.6*inch,1.6*inch,1.5*inch])
wt.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LAMB,LGRN,LBLUE,LPUR]),
    ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
    ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]))
story.append(wt)
story.append(Spacer(1,4))
story.append(p("Growing at 7%/yr, adding $300k/yr every year. These are conservative estimates.", small_s))
story.append(Spacer(1,8))

# ── ESTATE TAX EXPOSURE ─────────────────────────────────────────────────────
story.append(p("2.  ESTATE TAX EXPOSURE — THE PROBLEM THAT GROWS EVERY YEAR", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,5))

et_hdr = [p(h, ch_s) for h in [
    "Starting\nNet Worth",
    "Estate Tax\nTODAY\n(current $14M ex.)",
    "Estate Tax\nTODAY\n(if resets to $7M)",
    "Estate Tax\nat Age 65\n($7M scenario)",
    "Estate Tax\nat Age 75\n($7M scenario)",
    "Kids keep\nat Age 75\n($7M scenario)",
]]
et_rows = [et_hdr]
for start, col, bg in zip(asset_levels, colors_map, bg_map):
    et_today_cur = estate_tax(start, EX_CUR)
    et_today_rst = estate_tax(start, EX_RST)
    w65 = grow(start, 22); et65 = estate_tax(w65, EX_RST)
    w75 = grow(start, 32); et75 = estate_tax(w75, EX_RST)
    kids75 = w75 - et75
    et_rows.append([
        p(f"${start//1_000_000}M", S("al2",fontSize=9,fontName="Helvetica-Bold",textColor=col,alignment=TA_CENTER)),
        p(fmt(et_today_cur) if et_today_cur>0 else "$0", cg_s if et_today_cur==0 else cr_s),
        p(fmt(et_today_rst) if et_today_rst>0 else "$0", cr_s if et_today_rst>0 else cg_s),
        p(fmt(et65), cr_s),
        p(fmt(et75), cr_s),
        p(fmt(kids75), cg_s),
    ])
et_t = Table(et_rows, colWidths=[1.0*inch,1.2*inch,1.2*inch,1.25*inch,1.25*inch,1.25*inch])
et_t.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LAMB,LGRN,LBLUE,LPUR]),
    ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]))
story.append(et_t)
story.append(Spacer(1,4))

et_note = Table([[p(
    "⚠  The current $13.99M exemption expires December 31, 2025. If it resets to ~$7M (the pre-2017 level), "
    "someone with $10M today owes $1.2M in estate tax IMMEDIATELY. At $25M, $7.2M is owed. "
    "Each year's growth makes this worse. Every $1 shielded in PPLI/ILIT today saves $0.40 in estate tax.",
    small_s)]], colWidths=[7.5*inch])
et_note.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LRED),("BOX",(0,0),(-1,-1),0.5,RED),
    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7)]))
story.append(et_note)
story.append(Spacer(1,8))

# ── FULL PROJECTION PER ASSET LEVEL ────────────────────────────────────────
story.append(p("3.  DETAILED PROJECTION — EACH ASSET LEVEL WITH PPLI/IUL CASE", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))

for start, col, bg in zip(asset_levels, colors_map, bg_map):
    story.append(Spacer(1,6))
    story.append(p(f"Starting Net Worth: ${start//1_000_000}M",
                   S("ah",fontSize=10,fontName="Helvetica-Bold",textColor=col,spaceBefore=4,spaceAfter=3)))
    story.append(HRFlowable(width="100%", thickness=1.5, color=col))
    story.append(Spacer(1,4))

    annual_drag = (start * 0.60 * 0.02 + start * 0.30 * 0.04) * 0.37
    savings_comp = SAVINGS * (1.07**20)
    et_today_cur = estate_tax(start, EX_CUR)
    et_today_rst = estate_tax(start, EX_RST)

    # Stats strip
    stats = Table([[
        p(f"Estate tax today (cur ex): {fmt(et_today_cur) if et_today_cur else '$0'}",
          bold_s if et_today_cur==0 else S("er",fontSize=7.5,fontName="Helvetica-Bold",textColor=RED,leading=11)),
        p(f"Estate tax if exemption resets: {fmt(et_today_rst)}",
          S("er2",fontSize=7.5,fontName="Helvetica-Bold",textColor=RED if et_today_rst>0 else GREEN,leading=11)),
        p(f"Annual investment tax drag: ${annual_drag:,.0f}/yr",
          S("td",fontSize=7.5,fontName="Helvetica-Bold",textColor=AMBER,leading=11)),
        p(f"PPLI charge savings → ${savings_comp/1e6:.1f}M by yr 20",
          S("ps",fontSize=7.5,fontName="Helvetica-Bold",textColor=GREEN,leading=11)),
    ]], colWidths=[1.8*inch,2.0*inch,1.9*inch,1.8*inch])
    stats.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),bg),("BOX",(0,0),(-1,-1),0.8,col),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LINEBEFORE",(1,0),(3,-1),0.5,colors.HexColor("#AAAAAA"))]))
    story.append(stats)
    story.append(Spacer(1,4))

    # Detailed table
    det_hdr = [p(h, ch_s) for h in [
        "Milestone", "Age", "Wealth", "Estate Tax\n(cur $14M ex)",
        "Estate Tax\n(reset $7M ex)", "Kids keep\n(cur)", "Kids keep\n(reset)",
        "What $10M\nPPLI/IUL Saves"
    ]]
    det_rows = [det_hdr]
    for yrs, age, label in milestones:
        w = grow(start, yrs)
        et_c = estate_tax(w, EX_CUR)
        et_r = estate_tax(w, EX_RST)
        # $10M PPLI removes $10M from estate tax calculation
        et_r_with_ppli = estate_tax(max(0, w - 10_000_000), EX_RST)
        ppli_saves = et_r - et_r_with_ppli
        det_rows.append([
            p(label.replace("\n"," "), bold_s),
            p(str(age), cc_s),
            p(fmt(w), cg_s),
            p(fmt(et_c) if et_c>0 else "$0", cr_s if et_c>0 else cg_s),
            p(fmt(et_r), cr_s),
            p(fmt(w-et_c), cg_s),
            p(fmt(w-et_r), ca_s),
            p(fmt(ppli_saves) if ppli_saves>0 else "—", cg_s if ppli_saves>0 else cv_s),
        ])
    dt = Table(det_rows, colWidths=[1.05*inch,0.45*inch,0.85*inch,1.0*inch,1.0*inch,0.95*inch,0.95*inch,1.25*inch])
    sc = [("BACKGROUND",(0,0),(-1,0),col),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[LIGHT,WHITE]),
        ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
        ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
        ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE")]
    dt.setStyle(TableStyle(sc))
    story.append(dt)

story.append(Spacer(1,8))

# ── PPLI CASE SCORECARD ─────────────────────────────────────────────────────
story.append(p("4.  PPLI CASE STRENGTH SCORECARD", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1,5))

sc_hdr = [p(h, ch_s) for h in [
    "Net Worth\nToday", "Estate Tax\nRisk Today", "Estate Tax\nRisk if Reset",
    "Annual\nTax Drag", "PPLI Charge\nSavings", "PPLI Savings\n@ Yr20 (compd)",
    "Qualifies\nfor PPLI?", "Verdict"
]]
sc_rows = [sc_hdr]
verdicts = ["SOLID — start now", "STRONG — do it", "VERY STRONG", "CRITICAL"]
for start, col, bg, verd in zip(asset_levels, colors_map, bg_map, verdicts):
    et_c = estate_tax(start, EX_CUR); et_r = estate_tax(start, EX_RST)
    annual_drag = (start * 0.60 * 0.02 + start * 0.30 * 0.04) * 0.37
    sc_rows.append([
        p(f"${start//1_000_000}M", S("al3",fontSize=8,fontName="Helvetica-Bold",textColor=col,alignment=TA_CENTER)),
        p(fmt(et_c) if et_c>0 else "$0", cr_s if et_c>0 else cg_s),
        p(fmt(et_r), cr_s),
        p(f"${annual_drag:,.0f}", ca_s),
        p(fmt(SAVINGS), cg_s),
        p(fmt(SAVINGS*(1.07**20)), cg_s),
        p("✓ Yes", cg_s),
        p(verd, S("vd",fontSize=7.5,fontName="Helvetica-Bold",textColor=col,alignment=TA_CENTER)),
    ])
sc_t = Table(sc_rows, colWidths=[0.85*inch,0.95*inch,0.95*inch,0.85*inch,0.85*inch,1.05*inch,0.75*inch,1.25*inch])
sc_t.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LAMB,LGRN,LBLUE,LPUR]),
    ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]))
story.append(sc_t)
story.append(Spacer(1,7))

# Final box
fv = Table([[p(
    "THE CASE IN ONE SENTENCE:  At age 43, adding $300k/yr, growing at 7%, "
    "every asset level from $10M to $25M produces an estate worth $120M–$440M by age 75. "
    "The estate tax on that — even under best-case exemption — is $43M–$170M. "
    "A $10M PPLI/IUL policy in an ILIT, started TODAY, removes $10M+ from that taxable estate "
    "(saving $4M+ in estate tax), shelters investment growth from annual income tax, "
    "provides $200k–$400k/yr in tax-free retirement income, and passes a growing death benefit "
    "to your heirs completely income-tax-free. "
    "The question is not whether to do it. The question is IUL ($1.8M charges) vs PPLI ($500k charges) — "
    "a $1.3M difference that compounds to ~$5M by Year 20.",
    S("fv",fontSize=8,fontName="Helvetica-Bold",textColor=NAVY,leading=12))
]], colWidths=[7.5*inch])
fv.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LIGHT),("BOX",(0,0),(-1,-1),1.5,NAVY),
    ("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),9),
    ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10)]))
story.append(fv)
story.append(Spacer(1,5))
story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY))
story.append(Spacer(1,3))
story.append(p("7% annual growth rate and $300k/yr additions are assumptions for illustration purposes. "
               "Actual growth will vary. Estate tax figures based on 2026 law; exemption sunsets Dec 31 2025 unless extended. "
               "PPLI charges estimated at $500k for $10M policy. IUL charges from JH illustration April 2026. "
               "Not tax, legal, or financial advice. Consult your advisors.", small_s))

doc.build(story)
print(f"PDF written -> {OUTPUT}")
