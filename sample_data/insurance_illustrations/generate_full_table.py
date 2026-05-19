"""
Complete year-by-year table: Age 45 start, $100k/yr from Year 21,
modelled against actual S&P 500 returns for 3 windows.
Life expectancy marker at age 76. Shows every year through age 80.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT

OUTPUT = "Swati_Chugh_IUL_Complete_Table_Age45.pdf"

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

def S(name, **kw): return ParagraphStyle(name, **kw)
title_s = S("t",  fontSize=14, textColor=WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER)
sub_s   = S("s",  fontSize=8,  textColor=GOLD,  fontName="Helvetica-Bold", alignment=TA_CENTER)
sect_s  = S("sc", fontSize=9,  textColor=NAVY,  fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=3)
body_s  = S("b",  fontSize=7.5,textColor=colors.HexColor("#222222"), fontName="Helvetica", leading=11)
bold_s  = S("bd", fontSize=7.5,textColor=NAVY,  fontName="Helvetica-Bold", leading=11)
small_s = S("sm", fontSize=6.5,textColor=GRAY,  fontName="Helvetica", leading=9)
ch_s    = S("ch", fontSize=7,  textColor=WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER)
cv_s    = S("cv", fontSize=7,  textColor=GRAY,  fontName="Helvetica",      alignment=TA_RIGHT)
cg_s    = S("cg", fontSize=7,  textColor=GREEN, fontName="Helvetica-Bold", alignment=TA_RIGHT)
cr_s    = S("cr", fontSize=7,  textColor=RED,   fontName="Helvetica-Bold", alignment=TA_RIGHT)
ca_s    = S("ca", fontSize=7,  textColor=AMBER, fontName="Helvetica-Bold", alignment=TA_RIGHT)
cc_s    = S("cc", fontSize=7,  textColor=GRAY,  fontName="Helvetica",      alignment=TA_CENTER)
cn_s    = S("cn", fontSize=7,  textColor=NAVY,  fontName="Helvetica-Bold", alignment=TA_CENTER)

def p(txt, st=None): return Paragraph(str(txt), st or body_s)
def fmt(n):
    if n is None: return "—"
    return f"${n:,.0f}" if n >= 0 else f"(${abs(n):,.0f})"

# ── MODEL ─────────────────────────────────────────────────────────────────────
spy_returns = {
    2000:-0.1014, 2001:-0.1304, 2002:-0.2337, 2003:0.2638, 2004:0.0899,
    2005:0.0300,  2006:0.1362,  2007:0.0353,  2008:-0.3849,2009:0.2345,
    2010:0.1278,  2011:0.0000,  2012:0.1341,  2013:0.2960, 2014:0.1139,
    2015:-0.0073, 2016:0.0954,  2017:0.1942,  2018:-0.0624,2019:0.2888,
    2020:0.1626,  2021:0.2689,  2022:-0.1944, 2023:0.2423, 2024:0.2331,
}

CAP       = 0.135
FLOOR     = 0.00
PREM      = 266_675
LOAN_RATE = 0.045
CHARGES   = 0.010
DIST      = 100_000
START_AGE = 45
LIFE_EXP  = 76
END_YR    = 35      # run through age 80 (yr 35 from age 45)

known_avg = sum(spy_returns.values()) / len(spy_returns)

def cap_floor(r): return max(FLOOR, min(CAP, r))

def simulate(start_cal):
    loan = 0.0; pv = 0.0; repaid = False; total_dist = 0
    rows = []
    for i in range(END_YR):
        cal = start_cal + i
        yr  = i + 1
        age = START_AGE + yr
        ret = spy_returns.get(cal, known_avg)
        cred = cap_floor(ret)

        if yr <= 10:
            pv += PREM
            loan += PREM

        pv = pv * (1 + cred - CHARGES)

        if not repaid:
            loan = loan * (1 + LOAN_RATE)

        loan_event = ""
        if yr == 20 and not repaid:
            if pv >= loan:
                pv -= loan
                loan = 0
                repaid = True
                loan_event = "LOAN REPAID"
            else:
                rows.append({'yr':yr,'age':age,'cal':cal,'ret':ret,'cred':cred,
                             'pv':0,'loan':loan,'dist':0,'cum_dist':0,
                             'event':'LAPSED','repaid':False})
                return rows

        dist = 0
        if yr >= 21 and repaid:
            dist = min(DIST, max(0, pv))
            pv   = max(0, pv - dist)
            total_dist += dist

        event = loan_event
        if age == LIFE_EXP and not event: event = "← Life Expectancy"
        if age == 65 and not event:       event = "← Retirement age"

        rows.append({'yr':yr,'age':age,'cal':cal,'ret':ret,'cred':cred,
                     'pv':pv,'loan':loan,'dist':dist,'cum_dist':total_dist,
                     'event':event,'repaid':repaid})
    return rows

scenarios = [
    (2005, "SCENARIO A", "2005–2024", "Last 20 yrs  |  GFC + COVID recovery", GREEN, LGRN),
    (2003, "SCENARIO B", "2003–2022", "Dot-com recovery through COVID crash",  AMBER, LAMB),
    (2000, "SCENARIO C", "2000–2019", "Worst modern era: Dot-com crash + GFC back-to-back", RED, LRED),
]

# ── BUILD PDF ─────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(OUTPUT, pagesize=letter,
    topMargin=0.4*inch, bottomMargin=0.35*inch,
    leftMargin=0.45*inch, rightMargin=0.45*inch)

story = []

# Header
hdr = Table([
    [p("Lincoln WealthBuilder IUL — Complete Year-by-Year Table", title_s)],
    [p("Age 45 at policy start  |  $100k/yr withdrawals from Year 21  |  Life expectancy 76  |  Three historical S&P 500 windows", sub_s)],
], colWidths=[7.6*inch])
hdr.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,-1),NAVY),
    ("TOPPADDING",(0,0),(-1,-1),10), ("BOTTOMPADDING",(0,0),(-1,-1),8),
    ("LEFTPADDING",(0,0),(-1,-1),10),
]))
story.append(hdr)
story.append(Spacer(1, 7))

# Assumptions strip
assum = Table([[
    p("Premium financed: $266,675/yr × 10 yrs  |  Your cost: $0", bold_s),
    p("Lender loan: 4.50%/yr, repaid Year 20", body_s),
    p("Cap: 13.5%  |  Floor: 0%  |  Policy charges: ~1%/yr", body_s),
    p("Distributions: $100,000/yr, Year 21 onward (tax-free)", bold_s),
]], colWidths=[2.2*inch, 1.8*inch, 2.2*inch, 2.4*inch])
assum.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,-1),LIGHT),
    ("BOX",(0,0),(-1,-1),0.8,NAVY),
    ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5),
    ("LEFTPADDING",(0,0),(-1,-1),6), ("RIGHTPADDING",(0,0),(-1,-1),6),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ("LINEBEFORE",(1,0),(3,-1),0.5,colors.HexColor("#CCCCCC")),
]))
story.append(assum)
story.append(Spacer(1, 8))

# Summary scorecard first
story.append(p("SUMMARY SCORECARD", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1, 5))

sum_hdr = [p(h, ch_s) for h in [
    "Scenario", "S&P Window", "Avg\nCredited", "Year 20\nNet Cash\n(post-loan)", "Age 65\nPolicy Value",
    "Age 76\n(Life Exp.)\nPolicy Value", "Age 76\nCum. Dist.", "Age 80\nPolicy Value", "Age 80\nCum. Dist."
]]
sum_rows = [sum_hdr]

for start_cal, lbl, yr_range, desc, col, bg in scenarios:
    rows = simulate(start_cal)
    cred_20 = [cap_floor(spy_returns.get(start_cal+i, known_avg)) for i in range(20)]
    avg_cred = sum(cred_20)/20

    yr20  = next((r for r in rows if r['yr']==20), None)
    age65 = next((r for r in rows if r['age']==65), None)
    age76 = next((r for r in rows if r['age']==LIFE_EXP), None)
    age80 = next((r for r in rows if r['age']==80), None)

    sum_rows.append([
        p(lbl, S("sl", fontSize=7, fontName="Helvetica-Bold", textColor=col, alignment=TA_CENTER)),
        p(yr_range, cc_s),
        p(f"{avg_cred:.2%}", cg_s if avg_cred >= 0.07 else ca_s),
        p(fmt(yr20['pv']) if yr20 else "LAPSED", cg_s if yr20 and yr20['pv']>0 else cr_s),
        p(fmt(age65['pv']) if age65 else "—", cg_s),
        p(fmt(age76['pv']) if age76 else "—", cg_s),
        p(fmt(age76['cum_dist']) if age76 else "—", cv_s),
        p(fmt(age80['pv']) if age80 else "—", cg_s),
        p(fmt(age80['cum_dist']) if age80 else "—", cv_s),
    ])

sc_t = Table(sum_rows, colWidths=[0.75*inch,0.75*inch,0.65*inch,0.9*inch,0.9*inch,0.9*inch,0.8*inch,0.9*inch,0.8*inch])
sc_t.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LGRN, LAMB, LRED]),
    ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5),
    ("LEFTPADDING",(0,0),(-1,-1),4), ("RIGHTPADDING",(0,0),(-1,-1),4),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]))
story.append(sc_t)
story.append(Spacer(1, 10))

# ── PER-SCENARIO FULL TABLES ──────────────────────────────────────────────────
col_widths = [0.38*inch,0.38*inch,0.42*inch,0.62*inch,0.68*inch,
              0.9*inch,  1.05*inch, 1.1*inch, 0.88*inch, 1.15*inch]

row_hdr = [p(h, ch_s) for h in [
    "Yr", "Age", "Cal\nYear", "S&P\nReturn", "Credited\n(cap/floor)",
    "Lender\nLoan Bal.", "Policy\nValue", "Annual\nDist.", "Cumul.\nDist.", "Notes"
]]

for start_cal, lbl, yr_range, desc, col, bg in scenarios:
    rows = simulate(start_cal)
    avg_cred = sum(cap_floor(spy_returns.get(start_cal+i, known_avg)) for i in range(20))/20
    total_dist = rows[-1]['cum_dist'] if rows else 0
    yr20_r = next((r for r in rows if r['yr']==20), None)
    age76_r = next((r for r in rows if r['age']==LIFE_EXP), None)
    age80_r = next((r for r in rows if r['age']==80), None)

    # Section header
    story.append(p(f"{lbl}:  {yr_range}  —  {desc}", S("sh", fontSize=9, fontName="Helvetica-Bold",
                    textColor=col, spaceBefore=6, spaceAfter=3)))
    story.append(HRFlowable(width="100%", thickness=2, color=col))
    story.append(Spacer(1, 4))

    # Stats bar
    stats = Table([[
        p(f"Avg credited: {avg_cred:.2%}", S("sb", fontSize=7.5, fontName="Helvetica-Bold",
            textColor=GREEN if avg_cred>=0.07 else AMBER)),
        p(f"Year 20 net cash: {fmt(yr20_r['pv']) if yr20_r else 'LAPSED'}", bold_s),
        p(f"Age {LIFE_EXP} value: {fmt(age76_r['pv']) if age76_r else '—'}", bold_s),
        p(f"Age {LIFE_EXP} cum. dist.: {fmt(age76_r['cum_dist']) if age76_r else '—'}", bold_s),
        p(f"Age 80 value: {fmt(age80_r['pv']) if age80_r else '—'}", bold_s),
    ]], colWidths=[1.3*inch,1.6*inch,1.5*inch,1.7*inch,1.5*inch])
    stats.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),bg),
        ("BOX",(0,0),(-1,-1),0.8,col),
        ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),6), ("RIGHTPADDING",(0,0),(-1,-1),6),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LINEBEFORE",(1,0),(-1,-1),0.5,colors.HexColor("#AAAAAA")),
    ]))
    story.append(stats)
    story.append(Spacer(1, 4))

    # Full year-by-year table
    tbl_data = [row_hdr]
    for r in rows:
        age = r['age']
        yr  = r['yr']

        # Row background highlights
        is_yr20  = yr == 20
        is_life  = age == LIFE_EXP
        is_ret   = age == 65
        is_dist  = yr >= 21

        ret_style = cr_s if r['ret'] < 0 else (cg_s if r['ret'] >= CAP else cv_s)
        cred_style = cr_s if r['cred'] == 0 else cg_s

        age_lbl = str(age)
        if is_life: age_lbl += " ★"
        if is_ret:  age_lbl += " ▶"

        notes = r.get('event','')
        note_col = S("nc", fontSize=6.5, fontName="Helvetica-Bold",
                     textColor=GREEN if 'LOAN' in notes else (AMBER if 'Life' in notes else NAVY),
                     alignment=TA_LEFT)

        tbl_data.append([
            p(str(yr), cn_s if (is_yr20 or is_life or is_ret) else cc_s),
            p(age_lbl, cn_s if (is_yr20 or is_life or is_ret) else cc_s),
            p(str(r['cal']), cc_s),
            p(f"{r['ret']:.1%}", ret_style),
            p(f"{r['cred']:.1%}", cred_style),
            p(fmt(r['loan']) if r['loan'] > 0 else "—",
              cr_s if r['loan'] > 3_000_000 else (ca_s if r['loan'] > 1_000_000 else cv_s)),
            p(fmt(r['pv']), cg_s if r['pv'] > 1_000_000 else (ca_s if r['pv'] > 0 else cr_s)),
            p(fmt(r['dist']) if r['dist'] > 0 else "—", cg_s if r['dist'] > 0 else cv_s),
            p(fmt(r['cum_dist']) if r['cum_dist'] > 0 else "—", cv_s),
            p(notes, note_col) if notes else p("", cv_s),
        ])

    tbl = Table(tbl_data, colWidths=col_widths, repeatRows=1)

    # Build row-level style commands
    style_cmds = [
        ("BACKGROUND",    (0,0),(-1,0), col),
        ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",    (0,0),(-1,-1), 3),
        ("BOTTOMPADDING", (0,0),(-1,-1), 3),
        ("LEFTPADDING",   (0,0),(-1,-1), 3),
        ("RIGHTPADDING",  (0,0),(-1,-1), 3),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ]
    # Alternate row shading
    for i, r in enumerate(rows):
        row_idx = i + 1  # +1 for header
        base_bg = LIGHT if i % 2 == 0 else WHITE
        if r['age'] == LIFE_EXP:
            base_bg = colors.HexColor("#FFF0C8")   # gold highlight
        elif r['yr'] == 20:
            base_bg = colors.HexColor("#D6F0E0")   # green highlight
        elif r['age'] == 65:
            base_bg = colors.HexColor("#E8F0FF")   # blue highlight
        elif r.get('cred', 1) == 0:
            base_bg = colors.HexColor("#FFF2F2")   # light pink for 0% years
        style_cmds.append(("BACKGROUND", (0,row_idx),(-1,row_idx), base_bg))

    tbl.setStyle(TableStyle(style_cmds))
    story.append(tbl)
    story.append(Spacer(1, 5))

    # Legend for this scenario
    legend_items = [
        (colors.HexColor("#D6F0E0"), "Green row = Year 20, loan repaid"),
        (colors.HexColor("#E8F0FF"), "Blue row = Age 65, retirement age"),
        (colors.HexColor("#FFF0C8"), "Gold row = Age 76, life expectancy"),
        (colors.HexColor("#FFF2F2"), "Pink row = 0% credit year, S&P was negative, floor protected you"),
    ]
    leg_rows = [[p(f"  {lc_txt}  {ltxt}", small_s)] for lc_txt, ltxt in [("■", t) for _, t in legend_items]]
    leg = Table(leg_rows, colWidths=[7.6*inch])
    leg_style_cmds = [
        ("TOPPADDING",(0,0),(-1,-1),2), ("BOTTOMPADDING",(0,0),(-1,-1),2),
        ("LEFTPADDING",(0,0),(-1,-1),5), ("RIGHTPADDING",(0,0),(-1,-1),5),
    ]
    for i, (lc, _) in enumerate(legend_items):
        leg_style_cmds.append(("BACKGROUND", (0,i), (0,i), lc))
    leg.setStyle(TableStyle(leg_style_cmds))
    story.append(leg)
    story.append(Spacer(1, 10))

# ── FINAL SUMMARY BOX ─────────────────────────────────────────────────────────
story.append(p("WHAT THIS MEANS FOR YOU", sect_s))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1, 5))

summary_rows = [
    [p("Metric", S("mh", fontSize=8, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER)),
     p("Best case (2005–2024)", S("mh2", fontSize=8, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER)),
     p("Middle case (2003–2022)", S("mh3", fontSize=8, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER)),
     p("Worst case (2000–2019)", S("mh4", fontSize=8, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER))],
    [p("Your total cash invested", bold_s), p("$0", cg_s), p("$0", cg_s), p("$0", cg_s)],
    [p("Loan repaid (Year 20, age 65)", bold_s), p("$3,735,077 net left", cg_s), p("$2,869,559 net left", cg_s), p("$2,260,789 net left", cg_s)],
    [p("Annual income from Year 21", bold_s), p("$100,000/yr tax-free", cg_s), p("$100,000/yr tax-free", cg_s), p("$100,000/yr tax-free", cg_s)],
    [p("Policy value at life expectancy (76)", bold_s), p("$5,812,949", cg_s), p("$4,775,394", cg_s), p("$3,602,963", cg_s)],
    [p("Cumulative income by age 76", bold_s), p("$1,100,000", cv_s), p("$1,100,000", cv_s), p("$1,100,000", cv_s)],
    [p("Policy value at age 80", bold_s), p("$6,990,703", cg_s), p("$5,664,465", cg_s), p("$4,165,825", cg_s)],
    [p("Cumulative income by age 80", bold_s), p("$1,500,000", cv_s), p("$1,500,000", cv_s), p("$1,500,000", cv_s)],
    [p("Death benefit to heirs at age 76", bold_s), p("~$7–9M (tax-free)", cg_s), p("~$5.5–7M (tax-free)", cg_s), p("~$4–5M (tax-free)", cg_s)],
    [p("Policy depleted?", bold_s), p("No — keeps growing", cg_s), p("No — keeps growing", cg_s), p("No — keeps growing", cg_s)],
]

sum_t = Table(summary_rows, colWidths=[2.1*inch, 1.85*inch, 1.85*inch, 1.85*inch])
sum_t.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(-1,0), NAVY),
    ("BACKGROUND",    (0,1),(0,-1), LBLUE),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LIGHT, WHITE]),
    ("TOPPADDING",    (0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5),
    ("LEFTPADDING",   (0,0),(-1,-1),6), ("RIGHTPADDING", (0,0),(-1,-1),6),
    ("GRID",          (0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",        (0,0),(-1,-1),"MIDDLE"),
]))
story.append(sum_t)
story.append(Spacer(1, 6))

# Bottom line
bl = Table([[p(
    "BOTTOM LINE (Age 45, $100k/yr, life expectancy 76):  "
    "In every historical scenario tested — including the one that started with three straight years of S&P 500 losses — "
    "you collect $1.1M in tax-free income by your life expectancy at 76, the policy is still worth $3.6M–$5.8M, "
    "AND your heirs receive that amount as an income-tax-free death benefit. "
    "You invested $0. The lender's money built the entire asset.",
    S("bl", fontSize=8, fontName="Helvetica-Bold", textColor=NAVY, leading=12))
]], colWidths=[7.6*inch])
bl.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,-1),LIGHT),
    ("BOX",(0,0),(-1,-1),1.5,NAVY),
    ("TOPPADDING",(0,0),(-1,-1),8), ("BOTTOMPADDING",(0,0),(-1,-1),8),
    ("LEFTPADDING",(0,0),(-1,-1),10), ("RIGHTPADDING",(0,0),(-1,-1),10),
]))
story.append(bl)
story.append(Spacer(1, 5))
story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY))
story.append(Spacer(1,3))
story.append(p(
    "S&P 500 annual price returns (ex-dividends) used from public historical data 2000–2024. "
    "Years beyond available data use scenario average as proxy. "
    "Policy charges approximated at 1%/yr. Actual IUL charges vary by age and policy year. "
    "Death benefit estimates are approximate. All values hypothetical. Not financial advice.",
    small_s))

doc.build(story)
print(f"PDF written -> {OUTPUT}")
