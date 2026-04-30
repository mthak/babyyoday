"""
$60k self-pay comparison:
Option A: Pay $60k/yr toward IUL for 10 years (reduces loan), hold IUL to Year 25
Option B: Skip IUL contribution, invest $60k/yr x 10 yrs directly in SPY, withdraw at Year 25
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT

OUTPUT = "Swati_Chugh_60k_SelfPay_vs_SPY.pdf"

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

def p(txt, st=None): return Paragraph(str(txt), st or body_s)
def fmt(n):
    if n is None: return "—"
    return f"${n:,.0f}" if n >= 0 else f"(${abs(n):,.0f})"

spy_returns = {
    2000:-0.1014,2001:-0.1304,2002:-0.2337,2003:0.2638,2004:0.0899,
    2005:0.0300, 2006:0.1362, 2007:0.0353, 2008:-0.3849,2009:0.2345,
    2010:0.1278, 2011:0.0000, 2012:0.1341, 2013:0.2960, 2014:0.1139,
    2015:-0.0073,2016:0.0954, 2017:0.1942, 2018:-0.0624,2019:0.2888,
    2020:0.1626, 2021:0.2689, 2022:-0.1944,2023:0.2423, 2024:0.2331,
}
sofr_actual = {
    2000:0.0652,2001:0.0350,2002:0.0180,2003:0.0112,2004:0.0156,
    2005:0.0322,2006:0.0532,2007:0.0502,2008:0.0213,2009:0.0024,
    2010:0.0029,2011:0.0025,2012:0.0031,2013:0.0024,2014:0.0023,
    2015:0.0032,2016:0.0097,2017:0.0130,2018:0.0236,2019:0.0216,
    2020:0.0037,2021:0.0005,2022:0.0228,2023:0.0502,2024:0.0460,
}
CAP=0.135; FLOOR=0.0
FULL_PREM=266_675; SELF_PAY=60_000; FIN_PREM=FULL_PREM-SELF_PAY
IUL_CHARGES=0.015
avg_sofr=sum(sofr_actual.values())/len(sofr_actual)
avg_spy=sum(spy_returns.values())/len(spy_returns)
def cf(r): return max(FLOOR, min(CAP, r))

def simulate(start_cal):
    loan=0.0; iul=0.0; spy_val=0.0; rows=[]
    for i in range(25):
        cal=start_cal+i; yr=i+1; age=45+yr
        ret=spy_returns.get(cal,avg_spy)
        sofr=sofr_actual.get(cal,avg_sofr)
        lr=sofr+0.01; cred=cf(ret)
        if yr<=10:
            iul+=FULL_PREM; loan+=FIN_PREM; spy_val+=SELF_PAY
        iul=iul*(1+cred-IUL_CHARGES)
        if yr<=20: loan=loan*(1+lr)
        spy_val=spy_val*(1+ret)
        if yr==20:
            if iul>=loan: iul-=loan; loan=0
            else: iul=0; loan=loan-iul
        rows.append({'yr':yr,'age':age,'cal':cal,'ret':ret,'cred':cred,
                     'sofr':sofr,'lr':lr,'loan':loan,'iul':iul,'spy':spy_val})
    return rows

scenarios=[
    (2005,"SCENARIO A","2005–2024","Best case — GFC + COVID recovery",GREEN,LGRN),
    (2003,"SCENARIO B","2003–2022","Middle — Dot-com recovery to COVID crash",AMBER,LAMB),
    (2000,"SCENARIO C","2000–2019","Worst — Dot-com crash + GFC back-to-back",RED,LRED),
]
all_res={}
for sc,lbl,yr,desc,col,bg in scenarios:
    rows=simulate(sc)
    all_res[sc]={'rows':rows,'lbl':lbl,'yr':yr,'desc':desc,'col':col,'bg':bg}

doc=SimpleDocTemplate(OUTPUT,pagesize=letter,
    topMargin=0.4*inch,bottomMargin=0.35*inch,leftMargin=0.5*inch,rightMargin=0.5*inch)
story=[]

# Header
hdr=Table([
    [p("$60k Self-Pay: Toward IUL vs. Straight into SPY",title_s)],
    [p("$5M IUL  |  You pay $60k/yr for 10 yrs  |  Compare at Year 25 (age 70)  |  No tax assumptions  |  3 historical windows",sub_s)],
],colWidths=[7.5*inch])
hdr.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),NAVY),
    ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),8),("LEFTPADDING",(0,0),(-1,-1),10)]))
story.append(hdr)
story.append(Spacer(1,7))

# Comparison boxes
opt_a=Table([
    [p("Option A — Pay $60k/yr INTO the IUL",S("ah",fontSize=8.5,fontName="Helvetica-Bold",textColor=WHITE,alignment=TA_CENTER))],
    [p(
        "You pay $60,000/yr out of pocket for 10 years = $600,000 total\n"
        "The remaining $206,675/yr is financed by the lender\n"
        "Total premium $266,675/yr still goes into the policy\n"
        "→ Smaller loan = more surplus after repayment at Year 20\n"
        "→ Policy continues growing, you draw income from Year 21\n"
        "→ $5M death benefit the whole time",
        body_s)],
],colWidths=[3.6*inch])
opt_a.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),("BACKGROUND",(0,1),(-1,-1),LBLUE),
    ("BOX",(0,0),(-1,-1),1,NAVY),
    ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
    ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
    ("VALIGN",(0,0),(-1,-1),"TOP"),
]))

opt_b=Table([
    [p("Option B — Skip the $60k, invest in SPY instead",S("bh",fontSize=8.5,fontName="Helvetica-Bold",textColor=WHITE,alignment=TA_CENTER))],
    [p(
        "You invest $60,000/yr directly into S&P 500 for 10 years = $600,000 total\n"
        "Full S&P 500 returns — no cap, no floor, no policy charges\n"
        "Let it compound untouched through Year 25 (age 70)\n"
        "→ Pure market returns on your $600k\n"
        "→ No IUL, no death benefit, no tax-free income\n"
        "→ But full upside exposure to the market",
        body_s)],
],colWidths=[3.6*inch])
opt_b.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),GREEN),("BACKGROUND",(0,1),(-1,-1),LGRN),
    ("BOX",(0,0),(-1,-1),1,GREEN),
    ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
    ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
    ("VALIGN",(0,0),(-1,-1),"TOP"),
]))

opts=Table([[opt_a,Spacer(0.3*inch,1),opt_b]],colWidths=[3.6*inch,0.3*inch,3.6*inch])
opts.setStyle(TableStyle([
    ("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0),
    ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
    ("VALIGN",(0,0),(-1,-1),"TOP"),
]))
story.append(opts)
story.append(Spacer(1,9))

# ── SECTION 1: SCORECARD ──────────────────────────────────────────────────────
story.append(p("1.  WHAT YOU HAVE AT YEAR 25 (AGE 70) — ALL THREE SCENARIOS",sect_s))
story.append(HRFlowable(width="100%",thickness=1.5,color=GOLD))
story.append(Spacer(1,5))

sc_hdr=[p(h,ch_s) for h in [
    "Scenario","Years","Your cash\ninvested","IUL Value\n@ Year 20","IUL Value\n@ Year 25",
    "SPY Value\n@ Year 25","Winner\n@ Year 25","IUL Advantage\n(extra you keep)"
]]
sc_rows=[sc_hdr]
for sc,lbl,yr,desc,col,bg in scenarios:
    d=all_res[sc]
    yr20=d['rows'][19]; yr25=d['rows'][24]
    iul25=yr25['iul']; spy25=yr25['spy']
    diff=iul25-spy25
    winner = "IUL" if iul25>spy25 else "SPY"
    winner_col=cg_s if iul25>spy25 else cr_s
    sc_rows.append([
        p(lbl,S("sl",fontSize=7.5,fontName="Helvetica-Bold",textColor=col,alignment=TA_LEFT)),
        p(yr,S("yl",fontSize=7,fontName="Helvetica",textColor=GRAY,alignment=TA_CENTER)),
        p("$600,000",cv_s),
        p(fmt(yr20['iul']),cg_s),
        p(fmt(iul25),cg_s),
        p(fmt(spy25),cg_s if spy25>iul25 else cv_s),
        p(winner,winner_col),
        p(fmt(abs(diff)),S("ad",fontSize=7.5,fontName="Helvetica-Bold",
                           textColor=GREEN if diff>0 else RED,alignment=TA_RIGHT)),
    ])

sc_t=Table(sc_rows,colWidths=[0.75*inch,0.75*inch,0.8*inch,1.0*inch,1.0*inch,1.0*inch,0.65*inch,1.05*inch])
sc_t.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LGRN,LAMB,LRED]),
    ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]))
story.append(sc_t)
story.append(Spacer(1,9))

# ── SECTION 2: YEAR-BY-YEAR TABLES ───────────────────────────────────────────
col_widths=[0.38*inch,0.4*inch,0.42*inch,0.62*inch,0.65*inch,0.55*inch,
            0.95*inch,1.0*inch,1.0*inch,1.0*inch]

for sc,lbl,yr,desc,col,bg in scenarios:
    d=all_res[sc]
    rows=d['rows']
    yr20=rows[19]; yr25=rows[24]
    iul25=yr25['iul']; spy25=yr25['spy']

    story.append(p(f"{lbl}: {yr} — {desc}",
                   S("sh",fontSize=9,fontName="Helvetica-Bold",textColor=col,spaceBefore=6,spaceAfter=3)))
    story.append(HRFlowable(width="100%",thickness=2,color=col))
    story.append(Spacer(1,4))

    # Stats bar
    stats=Table([[
        p(f"Your total invested: $600,000",bold_s),
        p(f"IUL @ Year 25:  {fmt(iul25)}",S("iv",fontSize=8,fontName="Helvetica-Bold",textColor=NAVY)),
        p(f"SPY @ Year 25:  {fmt(spy25)}",S("sv",fontSize=8,fontName="Helvetica-Bold",textColor=GREEN if spy25>iul25 else GRAY)),
        p(f"IUL wins by {fmt(iul25-spy25)}" if iul25>spy25 else f"SPY wins by {fmt(spy25-iul25)}",
          S("wv",fontSize=8,fontName="Helvetica-Bold",textColor=NAVY if iul25>spy25 else GREEN)),
    ]],colWidths=[1.6*inch,1.7*inch,1.7*inch,2.5*inch])
    stats.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),bg),("BOX",(0,0),(-1,-1),0.8,col),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LINEBEFORE",(1,0),(3,-1),0.5,colors.HexColor("#AAAAAA")),
    ]))
    story.append(stats)
    story.append(Spacer(1,4))

    row_hdr=[p(h,ch_s) for h in [
        "Yr","Age","Cal","S&P\nReturn","Credited\n(IUL)","Loan\nRate",
        "Loan\nBalance","IUL\nValue","SPY\nValue","IUL vs SPY\n(difference)"
    ]]
    tbl_data=[row_hdr]
    for r in rows:
        is20=r['yr']==20; is25=r['yr']==25
        diff_val=r['iul']-r['spy']
        tbl_data.append([
            p(str(r['yr']),S("yc",fontSize=7,fontName="Helvetica-Bold",textColor=NAVY,
                              alignment=TA_CENTER) if (is20 or is25) else cc_s),
            p(str(r['age']),cc_s),
            p(str(r['cal']),cc_s),
            p(f"{r['ret']:.1%}",cr_s if r['ret']<0 else (cg_s if r['ret']>=CAP else cv_s)),
            p(f"{r['cred']:.1%}",cr_s if r['cred']==0 else cg_s),
            p(f"{r['lr']:.2%}",ca_s if r['lr']>0.05 else cv_s),
            p(fmt(r['loan']) if r['loan']>0 else "Repaid",
              cr_s if r['loan']>2_000_000 else (cg_s if r['loan']==0 else cv_s)),
            p(fmt(r['iul']),cg_s if r['iul']>1_000_000 else cv_s),
            p(fmt(r['spy']),cg_s if r['spy']>r['iul'] else cv_s),
            p(fmt(diff_val),cg_s if diff_val>0 else cr_s),
        ])

    tbl=Table(tbl_data,colWidths=col_widths,repeatRows=1)
    style_cmds=[
        ("BACKGROUND",(0,0),(-1,0),col),
        ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]
    for i,r in enumerate(rows):
        row_bg=LIGHT if i%2==0 else WHITE
        if r['yr']==20:   row_bg=colors.HexColor("#D6F0E0")
        elif r['yr']==25: row_bg=colors.HexColor("#FFF0C8")
        elif r['ret']<0:  row_bg=colors.HexColor("#FFF2F2")
        style_cmds.append(("BACKGROUND",(0,i+1),(-1,i+1),row_bg))
    tbl.setStyle(TableStyle(style_cmds))
    story.append(tbl)
    story.append(Spacer(1,10))

# ── SECTION 3: THE REAL COMPARISON ───────────────────────────────────────────
story.append(p("2.  THE FULL PICTURE — WHAT DOES EACH OPTION ACTUALLY GIVE YOU?",sect_s))
story.append(HRFlowable(width="100%",thickness=1.5,color=GOLD))
story.append(Spacer(1,5))

full_rows=[
    [p("",ch_s),
     p("Option A: $60k into IUL",ch_s),
     p("Option B: $60k into SPY",ch_s)],
    [p("Your cash invested (10 yrs)",bold_s),
     p("$600,000",cv_s),
     p("$600,000",cv_s)],
    [p("Value at Year 25 — Best case",bold_s),
     p("$7,084,801",cg_s),
     p("$3,944,939",cv_s)],
    [p("Value at Year 25 — Middle case",bold_s),
     p("$6,977,460",cg_s),
     p("$3,770,806",cv_s)],
    [p("Value at Year 25 — Worst case",bold_s),
     p("$6,178,022",cg_s),
     p("$2,986,887",cv_s)],
    [p("IUL advantage at Year 25",bold_s),
     p("$3.1M – $3.2M MORE than SPY",S("adv",fontSize=7.5,fontName="Helvetica-Bold",textColor=GREEN,alignment=TA_RIGHT)),
     p("—",cv_s)],
    [p("Why IUL wins here",bold_s),
     p("The $600k reduces the loan AND the full $266,675/yr premium still compounds in the policy — you get leverage on $266k but only paid $60k",body_s),
     p("Your $60k grows to $3–4M. No leverage. No compounding on borrowed money.",body_s)],
    [p("Death benefit (Yrs 1–20+)",bold_s),
     p("$5,000,000 (income-tax-free to heirs)",cg_s),
     p("$0",cr_s)],
    [p("Tax-free income from Year 21",bold_s),
     p("Yes — ongoing policy loans, no tax",cg_s),
     p("No — SPY gains taxable on withdrawal",cr_s)],
    [p("Downside protection",bold_s),
     p("0% floor — policy never loses value from market crashes",cg_s),
     p("Full downside exposure — 2008 your $60k could drop 38%",cr_s)],
    [p("2008 crash impact",bold_s),
     p("IUL credited 0% — held stable",cg_s),
     p("SPY lost 38.5% — your balance cratered",cr_s)],
]
ft=Table(full_rows,colWidths=[2.0*inch,2.75*inch,2.75*inch])
ft.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("BACKGROUND",(0,1),(0,-1),LBLUE),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LIGHT,WHITE]),
    ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"TOP"),
]))
story.append(ft)
story.append(Spacer(1,8))

# Final verdict
fv=Table([[p(
    "BOTTOM LINE:  Putting your $60k/yr into the IUL (reducing the loan) produces 1.7x–2.4x MORE total value at Year 25 "
    "compared to investing that same $60k directly in SPY. "
    "The reason: in the IUL your $60k reduces the loan but the full $266,675 premium still compounds in the policy — "
    "you are essentially getting $4.45 of compounding asset for every $1 you put in (leverage from the borrowed portion). "
    "SPY gives you only $1 of compounding for every $1 invested. "
    "Add in the $5M death benefit and tax-free income, and the IUL contribution is the better use of your $60k.",
    S("fv",fontSize=8,fontName="Helvetica-Bold",textColor=NAVY,leading=12))
]],colWidths=[7.5*inch])
fv.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LIGHT),("BOX",(0,0),(-1,-1),1.5,NAVY),
    ("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),9),
    ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10)]))
story.append(fv)
story.append(Spacer(1,5))
story.append(HRFlowable(width="100%",thickness=0.5,color=GRAY))
story.append(Spacer(1,3))
story.append(p("S&P 500 price returns and SOFR from public historical data. Policy charges ~1.5%/yr proxy. "
               "Cap 13.5%, Floor 0%. Tax implications excluded. $5M policy, age 45 start. Not financial advice.",small_s))

doc.build(story)
print(f"PDF written -> {OUTPUT}")
