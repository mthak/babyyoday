"""
The COI Problem: What does the insurance cost actually cost you?
IUL vs Buy Term + Invest the Difference (BTID)
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT

OUTPUT = "Swati_Chugh_COI_Analysis_vs_TermPlusInvest.pdf"

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

# Data
annual_charges = {
    1:46000,2:49654,3:53613,4:57625,5:62165,
    6:66918,7:71966,8:77308,9:83299,10:89857,
    11:51529,12:55517,13:59883,14:64348,15:69207,
    16:74820,17:80454,18:86284,19:92334
}
TERM_COST=5_000
spy_returns={
    2000:-0.1014,2001:-0.1304,2002:-0.2337,2003:0.2638,2004:0.0899,
    2005:0.0300, 2006:0.1362, 2007:0.0353, 2008:-0.3849,2009:0.2345,
    2010:0.1278, 2011:0.0000, 2012:0.1341, 2013:0.2960, 2014:0.1139,
    2015:-0.0073,2016:0.0954, 2017:0.1942, 2018:-0.0624,2019:0.2888,
    2020:0.1626, 2021:0.2689, 2022:-0.1944,2023:0.2423, 2024:0.2331,
}
avg_spy=sum(spy_returns.values())/len(spy_returns)
IUL_NET_YR20=1_348_332
IUL_NET_YR25=1_288_841

def run_btid(start_cal):
    spy_p=0; rows=[]
    for yr in range(1,20):
        cal=start_cal+yr-1
        ret=spy_returns.get(cal,avg_spy)
        charge=annual_charges.get(yr,68000)
        saved=charge-TERM_COST
        spy_p+=saved
        spy_p=spy_p*(1+ret)
        rows.append({'yr':yr,'cal':cal,'ret':ret,'charge':charge,'saved':saved,'spy':spy_p})
    spy_p25=spy_p
    for yr in range(21,26):
        cal=start_cal+yr-1
        ret=spy_returns.get(cal,avg_spy)
        spy_p25=spy_p25*(1+ret)
    return rows, spy_p, spy_p25

scenarios=[
    (2005,"SCENARIO A","2005–2024","Best case — GFC + COVID recovery",GREEN,LGRN),
    (2003,"SCENARIO B","2003–2022","Middle — Dot-com recovery to COVID crash",AMBER,LAMB),
    (2000,"SCENARIO C","2000–2019","Worst — Dot-com crash + GFC",RED,LRED),
]
all_res={}
for sc,lbl,yr,desc,col,bg in scenarios:
    rows,yr20,yr25=run_btid(sc)
    all_res[sc]={'rows':rows,'yr20':yr20,'yr25':yr25,'lbl':lbl,'yr_range':yr,'desc':desc,'col':col,'bg':bg}

doc=SimpleDocTemplate(OUTPUT,pagesize=letter,
    topMargin=0.4*inch,bottomMargin=0.35*inch,leftMargin=0.5*inch,rightMargin=0.5*inch)
story=[]

# Header
hdr=Table([
    [p("The COI Problem: IUL vs Buy Term + Invest the Difference",title_s)],
    [p("What does the insurance cost actually cost you? | $5M policy | Age 43 | 3 historical windows",sub_s)],
],colWidths=[7.5*inch])
hdr.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),NAVY),
    ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),8),("LEFTPADDING",(0,0),(-1,-1),10)]))
story.append(hdr)
story.append(Spacer(1,7))

# COI facts box
coi_data=Table([
    [p("What the IUL charges you (Cost of Insurance + Admin)", S("hh",fontSize=8.5,fontName="Helvetica-Bold",textColor=WHITE,alignment=TA_CENTER))],
    [Table([
        [p("Year 1 annual charge:", bold_s), p("$46,000",cr_s)],
        [p("Year 10 annual charge:", bold_s), p("$89,857",cr_s)],
        [p("Year 19 annual charge:", bold_s), p("$92,334",cr_s)],
        [p("Average annual charge (19 yrs):", bold_s), p("$68,041",cr_s)],
        [p("Total charges over 19 years:", bold_s), p("$1,292,781",S("big",fontSize=9,fontName="Helvetica-Bold",textColor=RED,alignment=TA_RIGHT))],
        [p("20-yr term $5M, female 43, preferred:", bold_s), p("~$5,000/yr",cg_s)],
        [p("IUL charges are:", bold_s), p("14–18× more expensive than term insurance",S("mult",fontSize=8,fontName="Helvetica-Bold",textColor=RED,alignment=TA_RIGHT))],
    ],colWidths=[3.2*inch,4.0*inch])],
],colWidths=[7.4*inch])
coi_data.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),RED),
    ("BACKGROUND",(0,1),(-1,-1),LRED),
    ("BOX",(0,0),(-1,-1),1,RED),
    ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
    ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
    ("VALIGN",(0,0),(-1,-1),"TOP"),
]))
story.append(coi_data)
story.append(Spacer(1,9))

# Annual charges table
story.append(p("1.  WHAT THE IUL CHARGES YOU EVERY YEAR (Derived from illustration)",sect_s))
story.append(HRFlowable(width="100%",thickness=1.5,color=GOLD))
story.append(Spacer(1,5))

ac_hdr=[p(h,ch_s) for h in ["Policy\nYear","Policy\nAge","No-Charge\nValue (7.19%)",
    "Actual IUL\nValue","Annual\nCharge","Term Insurance\nEquivalent","Charge vs\nTerm (×)"]]
ac_rows=[ac_hdr]
no_charge={1:285849,2:592250,3:920682,4:1272728,5:1650086,
           6:2054576,7:2488149,8:2952896,9:3451058,10:3985038,
           11:4271563,12:4578688,13:4907896,14:5260773,15:5639023,
           16:6044469,17:6479066,18:6944911,19:7444250}
actual={1:239849,2:496596,3:771415,4:1065836,5:1381029,
        6:1718601,7:2080208,8:2467647,9:2882510,10:3326633,
        11:3561628,12:3813236,13:4082561,14:4371091,15:4680134,
        16:5010760,17:5364903,18:5744464,19:6151469}
cumul=0
for yr in range(1,20):
    age=43+yr
    charge=annual_charges[yr]
    cumul+=charge
    mult=charge/TERM_COST
    ac_rows.append([
        p(str(yr),cc_s),p(str(age),cc_s),
        p(fmt(no_charge[yr]),cv_s),
        p(fmt(actual[yr]),cv_s),
        p(fmt(charge),cr_s if charge>60000 else ca_s),
        p(fmt(TERM_COST),cg_s),
        p(f"{mult:.0f}×",cr_s),
    ])

# Add totals row
ac_rows.append([
    p("TOTAL",S("tot",fontSize=7.5,fontName="Helvetica-Bold",textColor=NAVY,alignment=TA_CENTER)),
    p("",cc_s),p("",cv_s),p("",cv_s),
    p(fmt(1292781),S("trc",fontSize=7.5,fontName="Helvetica-Bold",textColor=RED,alignment=TA_RIGHT)),
    p(fmt(TERM_COST*19),S("tgc",fontSize=7.5,fontName="Helvetica-Bold",textColor=GREEN,alignment=TA_RIGHT)),
    p(f"{1292781/(TERM_COST*19):.0f}×",S("tmult",fontSize=7.5,fontName="Helvetica-Bold",textColor=RED,alignment=TA_RIGHT)),
])

ac_t=Table(ac_rows,colWidths=[0.55*inch,0.55*inch,1.25*inch,1.2*inch,0.95*inch,1.15*inch,0.85*inch])
ac_t.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("ROWBACKGROUNDS",(0,1),(-1,-2),[WHITE,LIGHT]),
    ("BACKGROUND",(0,-1),(-1,-1),colors.HexColor("#F0F0F0")),
    ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
    ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ("LINEABOVE",(0,-1),(-1,-1),1,NAVY),
]))
story.append(ac_t)
story.append(Spacer(1,9))

# ── SECTION 2: BUY TERM + INVEST (BTID) MODEL ────────────────────────────────
story.append(p("2.  BUY TERM + INVEST THE DIFFERENCE (BTID) — YEAR-BY-YEAR",sect_s))
story.append(HRFlowable(width="100%",thickness=1.5,color=GOLD))
story.append(Spacer(1,5))

btid_note=Table([[p(
    "Strategy: Buy a $5M 20-year term policy for ~$5,000/yr. "
    "Take the difference between IUL charges and term cost ($41k–$87k/yr) and invest it directly in the S&P 500. "
    "No caps. No floors. No policy charges. After 19 years, compare to what the IUL net cash value would be.",
    body_s)]],colWidths=[7.5*inch])
btid_note.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,-1),LGOLD),("BOX",(0,0),(-1,-1),0.5,GOLD),
    ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
]))
story.append(btid_note)
story.append(Spacer(1,5))

# Scorecard
sc_hdr=[p(h,ch_s) for h in ["Scenario","Years","Charges\nSaved (19yr)","BTID Value\n@ Year 20","IUL Net\n@ Year 20","BTID\nWins By","BTID Value\n@ Year 25","IUL Net\n@ Year 25","BTID\nWins By"]]
sc_rows=[sc_hdr]
for sc,lbl,yr,desc,col,bg in scenarios:
    d=all_res[sc]
    total_saved=sum(annual_charges[y]-TERM_COST for y in range(1,20))
    sc_rows.append([
        p(lbl,S("sl",fontSize=7.5,fontName="Helvetica-Bold",textColor=col,alignment=TA_LEFT)),
        p(yr,S("yl",fontSize=7,fontName="Helvetica",textColor=GRAY,alignment=TA_CENTER)),
        p(fmt(total_saved),cv_s),
        p(fmt(d['yr20']),cg_s),
        p(fmt(IUL_NET_YR20),cv_s),
        p(fmt(d['yr20']-IUL_NET_YR20),cg_s),
        p(fmt(d['yr25']),cg_s),
        p(fmt(IUL_NET_YR25),cv_s),
        p(fmt(d['yr25']-IUL_NET_YR25),cg_s),
    ])
sct=Table(sc_rows,colWidths=[0.7*inch,0.7*inch,0.85*inch,0.85*inch,0.85*inch,0.75*inch,0.85*inch,0.85*inch,0.75*inch])
sct.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LGRN,LAMB,LRED]),
    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]))
story.append(sct)
story.append(Spacer(1,9))

# Year by year BTID for best case
story.append(p("3.  BTID YEAR-BY-YEAR DETAIL: Scenario A (2005–2024)",sect_s))
story.append(HRFlowable(width="100%",thickness=1.5,color=GREEN))
story.append(Spacer(1,4))

d=all_res[2005]
yr_hdr=[p(h,ch_s) for h in ["Yr","Age","Cal","S&P\nReturn","IUL\nCharge","Term\nCost","Saved to\nSPY","SPY\nPortfolio","IUL Value\n(same yr)"]]
yr_rows=[yr_hdr]
for r in d['rows']:
    age=43+r['yr']
    iUL_val=actual.get(r['yr'],0)
    yr_rows.append([
        p(str(r['yr']),cc_s),p(str(age),cc_s),p(str(r['cal']),cc_s),
        p(f"{r['ret']:.1%}",cr_s if r['ret']<0 else (cg_s if r['ret']>=0.135 else cv_s)),
        p(fmt(r['charge']),cr_s),
        p(fmt(TERM_COST),cg_s),
        p(fmt(r['saved']),ca_s),
        p(fmt(r['spy']),cg_s),
        p(fmt(iUL_val),cv_s),
    ])

yt=Table(yr_rows,colWidths=[0.38*inch,0.4*inch,0.42*inch,0.65*inch,0.82*inch,0.65*inch,0.82*inch,0.95*inch,0.95*inch])
style_cmds=[
    ("BACKGROUND",(0,0),(-1,0),GREEN),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
    ("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]
for i,r in enumerate(d['rows']):
    bg=LIGHT if i%2==0 else WHITE
    if r['ret']<0: bg=colors.HexColor("#FFF2F2")
    style_cmds.append(("BACKGROUND",(0,i+1),(-1,i+1),bg))
yt.setStyle(TableStyle(style_cmds))
story.append(yt)
story.append(Spacer(1,9))

# ── SECTION 4: THE HONEST PROS/CONS ──────────────────────────────────────────
story.append(p("4.  WHAT THE IUL BUYS YOU WITH THAT EXTRA COST",sect_s))
story.append(HRFlowable(width="100%",thickness=1.5,color=GOLD))
story.append(Spacer(1,5))

tradeoffs=[
    ("Cost of IUL (what you pay extra)", "$1.29M in charges over 19 years vs $95k in term premiums — difference: $1.19M",
     "This is the real price of the IUL features", RED),
    ("What the 0% floor costs you", "In 2008 (-38.5%), BTID portfolio dropped ~38%. IUL held flat. "
     "Floor saved roughly $500k-$800k in that one year alone in the best-case scenario.",
     "The floor has real dollar value in crash years", AMBER),
    ("What the cap costs you", "In strong years (2013: +29.6%, 2019: +28.9%, 2021: +26.9%), "
     "BTID captures full gain. IUL capped at 13.5%. You leave $15-17% on the table each big year.",
     "Cap costs you in bull markets — 9 out of 20 years hit the cap", RED),
    ("Tax-free income after Year 20", "IUL distributions are policy loans — no tax ever. "
     "BTID portfolio sells shares — capital gains tax on every withdrawal. Rough cost: 20% of gains.",
     "Worth roughly $200k–$500k in tax savings over the distribution years", GREEN),
    ("Death benefit", "BTID: $5M term for 20 years only, then expires. "
     "IUL: $5M+ permanent, grows to $8-12M+ by age 80, passes to heirs tax-free forever.",
     "Permanent growing death benefit has significant estate planning value", GREEN),
    ("Loan collateral safety", "BTID: In 2008 your SPY portfolio is DOWN 38.5% while loan is UP — "
     "lender could demand collateral when you least want to provide it. "
     "IUL: 0% floor means collateral never goes underwater from market losses.",
     "Critical risk difference in premium financing context", AMBER),
]

for title, body, note, col in tradeoffs:
    row=Table([[
        p("+" if col==GREEN else ("⚠" if col==AMBER else "−"),
          S("ic",fontSize=12,fontName="Helvetica-Bold",textColor=col,alignment=TA_CENTER)),
        Table([
            [p(title,S("tt",fontSize=8,fontName="Helvetica-Bold",textColor=col,leading=11))],
            [p(body,S("tb",fontSize=7.5,fontName="Helvetica",textColor=colors.HexColor("#222222"),leading=11))],
            [p(note,S("tn",fontSize=7,fontName="Helvetica-Oblique",textColor=GRAY,leading=10))],
        ],colWidths=[6.9*inch])
    ]],colWidths=[0.4*inch,6.9*inch])
    row.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,-1),LGRN if col==GREEN else (LAMB if col==AMBER else LRED)),
        ("BACKGROUND",(1,0),(1,-1),LIGHT),
        ("BOX",(0,0),(-1,-1),0.5,col),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LINEBEFORE",(1,0),(1,-1),0.5,col),
    ]))
    story.append(row)
    story.append(Spacer(1,4))

story.append(Spacer(1,5))
fv=Table([[p(
    "BOTTOM LINE — You are completely right: "
    "the IUL charges ~$68k/yr on average vs ~$5k/yr for term insurance. "
    "If you just invested that $63k/yr savings into SPY instead, "
    "you would have $2.2M–$6.9M more at Year 25 than the IUL net cash value. "
    "The COI is the real cost of three specific things: "
    "(1) permanent growing death benefit instead of 20-yr term, "
    "(2) the 0% floor protecting your collateral from ever going negative in crashes, "
    "and (3) tax-free income for life instead of taxable withdrawals. "
    "If those three things matter to you — the cost is justified. "
    "If you only need 20-year protection and are comfortable with market volatility — "
    "Buy Term + Invest is the financially superior strategy.",
    S("fv",fontSize=8,fontName="Helvetica-Bold",textColor=NAVY,leading=12))
]],colWidths=[7.5*inch])
fv.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LIGHT),("BOX",(0,0),(-1,-1),1.5,NAVY),
    ("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),9),
    ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10)]))
story.append(fv)
story.append(Spacer(1,5))
story.append(HRFlowable(width="100%",thickness=0.5,color=GRAY))
story.append(Spacer(1,3))
story.append(p("Annual charges derived from comparing no-charge policy simulation at 7.19% vs actual illustrated values. "
               "Term insurance cost estimate for female age 43 preferred non-tobacco. SPY returns from public historical data. "
               "Not financial or tax advice.", small_s))

doc.build(story)
print(f"PDF written -> {OUTPUT}")
