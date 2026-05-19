"""
Lincoln $10M vs JH $10M — true apples-to-apples
Same cap: 13.5% / Same floor: 0% / Same index: S&P 500 / Same SOFR+1% loan
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT

OUTPUT = "Swati_Chugh_Lincoln10M_vs_JH10M_ApplesToApples.pdf"

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
title_s = S("t", fontSize=14, textColor=WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER)
sub_s   = S("s", fontSize=8,  textColor=GOLD,  fontName="Helvetica-Bold", alignment=TA_CENTER)
sect_s  = S("sc",fontSize=9,  textColor=NAVY,  fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=3)
body_s  = S("b", fontSize=7.5,textColor=colors.HexColor("#222222"), fontName="Helvetica", leading=11)
bold_s  = S("bd",fontSize=7.5,textColor=NAVY,  fontName="Helvetica-Bold", leading=11)
small_s = S("sm",fontSize=6.5,textColor=GRAY,  fontName="Helvetica", leading=9)
ch_s    = S("ch",fontSize=7.5,textColor=WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER)
cv_s    = S("cv",fontSize=7.5,textColor=GRAY,  fontName="Helvetica",      alignment=TA_RIGHT)
cg_s    = S("cg",fontSize=7.5,textColor=GREEN, fontName="Helvetica-Bold", alignment=TA_RIGHT)
cr_s    = S("cr",fontSize=7.5,textColor=RED,   fontName="Helvetica-Bold", alignment=TA_RIGHT)
ca_s    = S("ca",fontSize=7.5,textColor=AMBER, fontName="Helvetica-Bold", alignment=TA_RIGHT)
cc_s    = S("cc",fontSize=7.5,textColor=GRAY,  fontName="Helvetica",      alignment=TA_CENTER)

def p(txt, st=None): return Paragraph(str(txt), st or body_s)
def fmt(n):
    if n is None: return "—"
    return f"${n:,.0f}" if n >= 0 else f"(${abs(n):,.0f})"

# ── MODEL ─────────────────────────────────────────────────────────────────────
spy = {2000:-0.1014,2001:-0.1304,2002:-0.2337,2003:0.2638,2004:0.0899,
    2005:0.0300,2006:0.1362,2007:0.0353,2008:-0.3849,2009:0.2345,
    2010:0.1278,2011:0.0000,2012:0.1341,2013:0.2960,2014:0.1139,
    2015:-0.0073,2016:0.0954,2017:0.1942,2018:-0.0624,2019:0.2888,
    2020:0.1626,2021:0.2689,2022:-0.1944,2023:0.2423,2024:0.2331}
sofr={2000:0.0652,2001:0.0350,2002:0.0180,2003:0.0112,2004:0.0156,
    2005:0.0322,2006:0.0532,2007:0.0502,2008:0.0213,2009:0.0024,
    2010:0.0029,2011:0.0025,2012:0.0031,2013:0.0024,2014:0.0023,
    2015:0.0032,2016:0.0097,2017:0.0130,2018:0.0236,2019:0.0216,
    2020:0.0037,2021:0.0005,2022:0.0228,2023:0.0502,2024:0.0460}
avg_spy=sum(spy.values())/len(spy); avg_sofr=sum(sofr.values())/len(sofr)
CAP=0.135; FLOOR=0.0
def cf(r): return max(FLOOR,min(CAP,r))

LNC_C={1:92000,2:92694,3:93470,4:93786,5:94580,6:95146,7:95618,8:95954,9:96820,10:97956,
    11:8380,12:8946,13:9694,14:10012,15:10476,16:11752,17:12262,18:12352,19:12044,20:15564,
    21:8000,22:8200,23:8400,24:8600,25:8800,26:9000,27:9200,28:9400,29:9600,30:9800}
JH_PREM={1:689782,2:689782,3:689782,4:689782,5:118973,6:295509,7:528939,8:528939,9:528939,10:528939,11:528939,12:528939}
JH_C={1:91282,2:89868,3:95949,4:99398,5:65708,6:77643,7:95608,8:99497,9:103903,10:107403,
    11:90285,12:94347,13:84179,14:84565,15:85034,16:47800,17:48498,18:49392,19:50697,20:52390,
    21:31929,22:33748,23:36515,24:39983,25:43858,26:48102,27:51001,28:54369,29:57713,30:62085}

def sim_lnc(start):
    loan=0.0; pv=0.0; total_dist=0; rows=[]
    for i in range(30):
        yr=i+1; cal=start+i; age=43+yr
        ret=spy.get(cal,avg_spy); cred=cf(ret); lr=sofr.get(cal,avg_sofr)+0.01
        charge=LNC_C.get(yr,9000)
        pv+=533_350 if yr<=10 else 0; loan+=533_350 if yr<=10 else 0
        pv=pv*(1+cred); pv=max(0,pv-charge); loan=loan*(1+lr)
        dist=0
        if yr==20:
            if pv>=loan: pv-=loan; loan=0
            else: pv=0; loan=0
        if yr>=21 and loan==0 and pv>0:
            dist=min(408_172,pv); pv=max(0,pv-dist); total_dist+=dist
        rows.append({'yr':yr,'age':age,'cal':cal,'ret':ret,'cred':cred,'lr':lr,
                     'pv':pv,'dist':dist,'cum':total_dist,'charge':charge,'loan':loan})
    return rows

def sim_jh(start):
    loan=0.0; pv=0.0; total_dist=0; rows=[]
    for i in range(30):
        yr=i+1; cal=start+i; age=44+yr
        ret=spy.get(cal,avg_spy); cred=cf(ret); lr=sofr.get(cal,avg_sofr)+0.01
        charge=JH_C.get(yr,45000); prem=JH_PREM.get(yr,0)
        pv+=prem; loan+=prem if prem>0 else 0
        pv=pv*(1+cred); pv=max(0,pv-charge); loan=loan*(1+lr)
        dist=0
        if yr==21:
            if pv>=loan: pv-=loan; loan=0
            else: pv=0; loan=0
        if yr>=22 and loan==0 and pv>0:
            dist=min(200_000,pv); pv=max(0,pv-dist); total_dist+=dist
        rows.append({'yr':yr,'age':age,'cal':cal,'ret':ret,'cred':cred,'lr':lr,
                     'pv':pv,'dist':dist,'cum':total_dist,'charge':charge,'loan':loan})
    return rows

scenarios=[(2005,"A","2005–2024","Best — GFC + COVID recovery",GREEN,LGRN),
           (2003,"B","2003–2022","Middle — Dot-com recovery through COVID",AMBER,LAMB),
           (2000,"C","2000–2019","Worst — Dot-com crash + GFC",RED,LRED)]

doc=SimpleDocTemplate(OUTPUT,pagesize=letter,
    topMargin=0.4*inch,bottomMargin=0.35*inch,leftMargin=0.5*inch,rightMargin=0.5*inch)
story=[]

# HEADER
hdr=Table([[p("Lincoln $10M vs JH $10M — True Apples-to-Apples Comparison",title_s)],
           [p("Same cap: 13.5%  |  Same floor: 0%  |  Same index: S&P 500  |  Same SOFR+1% loan rate  |  3 historical windows",sub_s)]],
          colWidths=[7.5*inch])
hdr.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),NAVY),
    ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),8),("LEFTPADDING",(0,0),(-1,-1),10)]))
story.append(hdr)
story.append(Spacer(1,7))

# STRUCTURE COMPARISON
story.append(p("1.  STRUCTURE COMPARISON — WHAT'S ACTUALLY DIFFERENT?", sect_s))
story.append(HRFlowable(width="100%",thickness=1.5,color=GOLD))
story.append(Spacer(1,4))

struct_hdr=[p(h,ch_s) for h in ["Factor","Lincoln $10M\n(scaled from $5M illustration)","JH $10M\n(actual illustration)","Impact"]]
struct_rows=[struct_hdr,
    [p("Death benefit",bold_s),p("$10,000,000",cg_s),p("$10,000,000",cg_s),p("Same",cc_s)],
    [p("Annual premium",bold_s),p("$533,350/yr × 10 yrs (flat)",cv_s),p("$689,782/yr yrs 1-4, varies yrs 5-12",cv_s),p("JH front-loaded",ca_s)],
    [p("Total borrowed",bold_s),p("$5,333,500",cg_s),p("$6,347,244",cr_s),p("Lincoln borrows less",S("lw",fontSize=7.5,fontName="Helvetica-Bold",textColor=GREEN,alignment=TA_CENTER))],
    [p("Premium years",bold_s),p("10 years",cg_s),p("12 years",cv_s),p("Lincoln shorter",S("ls",fontSize=7.5,fontName="Helvetica-Bold",textColor=GREEN,alignment=TA_CENTER))],
    [p("Total policy charges",bold_s),p("~$1,101,506",cg_s),p("$1,799,479",cr_s),p("Lincoln $698k cheaper",S("lc",fontSize=7.5,fontName="Helvetica-Bold",textColor=GREEN,alignment=TA_CENTER))],
    [p("Loan repayment year",bold_s),p("Year 20 (Age 63)",cv_s),p("Year 21 (Age 65)",cv_s),p("1yr diff",cc_s)],
    [p("Annual distributions",bold_s),p("$408,172/yr from Year 21",cg_s),p("$200,000/yr from Year 22",cr_s),p("Lincoln 2× more/yr",S("lm",fontSize=7.5,fontName="Helvetica-Bold",textColor=GREEN,alignment=TA_CENTER))],
    [p("Cap / Floor",bold_s),p("13.5% / 0% (both same for this analysis)",cg_s),p("13.5% / 0% (same)",cg_s),p("Same",cc_s)],
    [p("Index",bold_s),p("S&P 500 TCA 15",cv_s),p("S&P 500 (same for this model)",cv_s),p("Same",cc_s)],
    [p("Distribution loan rate",bold_s),p("~4% fixed",cv_s),p("3.25% → 3.00%",cg_s),p("JH slightly better",S("jb",fontSize=7.5,fontName="Helvetica-Bold",textColor=GREEN,alignment=TA_CENTER))],
    [p("Vitality health bonus",bold_s),p("None",cr_s),p("Yes — Vitality PLUS",cg_s),p("JH unique benefit",S("ju",fontSize=7.5,fontName="Helvetica-Bold",textColor=GREEN,alignment=TA_CENTER))],
]
st=Table(struct_rows,colWidths=[1.4*inch,2.35*inch,2.35*inch,1.1*inch])
st.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),("BACKGROUND",(0,1),(0,-1),LBLUE),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LIGHT,WHITE]),
    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]))
story.append(st)
story.append(Spacer(1,8))

# SCORECARD
story.append(p("2.  TOTAL WEALTH SCORECARD (Policy Value + All Distributions Received)", sect_s))
story.append(HRFlowable(width="100%",thickness=1.5,color=GOLD))
story.append(Spacer(1,5))

sc_hdr=[p(h,ch_s) for h in ["Scenario","Yr 20 Gross\n(Lincoln)","Yr 20 Gross\n(JH)","Yr 25\nTotal Wealth\n(Lincoln)","Yr 25\nTotal Wealth\n(JH)","Yr 30\nTotal Wealth\n(Lincoln)","Yr 30\nTotal Wealth\n(JH)","Yr 30\nWinner"]]
sc_rows=[sc_hdr]
for start,sid,yr_range,desc,col,bg in scenarios:
    lnc=sim_lnc(start); jh=sim_jh(start)
    lnc25_tw=lnc[24]['pv']+lnc[24]['cum']; jh25_tw=jh[24]['pv']+jh[24]['cum']
    lnc30_tw=lnc[29]['pv']+lnc[29]['cum']; jh30_tw=jh[29]['pv']+jh[29]['cum']
    winner = "JH" if jh30_tw>lnc30_tw else "Lincoln"
    w_col = cg_s if winner=="JH" else S("lw",fontSize=7.5,fontName="Helvetica-Bold",textColor=NAVY,alignment=TA_RIGHT)
    sc_rows.append([
        p(f"Scen {sid}: {yr_range}",S("sl",fontSize=7.5,fontName="Helvetica-Bold",textColor=col,alignment=TA_LEFT)),
        p(fmt(lnc[19]['pv']),cv_s), p(fmt(jh[19]['pv']),cg_s),
        p(fmt(lnc25_tw),cv_s),      p(fmt(jh25_tw),cg_s),
        p(fmt(lnc30_tw),cv_s),      p(fmt(jh30_tw),cg_s),
        p(winner,w_col),
    ])
sc_t=Table(sc_rows,colWidths=[1.05*inch,0.85*inch,0.85*inch,0.9*inch,0.9*inch,0.9*inch,0.9*inch,0.65*inch])
sc_t.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LGRN,LAMB,LRED]),
    ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]))
story.append(sc_t)
story.append(Spacer(1,5))

tw_note=Table([[p(
    "Total Wealth = Ending Policy Value + All Cumulative Distributions Received. "
    "This is the fairest comparison since Lincoln pays $408k/yr and JH pays $200k/yr — "
    "different cash flows need to be added back to compare true wealth created.",
    small_s)]],colWidths=[7.5*inch])
tw_note.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LGOLD),("BOX",(0,0),(-1,-1),0.5,GOLD),
    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7)]))
story.append(tw_note)
story.append(Spacer(1,8))

# DETAILED TABLE FOR EACH SCENARIO
milestones={1,5,10,15,20,21,22,25,30}
col_w=[0.38*inch,0.38*inch,0.42*inch,0.6*inch,0.6*inch,0.9*inch,0.7*inch,0.75*inch,0.9*inch,0.7*inch,0.75*inch]

story.append(p("3.  YEAR-BY-YEAR DETAIL — ALL THREE SCENARIOS", sect_s))
story.append(HRFlowable(width="100%",thickness=1.5,color=GOLD))

for start,sid,yr_range,desc,col,bg in scenarios:
    lnc=sim_lnc(start); jh=sim_jh(start)
    lnc_ch=sum(r['charge'] for r in lnc[:25]); jh_ch=sum(r['charge'] for r in jh[:25])
    lnc30=lnc[29]; jh30=jh[29]

    story.append(Spacer(1,5))
    story.append(p(f"Scenario {sid}: {yr_range} — {desc}",
                   S("sh",fontSize=9,fontName="Helvetica-Bold",textColor=col,spaceBefore=4,spaceAfter=3)))
    story.append(HRFlowable(width="100%",thickness=1.5,color=col))
    story.append(Spacer(1,3))

    # Stats bar
    stats=Table([[
        p(f"Lincoln charges: {fmt(lnc_ch)}",S("lc",fontSize=7.5,fontName="Helvetica-Bold",textColor=NAVY)),
        p(f"JH charges: {fmt(jh_ch)}",S("jc",fontSize=7.5,fontName="Helvetica-Bold",textColor=RED)),
        p(f"JH costs {fmt(jh_ch-lnc_ch)} MORE",S("diff",fontSize=7.5,fontName="Helvetica-Bold",textColor=RED)),
        p(f"Yr30 Total Wealth: Lincoln {fmt(lnc30['pv']+lnc30['cum'])} | JH {fmt(jh30['pv']+jh30['cum'])}",
          S("tw",fontSize=7.5,fontName="Helvetica-Bold",textColor=GREEN if jh30['pv']+jh30['cum']>lnc30['pv']+lnc30['cum'] else NAVY)),
    ]],colWidths=[1.5*inch,1.5*inch,1.6*inch,2.9*inch])
    stats.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),bg),("BOX",(0,0),(-1,-1),0.8,col),
        ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
        ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LINEBEFORE",(1,0),(3,-1),0.5,colors.HexColor("#AAAAAA"))]))
    story.append(stats)
    story.append(Spacer(1,3))

    row_hdr=[p(h,ch_s) for h in ["Yr","Age","Cal","S&P","Crd","Lincoln\nValue","Lnc\nDist","Lnc\nTot W","JH\nValue","JH\nDist","JH\nTot W"]]
    tbl_data=[row_hdr]
    for rl,rj in zip(lnc,jh):
        yr=rl['yr']
        if yr not in milestones: continue
        lnc_tw=rl['pv']+rl['cum']; jh_tw=rj['pv']+rj['cum']
        jh_ahead=jh_tw>lnc_tw
        is20=(yr==20); is30=(yr==30); neg=(rl['ret']<0)
        tbl_data.append([
            p(str(yr),S("y",fontSize=7,fontName="Helvetica-Bold" if (is20 or is30) else "Helvetica",textColor=NAVY if (is20 or is30) else GRAY,alignment=TA_CENTER)),
            p(str(rl['age']),cc_s),p(str(rl['cal']),cc_s),
            p(f"{rl['ret']:.1%}",cr_s if neg else (cg_s if rl['ret']>=CAP else cv_s)),
            p(f"{rl['cred']:.1%}",cr_s if rl['cred']==0 else cg_s),
            p(fmt(rl['pv']),cv_s),
            p(fmt(rl['dist']) if rl['dist']>0 else "—",cg_s if rl['dist']>0 else cv_s),
            p(fmt(lnc_tw),S("lt",fontSize=7,fontName="Helvetica-Bold" if not jh_ahead else "Helvetica",textColor=NAVY if not jh_ahead else GRAY,alignment=TA_RIGHT)),
            p(fmt(rj['pv']),cv_s),
            p(fmt(rj['dist']) if rj['dist']>0 else "—",cg_s if rj['dist']>0 else cv_s),
            p(fmt(jh_tw),S("jt",fontSize=7,fontName="Helvetica-Bold" if jh_ahead else "Helvetica",textColor=GREEN if jh_ahead else GRAY,alignment=TA_RIGHT)),
        ])
    tbl=Table(tbl_data,colWidths=col_w,repeatRows=1)
    sc=[("BACKGROUND",(0,0),(-1,0),col),
        ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE")]
    for i,(rl,rj) in enumerate(zip(lnc,jh)):
        yr=rl['yr']
        if yr not in milestones: continue
        idx=list(sorted(milestones)).index(yr)+1
        rb=LIGHT if i%2==0 else WHITE
        if yr==20: rb=colors.HexColor("#D6F0E0")
        elif yr==30: rb=colors.HexColor("#FFF0C8")
        elif rl['ret']<0: rb=colors.HexColor("#FFF2F2")
        sc.append(("BACKGROUND",(0,idx),(-1,idx),rb))
    tbl.setStyle(TableStyle(sc))
    story.append(tbl)

story.append(Spacer(1,8))

# FINAL ANSWER
story.append(p("4.  THE HONEST ANSWER — LINCOLN vs JH FOR $10M COVERAGE", sect_s))
story.append(HRFlowable(width="100%",thickness=1.5,color=GOLD))
story.append(Spacer(1,5))

ans_hdr=[p(h,ch_s) for h in ["Factor","Lincoln $10M","JH $10M","Winner"]]
ans_rows=[ans_hdr,
    [p("Total premiums borrowed",bold_s),p("$5,333,500",cg_s),p("$6,347,244",cr_s),p("Lincoln\n($1M less debt)",S("lw",fontSize=7.5,fontName="Helvetica-Bold",textColor=GREEN,alignment=TA_CENTER))],
    [p("Total policy charges (25 yrs)",bold_s),p("~$1,101,506",cg_s),p("$1,799,479",cr_s),p("Lincoln\n($698k cheaper)",S("lw2",fontSize=7.5,fontName="Helvetica-Bold",textColor=GREEN,alignment=TA_CENTER))],
    [p("Annual distribution from Year 21",bold_s),p("$408,172/yr",cg_s),p("$200,000/yr",cr_s),p("Lincoln\n(2× more/yr)",S("lw3",fontSize=7.5,fontName="Helvetica-Bold",textColor=GREEN,alignment=TA_CENTER))],
    [p("Year 30 total wealth (best)",bold_s),p("$17,427,831",cv_s),p("$19,417,678",cg_s),p("JH\n(+$2M)",S("jw",fontSize=7.5,fontName="Helvetica-Bold",textColor=GREEN,alignment=TA_CENTER))],
    [p("Year 30 total wealth (worst)",bold_s),p("$13,581,433",cv_s),p("$17,432,023",cg_s),p("JH\n(+$3.9M)",S("jw2",fontSize=7.5,fontName="Helvetica-Bold",textColor=GREEN,alignment=TA_CENTER))],
    [p("Death benefit",bold_s),p("$10M (same)",cg_s),p("$10M (same)",cg_s),p("Same",cc_s)],
    [p("Simplicity / ease",bold_s),p("Flat $533k/yr × 10 yrs",cg_s),p("Variable premiums, complex structure",cr_s),p("Lincoln\n(simpler)",S("lw4",fontSize=7.5,fontName="Helvetica-Bold",textColor=GREEN,alignment=TA_CENTER))],
    [p("Vitality health bonus",bold_s),p("None",cr_s),p("Yes — earn credits for staying healthy",cg_s),p("JH unique",S("jw3",fontSize=7.5,fontName="Helvetica-Bold",textColor=GREEN,alignment=TA_CENTER))],
    [p("WHY JH WINS ON WEALTH",bold_s),p("Lincoln's structure is efficient but premiums stop at Year 10",cv_s),p("JH's larger/longer premiums ($6.35M vs $5.33M) put $1M MORE to work compounding over 12 years — outweighs higher charges",cg_s),p("",cc_s)],
]
at=Table(ans_rows,colWidths=[1.7*inch,2.3*inch,2.3*inch,1.1*inch])
at.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),("BACKGROUND",(0,1),(0,-1),LBLUE),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LIGHT,WHITE]),
    ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"TOP"),
]))
story.append(at)
story.append(Spacer(1,7))

fv=Table([[p(
    "BOTTOM LINE (Apples-to-Apples, $10M vs $10M, same cap/floor):  "
    "Lincoln is cheaper per dollar of coverage ($1.1M charges vs JH's $1.8M) and pays 2× more per year ($408k vs $200k). "
    "JH wins on total wealth at Year 30 in every scenario by $2M–$4M — not because it's cheaper, "
    "but because it puts $1M MORE of borrowed money to work ($6.35M vs $5.33M) over 12 years instead of 10. "
    "The extra $1M in JH premiums compounds at 13.5%/yr and overwhelms the extra $700k in charges. "
    "If you want higher annual income: Lincoln ($408k/yr).  "
    "If you want higher total wealth at Year 30: JH.  "
    "Both deliver $10M death benefit. Both cost $0 out of pocket.",
    S("fv",fontSize=8,fontName="Helvetica-Bold",textColor=NAVY,leading=12))
]],colWidths=[7.5*inch])
fv.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LIGHT),("BOX",(0,0),(-1,-1),1.5,NAVY),
    ("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),9),
    ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10)]))
story.append(fv)
story.append(Spacer(1,5))
story.append(HRFlowable(width="100%",thickness=0.5,color=GRAY))
story.append(Spacer(1,3))
story.append(p("Lincoln $10M values are scaled 2× from the Lincoln $5M illustration (March 2026, 7.19% assumed). "
               "JH charges from Annual Account Summary (April 2026). Both modelled at 13.5%/0% with actual S&P 500 returns and SOFR+1% loan. "
               "Lincoln charges estimated/derived; JH charges explicitly stated. Not financial advice.",small_s))

doc.build(story)
print(f"PDF written -> {OUTPUT}")
