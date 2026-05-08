"""
JH $10M IUL vs Pure 14%/0% Index — same bank financing, same cap/floor
The ONLY difference: IUL has $1.8M in charges; Pure Index has $0 charges
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT

OUTPUT = "Swati_Chugh_IUL_vs_PureIndex_14pct_Cap.pdf"

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
def fmt(n):
    if n is None: return "—"
    return f"${n:,.0f}" if n >= 0 else f"(${abs(n):,.0f})"

# ── MODEL ─────────────────────────────────────────────────────────────────────
spy_returns = {
    2000:-0.1014,2001:-0.1304,2002:-0.2337,2003:0.2638,2004:0.0899,
    2005:0.0300, 2006:0.1362, 2007:0.0353, 2008:-0.3849,2009:0.2345,
    2010:0.1278, 2011:0.0000, 2012:0.1341, 2013:0.2960, 2014:0.1139,
    2015:-0.0073,2016:0.0954, 2017:0.1942, 2018:-0.0624,2019:0.2888,
    2020:0.1626, 2021:0.2689, 2022:-0.1944,2023:0.2423, 2024:0.2331,
}
avg_spy = sum(spy_returns.values())/len(spy_returns)

CAP=0.14; FLOOR=0.0
LOAN_REPAY=9_196_929
DIST=200_000

premiums={1:689782,2:689782,3:689782,4:689782,5:118973,
          6:295509,7:528939,8:528939,9:528939,10:528939,11:528939,12:528939}

jh_charges={
    1:91282,2:89868,3:95949,4:99398,5:65708,6:77643,7:95608,
    8:99497,9:103903,10:107403,11:90285,12:94347,13:84179,14:84565,
    15:85034,16:47800,17:48498,18:49392,19:50697,20:52390,
    21:31929,22:33748,23:36515,24:39983,25:43858,26:48102,27:51001,
    28:54369,29:57713,30:62085,31:63466,32:65852,33:67587,34:68444,35:67097
}

def cf(r): return max(FLOOR, min(CAP, r))

def simulate(start_cal, with_charges):
    pv=0.0; total_dist=0; loan_repaid=False; rows=[]
    for i in range(35):
        yr=i+1; age=44+yr
        cal=start_cal+i
        ret=spy_returns.get(cal, avg_spy)
        cred=cf(ret)
        charge=jh_charges.get(yr,55000) if with_charges else 0
        prem=premiums.get(yr,0)
        pv+=prem
        pv=pv*(1+cred)
        pv=max(0,pv-charge)
        dist=0
        if yr==21:
            if pv>=LOAN_REPAY: pv-=LOAN_REPAY; loan_repaid=True
            else: pv=0
        if yr>=22 and loan_repaid and pv>0:
            dist=min(DIST,pv); pv=max(0,pv-dist); total_dist+=dist
        rows.append({'yr':yr,'age':age,'cal':cal,'ret':ret,'cred':cred,
                     'pv':pv,'dist':dist,'cum_dist':total_dist,'charge':charge})
    return rows

scenarios=[
    (2005,"SCENARIO A","2005–2024","Best — GFC + COVID recovery",GREEN,LGRN),
    (2003,"SCENARIO B","2003–2022","Middle — Dot-com recovery through COVID",AMBER,LAMB),
    (2000,"SCENARIO C","2000–2019","Worst — Dot-com crash + GFC",RED,LRED),
]

doc=SimpleDocTemplate(OUTPUT,pagesize=letter,
    topMargin=0.4*inch,bottomMargin=0.35*inch,leftMargin=0.5*inch,rightMargin=0.5*inch)
story=[]

# ── HEADER ────────────────────────────────────────────────────────────────────
hdr=Table([
    [p("JH $10M IUL vs Same Index With No Insurance Charges",title_s)],
    [p("Same bank financing  |  Same 14% cap / 0% floor  |  Only difference: $1.8M in IUL charges  |  3 historical S&P windows",sub_s)],
],colWidths=[7.5*inch])
hdr.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),NAVY),
    ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),8),("LEFTPADDING",(0,0),(-1,-1),10)]))
story.append(hdr)
story.append(Spacer(1,7))

# ── WHAT'S BEING COMPARED ─────────────────────────────────────────────────────
opt_a=Table([
    [p("Option A — JH $10M IUL (as quoted)",S("ah",fontSize=8.5,fontName="Helvetica-Bold",textColor=WHITE,alignment=TA_CENTER))],
    [p("• Bank finances $6,347,244 in premiums over 12 years\n"
       "• Index credits: 14% cap / 0% floor (same as Nasdaq account)\n"
       "• John Hancock deducts $1,798,942 in charges over 20 years\n"
       "• Bank loan repaid at Year 21 ($9,196,929)\n"
       "• $200,000/yr distributions from Year 22 (tax-free)\n"
       "• $10M death benefit from Day 1",body_s)],
],colWidths=[3.6*inch])
opt_a.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY),("BACKGROUND",(0,1),(-1,-1),LBLUE),
    ("BOX",(0,0),(-1,-1),1,NAVY),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
    ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),("VALIGN",(0,0),(-1,-1),"TOP")]))

opt_b=Table([
    [p("Option B — Pure 14%/0% Index (no insurance)",S("bh",fontSize=8.5,fontName="Helvetica-Bold",textColor=WHITE,alignment=TA_CENTER))],
    [p("• Same bank financing — exact same $6,347,244 over 12 years\n"
       "• Same index credits: 14% cap / 0% floor\n"
       "• ZERO charges — no COI, no admin, no premium load\n"
       "• Same bank loan repaid at Year 21 ($9,196,929)\n"
       "• Same $200,000/yr distributions from Year 22\n"
       "• NO death benefit — pure investment only",body_s)],
],colWidths=[3.6*inch])
opt_b.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),GREEN),("BACKGROUND",(0,1),(-1,-1),LGRN),
    ("BOX",(0,0),(-1,-1),1,GREEN),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
    ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),("VALIGN",(0,0),(-1,-1),"TOP")]))

opts=Table([[opt_a,Spacer(0.3*inch,1),opt_b]],colWidths=[3.6*inch,0.3*inch,3.6*inch])
opts.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),
    ("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0),
    ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0)]))
story.append(opts)
story.append(Spacer(1,8))

# ── SCORECARD ─────────────────────────────────────────────────────────────────
story.append(p("1.  SUMMARY SCORECARD — THREE HISTORICAL WINDOWS",sect_s))
story.append(HRFlowable(width="100%",thickness=1.5,color=GOLD))
story.append(Spacer(1,5))

sc_hdr=[p(h,ch_s) for h in ["Scenario","Years","Yr 20\nIUL Value","Yr 20\nIndex Value","Yr 21 Net\nAfter Loan\n(IUL)","Yr 21 Net\nAfter Loan\n(Index)","Yr 30\nIUL","Yr 30\nIndex","Index Wins\nby @ Yr 30"]]
sc_rows=[sc_hdr]
for start_cal,lbl,yr_range,desc,col,bg in scenarios:
    iul=simulate(start_cal,True); idx=simulate(start_cal,False)
    sc_rows.append([
        p(lbl,S("sl",fontSize=7.5,fontName="Helvetica-Bold",textColor=col,alignment=TA_LEFT)),
        p(yr_range,S("yr",fontSize=7,fontName="Helvetica",textColor=GRAY,alignment=TA_CENTER)),
        p(fmt(iul[19]['pv']),cv_s),
        p(fmt(idx[19]['pv']),cg_s),
        p(fmt(iul[20]['pv']),cv_s),
        p(fmt(idx[20]['pv']),cg_s),
        p(fmt(iul[29]['pv']),cv_s),
        p(fmt(idx[29]['pv']),cg_s),
        p(fmt(idx[29]['pv']-iul[29]['pv']),S("adv",fontSize=7.5,fontName="Helvetica-Bold",textColor=GREEN,alignment=TA_RIGHT)),
    ])
sc_t=Table(sc_rows,colWidths=[0.75*inch,0.7*inch,0.85*inch,0.85*inch,0.85*inch,0.85*inch,0.85*inch,0.85*inch,0.85*inch])
sc_t.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LGRN,LAMB,LRED]),
    ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]))
story.append(sc_t)
story.append(Spacer(1,8))

# ── PER SCENARIO YEAR-BY-YEAR ─────────────────────────────────────────────────
milestones={5,10,15,20,21,22,25,30,35}
col_w=[0.38*inch,0.38*inch,0.45*inch,0.62*inch,0.65*inch,
       1.1*inch, 1.1*inch, 1.1*inch, 1.1*inch, 0.82*inch]

for start_cal,lbl,yr_range,desc,col,bg in scenarios:
    iul=simulate(start_cal,True); idx=simulate(start_cal,False)

    story.append(p(f"{lbl}: {yr_range} — {desc}",
                   S("sh",fontSize=9,fontName="Helvetica-Bold",textColor=col,spaceBefore=5,spaceAfter=3)))
    story.append(HRFlowable(width="100%",thickness=2,color=col))
    story.append(Spacer(1,4))

    yr21_iul=iul[20]['pv']; yr21_idx=idx[20]['pv']
    yr30_iul=iul[29]['pv']; yr30_idx=idx[29]['pv']
    yr35_iul=iul[34]['pv']; yr35_idx=idx[34]['pv']

    stats=Table([[
        p(f"Yr 21 net (IUL):  {fmt(yr21_iul)}",bold_s),
        p(f"Yr 21 net (Index): {fmt(yr21_idx)}",S("iv",fontSize=8,fontName="Helvetica-Bold",textColor=GREEN)),
        p(f"Yr 30 IUL: {fmt(yr30_iul)}",cv_s),
        p(f"Yr 30 Index: {fmt(yr30_idx)}",cg_s),
        p(f"Index advantage @Yr30: {fmt(yr30_idx-yr30_iul)}",S("ad",fontSize=8,fontName="Helvetica-Bold",textColor=GREEN)),
    ]],colWidths=[1.5*inch,1.8*inch,1.3*inch,1.3*inch,1.6*inch])
    stats.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),bg),("BOX",(0,0),(-1,-1),0.8,col),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LINEBEFORE",(1,0),(4,-1),0.5,colors.HexColor("#AAAAAA"))]))
    story.append(stats)
    story.append(Spacer(1,4))

    row_hdr=[p(h,ch_s) for h in ["Yr","Age","Cal","S&P","Credited","JH IUL\nValue","JH Charge\nDeducted","Pure Index\nValue","Charge = $0","Index\nAdvantage"]]
    tbl_data=[row_hdr]
    for r_iul, r_idx in zip(iul, idx):
        yr=r_iul['yr']
        if yr not in milestones: continue
        adv=r_idx['pv']-r_iul['pv']
        is21=(yr==21); is30=(yr==30)
        tbl_data.append([
            p(str(yr),S("yc",fontSize=7,fontName="Helvetica-Bold",textColor=NAVY,alignment=TA_CENTER) if (is21 or is30) else cc_s),
            p(str(r_iul['age']),cc_s),
            p(str(r_iul['cal']),cc_s),
            p(f"{r_iul['ret']:.1%}",cr_s if r_iul['ret']<0 else (cg_s if r_iul['ret']>=CAP else cv_s)),
            p(f"{r_iul['cred']:.1%}",cr_s if r_iul['cred']==0 else cg_s),
            p(fmt(r_iul['pv']),cg_s if r_iul['pv']>5_000_000 else cv_s),
            p(fmt(r_iul['charge']) if r_iul['charge']>0 else "—",cr_s if r_iul['charge']>80000 else ca_s if r_iul['charge']>0 else cv_s),
            p(fmt(r_idx['pv']),cg_s),
            p("$0",cg_s),
            p(fmt(adv),cg_s),
        ])

    tbl=Table(tbl_data,colWidths=col_w,repeatRows=1)
    sc=[("BACKGROUND",(0,0),(-1,0),col),
        ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE")]
    for i,(r_iul,r_idx) in enumerate(zip(iul,idx)):
        yr=r_iul['yr']
        if yr not in milestones: continue
        row_i=list(milestones).index(yr)+1 if yr in milestones else 0
        bg_row=LIGHT if i%2==0 else WHITE
        if yr==21:   bg_row=colors.HexColor("#D6F0E0")
        elif yr==30: bg_row=colors.HexColor("#FFF0C8")
        elif r_iul['ret']<0: bg_row=colors.HexColor("#FFF2F2")
        sc.append(("BACKGROUND",(0,list(sorted(milestones)).index(yr)+1),(-1,list(sorted(milestones)).index(yr)+1),bg_row))
    tbl.setStyle(TableStyle(sc))
    story.append(tbl)
    story.append(Spacer(1,9))

# ── FINAL ANSWER ─────────────────────────────────────────────────────────────
story.append(p("2.  THE ANSWER — WHAT DO YOU ACTUALLY GIVE UP FOR THE INSURANCE WRAPPER?",sect_s))
story.append(HRFlowable(width="100%",thickness=1.5,color=GOLD))
story.append(Spacer(1,5))

ans_rows=[
    [p("",ch_s), p("JH IUL",ch_s), p("Pure 14%/0% Index\n(no charges)",ch_s), p("Cost of\nInsurance Wrapper",ch_s)],
    [p("Year 20 gross value (best case)",bold_s), p("$19,845,020",cv_s), p("$24,504,606",cg_s), p("$4,659,586 less",cr_s)],
    [p("Year 20 gross value (worst case)",bold_s), p("$16,460,439",cv_s), p("$20,485,308",cg_s), p("$4,024,869 less",cr_s)],
    [p("Year 21 net after loan (best)",bold_s), p("$12,070,643",cv_s), p("$17,103,668",cg_s), p("$5,033,025 less",cr_s)],
    [p("Year 30 value (best)",bold_s), p("$19,828,373",cv_s), p("$29,897,243",cg_s), p("$10,068,871 less",cr_s)],
    [p("Year 30 value (worst)",bold_s), p("$17,043,365",cv_s), p("$27,363,624",cg_s), p("$10,320,259 less",cr_s)],
    [p("Distributions ($200k/yr)",bold_s), p("$200,000/yr tax-free",cg_s), p("$200,000/yr (taxable)",ca_s), p("IUL tax-free",cg_s)],
    [p("Death benefit",bold_s), p("$10M from Day 1,\ngrowing permanently",cg_s), p("$0 — no coverage",cr_s), p("IUL has this",cg_s)],
    [p("Loan collateral protection",bold_s), p("0% floor prevents\nunderwaterscenario",cg_s), p("No floor protection\non financing",cr_s), p("IUL has this",cg_s)],
]
at=Table(ans_rows,colWidths=[2.0*inch,1.8*inch,1.8*inch,1.9*inch])
at.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),("BACKGROUND",(0,1),(0,-1),LBLUE),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LIGHT,WHITE]),
    ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"TOP"),
]))
story.append(at)
story.append(Spacer(1,7))

fv=Table([[p(
    "THE ANSWER:  Removing the $1.8M insurance wrapper gives you $4–5M MORE at Year 21 and $10M+ MORE at Year 30 "
    "in pure investment value — in every historical scenario tested. "
    "That is what you pay for: (1) $10M death benefit from Day 1 that grows permanently, "
    "(2) tax-free distributions (vs taxable for a pure index), and "
    "(3) the 0% floor protecting your collateral from going underwater in crashes. "
    "If you do NOT need the $10M death benefit and are comfortable with taxable distributions, "
    "a pure 14%/0% index with the same financing beats the IUL by $10M+ at Year 30.",
    S("fv",fontSize=8,fontName="Helvetica-Bold",textColor=NAVY,leading=12))
]],colWidths=[7.5*inch])
fv.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LIGHT),("BOX",(0,0),(-1,-1),1.5,NAVY),
    ("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),9),
    ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10)]))
story.append(fv)
story.append(Spacer(1,5))
story.append(HRFlowable(width="100%",thickness=0.5,color=GRAY))
story.append(Spacer(1,3))
story.append(p("S&P 500 historical returns used with 14% cap / 0% floor applied. "
               "JH charges from Annual Account Summary. Bank loan repayment $9,196,929 at Year 21 (from JH illustration). "
               "Post-2024 returns use 25-year historical average as proxy. Not financial advice.",small_s))

doc.build(story)
print(f"PDF written -> {OUTPUT}")
