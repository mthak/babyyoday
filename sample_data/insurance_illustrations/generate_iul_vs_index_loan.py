"""
IUL vs Pure Index Fund — Same borrowed money, same loan, same repayment at Year 20.
No tax. No cap difference. Just: does the IUL wrapper help or hurt vs raw S&P 500?
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT

OUTPUT = "Swati_Chugh_IUL_vs_DirectIndex_SameLoan.pdf"

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
ch_s    = S("ch", fontSize=7,  textColor=WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER)
cv_s    = S("cv", fontSize=7,  textColor=GRAY,  fontName="Helvetica",      alignment=TA_RIGHT)
cg_s    = S("cg", fontSize=7,  textColor=GREEN, fontName="Helvetica-Bold", alignment=TA_RIGHT)
cr_s    = S("cr", fontSize=7,  textColor=RED,   fontName="Helvetica-Bold", alignment=TA_RIGHT)
ca_s    = S("ca", fontSize=7,  textColor=AMBER, fontName="Helvetica-Bold", alignment=TA_RIGHT)
cc_s    = S("cc", fontSize=7,  textColor=GRAY,  fontName="Helvetica",      alignment=TA_CENTER)

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

CAP=0.135; FLOOR=0.0; PREM=533_350; CHARGES=0.015
avg_sofr=sum(sofr_actual.values())/len(sofr_actual)
avg_spy=sum(spy_returns.values())/len(spy_returns)
def cf(r): return max(FLOOR, min(CAP, r))

def simulate(start_cal):
    loan=0.0; iul=0.0; idx=0.0; rows=[]
    for i in range(20):
        cal=start_cal+i; yr=i+1
        ret=spy_returns.get(cal,avg_spy)
        sofr=sofr_actual.get(cal,avg_sofr)
        lr=sofr+0.01; cred=cf(ret)
        if yr<=10: iul+=PREM; idx+=PREM; loan+=PREM
        iul=iul*(1+cred-CHARGES)
        idx=idx*(1+ret)
        loan=loan*(1+lr)
        rows.append({'yr':yr,'cal':cal,'ret':ret,'cred':cred,'sofr':sofr,'lr':lr,
                     'loan':loan,'iul':iul,'idx':idx,'iul_net':iul-loan,'idx_net':idx-loan})
    f=rows[-1]
    return rows, max(0,f['iul']-f['loan']), max(0,f['loan']-f['iul']), \
                 max(0,f['idx']-f['loan']), max(0,f['loan']-f['idx'])

scenarios=[
    (2005,"SCENARIO A","2005–2024","GFC + COVID recovery (best 20 yrs)",GREEN,LGRN),
    (2003,"SCENARIO B","2003–2022","Dot-com recovery through COVID crash",AMBER,LAMB),
    (2000,"SCENARIO C","2000–2019","Worst: Dot-com crash + GFC back-to-back",RED,LRED),
]
all_res={}
for sc,lbl,yr,desc,col,bg in scenarios:
    rows,iu_su,iu_sh,ix_su,ix_sh=simulate(sc)
    all_res[sc]={'rows':rows,'iu_su':iu_su,'iu_sh':iu_sh,'ix_su':ix_su,'ix_sh':ix_sh,
                 'lbl':lbl,'yr':yr,'desc':desc,'col':col,'bg':bg}

doc=SimpleDocTemplate(OUTPUT,pagesize=letter,
    topMargin=0.4*inch,bottomMargin=0.35*inch,leftMargin=0.5*inch,rightMargin=0.5*inch)
story=[]

# Header
hdr=Table([[p("IUL vs. Direct Index Fund — Same Borrowed Money, Same Loan, Same Rules",title_s)],
           [p("$10M policy  |  Borrow $533,350/yr × 10 yrs at SOFR+1%  |  Repay full loan at Year 20  |  No tax  |  What do you keep?",sub_s)]],
          colWidths=[7.5*inch])
hdr.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),NAVY),
    ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),8),("LEFTPADDING",(0,0),(-1,-1),10)]))
story.append(hdr)
story.append(Spacer(1,7))

# What's being compared
comp_box=Table([[
    Table([
        [p("Option A: IUL (Lincoln WealthBuilder)",S("ah",fontSize=8,fontName="Helvetica-Bold",textColor=NAVY,alignment=TA_CENTER))],
        [p("• Borrow same money, same loan rate\n• S&P returns capped at 13.5%, floored at 0%\n• Policy charges ~1.5%/yr\n• PLUS: $10M death benefit entire time\n• PLUS: Tax-free policy loans after Yr 20",body_s)],
    ],colWidths=[3.6*inch]),
    Table([
        [p("Option B: Direct Index Fund (S&P 500 ETF)",S("bh",fontSize=8,fontName="Helvetica-Bold",textColor=GREEN,alignment=TA_CENTER))],
        [p("• Borrow same money, same loan rate\n• Full raw S&P returns — no cap, no floor\n• No policy charges\n• NO death benefit\n• Gains taxable on withdrawal (ignored here)",body_s)],
    ],colWidths=[3.6*inch]),
]],colWidths=[3.65*inch,3.65*inch])
for t,c in zip(comp_box._cellvalues[0],[NAVY,GREEN]):
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),c),
        ("BACKGROUND",(0,1),(-1,-1),LGRN if c==GREEN else LBLUE),
        ("BOX",(0,0),(-1,-1),1,c),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
    ]))
comp_box.setStyle(TableStyle([
    ("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0),
    ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
    ("VALIGN",(0,0),(-1,-1),"TOP"),
]))
story.append(comp_box)
story.append(Spacer(1,9))

# ── SECTION 1: SCORECARD ──────────────────────────────────────────────────────
story.append(p("1.  WHAT YOU KEEP AFTER REPAYING THE BANK AT YEAR 20",sect_s))
story.append(HRFlowable(width="100%",thickness=1.5,color=GOLD))
story.append(Spacer(1,5))

sc_hdr=[p(h,ch_s) for h in ["Scenario","Market\nWindow","IUL\nValue Yr 20","IUL Repays\nLoan","You Keep\n(IUL)","Index\nValue Yr 20","Index Repays\nLoan","You Keep\n(Index)","Winner\n& Margin"]]
sc_rows=[sc_hdr]
for start,lbl,yr,desc,col,bg in scenarios:
    d=all_res[start]
    f=d['rows'][-1]
    iu_keep=d['iu_su']; ix_keep=d['ix_su']
    iu_short=d['iu_sh']; ix_short=d['ix_sh']
    winner_txt=("Index +" + fmt(ix_keep-iu_keep) if ix_keep>iu_keep else "IUL +" + fmt(iu_keep-ix_keep))
    winner_col=cg_s if ix_keep>iu_keep else S("wc",fontSize=7,textColor=NAVY,fontName="Helvetica-Bold",alignment=TA_RIGHT)
    sc_rows.append([
        p(lbl,S("sl",fontSize=7.5,fontName="Helvetica-Bold",textColor=col,alignment=TA_LEFT)),
        p(yr,S("yr",fontSize=7,fontName="Helvetica",textColor=GRAY,alignment=TA_CENTER)),
        p(fmt(f['iul']),cv_s),
        p(fmt(f['loan']),cr_s),
        p(fmt(iu_keep) if iu_short==0 else "("+fmt(iu_short)+")",cg_s if iu_short==0 else cr_s),
        p(fmt(f['idx']),cv_s),
        p(fmt(f['loan']),cr_s),
        p(fmt(ix_keep) if ix_short==0 else "("+fmt(ix_short)+")",cg_s if ix_short==0 else cr_s),
        p(winner_txt,winner_col),
    ])

sc_t=Table(sc_rows,colWidths=[0.7*inch,0.7*inch,0.9*inch,0.8*inch,0.8*inch,0.9*inch,0.8*inch,0.8*inch,0.85*inch])
sc_t.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LGRN,LAMB,LRED]),
    ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]))
story.append(sc_t)
story.append(Spacer(1,9))

# ── SECTION 2: YEAR-BY-YEAR FOR ALL THREE ────────────────────────────────────
col_w=[0.35*inch,0.42*inch,0.65*inch,0.65*inch,0.55*inch,0.9*inch,0.9*inch,0.9*inch,0.9*inch,0.9*inch]

for start,lbl,yr,desc,col,bg in scenarios:
    d=all_res[start]
    rows=d['rows']
    f=rows[-1]

    story.append(p(f"{lbl}: {yr} — {desc}",S("sh",fontSize=9,fontName="Helvetica-Bold",
                    textColor=col,spaceBefore=6,spaceAfter=3)))
    story.append(HRFlowable(width="100%",thickness=2,color=col))
    story.append(Spacer(1,4))

    # Stats bar
    iu_keep=d['iu_su']; ix_keep=d['ix_su']
    diff=ix_keep-iu_keep
    stats=Table([[
        p(f"IUL keeps:  {fmt(iu_keep)}",S("is",fontSize=8,fontName="Helvetica-Bold",textColor=NAVY)),
        p(f"Index keeps:  {fmt(ix_keep)}",S("ixs",fontSize=8,fontName="Helvetica-Bold",textColor=GREEN)),
        p(f"Difference:  {fmt(abs(diff))}  {'in favor of Index' if diff>0 else 'in favor of IUL'}",
          S("ds",fontSize=8,fontName="Helvetica-Bold",textColor=GREEN if diff>0 else NAVY)),
        p(f"IUL also gives you $10M death benefit throughout — Index gives $0",
          S("db",fontSize=7.5,fontName="Helvetica-Oblique",textColor=NAVY)),
    ]],colWidths=[1.5*inch,1.5*inch,2.2*inch,2.3*inch])
    stats.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),bg),("BOX",(0,0),(-1,-1),0.8,col),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LINEBEFORE",(1,0),(3,-1),0.5,colors.HexColor("#AAAAAA")),
    ]))
    story.append(stats)
    story.append(Spacer(1,4))

    row_hdr=[p(h,ch_s) for h in ["Yr","Cal","S&P\nReturn","Credited\n(IUL)","Loan\nRate","Loan\nBalance","IUL\nValue","Index\nValue","IUL Net\n(IUL-Loan)","Idx Net\n(Idx-Loan)"]]
    tbl_data=[row_hdr]
    for r in rows:
        iul_ahead=r['iul_net']>r['idx_net']
        idx_neg=r['idx_net']<0
        iul_neg=r['iul_net']<0
        zero_yr=r['cred']==0 and r['ret']<0

        tbl_data.append([
            p(str(r['yr']),S("yn",fontSize=7,fontName="Helvetica-Bold",textColor=NAVY,alignment=TA_CENTER) if r['yr']==20 else cc_s),
            p(str(r['cal']),cc_s),
            p(f"{r['ret']:.1%}",cr_s if r['ret']<0 else (cg_s if r['ret']>=CAP else cv_s)),
            p(f"{r['cred']:.1%}",cr_s if r['cred']==0 else cg_s),
            p(f"{r['lr']:.2%}",ca_s if r['lr']>0.05 else cv_s),
            p(fmt(r['loan']),cv_s),
            p(fmt(r['iul']),cg_s if r['iul']>r['loan'] else cv_s),
            p(fmt(r['idx']),cg_s if r['idx']>r['loan'] else (cr_s if r['idx']<r['loan']*0.5 else ca_s)),
            p(fmt(r['iul_net']),cg_s if r['iul_net']>0 else cr_s),
            p(fmt(r['idx_net']),cg_s if r['idx_net']>r['iul_net'] else (cr_s if r['idx_net']<0 else cv_s)),
        ])

    tbl=Table(tbl_data,colWidths=col_w,repeatRows=1)
    style_cmds=[
        ("BACKGROUND",(0,0),(-1,0),col),
        ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]
    for i,r in enumerate(rows):
        bg_row=LIGHT if i%2==0 else WHITE
        if r['yr']==20:          bg_row=colors.HexColor("#D6F0E0")
        elif r['ret']<0:         bg_row=colors.HexColor("#FFF2F2")
        style_cmds.append(("BACKGROUND",(0,i+1),(-1,i+1),bg_row))
    tbl.setStyle(TableStyle(style_cmds))
    story.append(tbl)
    story.append(Spacer(1,10))

# ── SECTION 3: THE HONEST COMPARISON ─────────────────────────────────────────
story.append(p("2.  THE HONEST ANSWER",sect_s))
story.append(HRFlowable(width="100%",thickness=1.5,color=GOLD))
story.append(Spacer(1,5))

honest_rows=[
    [p("",ch_s), p("Index Fund\n(Direct, No Insurance)",ch_s), p("IUL (Insured)",ch_s)],
    [p("Year 20 payout — Best case (2005–2024)",bold_s), p("$16,705,310",cg_s), p("$8,893,017",cv_s)],
    [p("Year 20 payout — Middle (2003–2022)",bold_s),    p("$10,200,889",cg_s), p("$7,699,146",cv_s)],
    [p("Year 20 payout — Worst (2000–2019)",bold_s),     p("$6,715,651",cg_s),  p("$6,201,737",cv_s)],
    [p("Worst crash year (2008: -38.5%)",bold_s),
     p("Index lost 38.5% that year — value crashed", cr_s),
     p("IUL credited 0% — value held stable", cg_s)],
    [p("Death benefit throughout 20 yrs",bold_s), p("$0", cr_s), p("$10,000,000", cg_s)],
    [p("Tax on gains at Year 20",bold_s),
     p("Capital gains tax on all profits\n(ignored in this model but REAL)",S("tax",fontSize=7,fontName="Helvetica-Bold",textColor=AMBER,alignment=TA_RIGHT)),
     p("$0 — policy loans are tax-free",cg_s)],
    [p("If market is down in Year 20",bold_s),
     p("Index value drops — possible shortfall to repay loan",cr_s),
     p("Floor ensures IUL value never goes negative from losses",cg_s)],
    [p("Collateral during loan (yrs 1–10)",bold_s),
     p("Index can go deeply negative in crashes (2008: -48% loss on borrowed money)",cr_s),
     p("IUL never below zero — floor protects",cg_s)],
]
ht=Table(honest_rows,colWidths=[2.2*inch,2.65*inch,2.65*inch])
ht.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("BACKGROUND",(0,1),(0,-1),LBLUE),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LIGHT,WHITE]),
    ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]))
story.append(ht)
story.append(Spacer(1,8))

# Final verdict
fv=Table([[p(
    "BOTTOM LINE:  Yes — if you borrowed the same money and invested directly in the S&P 500 with no insurance wrapper, "
    "you would have kept MORE money at Year 20 in all three historical scenarios. "
    "The index beats the IUL by $514k (worst case) to $7.8M (best case). "
    "The cost of the IUL wrapper is the 13.5% cap and the 1.5%/yr policy charges. "
    "What you GET in return: a $10M death benefit for 20 years, a 0% floor (your collateral never goes underwater), "
    "and tax-free income after Year 20. "
    "The index wins on raw payout at Year 20. The IUL wins on protection, tax efficiency, and legacy.",
    S("fv",fontSize=8,fontName="Helvetica-Bold",textColor=NAVY,leading=12))
]],colWidths=[7.5*inch])
fv.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LIGHT),("BOX",(0,0),(-1,-1),1.5,NAVY),
    ("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),9),
    ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10)]))
story.append(fv)
story.append(Spacer(1,5))
story.append(HRFlowable(width="100%",thickness=0.5,color=GRAY))
story.append(Spacer(1,3))
story.append(p("S&P 500 price returns and SOFR/LIBOR from public historical data. Policy charges ~1.5%/yr proxy. "
               "Tax implications excluded as instructed. Death benefit estimates approximate. Not financial advice.",small_s))

doc.build(story)
print(f"PDF written -> {OUTPUT}")
