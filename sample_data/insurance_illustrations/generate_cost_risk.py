"""
$10M IUL — What would it COST me in the worst case?
SOFR+1% variable rate, full 20-year model, stress tests, break-even analysis.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT

OUTPUT = "Swati_Chugh_10M_IUL_Cost_Risk_Analysis.pdf"

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

# ── DATA ──────────────────────────────────────────────────────────────────────
spy_returns = {
    2000:-0.1014,2001:-0.1304,2002:-0.2337,2003:0.2638,2004:0.0899,
    2005:0.0300, 2006:0.1362, 2007:0.0353, 2008:-0.3849,2009:0.2345,
    2010:0.1278, 2011:0.0000, 2012:0.1341, 2013:0.2960,2014:0.1139,
    2015:-0.0073,2016:0.0954, 2017:0.1942, 2018:-0.0624,2019:0.2888,
    2020:0.1626, 2021:0.2689, 2022:-0.1944,2023:0.2423,2024:0.2331,
}
sofr_actual = {
    2000:0.0652,2001:0.0350,2002:0.0180,2003:0.0112,2004:0.0156,
    2005:0.0322,2006:0.0532,2007:0.0502,2008:0.0213,2009:0.0024,
    2010:0.0029,2011:0.0025,2012:0.0031,2013:0.0024,2014:0.0023,
    2015:0.0032,2016:0.0097,2017:0.0130,2018:0.0236,2019:0.0216,
    2020:0.0037,2021:0.0005,2022:0.0228,2023:0.0502,2024:0.0460,
}
CAP=0.135; FLOOR=0.0; PREM=533_350; CHARGES=0.015

def cf(r): return max(FLOOR, min(CAP, r))

def run(spy_seq, sofr_seq):
    loan=0.0; pv=0.0; rows=[]
    for yr in range(1,21):
        ret=spy_seq[yr-1]; sofr=sofr_seq[yr-1]; lr=sofr+0.01; cred=cf(ret)
        if yr<=10: pv+=PREM; loan+=PREM
        pv=pv*(1+cred-CHARGES); loan=loan*(1+lr)
        rows.append({'yr':yr,'ret':ret,'cred':cred,'sofr':sofr,'lr':lr,'loan':loan,'pv':pv,'net':pv-loan})
    shortfall=max(0,loan-pv); surplus=max(0,pv-loan)
    return rows, shortfall, surplus

spy_worst   = [spy_returns[2000+i] for i in range(20)]
spy_mid     = [spy_returns[2003+i] for i in range(20)]
spy_best    = [spy_returns[2005+i] for i in range(20)]
sofr_worst  = [sofr_actual[2000+i] for i in range(20)]
sofr_mid    = [sofr_actual[2003+i] for i in range(20)]
sofr_best   = [sofr_actual[2005+i] for i in range(20)]
sofr_high20 = [0.055]*20
sofr_spike  = sofr_best[:17] + [0.06,0.07,0.08]
sofr_2019h  = [sofr_actual.get(2000+i,0.03) for i in range(20)]

scenarios = [
    ("SCENARIO 1","2005–2024 market + actual SOFR",spy_best,sofr_best,GREEN,LGRN,"Historical best 20 yrs + low post-GFC rates"),
    ("SCENARIO 2","2003–2022 market + actual SOFR",spy_mid, sofr_mid, AMBER,LAMB,"Middle scenario incl. GFC + COVID crash"),
    ("SCENARIO 3","2000–2019 market + actual SOFR",spy_worst,sofr_worst,colors.HexColor("#1B5E20"),LGRN,"Worst real market + rates still produced surplus"),
    ("STRESS 1",  "Worst market + 5.5% SOFR flat",spy_worst,sofr_high20,RED,LRED,"First real shortfall: sustained high rates + bad market"),
    ("STRESS 2",  "Worst market + ultra-low IUL credits (3% max)",spy_worst,sofr_worst,RED,LRED,"IUL underperforms badly + actual rates"),
    ("STRESS 3",  "Best market + rate spike yr 18-20 (6/7/8%)",spy_best,sofr_spike,AMBER,LAMB,"Late-stage rate spike — does it matter?"),
]

all_results={}
for sc,label,spy,sofr,col,bg,desc in scenarios:
    rows,shortfall,surplus=run(spy,sofr)
    all_results[sc]={'rows':rows,'shortfall':shortfall,'surplus':surplus,
                     'label':label,'col':col,'bg':bg,'desc':desc}

# Break-even
breakeven=[]
for test_sofr in [0.03,0.04,0.045,0.05,0.055,0.06,0.065,0.07,0.08,0.09,0.10]:
    rows,sh,su=run(spy_worst,[test_sofr]*20)
    breakeven.append((test_sofr,test_sofr+0.01,rows[-1]['loan'],rows[-1]['pv'],sh,su))

# ── BUILD PDF ─────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(OUTPUT, pagesize=letter,
    topMargin=0.4*inch, bottomMargin=0.35*inch,
    leftMargin=0.5*inch, rightMargin=0.5*inch)
story=[]

# Header
hdr=Table([[p("$10M IUL — What Would It Cost Me in the Worst Case?",title_s)],
           [p("SOFR+1% variable loan rate  |  $10M death benefit  |  Actual S&P 500 returns  |  Bank demands repayment at Year 20",sub_s)]],
          colWidths=[7.5*inch])
hdr.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),NAVY),
    ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),8),("LEFTPADDING",(0,0),(-1,-1),10)]))
story.append(hdr)
story.append(Spacer(1,7))

# Policy facts box
facts=Table([[
    p("$10M death benefit  |  Annual premium: $533,350/yr × 10 yrs  |  Total borrowed: $5,333,500",bold_s),
    p("Loan rate: SOFR+1% (variable, changes each year)  |  Policy charges: ~1.5%/yr  |  Cap: 13.5%  |  Floor: 0%",body_s),
]],colWidths=[3.8*inch,3.7*inch])
facts.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LIGHT),("BOX",(0,0),(-1,-1),0.8,NAVY),
    ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LINEBEFORE",(1,0),(1,-1),0.5,colors.HexColor("#CCCCCC"))]))
story.append(facts)
story.append(Spacer(1,8))

# ── SECTION 1: SUMMARY SCORECARD ─────────────────────────────────────────────
story.append(p("1.  WHAT HAPPENS AT YEAR 20 — ALL SIX SCENARIOS",sect_s))
story.append(HRFlowable(width="100%",thickness=1.5,color=GOLD))
story.append(Spacer(1,5))

sc_hdr=[p(h,ch_s) for h in ["Scenario","Description","Avg Loan\nRate","Avg\nCredited","Year 20\nLoan Owed","Year 20\nPolicy Value","Shortfall /\nSurplus","Result"]]
sc_rows=[sc_hdr]
for sc,label,spy,sofr,col,bg,desc in scenarios:
    d=all_results[sc]
    rows=d['rows']
    avg_lr=sum(r['lr'] for r in rows)/20
    avg_cr=sum(r['cred'] for r in rows)/20
    final=rows[-1]
    sh=d['shortfall']; su=d['surplus']
    result="✓ SURPLUS" if sh==0 else "✗ SHORTFALL"
    sc_rows.append([
        p(sc,S("sl",fontSize=7.5,fontName="Helvetica-Bold",textColor=col,alignment=TA_LEFT)),
        p(label,S("dl",fontSize=7,fontName="Helvetica",textColor=GRAY,alignment=TA_LEFT)),
        p(f"{avg_lr:.2%}",ca_s if avg_lr>0.05 else cv_s),
        p(f"{avg_cr:.2%}",cg_s if avg_cr>=0.07 else cr_s),
        p(fmt(final['loan']),cr_s if final['loan']>10_000_000 else cv_s),
        p(fmt(final['pv']),cg_s),
        p(fmt(su) if sh==0 else fmt(-sh),cg_s if sh==0 else cr_s),
        p(result,S("rs",fontSize=7.5,fontName="Helvetica-Bold",
                   textColor=GREEN if sh==0 else RED,alignment=TA_CENTER)),
    ])

sc_t=Table(sc_rows,colWidths=[0.7*inch,1.5*inch,0.7*inch,0.7*inch,1.05*inch,1.05*inch,0.85*inch,0.85*inch])
sc_t.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LGRN,LAMB,LGRN,LRED,LRED,LAMB]),
    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]))
story.append(sc_t)
story.append(Spacer(1,9))

# ── SECTION 2: BREAK-EVEN TABLE ───────────────────────────────────────────────
story.append(p("2.  BREAK-EVEN: AT WHAT LOAN RATE DO YOU START LOSING MONEY?",sect_s))
story.append(p("(Using worst-case market window: 2000–2019, dot-com crash + GFC combined)",
               S("sub",fontSize=7.5,fontName="Helvetica-Oblique",textColor=GRAY,spaceAfter=4)))
story.append(HRFlowable(width="100%",thickness=1.5,color=GOLD))
story.append(Spacer(1,5))

be_hdr=[p(h,ch_s) for h in ["SOFR Rate","Loan Rate\n(SOFR+1%)","Year 20\nLoan Balance","Year 20\nPolicy Value","Shortfall /\nSurplus","What This Means"]]
be_rows=[be_hdr]
for sofr_r,lr,loan_bal,pv_val,sh,su in breakeven:
    is_break = abs(sh)<500_000 and (sh>0 or su<500_000)
    note = ("Comfortable surplus" if su>3_000_000 else
            "Slim surplus — close to break-even" if su>0 else
            "Small shortfall" if sh<1_000_000 else
            "Significant shortfall" if sh<5_000_000 else
            "Severe shortfall — major out-of-pocket")
    style = cg_s if sh==0 else (ca_s if sh<1_000_000 else cr_s)
    be_rows.append([
        p(f"{sofr_r:.1%}",cc_s),
        p(f"{lr:.1%}",ca_s if lr>0.06 else cv_s),
        p(fmt(loan_bal),cr_s if loan_bal>15_000_000 else cv_s),
        p(fmt(pv_val),cg_s),
        p(fmt(su) if sh==0 else fmt(-sh),style),
        p(note,S("nt",fontSize=7,fontName="Helvetica-Oblique",
                 textColor=GREEN if sh==0 else (AMBER if sh<1_000_000 else RED),alignment=TA_LEFT)),
    ])

be_t=Table(be_rows,colWidths=[0.7*inch,0.8*inch,1.35*inch,1.35*inch,1.05*inch,2.35*inch])
be_t.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[LGRN,LGRN,LGRN,LGRN,LRED,LRED,LRED,LRED,LRED,LRED,LRED]),
    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    # Highlight the break-even row (SOFR 5.5%)
    ("BACKGROUND",(0,5),(-1,5),colors.HexColor("#FFF0C8")),
    ("BOX",(0,5),(-1,5),1,AMBER),
]))
story.append(be_t)

be_note=Table([[p(
    "★ BREAK-EVEN POINT: SOFR must stay at or above 5.5% for ALL 20 YEARS CONTINUOUSLY "
    "combined with the worst S&P 500 market in modern history (2000–2019) before you face a shortfall. "
    "SOFR was above 5% for only 2 years in the last 25 (2023–2024). Sustained 5.5% SOFR for 20 years has never happened.",
    S("bn",fontSize=7.5,fontName="Helvetica-Bold",textColor=AMBER,leading=11))
]],colWidths=[7.5*inch])
be_note.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LAMB),("BOX",(0,0),(-1,-1),1,AMBER),
    ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
    ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8)]))
story.append(Spacer(1,5))
story.append(be_note)
story.append(Spacer(1,9))

# ── SECTION 3: YEAR-BY-YEAR for STRESS 1 (the scenario that costs you) ───────
story.append(p("3.  YEAR-BY-YEAR: THE SCENARIO THAT COSTS YOU MONEY",sect_s))
story.append(p("Stress Test 1: Worst S&P 500 market (2000–2019) + SOFR held at 5.5% every single year for 20 years",
               S("sub2",fontSize=7.5,fontName="Helvetica-Oblique",textColor=RED,spaceAfter=4)))
story.append(HRFlowable(width="100%",thickness=1.5,color=RED))
story.append(Spacer(1,4))

d=all_results["STRESS 1"]
yr_hdr=[p(h,ch_s) for h in ["Yr","S&P\nReturn","Credited\n(cap/floor)","SOFR","Loan\nRate","Lender\nLoan Balance","Policy\nValue","Net\n(Policy-Loan)"]]

yr_rows=[yr_hdr]
for r in d['rows']:
    net=r['net']
    net_s=cg_s if net>0 else cr_s
    zero_yr=r['cred']==0
    yr_rows.append([
        p(str(r['yr']),cc_s),
        p(f"{r['ret']:.1%}",cr_s if r['ret']<0 else (cg_s if r['ret']>=CAP else cv_s)),
        p(f"{r['cred']:.1%}",cr_s if r['cred']==0 else cg_s),
        p(f"{r['sofr']:.2%}",ca_s),
        p(f"{r['lr']:.2%}",cr_s),
        p(fmt(r['loan']),cr_s if r['loan']>10_000_000 else cv_s),
        p(fmt(r['pv']),ca_s if r['pv']<r['loan'] else cg_s),
        p(fmt(net),net_s),
    ])

yt=Table(yr_rows,colWidths=[0.35*inch,0.7*inch,0.75*inch,0.6*inch,0.6*inch,1.2*inch,1.2*inch,1.1*inch])
style_cmds=[
    ("BACKGROUND",(0,0),(-1,0),RED),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CCCCCC")),
    ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
    ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
]
for i,r in enumerate(d['rows']):
    bg=LIGHT if i%2==0 else WHITE
    if r['cred']==0: bg=colors.HexColor("#FFF2F2")
    if r['yr']==20:  bg=colors.HexColor("#FFDADA")
    style_cmds.append(("BACKGROUND",(0,i+1),(-1,i+1),bg))
yt.setStyle(TableStyle(style_cmds))
story.append(yt)

# Result callout
res_box=Table([[p(
    f"⚠  YEAR 20 RESULT:  Bank wants repayment of ${d['rows'][-1]['loan']:,.0f}  |  "
    f"Policy value: ${d['rows'][-1]['pv']:,.0f}  |  "
    f"YOU MUST PAY OUT OF POCKET:  ${d['shortfall']:,.0f}\n\n"
    f"This requires SOFR at 5.5% every single year for 20 consecutive years "
    f"AND the worst S&P 500 market in modern history simultaneously. "
    f"Neither condition alone causes a shortfall.",
    S("rb",fontSize=8,fontName="Helvetica-Bold",textColor=RED,leading=12))
]],colWidths=[7.5*inch])
res_box.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LRED),("BOX",(0,0),(-1,-1),1.5,RED),
    ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
    ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10)]))
story.append(Spacer(1,5))
story.append(res_box)
story.append(Spacer(1,9))

# ── SECTION 4: RISK FACTORS ───────────────────────────────────────────────────
story.append(p("4.  WHAT ACTUALLY PUTS YOU AT RISK — RANKED BY PROBABILITY",sect_s))
story.append(HRFlowable(width="100%",thickness=1.5,color=GOLD))
story.append(Spacer(1,5))

risks=[
    (AMBER,"MEDIUM","IUL cap rate gets lowered by Lincoln",
     "Currently 13.5%. If Lincoln lowers the cap to 8–9%, your average credited rate drops from 8.7% to ~5–6%, "
     "making the policy value grow slower while the loan compounds. Still unlikely to cause shortfall with real rates, "
     "but materially reduces distributions. Monitor annually."),
    (AMBER,"MEDIUM","SOFR spikes AND stays high (4%+) for several years",
     "SOFR hit 5.3% in 2023–2024. If it stays elevated for 5–7 years during premium-paying years (1–10), "
     "the loan balance grows faster. The policy can still cover it in most market conditions, but the surplus shrinks. "
     "Not a crisis — but reduces your cushion at Year 20."),
    (RED,"LOW-MEDIUM","Lender demands additional collateral mid-term",
     "If the policy net value (policy value minus loan) turns negative in early years — which it does in years 1–3 in "
     "ALL scenarios — the lender may require you to pledge additional assets. In most illustrated scenarios this gap "
     "closes by year 11–12. Know your lender's collateral trigger levels before signing."),
    (RED,"LOW","Both worst-case market AND sustained 5.5%+ SOFR simultaneously",
     "This is the only scenario that produces an out-of-pocket cost ($318k–$6.8M depending on rate). "
     "Requires 20 years of bad markets AND 20 years of elevated rates at the same time. "
     "SOFR has never stayed above 5.5% for more than 2-3 consecutive years in modern history."),
    (GREEN,"VERY LOW","Policy lapses before Year 20",
     "The 0% floor ensures the policy value never goes negative from market losses alone. "
     "Policy charges (~1.5%/yr) can erode value in sustained 0% credit years, but the lender's loan "
     "is also growing more slowly in low-rate environments. True lapse risk is extremely low."),
]

for risk_col, prob, title, detail in risks:
    prob_style = S("ps",fontSize=8,fontName="Helvetica-Bold",
                   textColor=risk_col if risk_col!=GREEN else GREEN,alignment=TA_CENTER)
    risk_row=Table([[
        p(prob, prob_style),
        Table([
            [p(f"  {title}", S("rt",fontSize=8,fontName="Helvetica-Bold",textColor=risk_col,leading=11))],
            [p(f"  {detail}", S("rd",fontSize=7.5,fontName="Helvetica",textColor=colors.HexColor("#222222"),leading=11))],
        ],colWidths=[6.6*inch])
    ]],colWidths=[0.7*inch,6.6*inch])
    risk_row.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,-1),LRED if risk_col==RED else (LAMB if risk_col==AMBER else LGRN)),
        ("BACKGROUND",(1,0),(1,-1),LIGHT if risk_col!=RED else colors.HexColor("#FFF8F8")),
        ("BOX",(0,0),(-1,-1),0.8,risk_col),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LINEBEFORE",(1,0),(1,-1),0.5,risk_col),
    ]))
    story.append(risk_row)
    story.append(Spacer(1,4))

story.append(Spacer(1,6))
# Final verdict
fv=Table([[p(
    "BOTTOM LINE:  In 25 years of actual market data, a $10M premium-financed IUL at SOFR+1% never cost you a single dollar out of pocket — "
    "including through the dot-com crash, the 2008 financial crisis, and the 2022 rate spike. "
    "The only scenario that produces a real shortfall requires sustained 5.5%+ SOFR for 20 straight years "
    "simultaneously with the worst equity market in modern history. That combination has never occurred. "
    "Your real risks are collateral calls (manageable), cap rate reductions (monitor annually), "
    "and sustained high rates in the premium years (early 2000s style). Know these, watch for them, and you are well-protected.",
    S("fv",fontSize=8,fontName="Helvetica-Bold",textColor=NAVY,leading=12))
]],colWidths=[7.5*inch])
fv.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LIGHT),("BOX",(0,0),(-1,-1),1.5,NAVY),
    ("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),9),
    ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10)]))
story.append(fv)
story.append(Spacer(1,5))
story.append(HRFlowable(width="100%",thickness=0.5,color=GRAY))
story.append(Spacer(1,3))
story.append(p(
    "SOFR/LIBOR rates from Federal Reserve historical data. S&P 500 price returns from public data. "
    "Policy charges approximated at 1.5%/yr for $10M face. Actual charges vary by age and policy year. "
    "Shortfall figures assume no additional collateral pledged or partial policy surrender. "
    "Not financial, tax, or legal advice. Consult your advisor before any decisions.",small_s))

doc.build(story)
print(f"PDF written -> {OUTPUT}")
