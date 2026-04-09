from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

OUTPUT = "Swati_Chugh_IUL_Summary.pdf"

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=letter,
    topMargin=0.45 * inch,
    bottomMargin=0.4 * inch,
    leftMargin=0.6 * inch,
    rightMargin=0.6 * inch,
)

# ── Colour palette ──────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#1B2A4A")
GOLD   = colors.HexColor("#C9A84C")
LIGHT  = colors.HexColor("#EAF0F8")
WHITE  = colors.white
GRAY   = colors.HexColor("#555555")
GREEN  = colors.HexColor("#1A6B3C")

styles = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

title_style = S("title", fontSize=17, textColor=WHITE, fontName="Helvetica-Bold",
                alignment=TA_CENTER, spaceAfter=2)
subtitle_style = S("subtitle", fontSize=9, textColor=GOLD, fontName="Helvetica-Bold",
                   alignment=TA_CENTER, spaceAfter=2)
section_style = S("section", fontSize=9, textColor=NAVY, fontName="Helvetica-Bold",
                  spaceBefore=6, spaceAfter=3)
body_style = S("body", fontSize=8, textColor=colors.HexColor("#222222"),
               fontName="Helvetica", leading=12, spaceAfter=2)
small_style = S("small", fontSize=7, textColor=GRAY, fontName="Helvetica",
                leading=10)
ans_style = S("ans", fontSize=8.5, textColor=GREEN, fontName="Helvetica-Bold",
              leading=12, spaceAfter=2)
bullet_style = S("bullet", fontSize=8, textColor=colors.HexColor("#222222"),
                 fontName="Helvetica", leading=12, leftIndent=10, spaceAfter=1)

story = []

# ── HEADER BANNER ────────────────────────────────────────────────────────────
header_data = [[
    Paragraph("Swati Chugh — Lincoln WealthBuilder IUL", title_style),
    Paragraph("One-Page Policy Summary  •  March 2026", subtitle_style),
]]
header_table = Table(header_data, colWidths=[7.3 * inch])
header_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), NAVY),
    ("TOPPADDING",    (0, 0), (-1, -1), 10),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ("LEFTPADDING",   (0, 0), (-1, -1), 12),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
    ("ROUNDEDCORNERS", [6]),
]))
story.append(header_table)
story.append(Spacer(1, 8))

# ── POLICY BASICS ────────────────────────────────────────────────────────────
story.append(Paragraph("POLICY AT A GLANCE", section_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1, 4))

basics = [
    ["Product",        "Lincoln WealthBuilder® IUL (Indexed Universal Life)"],
    ["Insured",        "Swati Chugh | Female | Age 43 | California | Preferred Non-Tobacco"],
    ["Death Benefit",  "$5,000,000 initial (increasing with cash value for first 10 years, then level)"],
    ["Annual Premium", "$266,675 / year for 10 years  →  Total scheduled: $2,666,750"],
    ["Growth Rate",    "7.19% assumed (S&P 500 Dynamic Intraday TCA 15, 1-yr indexed account)"],
    ["Riders",         "Enhanced Overloan Protection  •  Accelerated Death Benefit (ABR)  •  Performance Multiplier (PMR)"],
    ["Issued by",      "The Lincoln National Life Insurance Company, Fort Wayne, IN"],
]
basics_table = Table(
    [[Paragraph(k, S("bk", fontSize=8, fontName="Helvetica-Bold", textColor=NAVY)),
      Paragraph(v, body_style)] for k, v in basics],
    colWidths=[1.4 * inch, 5.9 * inch]
)
basics_table.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, -1), LIGHT),
    ("BACKGROUND",    (0, 0), (0, -1), colors.HexColor("#D6E4F0")),
    ("ROWBACKGROUNDS",(0, 0), (-1, -1), [LIGHT, WHITE]),
    ("TOPPADDING",    (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
    ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
]))
story.append(basics_table)
story.append(Spacer(1, 8))

# ── TWO COLUMN SECTION ───────────────────────────────────────────────────────
# Left: Q1  |  Right: Q2
def qa_box(question, answer_text, body_lines, bg):
    inner = []
    inner.append(Paragraph(question, S("q", fontSize=8, fontName="Helvetica-Bold",
                                        textColor=NAVY, spaceAfter=3)))
    inner.append(Paragraph(answer_text, ans_style))
    for line in body_lines:
        inner.append(Paragraph(f"• {line}", bullet_style))
    return inner

q1_content = qa_box(
    "Q1 — Do I pay anything out of pocket?",
    "No — $0 out of pocket, every year.",
    [
        "A third-party lender funds 100% of premiums ($266,675/yr × 10 yrs) through a Commercial Premium Financing loan.",
        "Loan interest (4.50%/yr) is also borrowed — never paid from your pocket.",
        "Interest rolls into the loan balance, growing it to $5,237,646 by Year 19.",
        "In Year 20, the policy uses its own cash value to repay the entire loan in one shot.",
        "Your Net After-Tax Outlay = $0 for all 50 illustrated years.",
    ],
    LIGHT
)

q2_content = qa_box(
    "Q2 — What if I take $150k/year starting Year 21?",
    "$150k/yr is tax-free — and sustainable.",
    [
        "Distributions are taken as indexed policy loans — not taxable income.",
        "The illustration shows $204k/yr is supportable; $150k/yr is even more conservative.",
        "At $150k/yr you'd receive $3.6M tax-free over 24 years (ages 64–87).",
        "Policy cash value and death benefit remain healthy — likely higher than the $204k scenario.",
        "Policy is not a MEC, so all loans remain income-tax-free.",
    ],
    LIGHT
)

def wrap_in_box(contents, title, title_color):
    rows = [[Paragraph(title, S("bh", fontSize=8.5, fontName="Helvetica-Bold",
                                textColor=WHITE))]]
    for item in contents:
        rows.append([item])
    t = Table(rows, colWidths=[3.45 * inch])
    style = [
        ("BACKGROUND",    (0, 0), (-1, 0), title_color),
        ("BACKGROUND",    (0, 1), (-1, -1), LIGHT),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 7),
        ("BOX",           (0, 0), (-1, -1), 0.8, title_color),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]
    t.setStyle(TableStyle(style))
    return t

box1 = wrap_in_box(q1_content, "OUT-OF-POCKET COST", NAVY)
box2 = wrap_in_box(q2_content, "TAKING MONEY OUT — YEAR 21+", GREEN)

two_col = Table([[box1, Spacer(0.1 * inch, 1), box2]], colWidths=[3.45 * inch, 0.1 * inch, 3.45 * inch])
two_col.setStyle(TableStyle([
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING",    (0, 0), (-1, -1), 0),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ("LEFTPADDING",   (0, 0), (-1, -1), 0),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
]))
story.append(two_col)
story.append(Spacer(1, 8))

# ── LOAN BALANCE TABLE ───────────────────────────────────────────────────────
story.append(Paragraph("HOW THE LOAN BALANCE BUILDS — THEN DISAPPEARS", section_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1, 4))

col_hdr = S("ch", fontSize=7.5, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER)
col_val = S("cv", fontSize=7.5, fontName="Helvetica",      textColor=GRAY,  alignment=TA_RIGHT)
col_hi  = S("cg", fontSize=7.5, fontName="Helvetica-Bold", textColor=GREEN, alignment=TA_RIGHT)

def cv(txt, hi=False):
    return Paragraph(txt, col_hi if hi else col_val)

loan_headers = [
    Paragraph(h, col_hdr) for h in [
        "Year", "Age", "Annual Premium\nBorrowed", "Interest\nAccrued @ 4.50%",
        "Loan Balance\n(End of Year)", "Net Out-of-Pocket"
    ]
]
loan_rows = [
    ["1",  "44", "$266,675", "$12,540",   "$279,216",   "$0"],
    ["2",  "45", "$266,675", "$25,671",   "$571,561",   "$0"],
    ["5",  "48", "$266,675", "$68,883",   "$1,533,700", "$0"],
    ["10", "53", "$266,675", "$155,559",  "$3,463,560", "$0"],
    ["19", "62", "$0",       "$235,238",  "$5,237,646", "$0"],
    ["20", "63", "$0",       "$0",        "$0 ← REPAID","$0"],
]

tbl_data = [loan_headers]
for i, row in enumerate(loan_rows):
    hi = (i == 5)
    tbl_data.append([cv(c, hi) for c in row])

loan_table = Table(tbl_data, colWidths=[0.45*inch, 0.45*inch, 1.3*inch, 1.35*inch, 1.65*inch, 1.35*inch])
loan_table.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
    ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT]),
    ("BACKGROUND",    (0, 6), (-1, 6), colors.HexColor("#D6F0E0")),
    ("TOPPADDING",    (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("LEFTPADDING",   (0, 0), (-1, -1), 5),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
    ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
    ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
]))
story.append(loan_table)
story.append(Spacer(1, 8))

# ── DISTRIBUTION SCENARIO TABLE ──────────────────────────────────────────────
story.append(Paragraph("ILLUSTRATED DISTRIBUTIONS AFTER YEAR 20  (@ 7.19% assumed rate)", section_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD))
story.append(Spacer(1, 4))

dist_headers = [Paragraph(h, col_hdr) for h in [
    "Year", "Age", "Annual Distribution\n(Policy Loan — Tax-Free)",
    "Policy Cash Value\n(Net)", "Death Benefit"
]]
dist_rows = [
    ["21", "64", "$204,086",  "$1,322,504", "$3,015,057"],
    ["25", "68", "$204,086",  "$1,288,841", "$2,958,027"],
    ["30", "73", "$204,086",  "$1,479,690", "$2,915,634"],
    ["35", "78", "$204,086",  "$2,126,170", "$3,045,790"],
    ["40", "83", "$204,086",  "$3,473,105", "$4,766,676"],
    ["43", "86", "$204,086",  "$4,739,521", "$6,323,876"],
]
note_row = [[Paragraph(
    "★  At $150k/yr (your scenario) — less drawn each year means higher cash value, larger death benefit, longer policy life.  "
    "Total tax-free income at $150k/yr over 24 yrs = $3,600,000",
    S("note", fontSize=7, fontName="Helvetica-Oblique", textColor=NAVY))]]

dist_data = [dist_headers] + [[cv(c) for c in r] for r in dist_rows]
dist_table = Table(dist_data, colWidths=[0.45*inch, 0.45*inch, 2.15*inch, 1.8*inch, 1.75*inch])
dist_table.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
    ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT]),
    ("TOPPADDING",    (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("LEFTPADDING",   (0, 0), (-1, -1), 5),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
    ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
    ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
]))
story.append(dist_table)

note_table = Table(note_row, colWidths=[7.3 * inch])
note_table.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#FFF8E6")),
    ("BOX",           (0, 0), (-1, -1), 0.5, GOLD),
    ("TOPPADDING",    (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("LEFTPADDING",   (0, 0), (-1, -1), 7),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 7),
]))
story.append(Spacer(1, 4))
story.append(note_table)
story.append(Spacer(1, 6))

# ── DISCLAIMER ───────────────────────────────────────────────────────────────
story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY))
story.append(Spacer(1, 3))
disclaimer = (
    "This one-page summary is for discussion purposes only and is derived from a Lincoln Financial illustration dated March 10, 2026. "
    "All values are hypothetical and based on a 7.19% non-guaranteed assumed interest rate. Actual results will vary. "
    "Policy loans and withdrawals reduce cash value and death benefit and may cause lapse. "
    "This is not a contract and does not constitute tax, legal, or investment advice. "
    "Consult your financial professional and tax advisor before making any decisions."
)
story.append(Paragraph(disclaimer, small_style))

doc.build(story)
print(f"PDF written → {OUTPUT}")
