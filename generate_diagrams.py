"""
Architecture diagram generator for the MCP + SLM article.
Produces three PNG diagrams:
  1. arch_overview.png         — end-to-end modern inference stack
  2. arch_token_reduction.png  — token budget comparison old vs new
  3. arch_routing.png          — complexity-based model routing
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ── shared style ───────────────────────────────────────────────────────────────
FONT = "DejaVu Sans"
BG   = "#0f1117"
CARD = "#1e2130"
CARD2= "#252840"

BLUE   = "#4a90e2"
TEAL   = "#00bcd4"
GREEN  = "#4caf50"
ORANGE = "#ff9800"
PURPLE = "#9c27b0"
RED    = "#f44336"
GREY   = "#78909c"
YELLOW = "#ffd600"
LIME   = "#cddc39"

TEXT_BRIGHT = "#e8eaf6"
TEXT_DIM    = "#90a4ae"

def box(ax, x, y, w, h, color, label, sublabel="", alpha=0.92, radius=0.02, fontsize=10, sublabel_fontsize=8):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        linewidth=1.5, edgecolor=color,
        facecolor=color + "33",
        alpha=alpha, zorder=3
    )
    ax.add_patch(patch)
    cy = y + h / 2 + (0.015 if sublabel else 0)
    ax.text(x + w/2, cy, label,
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            color=TEXT_BRIGHT, fontfamily=FONT, zorder=4)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.025, sublabel,
                ha="center", va="center",
                fontsize=sublabel_fontsize,
                color=TEXT_DIM, fontfamily=FONT, zorder=4)

def arrow(ax, x1, y1, x2, y2, color=TEXT_DIM, lw=1.5, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=5)

def label(ax, x, y, text, color=TEXT_DIM, fontsize=8, ha="center", bold=False):
    ax.text(x, y, text, ha=ha, va="center",
            fontsize=fontsize, color=color,
            fontfamily=FONT, fontweight="bold" if bold else "normal", zorder=6)


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 1 — End-to-end modern inference stack
# ══════════════════════════════════════════════════════════════════════════════
def diagram_overview():
    fig, ax = plt.subplots(figsize=(18, 11))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── title ──────────────────────────────────────────────────────────────────
    ax.text(0.5, 0.965, "Modern AI Inference Stack",
            ha="center", va="center", fontsize=18, fontweight="bold",
            color=TEXT_BRIGHT, fontfamily=FONT)
    ax.text(0.5, 0.942, "MCP Servers · Skills · Chat Interface · Local SLM",
            ha="center", va="center", fontsize=11, color=TEAL, fontfamily=FONT)

    # ── Layer 0: Users ─────────────────────────────────────────────────────────
    y0 = 0.86
    for i, (lbl, sub, col) in enumerate([
        ("Chat Widget", "web / mobile", BLUE),
        ("API Client",  "REST / SDK",   TEAL),
        ("Voice UI",    "ASR / TTS",    PURPLE),
    ]):
        x = 0.05 + i * 0.305
        box(ax, x, y0, 0.26, 0.062, col, lbl, sub, fontsize=9, sublabel_fontsize=7.5)

    # section header
    ax.text(0.5, y0 - 0.015, "USER / CLIENT LAYER",
            ha="center", va="center", fontsize=7.5, color=TEXT_DIM, fontfamily=FONT)

    # ── Arrow down ─────────────────────────────────────────────────────────────
    arrow(ax, 0.5, y0, 0.5, y0 - 0.055)
    label(ax, 0.535, y0 - 0.028, "OpenAI-compatible Chat API", TEXT_DIM, 7.5)

    # ── Layer 1: Inference Gateway ─────────────────────────────────────────────
    y1 = 0.74
    box(ax, 0.05, y1, 0.90, 0.075, ORANGE, "Inference Gateway",
        "Auth · Session Memory · Skill Registry · Complexity Router",
        fontsize=11, sublabel_fontsize=8.5)
    ax.text(0.5, y1 - 0.016, "GATEWAY LAYER",
            ha="center", va="center", fontsize=7.5, color=TEXT_DIM, fontfamily=FONT)

    # ── Arrows down from gateway ───────────────────────────────────────────────
    for x in [0.18, 0.50, 0.82]:
        arrow(ax, x, y1, x, y1 - 0.055)

    # ── Layer 2: Skill + Context + Routing ─────────────────────────────────────
    y2 = 0.595
    box(ax, 0.05,  y2, 0.26, 0.082, GREEN,  "Skill Layer",
        "check_booking\nget_pricing\nfind_policy", fontsize=9, sublabel_fontsize=7.5)
    box(ax, 0.37,  y2, 0.26, 0.082, TEAL,   "Context Layer",
        "Conversation history\nUser profile\nSession state", fontsize=9, sublabel_fontsize=7.5)
    box(ax, 0.69,  y2, 0.26, 0.082, PURPLE, "Routing Layer",
        "Complexity classifier\nSLM vs LLM decision\nCost policy", fontsize=9, sublabel_fontsize=7.5)
    ax.text(0.5, y2 - 0.016, "ORCHESTRATION LAYER",
            ha="center", va="center", fontsize=7.5, color=TEXT_DIM, fontfamily=FONT)

    # ── Arrows down from skill layer ───────────────────────────────────────────
    for x in [0.18, 0.50, 0.82]:
        arrow(ax, x, y2, x, y2 - 0.052)

    # ── Layer 3: MCP Servers ───────────────────────────────────────────────────
    y3 = 0.438
    for i, (lbl, sub, col) in enumerate([
        ("MCP: Documents",  "PDFs · Wiki · Notes", BLUE),
        ("MCP: Database",   "SQL · NoSQL · Vectors", TEAL),
        ("MCP: External APIs", "REST · GraphQL · WebHooks", ORANGE),
    ]):
        x = 0.05 + i * 0.305
        box(ax, x, y3, 0.26, 0.082, col, lbl, sub, fontsize=9, sublabel_fontsize=7.5)

    ax.text(0.5, y3 - 0.016, "MCP SERVER LAYER  (structured, typed, minimal context)",
            ha="center", va="center", fontsize=7.5, color=TEXT_DIM, fontfamily=FONT)

    # horizontal connector under MCP
    ax.plot([0.18, 0.50, 0.82], [y3 + 0.082/2]*3, color=TEXT_DIM + "55", lw=0, zorder=2)
    arrow(ax, 0.18, y3, 0.18, y3 - 0.052)
    arrow(ax, 0.50, y3, 0.50, y3 - 0.052)
    arrow(ax, 0.82, y3, 0.82, y3 - 0.052)
    # horizontal merge line
    ax.plot([0.18, 0.82], [y3 - 0.015, y3 - 0.015], color=TEXT_DIM, lw=1.2, zorder=5)
    ax.plot([0.50, 0.50], [y3 - 0.015, y3 - 0.040], color=TEXT_DIM, lw=1.2, zorder=5)
    label(ax, 0.5, y3 - 0.043, "Structured Context (low token count)", TEXT_DIM, 7.5)
    arrow(ax, 0.5, y3 - 0.048, 0.5, y3 - 0.070)

    # ── Layer 4: SLM + Cloud LLM ───────────────────────────────────────────────
    y4 = 0.26
    box(ax, 0.05,  y4, 0.26, 0.10, GREEN,  "Local SLM",
        "Phi-3 Mini · Gemma 2 · Qwen2.5\n~375 tokens · <300ms · $0.00",
        fontsize=9, sublabel_fontsize=7.5)
    box(ax, 0.37,  y4, 0.26, 0.10, YELLOW, "Mid SLM",
        "Llama-3 8B · Mistral 7B\nSelf-hosted · moderate cost",
        fontsize=9, sublabel_fontsize=7.5)
    box(ax, 0.69,  y4, 0.26, 0.10, RED,    "Cloud LLM (Fallback)",
        "GPT-4o · Claude 3.5\nComplex / creative only · 5% of queries",
        fontsize=9, sublabel_fontsize=7.5)
    ax.text(0.5, y4 - 0.016, "INFERENCE / MODEL LAYER",
            ha="center", va="center", fontsize=7.5, color=TEXT_DIM, fontfamily=FONT)

    # ── usage % labels ─────────────────────────────────────────────────────────
    for x, pct, col in [(0.18, "~75% of queries", GREEN), (0.50, "~20%", YELLOW), (0.82, "~5%", RED)]:
        ax.text(x, y4 - 0.032, pct, ha="center", va="center",
                fontsize=8, color=col, fontfamily=FONT, fontweight="bold")

    # ── legend strip ───────────────────────────────────────────────────────────
    y_leg = 0.03
    items = [("Data source (MCP)", BLUE), ("Skill / Orchestration", GREEN),
             ("Context / Memory", TEAL), ("Routing", PURPLE),
             ("Local SLM (free)", GREEN), ("Cloud LLM (pay-per-token)", RED)]
    for i, (txt, col) in enumerate(items):
        xpos = 0.03 + i * 0.162
        patch = mpatches.Patch(facecolor=col+"44", edgecolor=col, linewidth=1.2)
        ax.legend(handles=[patch], labels=[txt],
                  loc="lower left", bbox_to_anchor=(xpos, y_leg),
                  fontsize=7, framealpha=0,
                  labelcolor=TEXT_DIM, handlelength=1.2,
                  handleheight=0.8)

    plt.tight_layout(pad=0.3)
    plt.savefig("arch_overview.png", dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close()
    print("Saved arch_overview.png")


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 2 — Token budget: old vs new
# ══════════════════════════════════════════════════════════════════════════════
def diagram_token_reduction():
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor(BG)

    titles  = ["Traditional RAG + Large LLM", "MCP Skills + Local SLM"]
    colors_left  = [GREY, RED, BLUE, ORANGE, GREEN]
    colors_right = [GREY, GREEN, BLUE, TEAL, LIME]

    old_segments = [
        ("System Prompt",        500, GREY),
        ("RAG Context (noise)",  1200, RED),
        ("RAG Context (useful)", 600, ORANGE),
        ("History",              400, BLUE),
        ("User Query",           25,  GREEN),
    ]
    new_segments = [
        ("System Prompt",        120, GREY),
        ("Skill Result",         120, GREEN),
        ("MCP Context",          80,  TEAL),
        ("History (summarised)", 40,  BLUE),
        ("User Query",           22,  LIME),
    ]

    old_total = sum(s[1] for s in old_segments)
    new_total = sum(s[1] for s in new_segments)

    for ax, segments, total, title in zip(axes,
                                           [old_segments, new_segments],
                                           [old_total, new_total],
                                           titles):
        ax.set_facecolor(BG)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.18, 1.08)
        ax.axis("off")

        ax.text(0.5, 1.03, title, ha="center", va="center",
                fontsize=13, fontweight="bold", color=TEXT_BRIGHT, fontfamily=FONT)
        ax.text(0.5, 0.97, f"Total input: {total:,} tokens",
                ha="center", va="center", fontsize=10, color=TEAL, fontfamily=FONT)

        y_cur = 0.88
        bar_w = 0.60
        bar_x = 0.20
        bar_h_unit = 0.70 / total

        for name, count, col in segments:
            h = count * bar_h_unit
            patch = FancyBboxPatch(
                (bar_x, y_cur - h), bar_w, h,
                boxstyle="round,pad=0.004,rounding_size=0.008",
                linewidth=1.2, edgecolor=col,
                facecolor=col + "44", zorder=3
            )
            ax.add_patch(patch)
            pct = count / total * 100
            ax.text(bar_x + bar_w + 0.03, y_cur - h/2,
                    f"{name}\n{count} tokens  ({pct:.0f}%)",
                    va="center", fontsize=8, color=TEXT_DIM, fontfamily=FONT)
            y_cur -= h

        # reduction badge
        if total == new_total:
            reduction = 100 * (1 - new_total / old_total)
            ax.text(0.5, -0.06,
                    f"↓ {reduction:.0f}% fewer input tokens",
                    ha="center", va="center", fontsize=14,
                    fontweight="bold", color=GREEN, fontfamily=FONT)
            ax.text(0.5, -0.13,
                    f"+ runs locally at $0.00 / query",
                    ha="center", va="center", fontsize=10, color=LIME, fontfamily=FONT)
        else:
            ax.text(0.5, -0.06,
                    f"Cloud LLM  ·  ~$0.015 / query",
                    ha="center", va="center", fontsize=11,
                    fontweight="bold", color=RED, fontfamily=FONT)
            ax.text(0.5, -0.13,
                    f"3–8 second response latency",
                    ha="center", va="center", fontsize=10, color=GREY, fontfamily=FONT)

    fig.suptitle("Token Budget: Naive RAG vs. MCP + Skills + SLM",
                 fontsize=15, fontweight="bold", color=TEXT_BRIGHT, y=1.01)
    plt.tight_layout(pad=1.5)
    plt.savefig("arch_token_reduction.png", dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close()
    print("Saved arch_token_reduction.png")


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 3 — Complexity-based routing
# ══════════════════════════════════════════════════════════════════════════════
def diagram_routing():
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.96, "Complexity-Based Inference Routing",
            ha="center", va="center", fontsize=17, fontweight="bold",
            color=TEXT_BRIGHT, fontfamily=FONT)
    ax.text(0.5, 0.925, "Right model · Right context · Right cost",
            ha="center", va="center", fontsize=11, color=TEAL, fontfamily=FONT)

    # ── incoming query ─────────────────────────────────────────────────────────
    box(ax, 0.35, 0.83, 0.30, 0.06, BLUE, "Incoming User Query", fontsize=10)
    arrow(ax, 0.50, 0.83, 0.50, 0.775)

    # ── complexity classifier ──────────────────────────────────────────────────
    box(ax, 0.30, 0.70, 0.40, 0.07, ORANGE,
        "Complexity Classifier",
        "embedding distance · query length · tool count needed",
        fontsize=10, sublabel_fontsize=8)
    arrow(ax, 0.50, 0.70, 0.50, 0.645)

    # ── score bar ─────────────────────────────────────────────────────────────
    ax.text(0.50, 0.635, "Complexity Score  0.0 ──────────────────────── 1.0",
            ha="center", va="center", fontsize=8.5, color=TEXT_DIM, fontfamily=FONT)

    # colour gradient bar
    n = 300
    grad = np.linspace(0, 1, n).reshape(1, -1)
    ax.imshow(grad, aspect="auto", extent=(0.05, 0.95, 0.595, 0.620),
              cmap="RdYlGn_r", zorder=3, alpha=0.8)
    for x, lbl in [(0.20, "0.4"), (0.45, "0.7"), (0.65, "0.9")]:
        ax.plot([x, x], [0.592, 0.623], color="white", lw=1.2, zorder=4)
        ax.text(x, 0.585, lbl, ha="center", va="center",
                fontsize=8, color=TEXT_DIM, fontfamily=FONT)

    # ── branch arrows ──────────────────────────────────────────────────────────
    branch_xs   = [0.12, 0.37, 0.62, 0.87]
    branch_lbls = ["0.0 – 0.4", "0.4 – 0.7", "0.7 – 0.9", "0.9 – 1.0"]
    for bx, bl in zip(branch_xs, branch_lbls):
        ax.plot([bx, bx], [0.595, 0.545], color=TEXT_DIM, lw=1.2, zorder=5)
        ax.plot([0.12, 0.87], [0.595, 0.595], color=TEXT_DIM, lw=1.2, zorder=5)
        arrow(ax, bx, 0.545, bx, 0.505)
        ax.text(bx, 0.555, bl, ha="center", va="center",
                fontsize=7.5, color=TEXT_DIM, fontfamily=FONT)

    # ── model boxes ───────────────────────────────────────────────────────────
    model_defs = [
        (0.012, "Small Local SLM",
         "Gemma 2 2B · Qwen2.5 1.5B\n~200 tokens\n<100ms · $0.00\n\nSimple lookup\nClassification\nShort Q&A",
         GREEN, "~50% of queries"),
        (0.262, "Local SLM",
         "Phi-3 Mini 3.8B\nLlama-3.2 3B\n~375 tokens\n<400ms · $0.00\n\nMulti-step reasoning\nTool chaining\nDomain Q&A",
         TEAL, "~25% of queries"),
        (0.512, "Self-Hosted Mid LLM",
         "Llama-3 8B / 70B\nMistral 7B\n~800 tokens\n1–3s · very low cost\n\nComplex analysis\nCode generation\nMulti-hop",
         YELLOW, "~20% of queries"),
        (0.762, "Cloud Frontier LLM",
         "GPT-4o · Claude 3.5\nGemini 1.5 Pro\n~2700 tokens\n3–8s · $0.01–$0.05\n\nCreative writing\nOpen-ended tasks\nBroad knowledge",
         RED, "~5% of queries"),
    ]

    for bx, title, sub, col, pct in model_defs:
        box(ax, bx, 0.24, 0.226, 0.26, col, title, sub,
            fontsize=8.5, sublabel_fontsize=7.5)
        ax.text(bx + 0.113, 0.22, pct, ha="center", va="center",
                fontsize=9, fontweight="bold", color=col, fontfamily=FONT)

    # ── privacy / local indicators ─────────────────────────────────────────────
    ax.text(0.12,  0.175, "  Fully local", ha="center", va="center",
            fontsize=8.5, color=GREEN, fontfamily=FONT, fontweight="bold")
    ax.text(0.375, 0.175, "  Fully local", ha="center", va="center",
            fontsize=8.5, color=GREEN, fontfamily=FONT, fontweight="bold")
    ax.text(0.625, 0.175, "  Self-hosted", ha="center", va="center",
            fontsize=8.5, color=YELLOW, fontfamily=FONT, fontweight="bold")
    ax.text(0.875, 0.175, "  Cloud", ha="center", va="center",
            fontsize=8.5, color=RED, fontfamily=FONT, fontweight="bold")

    # ── cost summary bar ───────────────────────────────────────────────────────
    box(ax, 0.05, 0.06, 0.90, 0.08, TEAL,
        "Overall cost reduction: ~95% vs. sending all queries to GPT-4",
        "75% x FREE  +  25% x $0.001  +  20% x $0.003  +  5% x $0.015  =  $0.0010 avg/query   (was $0.015)",
        fontsize=10, sublabel_fontsize=8.5)

    plt.tight_layout(pad=0.3)
    plt.savefig("arch_routing.png", dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close()
    print("Saved arch_routing.png")


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 4 — MCP + Skill lifecycle (single query walkthrough)
# ══════════════════════════════════════════════════════════════════════════════
def diagram_query_lifecycle():
    fig, ax = plt.subplots(figsize=(17, 10))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.965, "Query Lifecycle: MCP + Skills + SLM",
            ha="center", va="center", fontsize=17, fontweight="bold",
            color=TEXT_BRIGHT, fontfamily=FONT)
    ax.text(0.5, 0.935, "From raw user message to grounded answer — without a cloud GPU",
            ha="center", va="center", fontsize=11, color=TEAL, fontfamily=FONT)

    steps = [
        # (y_center, step_num, left_label, box_color, right_detail)
        (0.855, "①", "User sends message", BLUE,
         '"What are my 3 largest charges last month — can I dispute any?"'),
        (0.755, "②", "Gateway receives query", ORANGE,
         "Creates session ID · loads conversation history (40 tokens)"),
        (0.655, "③", "Skill Registry lookup", GREEN,
         "Matches: get_transactions(period, sort, limit)  +  get_dispute_policy(topic)"),
        (0.555, "④", "MCP tool calls (parallel)", TEAL,
         "Finance.search_transactions → 3 rows (80 tok)  |  Policy.lookup → rules (60 tok)"),
        (0.455, "⑤", "Context assembly", PURPLE,
         "System 120 tok + Skill results 140 tok + History 40 tok + Query 22 tok = 322 tok"),
        (0.345, "⑥", "SLM inference (local)", LIME,
         "Phi-3 Mini 3.8B · 322 input tokens · 280ms · $0.00"),
        (0.230, "⑦", "Validator + streamer", YELLOW,
         "Checks source citations are present · streams tokens to client"),
        (0.120, "⑧", "Grounded answer delivered", GREEN,
         '"Your 3 largest charges: Amazon $88, Grubhub $79, Whole Foods $78.\n Dispute window is 60 days — all three qualify. [src: TXN-12, TXN-18, TXN-31]"'),
    ]

    for yc, num, lbl, col, detail in steps:
        # step circle
        circle = plt.Circle((0.055, yc), 0.028, color=col + "55",
                             ec=col, lw=1.8, zorder=4)
        ax.add_patch(circle)
        ax.text(0.055, yc, num, ha="center", va="center",
                fontsize=11, fontweight="bold", color=col, fontfamily=FONT, zorder=5)

        # left label
        ax.text(0.095, yc + 0.025, lbl,
                ha="left", va="center", fontsize=9.5, fontweight="bold",
                color=TEXT_BRIGHT, fontfamily=FONT, zorder=5)
        # detail box
        patch = FancyBboxPatch(
            (0.09, yc - 0.052), 0.895, 0.046,
            boxstyle="round,pad=0.005,rounding_size=0.01",
            linewidth=1.0, edgecolor=col + "66",
            facecolor=col + "15", zorder=3
        )
        ax.add_patch(patch)
        ax.text(0.105, yc - 0.028, detail,
                ha="left", va="center", fontsize=8,
                color=TEXT_DIM, fontfamily=FONT, zorder=5)

        # connector line
        if yc > 0.12:
            ax.plot([0.055, 0.055], [yc - 0.028, yc - 0.080],
                    color=TEXT_DIM + "55", lw=1.2, zorder=2)

    # total latency badge
    box(ax, 0.30, 0.025, 0.40, 0.055, GREEN,
        "Total end-to-end latency: ~320ms   |   Cost: $0.00",
        "(vs 3–8s and $0.015 with cloud GPT-4 + RAG)",
        fontsize=10, sublabel_fontsize=8.5)

    plt.tight_layout(pad=0.3)
    plt.savefig("arch_query_lifecycle.png", dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close()
    print("Saved arch_query_lifecycle.png")


if __name__ == "__main__":
    diagram_overview()
    diagram_token_reduction()
    diagram_routing()
    diagram_query_lifecycle()
    print("All diagrams generated.")
