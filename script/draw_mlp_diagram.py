"""Draw the full MLP block diagram showing computation flow,
measurement points, and compression targets."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 18))
ax.set_xlim(0, 14)
ax.set_ylim(0, 18)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── colour palette ──
C_RESIDUAL = "#e0e7ff"     # light indigo
C_NORM = "#dbeafe"         # light blue
C_COMPRESS = "#fef3c7"     # light amber  (compression targets)
C_ACTIVATION = "#fce7f3"   # light pink
C_OPERATE = "#d1fae5"      # light green
C_MEASURE_BG = "#f0fdf4"   # very light green for measure annotation bg
C_BORDER = "#374151"       # dark gray
C_ARROW = "#374151"
C_MEASURE = "#dc2626"      # red for measurement callouts
C_TARGET = "#b45309"       # amber-dark for compression target callouts

def box(x, y, w, h, label, color, fontsize=11, bold=False, sublabel=None):
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color, edgecolor=C_BORDER, linewidth=1.5,
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x, y + (0.12 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize, fontweight=weight,
            color="#111827")
    if sublabel:
        ax.text(x, y - 0.25, sublabel,
                ha="center", va="center", fontsize=8.5, fontstyle="italic",
                color="#6b7280")

def arrow(x1, y1, x2, y2, **kw):
    style = kw.pop("style", "->")
    color = kw.pop("color", C_ARROW)
    lw = kw.pop("lw", 1.5)
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle="arc3,rad=0"))

def curved_arrow(x1, y1, x2, y2, rad=0.3, **kw):
    color = kw.pop("color", C_ARROW)
    lw = kw.pop("lw", 1.5)
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"))

def measure_callout(x, y, text, side="right"):
    dx = 2.7 if side == "right" else -2.7
    ha = "left" if side == "right" else "right"
    ax.annotate(
        text, xy=(x, y), xytext=(x + dx, y),
        fontsize=8.5, color=C_MEASURE, fontweight="bold",
        ha=ha, va="center",
        arrowprops=dict(arrowstyle="-|>", color=C_MEASURE, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.25", fc=C_MEASURE_BG, ec=C_MEASURE, lw=0.8),
    )

def target_callout(x, y, text, side="left"):
    dx = -2.8 if side == "left" else 2.8
    ha = "right" if side == "left" else "left"
    ax.annotate(
        text, xy=(x, y), xytext=(x + dx, y),
        fontsize=8.5, color=C_TARGET, fontweight="bold",
        ha=ha, va="center",
        arrowprops=dict(arrowstyle="-|>", color=C_TARGET, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.25", fc="#fffbeb", ec=C_TARGET, lw=0.8),
    )


# ── y-coordinates (bottom to top) ──
Y_RES_IN   = 1.5
Y_ATTN     = 3.2
Y_ADD1     = 4.8
Y_RES_MID  = 6.0   # residual stream reference point
Y_NORM     = 7.5   # post_attention_layernorm
Y_GATE     = 9.5   # gate_proj
Y_UP       = 9.5   # up_proj (parallel)
Y_SILU     = 11.0
Y_MUL      = 12.2
Y_DOWN     = 13.5
Y_ADD2     = 15.0
Y_RES_OUT  = 16.5

X_MID      = 7.0
X_GATE     = 5.0
X_UP       = 9.0

# ── MLP sub-block boundary ──
mlp_rect = FancyBboxPatch(
    (2.5, Y_NORM - 0.7), 9.0, Y_ADD2 - Y_NORM + 1.4,
    boxstyle="round,pad=0.3",
    facecolor="#f9fafb", edgecolor="#9ca3af", linewidth=1.2,
    linestyle="--",
)
ax.add_patch(mlp_rect)
ax.text(11.2, Y_ADD2 + 0.3, "MLP Block", fontsize=10, color="#6b7280",
        fontstyle="italic", ha="right")

# ══════════════════════════════════════
#  BOXES
# ══════════════════════════════════════

# Residual stream input
box(X_MID, Y_RES_IN, 4.0, 0.7, "Residual Stream (input)", C_RESIDUAL, bold=True)

# Self-attention (simplified)
box(X_MID, Y_ATTN, 3.5, 0.7, "Self-Attention", "#e5e7eb", sublabel="(not compressed)")

# Add & Norm (first residual add)
box(X_MID, Y_ADD1, 1.2, 0.6, "+", "#f3f4f6", fontsize=14, bold=True)

# Residual stream mid-point
box(X_MID, Y_RES_MID, 4.2, 0.7, "Residual Stream (pre-MLP)", C_RESIDUAL, bold=True)

# Post-attention layernorm
box(X_MID, Y_NORM, 4.5, 0.8, "post_attention_layernorm", C_NORM, bold=True,
    sublabel="RMSNorm → MLP input x")

# Gate proj  (compression target)
box(X_GATE, Y_GATE, 3.0, 0.8, "gate_proj", C_COMPRESS, bold=True,
    sublabel="W_gate · x")

# Up proj  (compression target)
box(X_UP, Y_UP, 3.0, 0.8, "up_proj", C_COMPRESS, bold=True,
    sublabel="W_up · x")

# SiLU activation (only on gate branch)
box(X_GATE, Y_SILU, 2.2, 0.7, "SiLU(·)", C_ACTIVATION, bold=True)

# Element-wise multiply
box(X_MID, Y_MUL, 3.6, 0.7, "Element-wise  ⊙", C_OPERATE, bold=True,
    sublabel="h = SiLU(W_gate·x) ⊙ (W_up·x)")

# Down proj
box(X_MID, Y_DOWN, 3.0, 0.8, "down_proj", "#e5e7eb", bold=True,
    sublabel="W_down · h → MLP output")

# Second residual add
box(X_MID, Y_ADD2, 1.2, 0.6, "+", "#f3f4f6", fontsize=14, bold=True)

# Residual stream output
box(X_MID, Y_RES_OUT, 4.2, 0.7, "Residual Stream (output)", C_RESIDUAL, bold=True)

# ══════════════════════════════════════
#  ARROWS  (bottom-to-top flow)
# ══════════════════════════════════════

# Residual in → self-attention
arrow(X_MID, Y_RES_IN + 0.35, X_MID, Y_ATTN - 0.35)

# Self-attention → add
arrow(X_MID, Y_ATTN + 0.35, X_MID, Y_ADD1 - 0.3)

# Residual skip around self-attention
curved_arrow(X_MID - 2.0, Y_RES_IN + 0.35, X_MID - 0.6, Y_ADD1 - 0.3,
             rad=-0.25, color="#9ca3af")
ax.text(3.0, 3.2, "skip", fontsize=8, color="#9ca3af", fontstyle="italic")

# Add → residual mid
arrow(X_MID, Y_ADD1 + 0.3, X_MID, Y_RES_MID - 0.35)

# Residual mid → layernorm
arrow(X_MID, Y_RES_MID + 0.35, X_MID, Y_NORM - 0.4)

# Layernorm → gate_proj  and  layernorm → up_proj  (fan out)
arrow(X_MID - 0.5, Y_NORM + 0.4, X_GATE, Y_GATE - 0.4)
arrow(X_MID + 0.5, Y_NORM + 0.4, X_UP, Y_UP - 0.4)
ax.text(X_MID, Y_NORM + 0.65, "x", fontsize=10, ha="center", color="#4b5563",
        fontweight="bold")

# gate_proj → SiLU
arrow(X_GATE, Y_GATE + 0.4, X_GATE, Y_SILU - 0.35)

# SiLU → element-wise multiply
arrow(X_GATE, Y_SILU + 0.35, X_MID - 0.3, Y_MUL - 0.35)

# up_proj → element-wise multiply (straight down past SiLU)
arrow(X_UP, Y_UP + 0.4, X_MID + 0.3, Y_MUL - 0.35)

# element-wise multiply → down_proj
arrow(X_MID, Y_MUL + 0.35, X_MID, Y_DOWN - 0.4)
ax.text(X_MID + 0.35, Y_MUL + 0.62, "h", fontsize=10, ha="left", color="#4b5563",
        fontweight="bold")

# down_proj → add
arrow(X_MID, Y_DOWN + 0.4, X_MID, Y_ADD2 - 0.3)

# Residual skip around MLP block
curved_arrow(X_MID + 2.6, Y_RES_MID + 0.35, X_MID + 0.6, Y_ADD2 - 0.3,
             rad=0.35, color="#9ca3af")
ax.text(11.0, 10.5, "skip", fontsize=8, color="#9ca3af", fontstyle="italic")

# Add → residual out
arrow(X_MID, Y_ADD2 + 0.3, X_MID, Y_RES_OUT - 0.35)

# ══════════════════════════════════════
#  MEASUREMENT CALLOUTS (red)
# ══════════════════════════════════════

measure_callout(X_MID + 2.1, Y_RES_MID, "MEASURE: residual stream\n(uncentered trace)", side="right")
measure_callout(X_MID + 2.25, Y_NORM, "MEASURE: MLP input\n(centered & uncentered e-rank)", side="right")
measure_callout(X_MID + 1.5, Y_DOWN + 0.0, "MEASURE: MLP output\n(centered & uncentered e-rank,\n uncentered trace)", side="right")

# ══════════════════════════════════════
#  COMPRESSION TARGET CALLOUTS (amber)
# ══════════════════════════════════════

target_callout(X_GATE - 1.5, Y_GATE, "COMPRESS:\nlow-rank approx\nW ≈ U Σ V^T", side="left")
target_callout(X_UP + 1.5, Y_UP, "COMPRESS:\nlow-rank approx\nW ≈ U Σ V^T", side="right")

# ══════════════════════════════════════
#  DIMENSION ANNOTATIONS
# ══════════════════════════════════════
ax.text(X_GATE, Y_GATE - 0.65,
        "d_model → d_intermediate", fontsize=7.5, ha="center", color="#6b7280")
ax.text(X_UP, Y_UP - 0.65,
        "d_model → d_intermediate", fontsize=7.5, ha="center", color="#6b7280")
ax.text(X_MID, Y_DOWN - 0.65,
        "d_intermediate → d_model", fontsize=7.5, ha="center", color="#6b7280")

# ══════════════════════════════════════
#  TITLE
# ══════════════════════════════════════
ax.text(X_MID, 17.5,
        "SwiGLU MLP Block — Computation, Measurement & Compression",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#111827")

# ══════════════════════════════════════
#  LEGEND
# ══════════════════════════════════════
legend_elements = [
    mpatches.Patch(facecolor=C_RESIDUAL, edgecolor=C_BORDER, label="Residual stream"),
    mpatches.Patch(facecolor=C_NORM, edgecolor=C_BORDER, label="Layer normalization"),
    mpatches.Patch(facecolor=C_COMPRESS, edgecolor=C_BORDER, label="Linear projection (compression target)"),
    mpatches.Patch(facecolor=C_ACTIVATION, edgecolor=C_BORDER, label="Nonlinear activation"),
    mpatches.Patch(facecolor=C_OPERATE, edgecolor=C_BORDER, label="Element-wise operation"),
    mpatches.Patch(facecolor=C_MEASURE_BG, edgecolor=C_MEASURE, label="Stage-1 measurement point"),
    mpatches.Patch(facecolor="#fffbeb", edgecolor=C_TARGET, label="Stage-3 compression target"),
]
ax.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=8.5,
          frameon=True, fancybox=True, framealpha=0.9,
          bbox_to_anchor=(0.5, -0.01))

plt.tight_layout()
out_path = str(__file__).replace("script/draw_mlp_diagram.py",
                                  "result/mlp_block_diagram.pdf")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
out_png = out_path.replace(".pdf", ".png")
plt.savefig(out_png, dpi=200, bbox_inches="tight")
print(f"Saved {out_path}")
print(f"Saved {out_png}")
plt.show()
