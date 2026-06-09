#!/usr/bin/env python3
"""Draw the DynUNETR3D architecture diagram using matplotlib."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "paper_latest", "figures", "arch_dynunet.png")

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY    = "#1565C0"
TEAL    = "#007A99"
GREEN   = "#2E7D32"
ORANGE  = "#E65100"
PURPLE  = "#6A1B9A"
GRAY    = "#455A64"
LGRAY   = "#ECEFF1"
WHITE   = "#FFFFFF"
GOLD    = "#F9A825"
RED     = "#B71C1C"

# ── Stage definitions ─────────────────────────────────────────────────────────
# (label, kernel, stride, out_ch, color)
ENC_STAGES = [
    ("Enc 1\nConv-Res", "[3,3,1]", "[1,1,1]", 32,  NAVY),
    ("Enc 2\nConv-Res", "[3,3,3]", "[2,2,1]", 64,  NAVY),
    ("Enc 3\nConv-Res", "[3,3,3]", "[2,2,2]", 128, NAVY),
    ("Enc 4\nConv-Res", "[3,3,3]", "[2,2,2]", 256, NAVY),
    ("Bottleneck\nConv-Res","[3,3,3]","[2,2,2]",512,TEAL),
]
DEC_STAGES = [
    ("Dec 4\nUp+Res", 256, GRAY),
    ("Dec 3\nUp+Res", 128, GRAY),
    ("Dec 2\nUp+Res", 64,  GRAY),
    ("Dec 1\nUp+Res", 32,  GRAY),
]
AUX_STAGES = [1, 2]   # Dec 3 and Dec 2 emit aux outputs (0-indexed from Dec4)

# ── Figure layout ─────────────────────────────────────────────────────────────
FIG_W, FIG_H = 18, 7
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor(LGRAY)
ax.set_facecolor(LGRAY)

# Helper: rounded box
def box(cx, cy, w, h, color, text, fontsize=8.5, text_color=WHITE,
        bold=False, alpha=1.0):
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.05", linewidth=1.2,
        edgecolor="#222222", facecolor=color, alpha=alpha, zorder=3,
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight=weight, zorder=4, wrap=True,
            multialignment="center")

def arrow(x0, y0, x1, y1, color="#333333", lw=1.5, style="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0"))

def label(x, y, text, fontsize=7.5, color="#333333", ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize,
            color=color, zorder=5)

# ── Row positions ─────────────────────────────────────────────────────────────
ENC_Y   = 5.0
DEC_Y   = 2.8
AUX_Y   = 1.4
FILM_Y  = 2.8
OUT_Y   = 2.8

BW, BH        = 1.8, 0.85   # encoder/decoder box
FILM_W, FILM_H = 1.7, 0.85
INPUT_W        = 1.6
SKIP_Y_OFFSET  = 0.15

n_enc  = len(ENC_STAGES)
n_dec  = len(DEC_STAGES)

# x positions: encoder left-to-right, decoder right-to-left mirrored
ENC_X = [1.3 + i * 2.55 for i in range(n_enc)]
# decoder lines up with encoder 0..3 (not bottleneck)
DEC_X = [ENC_X[i] for i in range(n_dec)]   # Dec 4→Dec 1 aligned under Enc 4→Enc 1

# ── Input ─────────────────────────────────────────────────────────────────────
INP_X = 0.45
box(INP_X, ENC_Y, INPUT_W, BH, GRAY,
    "Input MR\n(B,1,D,H,W)", fontsize=8, bold=True)
arrow(INP_X + INPUT_W/2, ENC_Y, ENC_X[0] - BW/2, ENC_Y, color=NAVY, lw=2)

# ── Encoder stages ────────────────────────────────────────────────────────────
for i, (lbl, kern, stride, ch, col) in enumerate(ENC_STAGES):
    cx = ENC_X[i]
    box(cx, ENC_Y, BW, BH, col,
        f"{lbl}\nch={ch}", fontsize=8)
    # kernel/stride label above
    label(cx, ENC_Y + BH/2 + 0.22,
          f"k={kern}  s={stride}", fontsize=6.8, color="#1A237E")
    # arrow to next
    if i < n_enc - 1:
        arrow(cx + BW/2, ENC_Y, ENC_X[i+1] - BW/2, ENC_Y, color=NAVY, lw=1.8)

# ── Bottleneck label ─────────────────────────────────────────────────────────
label(ENC_X[-1], ENC_Y - BH/2 - 0.22, "bottleneck", fontsize=7, color=TEAL)

# ── Skip connections + decoder ────────────────────────────────────────────────
# Arrow from bottleneck down to dec4 (and across)
# Bottleneck -> Dec 4
arrow(ENC_X[-1], ENC_Y - BH/2, ENC_X[-1], DEC_Y + BH/2 + 0.12,
      color=TEAL, lw=1.8)
arrow(ENC_X[-1], DEC_Y + BH/2 + 0.12, DEC_X[-1] + BW/2 + 0.05, DEC_Y + BH/2 + 0.12,
      color=TEAL, lw=1.8)

for i, (lbl, ch, col) in enumerate(DEC_STAGES):
    cx = DEC_X[n_dec - 1 - i]   # Dec4 at enc_x[3], Dec3 at enc_x[2], ...
    box(cx, DEC_Y, BW, BH, col,
        f"{lbl}\nch={ch}", fontsize=8)
    # skip connection from encoder (dashed)
    enc_cx = ENC_X[n_dec - 1 - i]
    skip_y = DEC_Y + BH/2 + 0.05
    ax.annotate("", xy=(cx, DEC_Y + BH/2 + 0.05), xytext=(enc_cx, ENC_Y - BH/2 - 0.05),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4,
                                linestyle="dashed", connectionstyle="arc3,rad=0.0"))
    # arrow between decoder stages
    if i < n_dec - 1:
        next_cx = DEC_X[n_dec - 2 - i]
        arrow(cx - BW/2, DEC_Y, next_cx + BW/2, DEC_Y, color=GRAY, lw=1.8)

    # Auxiliary outputs
    if (n_dec - 1 - i - 1) in AUX_STAGES:   # Dec 3 and Dec 2 (i=1,2)
        pass

# Aux outputs from Dec 3 (i=1) and Dec 2 (i=2)
for aux_i, dec_i in enumerate([1, 2]):   # Dec3=index 1, Dec2=index 2
    cx = DEC_X[n_dec - 1 - dec_i]
    ax.annotate("", xy=(cx, AUX_Y + 0.3), xytext=(cx, DEC_Y - BH/2),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    box(cx, AUX_Y, 1.5, 0.55, RED, f"Aux Out {aux_i+1}\n(deep supervision)",
        fontsize=7.5)

# ── FiLM residual head ────────────────────────────────────────────────────────
FILM_X = DEC_X[0]   # after Dec 1
FILM_OFFSET_X = FILM_X - BW/2 - 1.5

box(FILM_OFFSET_X, DEC_Y, FILM_W, FILM_H, PURPLE,
    "FiLM Head\nConv→IN→LReLU→FiLM→Conv\nγ⊙h + β (anatomy-specific)", fontsize=7.5)
arrow(DEC_X[0] - BW/2, DEC_Y, FILM_OFFSET_X + FILM_W/2, DEC_Y, color=PURPLE, lw=2)

# ── tanh + output ─────────────────────────────────────────────────────────────
TANH_X = FILM_OFFSET_X - 1.4
box(TANH_X, DEC_Y, 1.2, BH, GREEN, "tanh\n+\nsCT out", fontsize=8.5, bold=True)
arrow(FILM_OFFSET_X - FILM_W/2, DEC_Y, TANH_X + 0.6, DEC_Y, color=GREEN, lw=2)

# ── Anatomy embedding input to FiLM ──────────────────────────────────────────
ANAT_X = FILM_OFFSET_X
ANAT_Y = DEC_Y - 1.55
box(ANAT_X, ANAT_Y, 1.7, 0.6, GOLD, "Anatomy Index a\n(HN=0 · TH=1 · AB=2)",
    fontsize=7.5, text_color="#333333")
arrow(ANAT_X, ANAT_Y + 0.3, ANAT_X, DEC_Y - FILM_H/2, color=GOLD, lw=1.5)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=NAVY,   label="Encoder (residual conv)"),
    mpatches.Patch(color=TEAL,   label="Bottleneck"),
    mpatches.Patch(color=GRAY,   label="Decoder (upsample + res)"),
    mpatches.Patch(color=ORANGE, label="Skip connections"),
    mpatches.Patch(color=RED,    label="Deep supervision outputs"),
    mpatches.Patch(color=PURPLE, label="FiLM residual head"),
    mpatches.Patch(color=GREEN,  label="Final tanh + sCT output"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=7.5,
          framealpha=0.9, ncol=1, edgecolor="#333333")

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(FIG_W / 2, FIG_H - 0.35,
        "DynUNETR3D Architecture — Anisotropic Kernels + FiLM Anatomy Conditioning + Deep Supervision",
        ha="center", va="center", fontsize=11, fontweight="bold", color=NAVY)

ax.text(FIG_W / 2, 0.3,
        "Anisotropic kernel [3,3,1] at Stage 1 avoids premature through-plane mixing (1:1:3 voxel spacing). "
        "FiLM head (γ,β) adapts output distribution per anatomy. "
        "Auxiliary outputs at Dec 3 & Dec 2 enforce multi-scale bone accuracy during training.",
        ha="center", va="center", fontsize=8, color=GRAY, style="italic",
        wrap=True)

plt.tight_layout(pad=0.3)
plt.savefig(OUT, dpi=180, bbox_inches="tight", facecolor=LGRAY)
print(f"✓  Saved → {OUT}")
