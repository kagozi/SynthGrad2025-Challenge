#!/usr/bin/env python3
"""Validation metric plots from 5-fold CSVs."""

import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

# ── Load all folds ────────────────────────────────────────────────
dfs = []
for f in sorted(glob.glob("results/fold*_val_metrics.csv")):
    df = pd.read_csv(f)
    df["fold"] = int(f.split("fold")[1].split("_")[0])
    dfs.append(df)
data = pd.concat(dfs, ignore_index=True)

# Anatomy label map
ANA_LABEL = {"HN": "Head & Neck", "TH": "Thorax", "AB": "Abdomen"}
ANA_COLOR = {"HN": "#3A86FF", "TH": "#FF6B6B", "AB": "#06D6A0"}
ANA_ORDER = ["HN", "TH", "AB"]

DARK   = "#0D1B2A"
CARD   = "#0F2540"
ACCENT = "#53A8E2"
WHITE  = "#FFFFFF"
GRAY   = "#AABBCC"

def style_ax(ax, ylabel="", title="", ylim=None):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=WHITE, labelsize=9)
    ax.set_ylabel(ylabel, color=GRAY, fontsize=10)
    ax.set_title(title, color=ACCENT, fontsize=11, fontweight="bold", pad=6)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color("#1E3A5F")
    if ylim:
        ax.set_ylim(*ylim)


# ════════════════════════════════════════════════════════════════
# FIGURE 1 — Box plots by anatomy  (5 metrics)
# ════════════════════════════════════════════════════════════════
metrics_box = [
    ("mae",     "MAE (HU)",     True,  None),
    ("psnr",    "PSNR (dB)",    False, None),
    ("ms_ssim", "MS-SSIM",      False, (0.0, 1.0)),
    ("mDice",   "Mean Dice",    False, (0.0, 1.0)),
    ("HD95",    "HD95 (mm)",    True,  None),
]

fig1, axes1 = plt.subplots(1, 5, figsize=(18, 6))
fig1.patch.set_facecolor(DARK)
fig1.suptitle("Validation Metrics by Anatomy  (5-fold, n=513 cases)",
              color=WHITE, fontsize=14, fontweight="bold", y=1.01)

for ax, (col, ylabel, lower_better, ylim) in zip(axes1, metrics_box):
    groups = [data.loc[data.anatomy == a, col].dropna().values for a in ANA_ORDER]
    bp = ax.boxplot(groups,
                    patch_artist=True,
                    widths=0.5,
                    medianprops=dict(color="#FFD700", linewidth=2),
                    whiskerprops=dict(color=GRAY, linewidth=1.2),
                    capprops=dict(color=GRAY, linewidth=1.2),
                    flierprops=dict(marker="o", markerfacecolor=GRAY,
                                   markersize=3, alpha=0.4, linestyle="none"))
    for patch, ana in zip(bp["boxes"], ANA_ORDER):
        patch.set_facecolor(ANA_COLOR[ana])
        patch.set_alpha(0.85)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([ANA_LABEL[a] for a in ANA_ORDER],
                       color=WHITE, fontsize=8.5, rotation=12)
    arrow = " ↓ better" if lower_better else " ↑ better"
    style_ax(ax, ylabel=ylabel + arrow, ylim=ylim)

    # median annotation
    for j, (grp, ana) in enumerate(zip(groups, ANA_ORDER)):
        med = np.median(grp)
        ax.text(j+1, med, f" {med:.1f}", va="center",
                fontsize=8, color="#FFD700", fontweight="bold")

plt.tight_layout()
out1 = "results/fig1_boxplots_by_anatomy.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig1)
print(f"Saved {out1}")


# ════════════════════════════════════════════════════════════════
# FIGURE 2 — Per-fold bar chart  (mean ± std of each metric)
# ════════════════════════════════════════════════════════════════
fold_stats = data.groupby("fold")[["mae","psnr","ms_ssim","mDice","HD95"]].agg(["mean","std"])

metrics_bar = [
    ("mae",     "MAE (HU)",   True),
    ("psnr",    "PSNR (dB)",  False),
    ("ms_ssim", "MS-SSIM",    False),
    ("mDice",   "Mean Dice",  False),
    ("HD95",    "HD95 (mm)",  True),
]
fold_colors = ["#3A86FF","#FF6B6B","#06D6A0","#FFD700","#FF9F1C"]
folds = sorted(data["fold"].unique())
x = np.arange(len(folds))
n_metrics = len(metrics_bar)

fig2, axes2 = plt.subplots(1, n_metrics, figsize=(18, 5.5))
fig2.patch.set_facecolor(DARK)
fig2.suptitle("Mean ± Std per Fold  (all anatomies combined)",
              color=WHITE, fontsize=14, fontweight="bold", y=1.02)

for ax, (col, ylabel, lower_better) in zip(axes2, metrics_bar):
    means = fold_stats[(col, "mean")].values
    stds  = fold_stats[(col, "std")].values
    bars  = ax.bar(x, means, color=fold_colors, edgecolor=ACCENT,
                   width=0.6, yerr=stds, capsize=4,
                   error_kw=dict(ecolor=GRAY, lw=1.2))
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + stds[list(means).index(m)] + 0.005*max(means),
                f"{m:.2f}", ha="center", va="bottom", color=WHITE, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds], color=WHITE, fontsize=9)
    arrow = " ↓" if lower_better else " ↑"
    style_ax(ax, ylabel=ylabel + arrow)
    # overall mean line
    overall = means.mean()
    ax.axhline(overall, color="#FFD700", lw=1.3, ls="--", alpha=0.8)
    ax.text(len(folds)-0.45, overall, f"avg={overall:.2f}",
            color="#FFD700", fontsize=8, va="bottom")

plt.tight_layout()
out2 = "results/fig2_per_fold_bars.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig2)
print(f"Saved {out2}")


# ════════════════════════════════════════════════════════════════
# FIGURE 3 — Scatter: PSNR vs MAE, coloured by anatomy
# ════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(9, 6.5))
fig3.patch.set_facecolor(DARK)
ax3.set_facecolor(CARD)

for ana in ANA_ORDER:
    sub = data[data.anatomy == ana]
    ax3.scatter(sub["mae"], sub["psnr"],
                c=ANA_COLOR[ana], label=ANA_LABEL[ana],
                alpha=0.65, s=30, edgecolors="none")

# trend line (all data)
m, b = np.polyfit(data["mae"], data["psnr"], 1)
xfit = np.linspace(data["mae"].min(), data["mae"].max(), 200)
ax3.plot(xfit, m*xfit + b, color="#FFD700", lw=1.5, ls="--",
         label=f"Trend  (slope={m:.3f})")

ax3.set_xlabel("MAE (HU)  ↓ lower = better", color=GRAY, fontsize=11)
ax3.set_ylabel("PSNR (dB)  ↑ higher = better", color=GRAY, fontsize=11)
style_ax(ax3, title="PSNR vs MAE by Anatomy  (5-fold validation, n=513)")

legend = ax3.legend(facecolor=DARK, edgecolor=ACCENT,
                    labelcolor=WHITE, fontsize=10, markerscale=1.5)

# per-anatomy medians annotated
for ana in ANA_ORDER:
    sub = data[data.anatomy == ana]
    mx, my = sub["mae"].median(), sub["psnr"].median()
    ax3.scatter(mx, my, c=ANA_COLOR[ana], s=130, marker="D",
                edgecolors=WHITE, linewidths=1.2, zorder=5)
    ax3.annotate(f"{ANA_LABEL[ana]}\nmed MAE={mx:.0f}\nmed PSNR={my:.1f}",
                 xy=(mx, my), xytext=(mx+4, my+0.2),
                 fontsize=8, color=ANA_COLOR[ana],
                 arrowprops=dict(arrowstyle="-", color=ANA_COLOR[ana],
                                 lw=0.8, alpha=0.6))

plt.tight_layout()
out3 = "results/fig3_scatter_psnr_mae.png"
fig3.savefig(out3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig3)
print(f"Saved {out3}")

print("\nAll done.")
