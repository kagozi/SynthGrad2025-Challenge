#!/usr/bin/env python3
"""
Generate two publication-ready figures for the SynthRAD2025 paper:

  paper_latest/figures/training_curves_train.png   (1×3: Total / MAE / GDL loss)
  paper_latest/figures/training_curves_val.png     (1×3: Val MAE / PSNR / MS-SSIM)

Rules:
  - X-axis is in epochs (steps are rescaled per-model using known epoch counts).
  - All series are clipped to the model with fewest epochs (100 epochs for 2D
    baselines) so every model is visible over the same epoch range.
  - Fold-averaged mean ± 1 std shown as solid line + shaded band.

Usage:
    python scripts/generate_paper_figure.py
    python scripts/generate_paper_figure.py --plots_dir plots \
        --out_dir paper_latest/figures
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── Per-model epoch/step calibration ──────────────────────────────────────────
# Known training lengths and empirically observed last WandB step.
# epoch = step * (EPOCHS / MAX_STEP)
MODEL_EPOCHS = {
    "unet2d":    100,
    "unet2.5d":  100,
    "swinunetr": 150,
    "dyunet":    150,
}
MODEL_MAX_STEP = {
    "unet2d":    137,
    "unet2.5d":  137,
    "swinunetr": 207,
    "dyunet":    207,
}

CLIP_EPOCHS = min(MODEL_EPOCHS.values())   # 100

# ── Appearance ─────────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "unet2d":    "#2871CC",
    "unet2.5d":  "#E07B00",
    "swinunetr": "#CC3333",
    "dyunet":    "#2BA04E",
}
MODEL_LABELS = {
    "unet2d":    "UNet2D",
    "unet2.5d":  "UNet2.5D",
    "swinunetr": "Swin-UNETR",
    "dyunet":    "DynUNet",
}
MODEL_ORDER = ["unet2d", "unet2.5d", "swinunetr", "dyunet"]

TRAIN_PANELS = [
    ("total",   "Total Loss",    True),
    ("mae",     "MAE Loss",      True),
    ("gdl",     "GDL Loss",      True),
]
VAL_PANELS = [
    ("mae",     "Val MAE (HU)",  True),
    ("psnr",    "Val PSNR (dB)", False),
    ("ms_ssim", "Val MS-SSIM",   False),
]

# ── Filename parsing ────────────────────────────────────────────────────────────
FNAME_RE = re.compile(
    r"^(?P<model>[^_]+(?:\.[^_]+)?)_(?P<split>train|val)_(?P<metric>.+)_fold(?P<fold>\d+)$"
)

def parse_filename(stem):
    m = FNAME_RE.fullmatch(stem)
    if not m:
        return None
    return m.group("model"), m.group("split"), m.group("metric"), int(m.group("fold"))

def load_csv(path):
    df = pd.read_csv(path)
    steps  = df.iloc[:, 0].values.astype(float)
    values = df.iloc[:, 1].values.astype(float)
    mask   = ~np.isnan(values) & ~np.isnan(steps)
    return steps[mask], values[mask]

def steps_to_epochs(steps, model):
    """Convert raw WandB steps to epoch numbers for a given model."""
    ratio = MODEL_EPOCHS.get(model, 100) / MODEL_MAX_STEP.get(model, 137)
    return steps * ratio

# ── Data loading ────────────────────────────────────────────────────────────────
def scan(plots_dir):
    """Returns data[split][metric][model][fold] = (epochs, values)."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for split in ("train", "val"):
        d = plots_dir / split
        if not d.exists():
            continue
        for f in sorted(d.glob("*.csv")):
            parsed = parse_filename(f.stem)
            if parsed is None:
                continue
            model, split_, metric, fold = parsed
            try:
                steps, values = load_csv(f)
                if len(steps) < 2:
                    continue
                epochs = steps_to_epochs(steps, model)
                # Clip to CLIP_EPOCHS
                mask   = epochs <= CLIP_EPOCHS
                if mask.sum() < 2:
                    continue
                data[split_][metric][model][fold] = (epochs[mask], values[mask])
            except Exception as e:
                print(f"[WARN] {f.name}: {e}")

    return data

# ── Fold averaging ──────────────────────────────────────────────────────────────
def fold_mean_std(fold_dict, n_grid=300):
    """Interpolate each fold to a common epoch grid, return (grid, mean, std)."""
    all_epochs, all_values = [], []
    for epochs, values in fold_dict.values():
        all_epochs.append(epochs)
        all_values.append(values)

    if not all_epochs:
        return None, None, None

    g_min = max(e[0]  for e in all_epochs)
    g_max = min(e[-1] for e in all_epochs)
    if g_max <= g_min:
        g_min = min(e[0]  for e in all_epochs)
        g_max = max(e[-1] for e in all_epochs)

    grid = np.linspace(g_min, g_max, n_grid)
    interp = np.array([np.interp(grid, ep, v) for ep, v in zip(all_epochs, all_values)])
    return grid, interp.mean(axis=0), interp.std(axis=0)

def smooth(values, weight=0.5):
    """Exponential moving average."""
    out, last = [], values[0]
    for v in values:
        last = weight * last + (1 - weight) * v
        out.append(last)
    return np.array(out)

# ── Single-figure builder ───────────────────────────────────────────────────────
def build_figure(data, split, panels, out_path):
    n_cols = len(panels)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 4.5),
                             constrained_layout=True)
    if n_cols == 1:
        axes = [axes]

    legend_handles, legend_labels = [], []

    for col, (metric, ylabel, lower_better) in enumerate(panels):
        ax = axes[col]
        ax.set_facecolor("#F8F8F8")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#BBBBBB")
        ax.spines["bottom"].set_color("#BBBBBB")
        ax.tick_params(colors="#444444", labelsize=9)
        ax.grid(color="#DDDDDD", linestyle="--", linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)

        metric_data = data.get(split, {}).get(metric, {})
        has_data = False

        for model in MODEL_ORDER:
            fold_dict = metric_data.get(model)
            if not fold_dict:
                continue

            grid, mean, std = fold_mean_std(fold_dict)
            if grid is None:
                continue

            has_data   = True
            color      = MODEL_COLORS.get(model, "#888888")
            label      = MODEL_LABELS.get(model, model)
            mean_s     = smooth(mean)
            std_s_lo   = smooth(mean - std)
            std_s_hi   = smooth(mean + std)

            band = ax.fill_between(grid, std_s_lo, std_s_hi,
                                   color=color, alpha=0.13)
            line, = ax.plot(grid, mean_s,
                            color=color, linewidth=2.0,
                            label=label, solid_capstyle="round")

            if col == 0:
                legend_handles.append(line)
                legend_labels.append(label)

        arrow = " $\\downarrow$" if lower_better else " $\\uparrow$"
        ax.set_title(ylabel + arrow, fontsize=11, fontweight="bold",
                     pad=6, color="#111111")
        ax.set_xlabel("Epoch", fontsize=10, color="#444444")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))

        if col == 0:
            ax.set_ylabel(ylabel, fontsize=9, color="#555555")

        if not has_data:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="#AAAAAA", fontsize=11)

        ax.set_xlim(left=0, right=CLIP_EPOCHS)

    # Shared legend below the figure
    fig.legend(
        legend_handles, legend_labels,
        loc="lower center",
        ncol=len(MODEL_ORDER),
        fontsize=9.5,
        framealpha=0.95,
        edgecolor="#CCCCCC",
        bbox_to_anchor=(0.5, -0.12),
    )

    split_title = "Training Loss" if split == "train" else "Validation Metrics"
    fig.suptitle(
        f"{split_title} — 5-Fold Mean $\\pm$ Std  (clipped to {CLIP_EPOCHS} epochs)",
        fontsize=12, fontweight="bold", color="#111111",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_path}")

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots_dir", default="plots")
    parser.add_argument("--out_dir",   default="paper_latest/figures")
    args = parser.parse_args()

    plots_dir = Path(args.plots_dir)
    out_dir   = Path(args.out_dir)

    print(f"Scanning {plots_dir} …")
    data = scan(plots_dir)

    build_figure(data, "train", TRAIN_PANELS,
                 out_dir / "training_curves_train.png")
    build_figure(data, "val",   VAL_PANELS,
                 out_dir / "training_curves_val.png")

    print("Done.")

if __name__ == "__main__":
    main()
