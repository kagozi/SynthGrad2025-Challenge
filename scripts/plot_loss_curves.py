#!/usr/bin/env python3
"""
Plot training and validation curves from WandB-exported CSVs.

Convention expected (after rename_plots.py):
    plots/train/{model}_train_{metric}_fold{N}.csv
    plots/val/{model}_val_{metric}_fold{N}.csv

Output: one figure per fold for each split, saved to plots/figures/
    plots/figures/train_fold{N}.png
    plots/figures/val_fold{N}.png

Each figure has one subplot per metric, with all available models overlaid.

Usage:
    python scripts/plot_loss_curves.py
    python scripts/plot_loss_curves.py --plots_dir plots --out_dir plots/figures
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_COLORS = {
    "unet2d":    "#4C9BE8",
    "unet2.5d":  "#F5A623",
    "dyunet":    "#50C878",
    "swinunetr": "#E85D5D",
}

MODEL_LABELS = {
    "unet2d":    "UNet2D",
    "unet2.5d":  "UNet2.5D",
    "dyunet":    "DynUNet",
    "swinunetr": "Swin-UNETR",
}

METRIC_LABELS = {
    # train
    "mae":   "MAE Loss",
    "gdl":   "GDL Loss",
    "total": "Total Loss",
    # val
    "psnr":    "PSNR (dB)",
    "ms_ssim": "MS-SSIM",
}

# Display order for metrics within a figure
TRAIN_METRIC_ORDER = ["total", "mae", "gdl"]
VAL_METRIC_ORDER   = ["mae", "psnr", "ms_ssim"]


# ── Helpers ────────────────────────────────────────────────────────────────────

FNAME_RE = re.compile(
    r"^(?P<model>[^_]+(?:\.[^_]+)?)_(?P<split>train|val)_(?P<metric>.+)_fold(?P<fold>\d+)$"
)


def parse_filename(stem: str):
    m = FNAME_RE.fullmatch(stem)
    if not m:
        return None
    return m.group("model"), m.group("split"), m.group("metric"), int(m.group("fold"))


def load_csv(path: Path):
    """Return (steps, values) arrays from a WandB-exported CSV."""
    df = pd.read_csv(path)
    steps  = df.iloc[:, 0].values.astype(float)
    values = df.iloc[:, 1].values.astype(float)
    # Drop NaN rows
    mask   = ~np.isnan(values)
    return steps[mask], values[mask]


def smooth(values: np.ndarray, weight: float = 0.85) -> np.ndarray:
    """Exponential moving average — same as TensorBoard's smoothing slider."""
    smoothed, last = [], values[0]
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return np.array(smoothed)


# ── Scan ───────────────────────────────────────────────────────────────────────

def scan(plots_dir: Path) -> dict:
    """
    Returns nested dict:
        data[split][fold][metric][model] = (steps, values)
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for split in ("train", "val"):
        d = plots_dir / split
        if not d.exists():
            continue
        for f in sorted(d.glob("*.csv")):
            parsed = parse_filename(f.stem)
            if parsed is None:
                print(f"[SKIP] {f.name}")
                continue
            model, split_, metric, fold = parsed
            try:
                steps, values = load_csv(f)
                data[split_][fold][metric][model] = (steps, values)
            except Exception as e:
                print(f"[WARN] {f.name}: {e}")

    return data


# ── Plot ───────────────────────────────────────────────────────────────────────

def clip_fold_to_min(fold_data: dict) -> dict:
    """Truncate all series in a fold to the shortest model's last step."""
    # Find the minimum last step across every model in every metric
    last_steps = []
    for metric_models in fold_data.values():
        for steps, _ in metric_models.values():
            if len(steps):
                last_steps.append(steps[-1])
    if not last_steps:
        return fold_data
    min_step = min(last_steps)

    clipped = {}
    for metric, metric_models in fold_data.items():
        clipped[metric] = {}
        for model, (steps, values) in metric_models.items():
            mask = steps <= min_step
            clipped[metric][model] = (steps[mask], values[mask])
    return clipped


def plot_fold(
    fold:        int,
    split:       str,
    fold_data:   dict,   # metric → model → (steps, values)
    metric_order: list,
    out_dir:     Path,
    smooth_w:    float = 0.85,
):
    available_metrics = [m for m in metric_order if m in fold_data]
    if not available_metrics:
        return

    n_cols = len(available_metrics)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    fig.patch.set_facecolor("#0F0F0F")

    if n_cols == 1:
        axes = [axes]

    for ax, metric in zip(axes, available_metrics):
        ax.set_facecolor("#1A1A1A")
        ax.tick_params(colors="white", labelsize=9)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.grid(color="#333333", linestyle="--", linewidth=0.5, alpha=0.7)

        model_data = fold_data[metric]
        # Sort models for consistent legend order
        for model in sorted(model_data.keys()):
            steps, values = model_data[model]
            color = MODEL_COLORS.get(model, "#AAAAAA")
            label = MODEL_LABELS.get(model, model)

            # Raw (faint)
            ax.plot(steps, values, color=color, alpha=0.2, linewidth=0.8)
            # Smoothed
            if len(values) > 3:
                ax.plot(steps, smooth(values, smooth_w),
                        color=color, linewidth=2.0, label=label)
            else:
                ax.plot(steps, values, color=color, linewidth=2.0, label=label)

        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Step", fontsize=9)
        ax.legend(
            fontsize=8,
            facecolor="#1A1A1A",
            edgecolor="#444444",
            labelcolor="white",
            loc="best",
        )

    split_label = "Training" if split == "train" else "Validation"
    fig.suptitle(
        f"{split_label} Curves — Fold {fold}",
        fontsize=13, color="white", fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    out_path = out_dir / f"{split}_fold{fold}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots_dir", default="plots",    help="Root CSV directory")
    parser.add_argument("--out_dir",   default="plots/figures", help="Output directory")
    parser.add_argument("--smooth",    type=float, default=0.85,
                        help="EMA smoothing weight (0=none, 0.99=max)")
    args = parser.parse_args()

    plots_dir = Path(args.plots_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = scan(plots_dir)

    for split, metric_order in [("train", TRAIN_METRIC_ORDER), ("val", VAL_METRIC_ORDER)]:
        if split not in data:
            print(f"[SKIP] No data found for split: {split}")
            continue
        print(f"\n── {split} ──────────────────────────")
        for fold in sorted(data[split].keys()):
            fold_data = clip_fold_to_min(data[split][fold])
            plot_fold(
                fold        = fold,
                split       = split,
                fold_data   = fold_data,
                metric_order= metric_order,
                out_dir     = out_dir,
                smooth_w    = args.smooth,
            )

    print(f"\n[DONE] Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
