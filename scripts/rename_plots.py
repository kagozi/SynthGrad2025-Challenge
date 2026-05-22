#!/usr/bin/env python3
"""
Normalise all CSV filenames in plots/train/ and plots/val/ to the convention:
    {model}_{split}_{metric}_fold{N}.csv

Model names  : unet2d | unet2.5d | dyunet | swinunetr
Train metrics: mae | gdl | total
Val metrics  : mae | psnr | ms_ssim

Run with --dry-run first to preview, then without to apply.

Usage:
    python scripts/rename_plots.py --dry-run
    python scripts/rename_plots.py
"""

import argparse
import re
import shutil
from pathlib import Path


# ── Normalisation tables ───────────────────────────────────────────────────────

MODEL_MAP = {
    "dunet":      "dyunet",
    "dyunet":     "dyunet",
    "DyUnet":     "dyunet",
    "Dyunet":     "dyunet",
    "swinunetr":  "swinunetr",
    "swinUnetr":  "swinunetr",
    "unet2.5":    "unet2.5d",
    "unet2.5d":   "unet2.5d",
    "unet2d":     "unet2d",
    "unet":       "unet2d",
}

METRIC_MAP = {
    # train
    "ssim":    "gdl",       # user confirmed: train metric is gdl, not ssim
    "mae":     "mae",
    "total":   "total",
    "gdl":     "gdl",
    # val
    "ms_ssim": "ms_ssim",
    "mssim":   "ms_ssim",
    "psnr":    "psnr",
    "psrn":    "psnr",      # typo fix
}


# ── Parser for each filename variant ──────────────────────────────────────────

def parse_train(name: str):
    """
    Matches:
        {model}_train_{metric}_fold{N}
    Returns (model, metric, fold) or None.
    """
    m = re.fullmatch(
        r"([^_]+(?:\.[^_]+)?)_train_([^_]+)_fold(\d+)",
        name, re.IGNORECASE,
    )
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def parse_val(name: str):
    """
    Matches several variants found in the val directory:
        {model}_{metric}_fold{N}          (missing _val_)
        {model}_val_{metric}_fold{N}      (already correct)
        unet_fold{N}_{metric}             (wrong order)
    Returns (model, metric, fold) or None.
    """
    # Already correct: model_val_metric_foldN
    m = re.fullmatch(
        r"([^_]+(?:\.[^_]+)?)_val_([^_]+)_fold(\d+)",
        name, re.IGNORECASE,
    )
    if m:
        return m.group(1), m.group(2), m.group(3)

    # Missing _val_: model_metric_foldN  (metric may contain underscore like ms_ssim)
    m = re.fullmatch(
        r"([^_]+(?:\.[^_]+)?)_(mae|psnr|psrn|ms_ssim|mssim)_fold(\d+)",
        name, re.IGNORECASE,
    )
    if m:
        return m.group(1), m.group(2), m.group(3)

    # Wrong order: model_foldN_metric
    m = re.fullmatch(
        r"([^_]+(?:\.[^_]+)?)_fold(\d+)_(mae|psnr|psrn|ms_ssim|mssim)",
        name, re.IGNORECASE,
    )
    if m:
        return m.group(1), m.group(3), m.group(2)

    return None


def normalise(model: str, metric: str) -> tuple[str, str] | None:
    model_norm  = MODEL_MAP.get(model)
    metric_norm = METRIC_MAP.get(metric.lower())
    if model_norm is None:
        print(f"  [WARN] Unknown model '{model}' — skipping")
        return None
    if metric_norm is None:
        print(f"  [WARN] Unknown metric '{metric}' — skipping")
        return None
    return model_norm, metric_norm


# ── Main ──────────────────────────────────────────────────────────────────────

def process_dir(directory: Path, split: str, dry_run: bool):
    renames = []

    for f in sorted(directory.glob("*.csv")):
        stem = f.stem   # filename without .csv

        parsed = parse_train(stem) if split == "train" else parse_val(stem)
        if parsed is None:
            print(f"  [SKIP] Could not parse: {f.name}")
            continue

        model_raw, metric_raw, fold = parsed
        result = normalise(model_raw, metric_raw)
        if result is None:
            continue

        model, metric = result
        new_name = f"{model}_{split}_{metric}_fold{fold}.csv"
        new_path  = directory / new_name

        if new_name == f.name:
            continue   # already correct, no rename needed

        renames.append((f, new_path))
        status = "[DRY RUN]" if dry_run else "[RENAME]"
        print(f"  {status}  {f.name}  →  {new_name}")

    if not dry_run:
        for src, dst in renames:
            if dst.exists() and dst != src:
                print(f"  [WARN] Target exists, skipping: {dst.name}")
                continue
            src.rename(dst)

    return len(renames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plots_dir", default="plots",
        help="Root plots directory (default: plots/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview renames without applying them",
    )
    args = parser.parse_args()

    plots_dir = Path(args.plots_dir)
    dry_run   = args.dry_run

    print(f"{'DRY RUN — ' if dry_run else ''}Normalising plot CSV filenames\n")

    for split in ("train", "val"):
        d = plots_dir / split
        if not d.exists():
            print(f"[SKIP] Directory not found: {d}")
            continue
        print(f"── {split}/ ──────────────────────────────")
        n = process_dir(d, split, dry_run)
        print(f"   {n} file(s) {'would be' if dry_run else ''} renamed\n")

    if dry_run:
        print("Run without --dry-run to apply.")


if __name__ == "__main__":
    main()
