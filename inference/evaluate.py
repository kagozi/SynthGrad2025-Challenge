#!/usr/bin/env python3
"""
SynthRAD2025 Task 1 — Local evaluation script

Computes MAE, PSNR, MS-SSIM, approximate mDice and HD95
for a folder of predicted sCTs against ground-truth CTs.

Usage:
    python inference/evaluate.py \
        --pred_dir  predictions/fold0_val \
        --gt_dir    data/raw \
        --output    results/fold0_val_metrics.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import evaluate_case, print_metrics


def load_hu(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def align_shapes(pred: np.ndarray, gt: np.ndarray) -> tuple:
    """Trim to minimum shared shape along each axis."""
    D = min(pred.shape[0], gt.shape[0])
    H = min(pred.shape[1], gt.shape[1])
    W = min(pred.shape[2], gt.shape[2])
    return pred[:D, :H, :W], gt[:D, :H, :W]


def main(args):
    pred_dir = Path(args.pred_dir)
    gt_dir   = Path(args.gt_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Find predicted cases
    pred_cases = sorted([d for d in pred_dir.iterdir()
                         if d.is_dir() and (d / "ct.mha").exists()])

    print(f"[Eval] Cases found: {len(pred_cases)}")
    if not pred_cases:
        print("[Eval] No predicted cases found. Expected: pred_dir/<case_id>/ct.mha")
        sys.exit(1)

    records = []
    anatomy_groups = {"HN": [], "TH": [], "AB": []}

    for pred_case_dir in tqdm(pred_cases, desc="Evaluating"):
        case_id = pred_case_dir.name
        anatomy = case_id[1:3].upper()

        # Find GT: try anatomy-grouped and flat layouts
        gt_case_dir = None
        for pattern in [gt_dir / anatomy.replace("HN","1HN").replace("TH","1TH").replace("AB","1AB") / case_id,
                         gt_dir / case_id]:
            if pattern.exists():
                gt_case_dir = pattern
                break

        if gt_case_dir is None or not (gt_case_dir / "ct.mha").exists():
            print(f"  [WARN] GT not found for {case_id}")
            continue

        pred_hu = load_hu(pred_case_dir / "ct.mha")
        gt_hu   = load_hu(gt_case_dir   / "ct.mha")
        pred_hu, gt_hu = align_shapes(pred_hu, gt_hu)

        mask_path = gt_case_dir / "mask.mha"
        mask = load_hu(mask_path).astype(bool) if mask_path.exists() else None
        if mask is not None:
            mask = mask[:pred_hu.shape[0], :pred_hu.shape[1], :pred_hu.shape[2]]

        metrics = evaluate_case(pred_hu, gt_hu, mask, compute_seg=args.compute_seg)
        metrics["case_id"] = case_id
        metrics["anatomy"] = anatomy
        records.append(metrics)
        anatomy_groups.get(anatomy, []).append(metrics)

        if args.verbose:
            print_metrics(metrics, prefix=f"  {case_id}")

    if not records:
        print("[Eval] No results computed.")
        sys.exit(1)

    df = pd.DataFrame(records)

    # Column order
    first_cols = ["case_id", "anatomy", "mae", "psnr", "ms_ssim"]
    seg_cols   = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + seg_cols]

    df.to_csv(out_path, index=False)
    print(f"\n[Eval] Results saved: {out_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" Per-anatomy summary")
    print("=" * 60)

    metric_cols = [c for c in ["mae", "psnr", "ms_ssim", "mDice", "HD95"] if c in df.columns]
    summary = df.groupby("anatomy")[metric_cols].agg(["mean", "std"])
    print(summary.to_string())

    print("\n" + "=" * 60)
    print(" Overall")
    print("=" * 60)
    overall = df[metric_cols].agg(["mean", "std"])
    print(overall.to_string())

    # Quick interpretation
    mae = df["mae"].mean()
    psnr = df["psnr"].mean()
    ms = df["ms_ssim"].mean()
    print(f"\n  Mean MAE:     {mae:.2f} HU")
    print(f"  Mean PSNR:    {psnr:.2f} dB")
    print(f"  Mean MS-SSIM: {ms:.4f}")
    if "mDice" in df.columns:
        print(f"  Mean mDice:   {df['mDice'].mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir",    type=str, required=True,
                        help="Dir with <case_id>/ct.mha predictions")
    parser.add_argument("--gt_dir",      type=str, required=True,
                        help="Dir with ground-truth cases (data/raw layout)")
    parser.add_argument("--output",      type=str, default="results/metrics.csv")
    parser.add_argument("--compute_seg", action="store_true",
                        help="Compute approximate Dice/HD95 (slow)")
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()
    main(args)
