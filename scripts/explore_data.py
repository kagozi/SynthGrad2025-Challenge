#!/usr/bin/env python3
"""
SynthRAD2025 Task 1 - Data Exploration Script

Scans data/raw/ and generates:
  - Per-anatomy/center statistics (spacing, shape, HU range)
  - Intensity distribution plots
  - Summary CSV saved to data/splits/dataset_info.csv

Usage:
    python scripts/explore_data.py --data_dir data/raw
"""

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


# ── Parsing ────────────────────────────────────────────────────────────────────

def parse_case_id(case_id: str) -> dict:
    """
    Parses naming convention: [Task][Anatomy][CenterID][PatientID]
    e.g. '1HNA001' → task=1, anatomy=HN, center=A, patient=001
    """
    m = re.match(r"(\d)(HN|TH|AB)([A-E])(\d+)", case_id, re.IGNORECASE)
    if not m:
        return {}
    return {
        "task":    int(m.group(1)),
        "anatomy": m.group(2).upper(),
        "center":  m.group(3).upper(),
        "patient": m.group(4),
    }


def load_volume_meta(path: Path) -> dict:
    """Reads .mha header only (no pixel data) for quick stats."""
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    return {
        "size":    reader.GetSize(),        # (W, H, D)
        "spacing": reader.GetSpacing(),     # mm (x, y, z)
        "origin":  reader.GetOrigin(),
    }


def load_volume_stats(path: Path, mask_path: Path = None) -> dict:
    """Full pixel load for intensity statistics."""
    img  = sitk.ReadImage(str(path))
    arr  = sitk.GetArrayFromImage(img).astype(np.float32)  # (D, H, W)

    if mask_path and mask_path.exists():
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))).astype(bool)
        vals = arr[mask]
    else:
        vals = arr.ravel()

    return {
        "min":  float(vals.min()),
        "max":  float(vals.max()),
        "mean": float(vals.mean()),
        "std":  float(vals.std()),
        "p1":   float(np.percentile(vals, 1)),
        "p99":  float(np.percentile(vals, 99)),
    }


# ── Main scan ──────────────────────────────────────────────────────────────────

def scan_dataset(data_dir: Path) -> pd.DataFrame:
    records = []

    # Support both flat and anatomy-grouped layouts
    case_dirs = []
    for pattern in ["*/*", "*"]:
        candidates = list(data_dir.glob(pattern))
        case_dirs = [d for d in candidates if d.is_dir() and re.match(r"\d(HN|TH|AB)[A-E]\d+", d.name, re.I)]
        if case_dirs:
            break

    if not case_dirs:
        print(f"[ERROR] No cases found in {data_dir}")
        print("  Expected pattern: 1HNA001, 1THA002, 1ABA003 ...")
        sys.exit(1)

    print(f"[INFO] Found {len(case_dirs)} cases")

    for case_dir in tqdm(sorted(case_dirs), desc="Scanning"):
        case_id = case_dir.name
        meta    = parse_case_id(case_id)
        if not meta:
            continue

        mr_path   = case_dir / "mr.mha"
        ct_path   = case_dir / "ct.mha"
        mask_path = case_dir / "mask.mha"

        if not mr_path.exists() or not ct_path.exists():
            print(f"  [WARN] Missing files in {case_id}")
            continue

        try:
            mr_meta = load_volume_meta(mr_path)
            ct_meta = load_volume_meta(ct_path)

            row = {
                "case_id":   case_id,
                "anatomy":   meta["anatomy"],
                "center":    meta["center"],
                "patient":   meta["patient"],
                # MR
                "mr_size_x": mr_meta["size"][0],
                "mr_size_y": mr_meta["size"][1],
                "mr_size_z": mr_meta["size"][2],
                "mr_sp_x":   round(mr_meta["spacing"][0], 3),
                "mr_sp_y":   round(mr_meta["spacing"][1], 3),
                "mr_sp_z":   round(mr_meta["spacing"][2], 3),
                # CT
                "ct_size_x": ct_meta["size"][0],
                "ct_size_y": ct_meta["size"][1],
                "ct_size_z": ct_meta["size"][2],
                "has_mask":  mask_path.exists(),
            }

            # Intensity stats (slower — skip with --fast)
            if not args.fast:
                mr_stats = load_volume_stats(mr_path, mask_path)
                ct_stats = load_volume_stats(ct_path, mask_path)
                for k, v in mr_stats.items():
                    row[f"mr_{k}"] = v
                for k, v in ct_stats.items():
                    row[f"ct_{k}"] = v

            records.append(row)

        except Exception as e:
            print(f"  [ERROR] {case_id}: {e}")

    return pd.DataFrame(records)


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_distributions(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Case count per anatomy × center
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = df.groupby(["anatomy", "center"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", ax=ax, colormap="Set2")
    ax.set_title("Case Count per Anatomy × Center")
    ax.set_xlabel("Anatomy")
    ax.set_ylabel("Count")
    ax.legend(title="Center")
    plt.tight_layout()
    fig.savefig(out_dir / "case_counts.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir}/case_counts.png")

    # 2. Volume sizes
    for axis, label in [("z", "Num Slices (z)"), ("x", "Width (x)"), ("y", "Height (y)")]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
        for i, anat in enumerate(["HN", "TH", "AB"]):
            sub = df[df["anatomy"] == anat]
            if sub.empty:
                continue
            axes[i].hist(sub[f"mr_size_{axis}"], bins=20, alpha=0.7, label="MR", color="steelblue")
            axes[i].hist(sub[f"ct_size_{axis}"], bins=20, alpha=0.7, label="CT", color="tomato")
            axes[i].set_title(f"{anat} — {label}")
            axes[i].legend()
        plt.suptitle(f"Volume {label} Distribution")
        plt.tight_layout()
        fig.savefig(out_dir / f"vol_size_{axis}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_dir}/vol_size_{axis}.png")

    # 3. Intensity distributions (if available)
    if "mr_mean" in df.columns:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for col, anat in enumerate(["HN", "TH", "AB"]):
            sub = df[df["anatomy"] == anat]
            if sub.empty:
                continue
            # MR
            axes[0, col].hist(sub["mr_mean"], bins=30, color="steelblue", alpha=0.8)
            axes[0, col].set_title(f"{anat} MR — Mean Intensity")
            # CT
            axes[1, col].hist(sub["ct_mean"], bins=30, color="tomato", alpha=0.8)
            axes[1, col].set_title(f"{anat} CT — Mean HU")
        plt.suptitle("Intensity Statistics per Anatomy")
        plt.tight_layout()
        fig.savefig(out_dir / "intensities.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_dir}/intensities.png")

    # 4. Spacing distribution
    if "mr_sp_z" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        for anat, grp in df.groupby("anatomy"):
            ax.hist(grp["mr_sp_z"], bins=20, alpha=0.6, label=anat)
        ax.set_title("MR Slice Thickness (z-spacing)")
        ax.set_xlabel("mm")
        ax.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "mr_spacing_z.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_dir}/mr_spacing_z.png")


# ── Entry ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw",
                        help="Root directory of downloaded training data")
    parser.add_argument("--out_dir",  type=str, default="data/splits",
                        help="Where to save CSV and plots")
    parser.add_argument("--fast",     action="store_true",
                        help="Skip intensity stats (header-only scan)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" SynthRAD2025 Task 1 — Data Exploration")
    print("=" * 60)

    df = scan_dataset(data_dir)

    csv_path = out_dir / "dataset_info.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[INFO] Saved dataset info: {csv_path} ({len(df)} cases)")

    print("\n── Per-anatomy summary ──────────────────────────────────")
    print(df.groupby("anatomy").agg(
        cases=("case_id", "count"),
        centers=("center", "nunique"),
        **({
            "mr_mean_mean": ("mr_mean", "mean"),
            "ct_mean_mean": ("ct_mean", "mean"),
        } if "mr_mean" in df.columns else {})
    ).to_string())

    print("\n── Per-center summary ───────────────────────────────────")
    print(df.groupby(["anatomy", "center"]).size().unstack(fill_value=0).to_string())

    print("\n[INFO] Generating plots...")
    plot_distributions(df, out_dir / "plots")

    print("\n[DONE]")
