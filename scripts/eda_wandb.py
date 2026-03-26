#!/usr/bin/env python3
"""
SynthRAD2025 Task 1 — Deep Data Exploration → WandB

Logs to WandB:
  - Per-case stats table  (shape, spacing, intensity stats for MR + CT)
  - Per-anatomy summary tables
  - Distribution charts (n_slices, H, W, spacing, HU range)
  - Representative sample panels: axial / coronal / sagittal
    views of MR, CT and mask for N cases per anatomy

Usage (local):
    python scripts/eda_wandb.py \\
        --data_dirs /pvc/data/synthRAD2025_Task1_Train/Task1 \\
                    /pvc/data/synthRAD2025_Task1_Train_D/Task1 \\
        --out_dir   /pvc/data/splits \\
        --n_samples 5

Usage (K8s job):
    kubectl apply -f nautilius/jobs/eda-wandb.yaml
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")                    # headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import SimpleITK as sitk
import wandb
from tqdm import tqdm


# ── Helpers ────────────────────────────────────────────────────────────────────

CASE_RE = re.compile(r"(\d)(HN|TH|AB)([A-E])(\d+)", re.I)


def parse_case_id(cid: str) -> Optional[Dict]:
    m = CASE_RE.match(cid)
    if not m:
        return None
    return {"anatomy": m.group(2).upper(), "center": m.group(3).upper()}


def discover_cases(data_dirs: List[str]) -> List[Dict]:
    """Walk data_dirs and return list of {case_id, anatomy, center, path}."""
    seen, cases = set(), []
    for root in data_dirs:
        root = Path(root)
        if not root.exists():
            print(f"[WARN] data_dir not found: {root}")
            continue
        # Try PVC layout first, then flat
        for pattern in ["*/*/", "*/"]:
            hits = [d for d in root.glob(pattern)
                    if d.is_dir() and CASE_RE.match(d.name) and (d / "mr.mha").exists()]
            if hits:
                break
        for d in sorted(hits):
            if d.name in seen:
                continue
            meta = parse_case_id(d.name)
            if not meta:
                continue
            seen.add(d.name)
            cases.append({"case_id": d.name, "path": str(d), **meta})
    return cases


def load_meta(path: Path) -> Dict:
    """Read header only — fast."""
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()
    size    = r.GetSize()       # (W, H, D)  SimpleITK convention
    spacing = r.GetSpacing()    # (sx, sy, sz) mm
    return {
        "W": int(size[0]), "H": int(size[1]), "D": int(size[2]),
        "sp_x": round(spacing[0], 4),
        "sp_y": round(spacing[1], 4),
        "sp_z": round(spacing[2], 4),
    }


def load_arr(path: Path) -> np.ndarray:
    """Load .mha → float32 numpy array (D, H, W)."""
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def intensity_stats(arr: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict:
    vals = arr[mask.astype(bool)] if mask is not None else arr.ravel()
    if vals.size == 0:
        vals = arr.ravel()
    return {
        "min":  float(vals.min()),
        "max":  float(vals.max()),
        "mean": float(vals.mean()),
        "std":  float(vals.std()),
        "p1":   float(np.percentile(vals, 1)),
        "p5":   float(np.percentile(vals, 5)),
        "p95":  float(np.percentile(vals, 95)),
        "p99":  float(np.percentile(vals, 99)),
    }


# ── Per-case scanning ──────────────────────────────────────────────────────────

def scan_case(case: Dict) -> Optional[Dict]:
    path = Path(case["path"])
    mr_path   = path / "mr.mha"
    ct_path   = path / "ct.mha"
    mask_path = path / "mask.mha"

    if not mr_path.exists() or not ct_path.exists():
        print(f"  [SKIP] missing files in {case['case_id']}")
        return None

    try:
        mr_meta = load_meta(mr_path)
        ct_meta = load_meta(ct_path)

        mr_arr   = load_arr(mr_path)
        ct_arr   = load_arr(ct_path)
        mask_arr = load_arr(mask_path) if mask_path.exists() else None

        mr_stats = intensity_stats(mr_arr, mask_arr)
        ct_stats = intensity_stats(ct_arr, mask_arr)

        # Count non-empty axial slices (>1% mask coverage)
        if mask_arr is not None:
            slice_coverage = mask_arr.mean(axis=(1, 2))           # shape (D,)
            n_content_slices = int((slice_coverage > 0.01).sum())
        else:
            n_content_slices = mr_meta["D"]

        row = {
            "case_id": case["case_id"],
            "anatomy": case["anatomy"],
            "center":  case["center"],
            "has_mask": mask_path.exists(),
            # MR geometry
            "mr_D": mr_meta["D"], "mr_H": mr_meta["H"], "mr_W": mr_meta["W"],
            "mr_sp_x": mr_meta["sp_x"], "mr_sp_y": mr_meta["sp_y"], "mr_sp_z": mr_meta["sp_z"],
            # CT geometry
            "ct_D": ct_meta["D"], "ct_H": ct_meta["H"], "ct_W": ct_meta["W"],
            # Slice counts
            "n_slices_total":   mr_meta["D"],
            "n_slices_content": n_content_slices,
            # MR intensity (raw, before normalisation)
            **{f"mr_{k}": v for k, v in mr_stats.items()},
            # CT intensity (HU)
            **{f"ct_{k}": v for k, v in ct_stats.items()},
        }
        return row, mr_arr, ct_arr, mask_arr

    except Exception as e:
        print(f"  [ERROR] {case['case_id']}: {e}")
        return None


# ── Sample image rendering ─────────────────────────────────────────────────────

def _norm_display(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip and scale to [0, 1] for display."""
    arr = np.clip(arr, lo, hi)
    return (arr - lo) / (hi - lo + 1e-8)


def make_sample_panel(
    case_id: str,
    mr: np.ndarray,  # (D, H, W)  raw MR
    ct: np.ndarray,  # (D, H, W)  raw CT HU
    mask: Optional[np.ndarray],
) -> plt.Figure:
    """
    3-row panel:
      Row 0 — axial mid slice
      Row 1 — coronal mid slice
      Row 2 — sagittal mid slice

    Columns: MR | CT | mask (if available) | MR+mask overlay
    """
    D, H, W = mr.shape
    mid_d, mid_h, mid_w = D // 2, H // 2, W // 2

    views = {
        "axial":    (mr[mid_d],    ct[mid_d],    mask[mid_d]    if mask is not None else None),
        "coronal":  (mr[:, mid_h, :], ct[:, mid_h, :], mask[:, mid_h, :] if mask is not None else None),
        "sagittal": (mr[:, :, mid_w], ct[:, :, mid_w], mask[:, :, mid_w] if mask is not None else None),
    }

    # MR display range: p1–p99
    mr_lo, mr_hi = float(np.percentile(mr, 1)), float(np.percentile(mr, 99))
    ct_lo, ct_hi = -1000.0, 1000.0     # HU window (soft tissue)

    n_cols = 4 if mask is not None else 3
    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 10))
    fig.suptitle(case_id, fontsize=13, fontweight="bold")

    col_titles = ["MR (raw)", "CT (HU)", "Mask", "MR + mask overlay"]
    for ax, title in zip(axes[0], col_titles[:n_cols]):
        ax.set_title(title, fontsize=10)

    for row_idx, (view_name, (mr_sl, ct_sl, mask_sl)) in enumerate(views.items()):
        axes[row_idx, 0].set_ylabel(view_name, fontsize=10, rotation=90, labelpad=4)

        mr_disp   = _norm_display(mr_sl, mr_lo, mr_hi)
        ct_disp   = _norm_display(ct_sl, ct_lo, ct_hi)

        axes[row_idx, 0].imshow(mr_disp, cmap="gray", origin="lower", aspect="equal")
        axes[row_idx, 1].imshow(ct_disp, cmap="gray", origin="lower", aspect="equal")

        if mask is not None and n_cols >= 3:
            axes[row_idx, 2].imshow(mask_sl, cmap="hot", origin="lower", aspect="equal", vmin=0, vmax=1)

        if mask is not None and n_cols >= 4:
            # Overlay: MR as background, mask as semi-transparent red
            axes[row_idx, 3].imshow(mr_disp, cmap="gray", origin="lower", aspect="equal")
            overlay = np.zeros((*mr_sl.shape, 4), dtype=np.float32)
            overlay[..., 0] = 1.0     # red channel
            overlay[..., 3] = (mask_sl > 0).astype(np.float32) * 0.45   # alpha
            axes[row_idx, 3].imshow(overlay, origin="lower", aspect="equal")

        for ax in axes[row_idx]:
            ax.axis("off")

    plt.tight_layout()
    return fig


# ── Distribution plots ─────────────────────────────────────────────────────────

def make_distribution_plots(df: pd.DataFrame) -> Dict[str, plt.Figure]:
    figs = {}
    anatomies = sorted(df["anatomy"].unique())
    colors = {"HN": "steelblue", "TH": "tomato", "AB": "seagreen"}

    # 1. n_slices, H, W per anatomy
    for col, title in [
        ("n_slices_total", "Total slices (D)"),
        ("mr_H", "Height (H, pixels)"),
        ("mr_W", "Width (W, pixels)"),
        ("mr_sp_z", "MR slice thickness z (mm)"),
        ("mr_sp_x", "MR in-plane spacing x (mm)"),
    ]:
        fig, axes = plt.subplots(1, len(anatomies), figsize=(5 * len(anatomies), 4), sharey=False)
        if len(anatomies) == 1:
            axes = [axes]
        for ax, anat in zip(axes, anatomies):
            sub = df[df["anatomy"] == anat][col].dropna()
            ax.hist(sub, bins=25, color=colors.get(anat, "gray"), alpha=0.8, edgecolor="white")
            ax.set_title(f"{anat}  (n={len(sub)})", fontsize=11)
            ax.set_xlabel(title)
            ax.set_ylabel("count")
            ax.axvline(sub.median(), color="k", linestyle="--", linewidth=1, label=f"median={sub.median():.1f}")
            ax.legend(fontsize=8)
        fig.suptitle(title, fontsize=12, fontweight="bold")
        plt.tight_layout()
        figs[f"dist_{col}"] = fig

    # 2. CT HU range per anatomy
    fig, axes = plt.subplots(1, len(anatomies), figsize=(5 * len(anatomies), 4))
    if len(anatomies) == 1:
        axes = [axes]
    for ax, anat in zip(axes, anatomies):
        sub = df[df["anatomy"] == anat]
        ax.scatter(sub["ct_p1"], sub["ct_p99"], alpha=0.5, s=20, color=colors.get(anat, "gray"))
        ax.set_title(f"{anat} CT HU range (p1 vs p99)", fontsize=11)
        ax.set_xlabel("CT p1 HU")
        ax.set_ylabel("CT p99 HU")
    fig.suptitle("CT HU Range per Case", fontsize=12, fontweight="bold")
    plt.tight_layout()
    figs["dist_ct_hu_range"] = fig

    # 3. MR intensity per anatomy
    fig, axes = plt.subplots(1, len(anatomies), figsize=(5 * len(anatomies), 4))
    if len(anatomies) == 1:
        axes = [axes]
    for ax, anat in zip(axes, anatomies):
        sub = df[df["anatomy"] == anat]
        ax.errorbar(range(len(sub)), sub["mr_mean"].values,
                    yerr=sub["mr_std"].values, fmt="none", alpha=0.3, color=colors.get(anat, "gray"))
        ax.scatter(range(len(sub)), sub["mr_mean"].values, s=10, color=colors.get(anat, "gray"))
        ax.set_title(f"{anat} MR mean ± std", fontsize=11)
        ax.set_xlabel("case index")
        ax.set_ylabel("mean intensity (raw)")
    fig.suptitle("MR Intensity per Case", fontsize=12, fontweight="bold")
    plt.tight_layout()
    figs["dist_mr_intensity"] = fig

    return figs


# ── Summary table ──────────────────────────────────────────────────────────────

def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    agg = {}
    for col in ["n_slices_total", "n_slices_content", "mr_D", "mr_H", "mr_W",
                "mr_sp_x", "mr_sp_y", "mr_sp_z",
                "mr_mean", "mr_std", "mr_p1", "mr_p99",
                "ct_mean", "ct_std", "ct_p1", "ct_p99"]:
        if col in df.columns:
            agg[col] = ["min", "median", "max", "mean"]

    summary = df.groupby("anatomy").agg(
        n_cases=("case_id", "count"),
        n_centers=("center", "nunique"),
    )
    for col, funcs in agg.items():
        for fn in funcs:
            summary[f"{col}_{fn}"] = df.groupby("anatomy")[col].agg(fn)
    return summary.reset_index()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", nargs="+", required=True)
    parser.add_argument("--out_dir",   default="/pvc/data/splits")
    parser.add_argument("--n_samples", type=int, default=4,
                        help="Number of sample panels per anatomy to log to WandB")
    parser.add_argument("--wandb_project", default="synthrad2025-task1")
    parser.add_argument("--wandb_entity",  default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── WandB init ──────────────────────────────────────────────────────────────
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name="eda-data-exploration",
        tags=["eda", "data-exploration"],
        config={
            "data_dirs":   args.data_dirs,
            "n_samples":   args.n_samples,
        },
    )

    # ── Discover cases ──────────────────────────────────────────────────────────
    cases = discover_cases(args.data_dirs)
    print(f"[INFO] Discovered {len(cases)} cases across {args.data_dirs}")

    if not cases:
        print("[ERROR] No cases found — check --data_dirs")
        sys.exit(1)

    wandb.log({"total_cases": len(cases)})

    # ── Scan every case ─────────────────────────────────────────────────────────
    records = []
    # Keep volumes for sample logging (per anatomy, first n_samples cases)
    sample_store: Dict[str, List] = {"HN": [], "TH": [], "AB": []}

    cases_by_anatomy: Dict[str, List] = {"HN": [], "TH": [], "AB": []}
    for c in cases:
        cases_by_anatomy[c["anatomy"]].append(c)

    for case in tqdm(cases, desc="Scanning"):
        result = scan_case(case)
        if result is None:
            continue
        row, mr_arr, ct_arr, mask_arr = result
        records.append(row)

        anat = case["anatomy"]
        if len(sample_store[anat]) < args.n_samples:
            sample_store[anat].append((case["case_id"], mr_arr, ct_arr, mask_arr))

    df = pd.DataFrame(records)
    csv_path = out_dir / "dataset_info.csv"
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved per-case CSV: {csv_path} ({len(df)} rows)")

    # ── Log per-case table ──────────────────────────────────────────────────────
    table_cols = [
        "case_id", "anatomy", "center", "has_mask",
        "mr_D", "mr_H", "mr_W", "mr_sp_x", "mr_sp_y", "mr_sp_z",
        "ct_D", "ct_H", "ct_W",
        "n_slices_total", "n_slices_content",
        "mr_min", "mr_max", "mr_mean", "mr_std", "mr_p1", "mr_p99",
        "ct_min", "ct_max", "ct_mean", "ct_std", "ct_p1", "ct_p99",
    ]
    table_cols = [c for c in table_cols if c in df.columns]
    wt = wandb.Table(dataframe=df[table_cols])
    wandb.log({"per_case_stats": wt})
    print("[INFO] Logged per-case stats table to WandB")

    # ── Log summary table ───────────────────────────────────────────────────────
    summary = make_summary(df)
    wandb.log({"anatomy_summary": wandb.Table(dataframe=summary)})
    print("[INFO] Logged anatomy summary table to WandB")

    # ── Print human-readable summary ────────────────────────────────────────────
    print("\n── Cases per anatomy × center ───────────────────────────")
    print(df.groupby(["anatomy", "center"]).size().unstack(fill_value=0).to_string())

    print("\n── Geometry summary ─────────────────────────────────────")
    geo_cols = [c for c in ["anatomy", "mr_D", "mr_H", "mr_W", "mr_sp_z", "n_slices_total"] if c in df.columns]
    print(df[geo_cols].groupby("anatomy").agg(["min", "median", "max"]).to_string())

    print("\n── CT HU summary ────────────────────────────────────────")
    hu_cols = [c for c in ["anatomy", "ct_min", "ct_max", "ct_mean", "ct_p1", "ct_p99"] if c in df.columns]
    print(df[hu_cols].groupby("anatomy").agg(["min", "median", "max"]).to_string())

    # ── Distribution plots → WandB ──────────────────────────────────────────────
    print("\n[INFO] Generating distribution plots...")
    dist_figs = make_distribution_plots(df)
    wandb_dist = {}
    for key, fig in dist_figs.items():
        wandb_dist[f"distributions/{key}"] = wandb.Image(fig)
        fig.savefig(out_dir / f"{key}.png", dpi=150)
        plt.close(fig)
    wandb.log(wandb_dist)
    print(f"[INFO] Logged {len(dist_figs)} distribution plots")

    # ── Sample image panels → WandB ─────────────────────────────────────────────
    print("\n[INFO] Generating sample image panels...")
    sample_images = {}
    for anat, samples in sample_store.items():
        for i, (cid, mr, ct, mask) in enumerate(samples):
            fig = make_sample_panel(cid, mr, ct, mask)
            key = f"samples/{anat}/{cid}"
            sample_images[key] = wandb.Image(fig, caption=cid)
            # Also save locally
            fig.savefig(out_dir / f"sample_{cid}.png", dpi=120)
            plt.close(fig)
            print(f"  Panel: {cid}  shape={mr.shape}")

    wandb.log(sample_images)
    print(f"[INFO] Logged {len(sample_images)} sample panels to WandB")

    # ── Scalar summary metrics ──────────────────────────────────────────────────
    scalar_logs = {}
    for anat in df["anatomy"].unique():
        sub = df[df["anatomy"] == anat]
        scalar_logs[f"{anat}/n_cases"]              = len(sub)
        scalar_logs[f"{anat}/median_D_slices"]      = float(sub["n_slices_total"].median())
        scalar_logs[f"{anat}/median_H"]             = float(sub["mr_H"].median())
        scalar_logs[f"{anat}/median_W"]             = float(sub["mr_W"].median())
        scalar_logs[f"{anat}/median_sp_z_mm"]       = float(sub["mr_sp_z"].median())
        scalar_logs[f"{anat}/content_slice_frac"]   = float(sub["n_slices_content"].sum() / sub["n_slices_total"].sum())
        if "ct_p99" in sub.columns:
            scalar_logs[f"{anat}/median_ct_p99_HU"] = float(sub["ct_p99"].median())
            scalar_logs[f"{anat}/median_ct_p1_HU"]  = float(sub["ct_p1"].median())

    wandb.log(scalar_logs)

    run.finish()
    print("\n[DONE] WandB run complete →", run.url)


if __name__ == "__main__":
    main()
