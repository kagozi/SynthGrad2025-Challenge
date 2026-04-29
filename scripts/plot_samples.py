#!/usr/bin/env python3
"""
Plot one sample case per anatomy (HN, TH, AB) showing MR and CT
in all 3 orthogonal views (axial, coronal, sagittal).

Layout:
    Rows    → anatomy (Head-Neck, Thorax, Abdomen)
    Columns → view × modality: Axial MR | Axial CT | Coronal MR | Coronal CT | Sagittal MR | Sagittal CT

Usage:
    python scripts/plot_samples.py --data_dir /pvc/data/synthRAD2025_Task1_Train/Task1
    python scripts/plot_samples.py --data_dir data/raw --out plots/samples.png
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import SimpleITK as sitk


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_volume(path: Path) -> np.ndarray:
    """Load .mha → (D, H, W) float32."""
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def normalise_mr(arr: np.ndarray) -> np.ndarray:
    """Robust normalisation to [0, 1] using 1st–99th percentile."""
    p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
    return np.clip((arr - p1) / (p99 - p1 + 1e-8), 0, 1)


def normalise_ct(arr: np.ndarray) -> np.ndarray:
    """Clip to soft-tissue window [-200, 800] HU and scale to [0, 1] for display."""
    lo, hi = -200, 800
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def centre_slice(vol: np.ndarray, axis: int) -> np.ndarray:
    """Return the centre slice along the given axis."""
    idx = vol.shape[axis] // 2
    return np.take(vol, idx, axis=axis)


def find_case(data_dir: Path, anatomy: str) -> Path:
    """
    Find the first available case directory for the given anatomy.
    Searches recursively for directories matching the naming convention.
    """
    pattern = re.compile(rf"1{anatomy}[A-E]\d+", re.IGNORECASE)
    for p in sorted(data_dir.rglob("*")):
        if p.is_dir() and pattern.fullmatch(p.name):
            if (p / "mr.mha").exists() and (p / "ct.mha").exists():
                return p
    return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main(data_dir: Path, out_path: Path):
    anatomies  = ["HN", "TH", "AB"]
    anat_names = {"HN": "Head & Neck", "TH": "Thorax", "AB": "Abdomen"}

    # ── Find one case per anatomy ──────────────────────────────────────────────
    cases = {}
    for anat in anatomies:
        p = find_case(data_dir, anat)
        if p is None:
            print(f"[WARN] No case found for anatomy {anat} in {data_dir}")
        else:
            cases[anat] = p
            print(f"[INFO] {anat}: {p.name}")

    if not cases:
        print("[ERROR] No cases found. Check --data_dir.")
        sys.exit(1)

    # ── Figure layout: 3 rows (anatomy) × 6 cols (view×modality) ──────────────
    # Col order: Axial MR | Axial CT | Coronal MR | Coronal CT | Sagittal MR | Sagittal CT
    views     = ["Axial", "Coronal", "Sagittal"]
    modalities = ["MR", "CT"]
    n_rows = len(cases)
    n_cols = len(views) * len(modalities)   # 6

    fig = plt.figure(figsize=(n_cols * 3, n_rows * 3.2))
    fig.patch.set_facecolor("#111111")

    outer = gridspec.GridSpec(
        n_rows, 1,
        figure=fig,
        hspace=0.35,
    )

    for row_idx, anat in enumerate(anatomies):
        if anat not in cases:
            continue

        case_path = cases[anat]
        mr_vol = normalise_mr(load_volume(case_path / "mr.mha"))
        ct_vol = normalise_ct(load_volume(case_path / "ct.mha"))

        # axis=0 → axial (D), axis=1 → coronal (H), axis=2 → sagittal (W)
        slices = {
            "Axial":    {"MR": centre_slice(mr_vol, 0), "CT": centre_slice(ct_vol, 0)},
            "Coronal":  {"MR": centre_slice(mr_vol, 1), "CT": centre_slice(ct_vol, 1)},
            "Sagittal": {"MR": centre_slice(mr_vol, 2), "CT": centre_slice(ct_vol, 2)},
        }

        inner = gridspec.GridSpecFromSubplotSpec(
            1, n_cols,
            subplot_spec=outer[row_idx],
            wspace=0.04,
            hspace=0.0,
        )

        col = 0
        for view in views:
            for mod in modalities:
                ax = fig.add_subplot(inner[col])
                img = slices[view][mod]

                # Flip so superior is up on axial/sagittal, anterior is up on coronal
                ax.imshow(np.flipud(img), cmap="gray", vmin=0, vmax=1,
                          aspect="equal", interpolation="bilinear")

                # Column header (only on first row)
                if row_idx == 0:
                    color = "#88BBFF" if mod == "MR" else "#FFAA66"
                    ax.set_title(f"{view}\n{mod}", fontsize=9, color=color,
                                 fontweight="bold", pad=4)

                ax.axis("off")
                col += 1

        # Row label on the left
        label_ax = fig.add_subplot(outer[row_idx])
        label_ax.set_axis_off()
        label_ax.text(
            -0.01, 0.5,
            f"{anat_names[anat]}\n({case_path.name})",
            transform=label_ax.transAxes,
            fontsize=9, color="white", va="center", ha="right",
            fontweight="bold",
        )

    fig.suptitle(
        "SynthRAD2025 — Sample Cases: MR & CT in 3 Views",
        fontsize=13, color="white", fontweight="bold", y=1.01,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[DONE] Saved → {out_path}")


# ── Entry ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot one MR+CT sample per anatomy in 3 views.")
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Root data directory (e.g. /pvc/data/synthRAD2025_Task1_Train/Task1 or data/raw)",
    )
    parser.add_argument(
        "--out", type=str, default="data/splits/plots/samples_3views.png",
        help="Output image path",
    )
    args = parser.parse_args()

    main(Path(args.data_dir), Path(args.out))
