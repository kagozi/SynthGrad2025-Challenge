#!/usr/bin/env python3
"""
SynthRAD2025 Task 1 - 5-Fold Cross-Validation Split Generator

Stratified by anatomy AND center to ensure balanced representation.
Saves fold assignments to folds.csv

Usage (local):
    python scripts/prepare_folds.py --data_dirs data --n_folds 5

Usage (on PVC via kubectl exec or K8s job):
    python scripts/prepare_folds.py \
        --data_dirs /pvc/data/synthRAD2025_Task1_Train /pvc/data/synthRAD2025_Task1_Train_D \
        --out_dir /pvc/data/splits
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def parse_case_id(case_id: str) -> dict:
    m = re.match(r"(\d)(HN|TH|AB)([A-E])(\d+)", case_id, re.IGNORECASE)
    if not m:
        return {}
    return {
        "anatomy": m.group(2).upper(),
        "center":  m.group(3).upper(),
    }


def find_cases(data_dirs) -> pd.DataFrame:
    """Scan one or more root dirs (handles PVC layout and flat layout)."""
    if isinstance(data_dirs, (str, Path)):
        data_dirs = [data_dirs]

    case_id_re = re.compile(r"\d(HN|TH|AB)[A-E]\d+", re.I)
    seen    = set()
    records = []

    for root in data_dirs:
        root = Path(root)
        if not root.exists():
            print(f"[WARN] Directory not found: {root}")
            continue

        dirs = []
        for pattern in ["*/Task1/*/*", "*/*", "*"]:
            hits = [d for d in root.glob(pattern)
                    if d.is_dir() and case_id_re.match(d.name)]
            if hits:
                dirs = hits
                break

        for d in sorted(dirs):
            if d.name in seen:
                continue
            meta = parse_case_id(d.name)
            if not meta:
                continue
            if (d / "mr.mha").exists() and (d / "ct.mha").exists():
                seen.add(d.name)
                records.append({
                    "case_id": d.name,
                    "anatomy": meta["anatomy"],
                    "center":  meta["center"],
                    "path":    str(d),
                })

    if not records:
        raise FileNotFoundError(f"No valid cases found in: {data_dirs}")

    return pd.DataFrame(records)


def make_folds(df: pd.DataFrame, n_folds: int, seed: int = 42) -> pd.DataFrame:
    """
    Stratify by anatomy+center combination.
    Falls back to anatomy-only if a stratum has fewer samples than n_folds.
    """
    df = df.copy()

    # Create stratum label: anatomy_center (e.g. HN_A)
    df["stratum"] = df["anatomy"] + "_" + df["center"]

    # Check if any stratum is too small → merge to anatomy-only
    counts = df["stratum"].value_counts()
    small  = counts[counts < n_folds].index.tolist()
    if small:
        print(f"[WARN] Small strata (<{n_folds} cases), falling back to anatomy-only: {small}")
        for s in small:
            anat = s.split("_")[0]
            df.loc[df["stratum"] == s, "stratum"] = anat

    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    labels = df["stratum"].values
    df["fold"] = -1

    for fold_idx, (_, val_idx) in enumerate(skf.split(df, labels)):
        df.iloc[val_idx, df.columns.get_loc("fold")] = fold_idx

    return df


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f" SynthRAD2025 Task 1 — {args.n_folds}-Fold CV Split")
    print("=" * 60)

    df = find_cases(args.data_dirs)
    print(f"[INFO] Total cases found: {len(df)}")

    df = make_folds(df, n_folds=args.n_folds, seed=args.seed)

    out_path = out_dir / "folds.csv"
    df.to_csv(out_path, index=False)
    print(f"[INFO] Saved: {out_path}")

    print("\n── Fold distribution ───────────────────────────────────")
    pivot = df.groupby(["fold", "anatomy"]).size().unstack(fill_value=0)
    pivot["total"] = pivot.sum(axis=1)
    print(pivot.to_string())

    print("\n── Center distribution per fold ────────────────────────")
    print(df.groupby(["fold", "center"]).size().unstack(fill_value=0).to_string())

    # Sanity check: each center appears in val sets
    print("\n── Validation coverage check ───────────────────────────")
    for fold in range(args.n_folds):
        val_centers  = df[df["fold"] == fold]["center"].unique()
        val_anatomy  = df[df["fold"] == fold]["anatomy"].unique()
        train_n      = (df["fold"] != fold).sum()
        val_n        = (df["fold"] == fold).sum()
        print(f"  Fold {fold}: train={train_n}, val={val_n} | "
              f"centers={sorted(val_centers)} | anatomy={sorted(val_anatomy)}")

    print("\n[DONE]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", type=str, nargs="+",
                        default=["data/raw"],
                        help="One or more root dirs to scan for training cases")
    parser.add_argument("--out_dir",  type=str, default="data/splits")
    parser.add_argument("--n_folds",  type=int, default=5)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()
    main(args)
