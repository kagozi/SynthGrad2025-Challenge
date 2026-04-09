#!/usr/bin/env python3
"""
SynthRAD2025 Task 1 — Full-volume inference script

Runs a trained model on all cases in an input directory and saves
predicted sCTs as .mha files, preserving original metadata.

Usage (local evaluation on val split):
    python inference/predict.py \
        --checkpoint checkpoints/baseline_unet2d/fold0_best.pth \
        --input_dir  data/raw \
        --output_dir predictions/fold0_val \
        --split val \
        --folds_csv  data/splits/folds.csv \
        --fold 0

Usage (predict on arbitrary input directory):
    python inference/predict.py \
        --checkpoint checkpoints/baseline_unet2d/fold0_best.pth \
        --input_dir  /path/to/validation/cases \
        --output_dir predictions/submission
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import (
    SynthRADInferenceDataset,
    build_case_list,
    denormalise_ct,
    ANATOMY_TO_IDX,
)
from src.models.unet2d import UNet2D, AttentionUNet2D


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    mc   = cfg["model"]

    kwargs = dict(
        in_channels=mc["in_channels"],
        out_channels=mc["out_channels"],
        base_features=mc["base_features"],
        depth=mc["depth"],
        n_anatomy=mc["n_anatomy"],
        use_anatomy=mc["use_anatomy"],
    )
    if mc["name"] == "attention_unet2d":
        model = AttentionUNet2D(**kwargs)
    else:
        model = UNet2D(**kwargs)

    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(device)
    n_context = cfg.get("data", {}).get("n_context", 0)
    print(f"[Predict] Loaded checkpoint (epoch {ckpt.get('epoch','?')}, "
          f"best MAE={ckpt.get('best_mae', '?'):.2f}, n_context={n_context})")
    return model, n_context


# ── Predict single case ────────────────────────────────────────────────────────

@torch.no_grad()
def predict_case(
    model: torch.nn.Module,
    case_path: Path,
    anatomy: str,
    device: torch.device,
    batch_size: int = 16,
    n_context: int = 0,
) -> np.ndarray:
    """
    Returns predicted sCT volume in HU, same shape as input MR.
    """
    ds     = SynthRADInferenceDataset(case_path, anatomy, n_context=n_context)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    anatomy_idx = torch.tensor(
        [ANATOMY_TO_IDX[anatomy]], dtype=torch.long, device=device
    )

    slices = []
    for batch in loader:
        mr = batch["mr"].to(device)
        ai = anatomy_idx.expand(mr.size(0))
        out = model(mr, ai)
        slices.append(out.squeeze(1).cpu().numpy())

    pred_norm = np.concatenate(slices, axis=0)    # (D, H, W), normalised [-1,1]
    pred_hu   = denormalise_ct(pred_norm)

    # Clip to physiologically plausible range
    pred_hu = np.clip(pred_hu, -1024, 3000)
    return pred_hu


# ── Save as .mha preserving original metadata ──────────────────────────────────

def save_mha(
    arr: np.ndarray,
    reference_path: Path,
    out_path: Path,
):
    """Save array with spacing/origin/direction copied from reference image."""
    ref = sitk.ReadImage(str(reference_path))
    out = sitk.GetImageFromArray(arr.astype(np.float32))
    out.SetSpacing(ref.GetSpacing())
    out.SetOrigin(ref.GetOrigin())
    out.SetDirection(ref.GetDirection())
    sitk.WriteImage(out, str(out_path))


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Predict] Device: {device}")

    model, n_context = load_model(args.checkpoint, device)

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build case list
    all_cases = build_case_list(input_dir)

    # Filter by fold/split if requested
    if args.folds_csv and Path(args.folds_csv).exists() and args.fold is not None:
        fold_df = pd.read_csv(args.folds_csv)
        if args.split == "val":
            ids     = set(fold_df[fold_df["fold"] == args.fold]["case_id"])
        else:
            ids     = set(fold_df[fold_df["fold"] != args.fold]["case_id"])
        all_cases = [c for c in all_cases if c["case_id"] in ids]

    print(f"[Predict] Cases to predict: {len(all_cases)}")

    results = []
    for case in tqdm(all_cases, desc="Predicting"):
        case_id = case["case_id"]
        anatomy = case["anatomy"]
        path    = Path(case["path"])

        try:
            pred_hu = predict_case(
                model, path, anatomy, device,
                batch_size=args.batch_size, n_context=n_context,
            )

            # Save predicted sCT
            out_case_dir = output_dir / case_id
            out_case_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_case_dir / "ct.mha"

            ref_path = path / "mr.mha"
            save_mha(pred_hu, ref_path, out_path)

            results.append({
                "case_id": case_id,
                "anatomy": anatomy,
                "status":  "ok",
                "shape":   str(pred_hu.shape),
            })

        except Exception as e:
            print(f"  [ERROR] {case_id}: {e}")
            results.append({"case_id": case_id, "anatomy": anatomy, "status": f"error: {e}"})

    # Summary
    ok  = [r for r in results if r["status"] == "ok"]
    err = [r for r in results if r["status"] != "ok"]
    print(f"\n[Predict] Done: {len(ok)} OK, {len(err)} errors")
    print(f"[Predict] Output: {output_dir}")

    if err:
        print("[Predict] Errors:")
        for r in err:
            print(f"  {r['case_id']}: {r['status']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--input_dir",   type=str, required=True)
    parser.add_argument("--output_dir",  type=str, required=True)
    parser.add_argument("--split",       type=str, default=None, choices=["train","val"])
    parser.add_argument("--folds_csv",   type=str, default=None)
    parser.add_argument("--fold",        type=int, default=None)
    parser.add_argument("--batch_size",  type=int, default=16)
    args = parser.parse_args()
    main(args)
