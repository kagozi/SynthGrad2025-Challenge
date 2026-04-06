#!/usr/bin/env python3
"""
SynthRAD2025 Task 1 — 5-Fold Ensemble Inference

Loads all fold checkpoints, averages predictions in normalised [-1,1] space,
saves flat submission files as:
    {output_dir}/sct_{case_id}.mha

Usage:
    python inference/predict_ensemble.py \
        --checkpoints \
            /pvc/checkpoints/round2_unet2d/fold0_best.pth \
            /pvc/checkpoints/round2_unet2d/fold1_best.pth \
            /pvc/checkpoints/round2_unet2d/fold2_best.pth \
            /pvc/checkpoints/round2_unet2d/fold3_best.pth \
            /pvc/checkpoints/round2_unet2d/fold4_best.pth \
        --input_dirs \
            /pvc/data/synthRAD2025_Task1_Val_Input/Task1 \
            /pvc/data/synthRAD2025_Task1_Val_Input_D/Task1 \
        --output_dir /pvc/submissions/round2_ensemble
"""

import argparse
import sys
from pathlib import Path

import numpy as np
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


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    mc   = cfg["model"]
    kwargs = dict(
        in_channels   = mc["in_channels"],
        out_channels  = mc["out_channels"],
        base_features = mc["base_features"],
        depth         = mc["depth"],
        n_anatomy     = mc["n_anatomy"],
        use_anatomy   = mc["use_anatomy"],
    )
    model = AttentionUNet2D(**kwargs) if mc["name"] == "attention_unet2d" else UNet2D(**kwargs)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    epoch    = ckpt.get("epoch", "?")
    best_mae = ckpt.get("best_mae", float("nan"))
    print(f"  Loaded {Path(checkpoint_path).name}  epoch={epoch}  best_MAE={best_mae:.2f}")
    return model


@torch.no_grad()
def predict_case_norm(model, case_path: Path, anatomy: str,
                      device: torch.device, batch_size: int) -> np.ndarray:
    """Run model on one case; return normalised predictions (D, H, W) in [-1,1]."""
    ds     = SynthRADInferenceDataset(case_path, anatomy)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    ai     = torch.tensor([ANATOMY_TO_IDX[anatomy]], dtype=torch.long, device=device)
    slices = []
    for batch in loader:
        mr  = batch["mr"].to(device)
        out = model(mr, ai.expand(mr.size(0)))
        slices.append(out.squeeze(1).cpu().numpy())
    return np.concatenate(slices, axis=0)


def save_mha(arr: np.ndarray, reference_path: Path, out_path: Path):
    ref = sitk.ReadImage(str(reference_path))
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetSpacing(ref.GetSpacing())
    img.SetOrigin(ref.GetOrigin())
    img.SetDirection(ref.GetDirection())
    sitk.WriteImage(img, str(out_path), True)   # True = use compression


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Ensemble] Device: {device}")

    # Load all models
    print(f"[Ensemble] Loading {len(args.checkpoints)} checkpoints...")
    models = [load_model(ckpt, device) for ckpt in args.checkpoints]

    # Build case list across all input dirs
    all_cases = build_case_list(args.input_dirs)
    print(f"[Ensemble] Cases found: {len(all_cases)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ok, err = 0, 0
    for case in tqdm(all_cases, desc="Predicting"):
        case_id = case["case_id"]
        anatomy = case["anatomy"]
        path    = Path(case["path"])

        try:
            # Average normalised predictions across all folds
            pred_norm = np.zeros(0, dtype=np.float32)
            for i, model in enumerate(models):
                p = predict_case_norm(model, path, anatomy, device, args.batch_size)
                pred_norm = p if i == 0 else pred_norm + p
            pred_norm /= len(models)

            # Denormalise and clip to plausible HU range
            pred_hu = denormalise_ct(pred_norm)
            pred_hu = np.clip(pred_hu, -1024, 3000)

            out_path = output_dir / f"sct_{case_id}.mha"
            save_mha(pred_hu, path / "mr.mha", out_path)

            tqdm.write(f"  {case_id}  shape={pred_hu.shape}  "
                       f"HU=[{pred_hu.min():.0f},{pred_hu.max():.0f}]")
            ok += 1

        except Exception as e:
            tqdm.write(f"  [ERROR] {case_id}: {e}")
            err += 1

    print(f"\n[Ensemble] Done: {ok} OK, {err} errors")
    print(f"[Ensemble] Output: {output_dir}")

    # Create zip for submission
    import zipfile
    zip_path = output_dir.parent / f"{output_dir.name}.zip"
    mha_files = sorted(output_dir.glob("sct_*.mha"))
    print(f"[Ensemble] Zipping {len(mha_files)} files → {zip_path}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in mha_files:
            zf.write(f, f.name)
    print(f"[Ensemble] Zip ready: {zip_path}  ({zip_path.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints",  nargs="+", required=True)
    parser.add_argument("--input_dirs",   nargs="+", required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--batch_size",   type=int, default=8)
    args = parser.parse_args()
    main(args)
