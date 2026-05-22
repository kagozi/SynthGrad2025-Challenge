#!/usr/bin/env python3
"""
SynthRAD2025 Task 1 — DDPM Inference (DDIM sampler)

Runs full-volume slice-by-slice inference using a trained DDPM checkpoint
and writes synthetic CT as .mha files with original spacing/origin/direction.

Usage:
    python inference/predict_ddpm.py \
        --checkpoint /pvc/checkpoints/ddpm/fold0_best.pth \
        --input_dir  /pvc/data/synthRAD2025_Task1_Train/Task1 \
        --output_dir /pvc/predictions/ddpm_fold0 \
        --steps 50

Ensemble (average multiple folds):
    python inference/predict_ddpm.py \
        --checkpoint /pvc/checkpoints/ddpm/fold{0,1,2,3,4}_best.pth \
        --input_dir ... --output_dir ... --steps 50

    (Pass each checkpoint as a separate --checkpoint argument; predictions
     are averaged in normalised [-1,1] space before denormalisation.)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import (
    SynthRADInferenceDataset,
    build_case_list,
    denormalise_ct,
    normalise_mr,
    normalise_ct,
)
from src.models.ddpm import DDPMUNet, GaussianDiffusion


# ── Anatomy index mapping ──────────────────────────────────────────────────────

_ANATOMY_IDX = {"HN": 0, "TH": 1, "AB": 2}


def anatomy_from_case_id(case_id: str) -> int:
    for tag, idx in _ANATOMY_IDX.items():
        if tag in case_id:
            return idx
    return 0


# ── Load model ─────────────────────────────────────────────────────────────────

def load_diffusion(checkpoint_path: str, device: torch.device) -> GaussianDiffusion:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt["cfg"]
    mc   = cfg["model"]
    dc   = cfg["diffusion"]

    unet = DDPMUNet(
        in_channels  = mc["in_channels"],
        out_channels = mc["out_channels"],
        base_ch      = mc["base_ch"],
        ch_mult      = tuple(mc["ch_mult"]),
        num_res      = mc["num_res"],
        attn_levels  = tuple(mc["attn_levels"]),
        time_emb_dim = mc["time_emb_dim"],
        n_anatomy    = mc["n_anatomy"],
        dropout      = 0.0,
    )
    diffusion = GaussianDiffusion(unet, T=dc["T"], s=dc.get("s", 0.008))
    diffusion.load_state_dict(ckpt["model"])
    diffusion.eval().to(device)
    return diffusion


# ── Single-volume inference ────────────────────────────────────────────────────

@torch.no_grad()
def predict_volume(
    diffusion:   GaussianDiffusion,
    mr_volume:   np.ndarray,   # (D, H, W) normalised MR [0, 1]
    mask_volume: np.ndarray,   # (D, H, W) binary body mask
    anatomy_idx: int,
    device:      torch.device,
    steps:       int   = 50,
    batch_size:  int   = 16,
    eta:         float = 0.0,
) -> np.ndarray:
    """Returns predicted sCT in normalised [-1, 1] space, shape (D, H, W)."""
    D, H, W = mr_volume.shape
    pred_vol = np.zeros((D, H, W), dtype=np.float32)

    anat_t = torch.tensor([anatomy_idx], device=device)

    for start in range(0, D, batch_size):
        end    = min(start + batch_size, D)
        B_curr = end - start

        mr_batch = torch.from_numpy(
            mr_volume[start:end, None]   # (B, 1, H, W)
        ).float().to(device)

        anat_batch = anat_t.expand(B_curr)
        pred_batch = diffusion.ddim_sample(
            mr_batch, anat_batch, steps=steps, eta=eta
        )  # (B, 1, H, W)

        pred_vol[start:end] = pred_batch[:, 0].cpu().numpy()

    return pred_vol


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", nargs="+", required=True,
                        help="One or more checkpoint paths (ensemble = average)")
    parser.add_argument("--input_dir",  required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--steps",      type=int,   default=50)
    parser.add_argument("--eta",        type=float, default=0.0,
                        help="DDIM stochasticity: 0=deterministic, 1=full DDPM")
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--gpu",        type=int,   default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all checkpoints (for ensemble)
    models = [load_diffusion(ckpt, device) for ckpt in args.checkpoint]
    print(f"Loaded {len(models)} model(s). DDIM steps: {args.steps}")

    cases = build_case_list([args.input_dir])
    print(f"Found {len(cases)} cases.")

    for case in tqdm(cases, desc="Predicting"):
        case_id = case["case_id"]
        anatomy = anatomy_from_case_id(case_id)

        # Load MR and mask
        mr_img   = sitk.ReadImage(str(case["mr_path"]))
        mask_img = sitk.ReadImage(str(case["mask_path"]))

        mr_arr   = sitk.GetArrayFromImage(mr_img).astype(np.float32)    # (D, H, W)
        mask_arr = sitk.GetArrayFromImage(mask_img).astype(np.float32)

        # Normalise MR (same as training)
        mr_norm = normalise_mr(mr_arr, mask_arr.astype(bool))

        # Pad/crop to expected size
        _, H, W = mr_norm.shape
        H_target, W_target = 512, 512
        pad_h = max(0, H_target - H)
        pad_w = max(0, W_target - W)
        if pad_h > 0 or pad_w > 0:
            mr_norm  = np.pad(mr_norm,  [(0,0),(0,pad_h),(0,pad_w)])
            mask_arr = np.pad(mask_arr, [(0,0),(0,pad_h),(0,pad_w)])

        mr_norm  = mr_norm [:, :H_target, :W_target]
        mask_arr = mask_arr[:, :H_target, :W_target]

        # Predict (average over ensemble models)
        pred_sum = np.zeros_like(mr_norm)
        for model in models:
            pred_sum += predict_volume(
                model, mr_norm, mask_arr, anatomy,
                device, args.steps, args.batch_size, args.eta,
            )
        pred_norm = pred_sum / len(models)

        # Crop back to original spatial size and denormalise to HU
        pred_norm = pred_norm[:, :H, :W]
        pred_hu   = denormalise_ct(pred_norm)

        # Write output .mha preserving original metadata
        pred_img = sitk.GetImageFromArray(pred_hu.astype(np.float32))
        pred_img.CopyInformation(mr_img)

        out_path = out_dir / f"{case_id}_ct.mha"
        sitk.WriteImage(pred_img, str(out_path))

    print(f"Done. Predictions written to {out_dir}")


if __name__ == "__main__":
    main()
