#!/usr/bin/env python3
"""
SynthRAD2025 Task 1 — VS-DDPM 3D Sliding Window Inference

Stochastic DDPM sampling (not DDIM) with per-step background masking,
matching the Faking_it submission pipeline.

Usage:
    # Single case:
    python inference/predict_vs_ddpm.py \
        --input-dir  /pvc/data/.../1HNA001 \
        --output-dir /pvc/predictions/vs_ddpm/fold0 \
        --checkpoint /pvc/checkpoints/vs_ddpm_3d/fold0_best.pth \
        --config     training/configs/vs_ddpm_3d.yaml \
        --anatomy    HN --steps 100

    # All cases:
    python inference/predict_vs_ddpm.py \
        --input-dir  /pvc/data/synthRAD2025_Task1_Train/Task1 \
        --output-dir /pvc/predictions/vs_ddpm/fold0 \
        --checkpoint /pvc/checkpoints/vs_ddpm_3d/fold0_best.pth \
        --config     training/configs/vs_ddpm_3d.yaml \
        --steps 100
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import (
    build_case_list,
    denormalise_ct,
    load_mha,
    normalise_mr_m11,
    ANATOMY_TO_IDX,
)
from src.models.vs_ddpm_3d import (
    UNetModel3D,
    build_model_and_diffusions,
    create_spaced_diffusion,
)


def load_model(checkpoint_path: str, cfg: dict, device: torch.device) -> UNetModel3D:
    mc = cfg["model"]
    model, _ = build_model_and_diffusions(
        model_channels = mc.get("model_channels", 64),
        dropout        = mc.get("dropout", 0.2),
        image_size     = mc.get("image_size", 128),
        noise_schedule = cfg["diffusion"].get("noise_schedule", "linear"),
    )
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    epoch    = ckpt.get("epoch", "?")
    best_mae = ckpt.get("best_mae", None)
    msg = f"[Predict] Loaded checkpoint epoch={epoch}"
    if best_mae:
        msg += f"  best_mae={best_mae:.2f} HU"
    print(msg)
    return model


def _compute_patch_starts(vol_size: int, patch_size: int, overlap: float) -> list:
    step   = max(1, int(patch_size * (1.0 - overlap)))
    starts = list(range(0, max(1, vol_size - patch_size + 1), step))
    if not starts or starts[-1] + patch_size < vol_size:
        starts.append(max(0, vol_size - patch_size))
    return starts


@torch.no_grad()
def predict_case(
    mr_path:     Path,
    mask_path:   Optional[Path],
    anatomy:     str,
    model:       UNetModel3D,
    device:      torch.device,
    cfg:         dict,
    steps:       int   = 100,
    overlap:     float = 0.5,
    infer_batch: int   = 1,
) -> tuple:
    patch_size = tuple(cfg["data"].get("patch_size", [32, 128, 128]))
    pd, ph, pw = patch_size

    mr_img  = sitk.ReadImage(str(mr_path))
    mr_raw  = sitk.GetArrayFromImage(mr_img).astype(np.float32)

    if mask_path is not None and mask_path.exists():
        mask_raw  = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))).astype(np.float32)
        mask_bool = (mask_raw > 0)
    else:
        mask_bool = (mr_raw > 0)

    mr_arr = normalise_mr_m11(mr_raw, anatomy, mask=mask_bool)
    D, H, W = mr_arr.shape

    pad_d = max(0, pd - D)
    pad_h = max(0, ph - H)
    pad_w = max(0, pw - W)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        mr_arr    = np.pad(mr_arr,    ((0, pad_d), (0, pad_h), (0, pad_w)),
                           constant_values=-1.0)
        mask_bool = np.pad(mask_bool, ((0, pad_d), (0, pad_h), (0, pad_w)))

    Dp, Hp, Wp = mr_arr.shape

    sum_tensor   = np.zeros((Dp, Hp, Wp), dtype=np.float64)
    count_tensor = np.zeros((Dp, Hp, Wp), dtype=np.float64)

    d_starts = _compute_patch_starts(Dp, pd, overlap)
    h_starts = _compute_patch_starts(Hp, ph, overlap)
    w_starts = _compute_patch_starts(Wp, pw, overlap)

    patch_coords = [
        (d0, h0, w0)
        for d0 in d_starts
        for h0 in h_starts
        for w0 in w_starts
    ]

    diff = create_spaced_diffusion(steps, cfg["diffusion"].get("noise_schedule", "linear"))

    for batch_start in tqdm(
        range(0, len(patch_coords), infer_batch),
        desc=f"  [{D}×{H}×{W}] {len(patch_coords)} patches",
        leave=False,
    ):
        batch_coords = patch_coords[batch_start : batch_start + infer_batch]

        mr_patches   = []
        mask_patches = []
        for d0, h0, w0 in batch_coords:
            mr_patches.append(mr_arr[d0:d0+pd, h0:h0+ph, w0:w0+pw])
            mask_patches.append(mask_bool[d0:d0+pd, h0:h0+ph, w0:w0+pw].astype(np.float32))

        mr_batch   = torch.from_numpy(
            np.stack(mr_patches,   axis=0)[:, None]
        ).to(device, dtype=torch.float32)
        mask_batch = torch.from_numpy(
            np.stack(mask_patches, axis=0)[:, None]
        ).to(device, dtype=torch.float32)

        preds = diff.p_sample_loop_mask(
            model,
            tuple(mr_batch.shape),
            mr_batch,
            mask_batch,
            clip_denoised=True,
            device=device,
        )

        for i, (d0, h0, w0) in enumerate(batch_coords):
            p = preds[i, 0].cpu().numpy()
            sum_tensor  [d0:d0+pd, h0:h0+ph, w0:w0+pw] += p
            count_tensor[d0:d0+pd, h0:h0+ph, w0:w0+pw] += 1.0

    pred_vol = (sum_tensor / np.maximum(count_tensor, 1e-8)).astype(np.float32)
    pred_vol = pred_vol[:D, :H, :W]
    pred_hu  = denormalise_ct(pred_vol)

    body_mask = mask_bool[:D, :H, :W]
    pred_hu[~body_mask] = -1024.0
    pred_hu = np.clip(pred_hu, -1024, 3000).astype(np.float32)

    return pred_hu, mr_img


def save_mha(arr: np.ndarray, reference_img: sitk.Image, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = sitk.GetImageFromArray(arr.astype(np.float32))
    out_img.SetSpacing(reference_img.GetSpacing())
    out_img.SetOrigin(reference_img.GetOrigin())
    out_img.SetDirection(reference_img.GetDirection())
    sitk.WriteImage(out_img, str(out_path))


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Predict VS-DDPM] Device: {device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model = load_model(args.checkpoint, cfg, device)

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps       = args.steps or cfg["training"].get("val_ddim_steps", 100)
    overlap     = args.overlap
    infer_batch = args.infer_batch

    if args.anatomy and (input_dir / "mr.mha").exists():
        all_cases = [{"case_id": input_dir.name,
                      "anatomy": args.anatomy.upper(),
                      "path":    str(input_dir)}]
    else:
        all_cases = build_case_list(input_dir)
        if args.anatomy:
            all_cases = [c for c in all_cases
                         if c["anatomy"].upper() == args.anatomy.upper()]

    print(f"[Predict VS-DDPM] Cases to predict: {len(all_cases)}  steps={steps}")

    results = []
    for case in tqdm(all_cases, desc="Predicting"):
        case_id = case["case_id"]
        anatomy = case["anatomy"]
        path    = Path(case["path"])
        try:
            pred_hu, mr_img = predict_case(
                mr_path    = path / "mr.mha",
                mask_path  = path / "mask.mha",
                anatomy    = anatomy,
                model      = model,
                device     = device,
                cfg        = cfg,
                steps      = steps,
                overlap    = overlap,
                infer_batch = infer_batch,
            )
            out_dir = output_dir / case_id
            out_dir.mkdir(parents=True, exist_ok=True)
            save_mha(pred_hu, mr_img, out_dir / "ct.mha")
            results.append({"case_id": case_id, "status": "ok"})
            print(f"  [OK] {case_id}  shape={pred_hu.shape}")
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({"case_id": case_id, "status": f"error: {e}"})
            print(f"  [ERR] {case_id}: {e}")

    ok  = sum(1 for r in results if r["status"] == "ok")
    err = sum(1 for r in results if r["status"] != "ok")
    print(f"\n[Predict VS-DDPM] Done: {ok} OK, {err} errors → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",   required=True)
    parser.add_argument("--output-dir",  required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--config",      required=True)
    parser.add_argument("--anatomy",     default=None, choices=["HN","TH","AB","hn","th","ab"])
    parser.add_argument("--steps",       type=int,   default=None)
    parser.add_argument("--overlap",     type=float, default=0.5)
    parser.add_argument("--infer-batch", type=int,   default=1)
    main(parser.parse_args())
