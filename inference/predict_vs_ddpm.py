#!/usr/bin/env python3
"""
SynthRAD2025 Task 1 — VS-DDPM 3D Sliding Window Inference

Performs overlapping patch-based DDIM inference for a single case
and saves the predicted sCT as a .mha file, preserving original metadata.

Usage:
    python inference/predict_vs_ddpm.py \
        --input-dir  /pvc/data/.../1HNA001 \
        --output-dir /pvc/predictions/vs_ddpm/fold0 \
        --checkpoint /pvc/checkpoints/vs_ddpm_3d/fold0_best.pth \
        --config     training/configs/vs_ddpm_3d.yaml \
        --anatomy    HN \
        --steps      20

    # Or predict all cases in a directory tree:
    python inference/predict_vs_ddpm.py \
        --input-dir  /pvc/data/synthRAD2025_Task1_Train/Task1 \
        --output-dir /pvc/predictions/vs_ddpm/fold0 \
        --checkpoint /pvc/checkpoints/vs_ddpm_3d/fold0_best.pth \
        --config     training/configs/vs_ddpm_3d.yaml \
        --steps      20
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
    normalise_mr,
    ANATOMY_TO_IDX,
)
from src.models.vs_ddpm_3d import DDPMUNet3D, GaussianDiffusion3D


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model_from_checkpoint(
    checkpoint_path: str,
    cfg: dict,
    device: torch.device,
) -> GaussianDiffusion3D:
    """Build GaussianDiffusion3D from cfg, load state dict from checkpoint."""
    mc = cfg["model"]
    dc = cfg["diffusion"]

    unet = DDPMUNet3D(
        in_channels  = mc["in_channels"],
        out_channels = mc["out_channels"],
        base_ch      = mc["base_ch"],
        time_emb_dim = mc["time_emb_dim"],
        n_anatomy    = mc["n_anatomy"],
        dropout      = mc.get("dropout", 0.0),
    )
    diffusion = GaussianDiffusion3D(
        model       = unet,
        T           = dc["T"],
        s           = dc.get("s", 0.008),
        lambda_vlb  = dc.get("lambda_vlb",  0.001),
        lambda_mae  = dc.get("lambda_mae",  1.0),
        lambda_ssim = dc.get("lambda_ssim", 1.0),
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Support both raw state_dict and wrapped checkpoint
    state = ckpt.get("model", ckpt)
    diffusion.load_state_dict(state)
    diffusion.eval()
    diffusion.to(device)

    epoch   = ckpt.get("epoch", "?")
    best_mae = ckpt.get("best_mae", None)
    msg = f"[Predict] Loaded checkpoint (epoch {epoch}"
    if best_mae is not None:
        msg += f", best MAE={best_mae:.2f}"
    print(msg + ")")
    return diffusion


# ── Sliding window helpers ─────────────────────────────────────────────────────

def _compute_patch_starts(vol_size: int, patch_size: int, overlap: float) -> list:
    """
    Compute start indices for overlapping 1D sliding window.
    The last window is always placed so it ends at vol_size (may overlap more).
    """
    step = max(1, int(patch_size * (1.0 - overlap)))
    starts = list(range(0, max(1, vol_size - patch_size + 1), step))
    if not starts or starts[-1] + patch_size < vol_size:
        starts.append(max(0, vol_size - patch_size))
    return starts


@torch.no_grad()
def predict_case(
    mr_path:         Path,
    mask_path:       Optional[Path],
    anatomy:         str,
    diffusion:       GaussianDiffusion3D,
    device:          torch.device,
    cfg:             dict,
    ddim_steps:      int   = 20,
    overlap:         float = 0.5,
    infer_batch:     int   = 4,
) -> tuple[np.ndarray, sitk.Image]:
    """
    Sliding window DDIM inference for a single case.

    Returns:
        pred_hu   : (D, H, W) float32 array in HU
        mr_img    : original SimpleITK image (for metadata copying)
    """
    patch_size = tuple(cfg["data"].get("patch_size", [32, 128, 128]))
    pd, ph, pw  = patch_size

    # ── Load and normalise MR ─────────────────────────────────────────────────
    mr_img  = sitk.ReadImage(str(mr_path))
    mr_raw  = sitk.GetArrayFromImage(mr_img).astype(np.float32)  # (D, H, W)

    if mask_path is not None and mask_path.exists():
        mask_raw  = sitk.GetArrayFromImage(
            sitk.ReadImage(str(mask_path))
        ).astype(np.float32)
        mask_bool = (mask_raw > 0)
    else:
        mask_bool = (mr_raw > 0)

    mr_arr = normalise_mr(mr_raw, anatomy, mask=mask_bool)   # (D, H, W) in [0,1]

    D, H, W = mr_arr.shape

    # ── Pad volume to at least patch_size ────────────────────────────────────
    pad_d = max(0, pd - D)
    pad_h = max(0, ph - H)
    pad_w = max(0, pw - W)

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        mr_arr    = np.pad(mr_arr,    ((0, pad_d), (0, pad_h), (0, pad_w)))
        mask_bool = np.pad(mask_bool, ((0, pad_d), (0, pad_h), (0, pad_w)))

    Dp, Hp, Wp = mr_arr.shape

    # ── Accumulation buffers ─────────────────────────────────────────────────
    sum_tensor   = np.zeros((Dp, Hp, Wp), dtype=np.float64)
    count_tensor = np.zeros((Dp, Hp, Wp), dtype=np.float64)

    # ── Build list of all patch coordinates ──────────────────────────────────
    d_starts = _compute_patch_starts(Dp, pd, overlap)
    h_starts = _compute_patch_starts(Hp, ph, overlap)
    w_starts = _compute_patch_starts(Wp, pw, overlap)

    patch_coords = [
        (d0, h0, w0)
        for d0 in d_starts
        for h0 in h_starts
        for w0 in w_starts
    ]

    anat_tensor = torch.tensor([ANATOMY_TO_IDX[anatomy]], device=device)

    # ── Batch inference ───────────────────────────────────────────────────────
    for batch_start in tqdm(
        range(0, len(patch_coords), infer_batch),
        desc=f"  sliding window [{D}×{H}×{W}]",
        leave=False,
    ):
        batch_coords = patch_coords[batch_start : batch_start + infer_batch]
        batch_patches = []

        for d0, h0, w0 in batch_coords:
            patch = mr_arr[d0:d0+pd, h0:h0+ph, w0:w0+pw]
            # Crop back if volume was exactly divisible (shouldn't happen but guard)
            patch = patch[:pd, :ph, :pw]
            batch_patches.append(patch)

        # Stack → (B, 1, pd, ph, pw)
        mr_batch = torch.from_numpy(
            np.stack(batch_patches, axis=0)[:, None]
        ).to(device, dtype=torch.float32)

        B_cur      = mr_batch.shape[0]
        anat_batch = anat_tensor.expand(B_cur)

        preds = diffusion.ddim_sample(
            mr_batch, anat_batch, steps=ddim_steps, eta=0.0
        )  # (B, 1, pd, ph, pw) in [-1,1]

        # Accumulate
        for i, (d0, h0, w0) in enumerate(batch_coords):
            p = preds[i, 0].cpu().numpy()   # (pd, ph, pw)
            sum_tensor[d0:d0+pd, h0:h0+ph, w0:w0+pw]   += p
            count_tensor[d0:d0+pd, h0:h0+ph, w0:w0+pw] += 1.0

    # ── Average predictions ───────────────────────────────────────────────────
    pred_volume = (sum_tensor / np.maximum(count_tensor, 1e-8)).astype(np.float32)

    # ── Crop back to original volume size ─────────────────────────────────────
    pred_volume = pred_volume[:D, :H, :W]

    # ── Denormalise to HU ────────────────────────────────────────────────────
    pred_hu = denormalise_ct(pred_volume)

    # ── Apply body mask (background → -1024 HU) ──────────────────────────────
    body_mask = mask_bool[:D, :H, :W]
    pred_hu[~body_mask] = -1024.0

    # ── Clip to physiologically plausible range ───────────────────────────────
    pred_hu = np.clip(pred_hu, -1024, 3000).astype(np.float32)

    return pred_hu, mr_img


# ── Save as .mha preserving original metadata ──────────────────────────────────

def save_mha(arr: np.ndarray, reference_img: sitk.Image, out_path: Path):
    """Save prediction with spacing/origin/direction copied from reference."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = sitk.GetImageFromArray(arr.astype(np.float32))
    out_img.SetSpacing(reference_img.GetSpacing())
    out_img.SetOrigin(reference_img.GetOrigin())
    out_img.SetDirection(reference_img.GetDirection())
    sitk.WriteImage(out_img, str(out_path))


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Predict VS-DDPM] Device: {device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    diffusion = load_model_from_checkpoint(args.checkpoint, cfg, device)

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ddim_steps  = args.steps  or cfg["training"].get("infer_ddim_steps", 20)
    overlap     = args.overlap
    infer_batch = args.infer_batch

    # ── Build case list ───────────────────────────────────────────────────────
    # If --anatomy is given and input_dir is a single case, wrap it directly
    if args.anatomy and (input_dir / "mr.mha").exists():
        import re
        case_id_re = re.compile(r"(\d)(HN|TH|AB)([A-E])(\d+)", re.I)
        m = case_id_re.match(input_dir.name)
        all_cases = [{
            "case_id": input_dir.name,
            "anatomy": args.anatomy.upper(),
            "path":    str(input_dir),
        }]
    else:
        all_cases = build_case_list(input_dir)
        if args.anatomy:
            all_cases = [c for c in all_cases
                         if c["anatomy"].upper() == args.anatomy.upper()]

    print(f"[Predict VS-DDPM] Cases to predict: {len(all_cases)}")

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
                diffusion  = diffusion,
                device     = device,
                cfg        = cfg,
                ddim_steps = ddim_steps,
                overlap    = overlap,
                infer_batch = infer_batch,
            )

            out_case_dir = output_dir / case_id
            out_case_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_case_dir / "ct.mha"
            save_mha(pred_hu, mr_img, out_path)

            results.append({
                "case_id": case_id,
                "anatomy": anatomy,
                "status":  "ok",
                "shape":   str(pred_hu.shape),
            })
            print(f"  [OK] {case_id}  shape={pred_hu.shape}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [ERROR] {case_id}: {e}")
            results.append({"case_id": case_id, "anatomy": anatomy, "status": f"error: {e}"})

    ok  = [r for r in results if r["status"] == "ok"]
    err = [r for r in results if r["status"] != "ok"]
    print(f"\n[Predict VS-DDPM] Done: {len(ok)} OK, {len(err)} errors")
    print(f"[Predict VS-DDPM] Output: {output_dir}")

    if err:
        print("[Predict VS-DDPM] Errors:")
        for r in err:
            print(f"  {r['case_id']}: {r['status']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VS-DDPM 3D sliding window inference")
    parser.add_argument("--input-dir",    type=str, required=True,
                        help="Root dir with cases (or single case dir if --anatomy set)")
    parser.add_argument("--output-dir",   type=str, required=True,
                        help="Output directory for predicted sCT .mha files")
    parser.add_argument("--checkpoint",   type=str, required=True,
                        help="Path to .pth checkpoint file")
    parser.add_argument("--config",       type=str, required=True,
                        help="Path to vs_ddpm_3d.yaml config file")
    parser.add_argument("--anatomy",      type=str, default=None,
                        choices=["HN", "TH", "AB", "hn", "th", "ab"],
                        help="Filter to a single anatomy (optional)")
    parser.add_argument("--steps",        type=int, default=None,
                        help="DDIM steps (default: from config infer_ddim_steps=20)")
    parser.add_argument("--overlap",      type=float, default=0.5,
                        help="Patch overlap fraction (default: 0.5)")
    parser.add_argument("--infer-batch",  type=int, default=4,
                        help="Number of patches to process per DDIM batch (default: 4)")
    args = parser.parse_args()
    main(args)
