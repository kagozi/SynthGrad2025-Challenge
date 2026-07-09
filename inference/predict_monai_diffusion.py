#!/usr/bin/env python3
"""
SynthRAD2025 Task 1 — MONAI 3D Diffusion Inference

Runs DDIM sliding-window inference for a single patient case.
For each patch position, the full DDIM reverse diffusion loop runs
conditioned on the corresponding MR patch.

Usage:
    python inference/predict_monai_diffusion.py \
        --checkpoint /pvc/checkpoints/monai_diffusion_3d/fold0_best.pth \
        --case_dir   /pvc/data/.../1HNA001 \
        --anatomy    HN \
        --out_dir    /pvc/predictions/monai_diffusion \
        --ddim_steps 50
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
    load_mha, load_mha_with_meta,
    normalise_mr, normalise_ct, denormalise_ct,
    ANATOMY_TO_IDX,
)
from src.models.monai_diffusion_3d import (
    MonaiDiffusion3D, GaussianDiffusion3D, EMA,
)

try:
    from monai.inferers import sliding_window_inference
except ImportError as e:
    raise ImportError("MONAI required: pip install 'monai[all]>=1.4'") from e


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device) -> GaussianDiffusion3D:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt["cfg"]
    mc   = cfg["model"]
    dc   = cfg["diffusion"]
    lc   = cfg["loss"]

    backbone = MonaiDiffusion3D(
        in_channels       = mc["in_channels"],
        out_channels      = mc["out_channels"],
        channels          = tuple(mc["channels"]),
        attention_levels  = tuple(mc["attention_levels"]),
        num_res_blocks    = mc["num_res_blocks"],
        num_head_channels = mc["num_head_channels"],
        norm_num_groups   = mc["norm_num_groups"],
        norm_eps          = mc.get("norm_eps", 1e-6),
        n_anatomy         = mc.get("n_anatomy", 3),
        dropout_cattn     = mc.get("dropout_cattn", 0.0),
        use_flash_attention = mc.get("use_flash_attention", False),
    )
    diffusion = GaussianDiffusion3D(
        model            = backbone,
        T                = dc["T"],
        cosine_s         = dc.get("cosine_s", 0.008),
        noise_mse_weight = lc.get("noise_mse_weight", 1.0),
        mae_weight       = lc.get("mae_weight", 0.25),
        ssim_weight      = lc.get("ssim_weight", 0.25),
    )

    # Load EMA weights if available, otherwise raw model weights
    if "ema" in ckpt:
        ema = EMA(diffusion.model)
        ema.load_state_dict(ckpt["ema"])
        ema.apply_to(diffusion.model)
        print("Loaded EMA weights.")
    else:
        diffusion.load_state_dict(ckpt["model"])

    diffusion = diffusion.to(device)
    diffusion.eval()
    return diffusion


# ── Sliding window DDIM predictor ─────────────────────────────────────────────

def _make_ddim_predictor(diffusion, anatomy_idx_t, ddim_steps):
    """
    Returns a function `predict(mr_batch) → sct_batch` suitable for
    MONAI's sliding_window_inference.

    Each call receives a batch of overlapping MR patches and runs the
    full DDIM loop independently per patch.
    """
    @torch.no_grad()
    def _predict(mr_batch: torch.Tensor) -> torch.Tensor:
        B = mr_batch.shape[0]
        anat = anatomy_idx_t.expand(B).to(mr_batch.device)
        return diffusion.ddim_sample(mr_batch, anat, steps=ddim_steps)
    return _predict


# ── Case inference ────────────────────────────────────────────────────────────

def predict_case(
    diffusion:   GaussianDiffusion3D,
    case_dir:    Path,
    anatomy:     str,
    device:      torch.device,
    roi_size:    tuple = (32, 128, 128),
    sw_batch:    int   = 4,
    overlap:     float = 0.5,
    ddim_steps:  int   = 50,
) -> tuple[np.ndarray, tuple, tuple]:
    """
    Run DDIM sliding-window inference for one case.

    Returns:
        pred_hu  (D, H, W) — predicted CT in HU
        spacing  (x, y, z)
        origin   (x, y, z)
    """
    mr_path   = next(case_dir.glob("mr.mha"), None) or next(case_dir.glob("*mr*.mha"))
    mask_path = next(case_dir.glob("mask.mha"), None) or next(case_dir.glob("*mask*.mha"), None)

    mr_arr, spacing, origin = load_mha_with_meta(mr_path)
    mask_arr = load_mha(mask_path).astype(bool) if mask_path else None

    mr_norm = normalise_mr(mr_arr, anatomy=anatomy, mask=mask_arr)

    mr_t = torch.from_numpy(mr_norm).float().unsqueeze(0).unsqueeze(0).to(device)

    anatomy_idx_t = torch.tensor(
        [ANATOMY_TO_IDX[anatomy]], dtype=torch.long, device=device,
    )

    predictor = _make_ddim_predictor(diffusion, anatomy_idx_t, ddim_steps)

    with torch.no_grad():
        pred_norm = sliding_window_inference(
            inputs      = mr_t,
            roi_size    = roi_size,
            sw_batch_size = sw_batch,
            predictor   = predictor,
            overlap     = overlap,
            mode        = "gaussian",
        )

    pred_np = pred_norm[0, 0].cpu().numpy()           # (D, H, W) in [-1, 1]
    pred_hu = denormalise_ct(pred_np)                  # → HU
    pred_hu = np.clip(pred_hu, -1024, 3000)

    if mask_arr is not None:
        pred_hu[~mask_arr] = -1024

    return pred_hu, spacing, origin


# ── Save .mha ─────────────────────────────────────────────────────────────────

def save_mha(arr: np.ndarray, spacing: tuple, origin: tuple, path: Path):
    img = sitk.GetImageFromArray(arr.astype(np.int16))
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--case_dir",   required=True, help="Case directory with mr.mha")
    parser.add_argument("--anatomy",    required=True, choices=["HN", "TH", "AB"])
    parser.add_argument("--out_dir",    required=True, help="Output directory")
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--roi_size",   nargs=3, type=int, default=[32, 128, 128])
    parser.add_argument("--sw_batch",   type=int, default=4)
    parser.add_argument("--overlap",    type=float, default=0.5)
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | DDIM steps: {args.ddim_steps}")

    diffusion = load_model(args.checkpoint, device)

    case_dir = Path(args.case_dir)
    out_dir  = Path(args.out_dir)

    print(f"Predicting: {case_dir.name}  anatomy={args.anatomy}")
    pred_hu, spacing, origin = predict_case(
        diffusion   = diffusion,
        case_dir    = case_dir,
        anatomy     = args.anatomy,
        device      = device,
        roi_size    = tuple(args.roi_size),
        sw_batch    = args.sw_batch,
        overlap     = args.overlap,
        ddim_steps  = args.ddim_steps,
    )

    out_path = out_dir / case_dir.name / "synthetic_ct.mha"
    save_mha(pred_hu, spacing, origin, out_path)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
