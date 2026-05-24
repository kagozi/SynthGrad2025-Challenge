#!/usr/bin/env python3
"""
SynthRAD2025 Task 1 — DynUNet Per-Anatomy Inference

Loads one anatomy-specific checkpoint per anatomy (HN / TH / AB) and routes
each patient to the matching model.  A fallback checkpoint is used for any
anatomy whose per-anatomy model has not been trained yet.

Usage:
    python inference/predict_per_anatomy_dynunet.py \
        --ckpt_HN  /pvc/checkpoints/dynunet_HN/fold0_best.pth \
        --ckpt_TH  /pvc/checkpoints/dynunet_TH/fold0_best.pth \
        --ckpt_AB  /pvc/checkpoints/dynunet_AB/fold0_best.pth \
        --input_dirs \
            /pvc/data/synthRAD2025_Task1_Val_Input/Task1 \
            /pvc/data/synthRAD2025_Task1_Val_Input_D/Task1 \
        --output_dir /pvc/submissions/dynunet_per_anatomy

    # If some per-anatomy checkpoints are not yet ready, pass a fallback:
    python inference/predict_per_anatomy_dynunet.py \
        --ckpt_HN /pvc/checkpoints/dynunet_HN/fold0_best.pth \
        --fallback /pvc/checkpoints/dynunet/fold0_best.pth \
        --input_dirs /pvc/data/synthRAD2025_Task1_Val_Input/Task1 \
        --output_dir /pvc/submissions/dynunet_per_anatomy
"""

import argparse
import sys
import zipfile
from functools import partial
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import build_case_list, denormalise_ct, normalise_mr, ANATOMY_TO_IDX
from src.models.dynunet import DynUNETR3D

try:
    from monai.inferers import sliding_window_inference
except ImportError as e:
    raise ImportError("MONAI is required: pip install monai>=1.3") from e


def load_mha(path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def load_model(checkpoint_path: str, device: torch.device) -> tuple[DynUNETR3D, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    mc   = cfg["model"]
    model = DynUNETR3D(
        in_channels      = mc["in_channels"],
        n_anatomy        = mc["n_anatomy"],
        use_anatomy      = mc["use_anatomy"],
        film_hidden      = mc["film_hidden"],
        deep_supervision = False,   # always off at inference
        res_block        = mc["res_block"],
        filters          = mc.get("filters"),
        dropout          = 0.0,
    )
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    epoch    = ckpt.get("epoch", "?")
    best_mae = ckpt.get("best_mae", float("nan"))
    print(f"  Loaded {Path(checkpoint_path).name}  epoch={epoch}  best_MAE={best_mae:.2f}")
    return model, cfg


def _predictor(x, model, anatomy_t):
    B  = x.shape[0]
    ai = anatomy_t.expand(B)
    return model(x, ai)


@torch.no_grad()
def predict_case(model, mr_norm: np.ndarray, anatomy: str,
                 device: torch.device, roi_size, sw_batch_size: int,
                 overlap: float, mode: str) -> np.ndarray:
    """Sliding-window 3D inference → normalised (D,H,W) in [-1,1]."""
    mr_t      = torch.from_numpy(mr_norm[None, None]).float().to(device)
    anatomy_t = torch.tensor([ANATOMY_TO_IDX[anatomy]], dtype=torch.long, device=device)
    pred = sliding_window_inference(
        inputs        = mr_t,
        roi_size      = roi_size,
        sw_batch_size = sw_batch_size,
        predictor     = partial(_predictor, model=model, anatomy_t=anatomy_t),
        overlap       = overlap,
        mode          = mode,
    )
    return pred.squeeze().cpu().numpy()


def save_mha(arr: np.ndarray, reference_path: Path, out_path: Path):
    ref = sitk.ReadImage(str(reference_path))
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetSpacing(ref.GetSpacing())
    img.SetOrigin(ref.GetOrigin())
    img.SetDirection(ref.GetDirection())
    sitk.WriteImage(img, str(out_path), True)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PerAnatomy-DynUNet] Device: {device}")

    # Build anatomy → model routing table
    # Per-anatomy checkpoints take priority; fallback fills any gaps.
    anatomy_model: dict[str, tuple] = {}   # anatomy → (model, cfg)

    for anat, ckpt_path in [("HN", args.ckpt_HN), ("TH", args.ckpt_TH), ("AB", args.ckpt_AB)]:
        if ckpt_path and Path(ckpt_path).exists():
            print(f"[PerAnatomy-DynUNet] Loading {anat} model: {ckpt_path}")
            anatomy_model[anat] = load_model(ckpt_path, device)
        elif args.fallback and Path(args.fallback).exists():
            print(f"[PerAnatomy-DynUNet] {anat}: no checkpoint provided → using fallback")
            if "__fallback__" not in anatomy_model:
                anatomy_model["__fallback__"] = load_model(args.fallback, device)
            anatomy_model[anat] = anatomy_model["__fallback__"]
        else:
            raise FileNotFoundError(
                f"No checkpoint for anatomy {anat} and no --fallback provided."
            )

    # Remove the internal key used for deduplication
    anatomy_model.pop("__fallback__", None)

    # Use inference config from the first loaded checkpoint
    first_cfg     = next(iter(anatomy_model.values()))[1]
    ic            = first_cfg["inference"]
    roi_size      = tuple(ic["roi_size"])
    sw_batch_size = args.sw_batch_size or ic["sw_batch_size"]
    overlap       = ic["overlap"]
    mode          = ic["mode"]
    print(f"[PerAnatomy-DynUNet] roi_size={roi_size}  sw_batch={sw_batch_size}  overlap={overlap}")

    all_cases  = build_case_list(args.input_dirs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[PerAnatomy-DynUNet] Cases: {len(all_cases)}")

    ok, err = 0, 0
    for case in tqdm(all_cases, desc="Predicting"):
        case_id = case["case_id"]
        anatomy = case["anatomy"]
        path    = Path(case["path"])

        try:
            mr_norm = normalise_mr(load_mha(path / "mr.mha"), anatomy)

            model, _ = anatomy_model[anatomy]
            pred_norm = predict_case(
                model, mr_norm, anatomy, device,
                roi_size, sw_batch_size, overlap, mode,
            )
            pred_hu = np.clip(denormalise_ct(pred_norm), -1024, 3000)

            out_path = output_dir / f"sct_{case_id}.mha"
            save_mha(pred_hu, path / "mr.mha", out_path)

            tqdm.write(f"  {case_id} [{anatomy}]  shape={pred_hu.shape}  "
                       f"HU=[{pred_hu.min():.0f},{pred_hu.max():.0f}]")
            ok += 1

        except Exception as e:
            tqdm.write(f"  [ERROR] {case_id}: {e}")
            err += 1

    print(f"\n[PerAnatomy-DynUNet] Done: {ok} OK, {err} errors")

    zip_path  = output_dir.parent / f"{output_dir.name}.zip"
    mha_files = sorted(output_dir.glob("sct_*.mha"))
    print(f"[PerAnatomy-DynUNet] Zipping {len(mha_files)} files → {zip_path}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in mha_files:
            zf.write(f, f.name)
    print(f"[PerAnatomy-DynUNet] Zip: {zip_path}  ({zip_path.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_HN",       type=str, default=None,
                        help="Head & Neck per-anatomy checkpoint")
    parser.add_argument("--ckpt_TH",       type=str, default=None,
                        help="Thorax per-anatomy checkpoint")
    parser.add_argument("--ckpt_AB",       type=str, default=None,
                        help="Abdomen per-anatomy checkpoint")
    parser.add_argument("--fallback",      type=str, default=None,
                        help="Multi-anatomy checkpoint used when a per-anatomy one is missing")
    parser.add_argument("--input_dirs",    nargs="+", required=True)
    parser.add_argument("--output_dir",    required=True)
    parser.add_argument("--sw_batch_size", type=int, default=None)
    args = parser.parse_args()

    if not any([args.ckpt_HN, args.ckpt_TH, args.ckpt_AB]) and not args.fallback:
        parser.error("Provide at least one of --ckpt_HN/TH/AB or --fallback")

    main(args)
