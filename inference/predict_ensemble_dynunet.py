#!/usr/bin/env python3
"""
SynthRAD2025 Task 1 — DynUNet 3D 5-Fold Ensemble Inference

Loads all 5 DynUNet fold checkpoints, averages normalised predictions via
sliding window inference, saves flat submission files as:
    {output_dir}/sct_{case_id}.mha
    {output_dir}/../dynunet_ensemble.zip

Usage:
    python inference/predict_ensemble_dynunet.py \
        --checkpoints \
            /pvc/checkpoints/dynunet/fold0_best.pth \
            /pvc/checkpoints/dynunet/fold1_best.pth \
            /pvc/checkpoints/dynunet/fold2_best.pth \
            /pvc/checkpoints/dynunet/fold3_best.pth \
            /pvc/checkpoints/dynunet/fold4_best.pth \
        --input_dirs \
            /pvc/data/synthRAD2025_Task1_Val_Input/Task1 \
            /pvc/data/synthRAD2025_Task1_Val_Input_D/Task1 \
        --output_dir /pvc/submissions/dynunet_ensemble
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
        deep_supervision = mc["deep_supervision"],
        deep_supr_num    = mc["deep_supr_num"],
        res_block        = mc["res_block"],
        filters          = mc.get("filters"),
        dropout          = mc.get("dropout", 0.0),
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
def predict_case_norm(model, mr_norm: np.ndarray, anatomy: str,
                      device: torch.device, roi_size, sw_batch_size: int,
                      overlap: float, mode: str) -> np.ndarray:
    """Run 3D sliding-window inference; return normalised pred (D, H, W)."""
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
    print(f"[Ensemble-DynUNet] Device: {device}")

    print(f"[Ensemble-DynUNet] Loading {len(args.checkpoints)} checkpoints...")
    models_cfgs = [load_model(ckpt, device) for ckpt in args.checkpoints]
    models = [m for m, _ in models_cfgs]

    # Use inference config from the first checkpoint
    ic            = models_cfgs[0][1]["inference"]
    roi_size      = tuple(ic["roi_size"])
    sw_batch_size = args.sw_batch_size or ic["sw_batch_size"]
    overlap       = ic["overlap"]
    mode          = ic["mode"]
    print(f"[Ensemble-DynUNet] roi_size={roi_size}  sw_batch={sw_batch_size}  overlap={overlap}")

    all_cases = build_case_list(args.input_dirs)
    print(f"[Ensemble-DynUNet] Cases found: {len(all_cases)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ok, err = 0, 0
    for case in tqdm(all_cases, desc="Predicting"):
        case_id = case["case_id"]
        anatomy = case["anatomy"]
        path    = Path(case["path"])

        try:
            mr_norm = normalise_mr(load_mha(path / "mr.mha"), anatomy)

            pred_norm = np.zeros(0, dtype=np.float32)
            for i, model in enumerate(models):
                p = predict_case_norm(
                    model, mr_norm, anatomy, device,
                    roi_size, sw_batch_size, overlap, mode,
                )
                pred_norm = p if i == 0 else pred_norm + p
            pred_norm /= len(models)

            pred_hu = np.clip(denormalise_ct(pred_norm), -1024, 3000)

            out_path = output_dir / f"sct_{case_id}.mha"
            save_mha(pred_hu, path / "mr.mha", out_path)

            tqdm.write(f"  {case_id}  shape={pred_hu.shape}  "
                       f"HU=[{pred_hu.min():.0f},{pred_hu.max():.0f}]")
            ok += 1

        except Exception as e:
            tqdm.write(f"  [ERROR] {case_id}: {e}")
            err += 1

    print(f"\n[Ensemble-DynUNet] Done: {ok} OK, {err} errors")
    print(f"[Ensemble-DynUNet] Output: {output_dir}")

    zip_path  = output_dir.parent / f"{output_dir.name}.zip"
    mha_files = sorted(output_dir.glob("sct_*.mha"))
    print(f"[Ensemble-DynUNet] Zipping {len(mha_files)} files → {zip_path}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in mha_files:
            zf.write(f, f.name)
    print(f"[Ensemble-DynUNet] Zip ready: {zip_path}  ({zip_path.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints",   nargs="+", required=True)
    parser.add_argument("--input_dirs",    nargs="+", required=True)
    parser.add_argument("--output_dir",    required=True)
    parser.add_argument("--sw_batch_size", type=int, default=None,
                        help="Override sw_batch_size from checkpoint config")
    args = parser.parse_args()
    main(args)
