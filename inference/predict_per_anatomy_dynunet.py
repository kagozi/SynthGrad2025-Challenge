#!/usr/bin/env python3
"""
SynthRAD2025 Task 1 — Two-Stage DynUNet Per-Anatomy Ensemble Inference

Loads all Stage-2 per-anatomy checkpoints (up to 5 folds each) and produces
a mean-ensemble prediction for every validation case, routing each patient to
its anatomy-specific fold ensemble.

Usage — full 5-fold ensemble (run after Stage 2 completes):
    python inference/predict_per_anatomy_dynunet.py \
        --ckpts_HN /pvc/checkpoints/dynunet_HN/fold{0..4}_best.pth \
        --ckpts_TH /pvc/checkpoints/dynunet_TH/fold{0..4}_best.pth \
        --ckpts_AB /pvc/checkpoints/dynunet_AB/fold{0..4}_best.pth \
        --input_dirs \
            /pvc/data/synthRAD2025_Task1_Val_Input/Task1 \
            /pvc/data/synthRAD2025_Task1_Val_Input_D/Task1 \
        --output_dir /pvc/submissions/dynunet_stage2_ensemble

Usage — partial (some folds still training):
    python inference/predict_per_anatomy_dynunet.py \
        --ckpts_HN /pvc/checkpoints/dynunet_HN/fold0_best.pth \
                   /pvc/checkpoints/dynunet_HN/fold1_best.pth \
        --ckpts_TH /pvc/checkpoints/dynunet_TH/fold0_best.pth \
        --ckpts_AB /pvc/checkpoints/dynunet_AB/fold0_best.pth \
        --fallback /pvc/checkpoints/dynunet/fold0_best.pth \
        --input_dirs /pvc/data/synthRAD2025_Task1_Val_Input/Task1 \
        --output_dir /pvc/submissions/dynunet_stage2_ensemble
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


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    mc   = cfg["model"]
    model = DynUNETR3D(
        in_channels      = mc["in_channels"],
        n_anatomy        = mc["n_anatomy"],
        use_anatomy      = mc["use_anatomy"],
        film_hidden      = mc["film_hidden"],
        deep_supervision = False,
        res_block        = mc["res_block"],
        filters          = mc.get("filters"),
        dropout          = 0.0,
    )
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    epoch    = ckpt.get("epoch", "?")
    best_mae = ckpt.get("best_mae", float("nan"))
    print(f"    {Path(checkpoint_path).name}  epoch={epoch}  best_MAE={best_mae:.2f}")
    return model, cfg


def _predictor(x, model, anatomy_t):
    return model(x, anatomy_t.expand(x.shape[0]))


@torch.no_grad()
def predict_case_single(model, mr_norm: np.ndarray, anatomy: str,
                        device: torch.device, roi_size, sw_batch_size: int,
                        overlap: float, mode: str) -> np.ndarray:
    """Sliding-window 3D inference for one model → (D,H,W) normalised [-1,1]."""
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


@torch.no_grad()
def predict_case_ensemble(models: list, mr_norm: np.ndarray, anatomy: str,
                          device: torch.device, roi_size, sw_batch_size: int,
                          overlap: float, mode: str) -> np.ndarray:
    """Mean-ensemble over all folds for one case."""
    preds = [
        predict_case_single(m, mr_norm, anatomy, device,
                            roi_size, sw_batch_size, overlap, mode)
        for m in models
    ]
    return np.mean(preds, axis=0)


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

    # Load fallback ensemble (Stage 1 multi-anatomy) if provided
    fallback_models = []
    if args.fallback:
        print(f"[Ensemble-DynUNet] Loading fallback checkpoints ...")
        for p in args.fallback:
            if Path(p).exists():
                m, cfg_fb = load_model(p, device)
                fallback_models.append(m)
            else:
                print(f"  [WARN] fallback not found: {p}")

    # Build anatomy → [models] routing table
    # ckpts_HN/TH/AB are lists; missing anatomies use fallback.
    anatomy_models: dict[str, list] = {}
    inference_cfg = None

    for anat, ckpt_list in [("HN", args.ckpts_HN),
                             ("TH", args.ckpts_TH),
                             ("AB", args.ckpts_AB)]:
        models = []
        if ckpt_list:
            print(f"[Ensemble-DynUNet] Loading {anat} ({len(ckpt_list)} fold(s)) ...")
            for p in ckpt_list:
                if Path(p).exists():
                    m, cfg = load_model(p, device)
                    models.append(m)
                    if inference_cfg is None:
                        inference_cfg = cfg
                else:
                    print(f"  [WARN] not found: {p}")

        if models:
            anatomy_models[anat] = models
        elif fallback_models:
            print(f"[Ensemble-DynUNet] {anat}: no Stage-2 checkpoints → using fallback ({len(fallback_models)} model(s))")
            anatomy_models[anat] = fallback_models
            if inference_cfg is None and args.fallback:
                _, cfg_fb = load_model(args.fallback[0], device)
                inference_cfg = cfg_fb
        else:
            raise FileNotFoundError(
                f"No checkpoints for anatomy {anat} and no --fallback provided."
            )

    for anat, ms in anatomy_models.items():
        print(f"[Ensemble-DynUNet] {anat}: {len(ms)} model(s) in ensemble")

    # Inference settings from checkpoint config
    ic            = inference_cfg["inference"]
    roi_size      = tuple(ic["roi_size"])
    sw_batch_size = args.sw_batch_size or ic.get("sw_batch_size", 2)
    overlap       = ic["overlap"]
    mode          = ic["mode"]
    print(f"[Ensemble-DynUNet] roi_size={roi_size}  sw_batch={sw_batch_size}  overlap={overlap}")

    all_cases  = build_case_list(args.input_dirs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Ensemble-DynUNet] Cases to predict: {len(all_cases)}")

    ok, err = 0, 0
    for case in tqdm(all_cases, desc="Predicting"):
        case_id = case["case_id"]
        anatomy = case["anatomy"]
        path    = Path(case["path"])

        try:
            mr_norm   = normalise_mr(load_mha(path / "mr.mha"), anatomy)
            pred_norm = predict_case_ensemble(
                anatomy_models[anatomy], mr_norm, anatomy, device,
                roi_size, sw_batch_size, overlap, mode,
            )
            pred_hu = np.clip(denormalise_ct(pred_norm), -1024, 3000)

            out_path = output_dir / f"sct_{case_id}.mha"
            save_mha(pred_hu, path / "mr.mha", out_path)

            tqdm.write(f"  {case_id} [{anatomy}] folds={len(anatomy_models[anatomy])}"
                       f"  shape={pred_hu.shape}  HU=[{pred_hu.min():.0f},{pred_hu.max():.0f}]")
            ok += 1

        except Exception as e:
            tqdm.write(f"  [ERROR] {case_id}: {e}")
            err += 1

    print(f"\n[Ensemble-DynUNet] Done: {ok} OK, {err} errors")

    zip_path  = output_dir.parent / f"{output_dir.name}.zip"
    mha_files = sorted(output_dir.glob("sct_*.mha"))
    print(f"[Ensemble-DynUNet] Zipping {len(mha_files)} files → {zip_path}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in mha_files:
            zf.write(f, f.name)
    print(f"[Ensemble-DynUNet] Zip: {zip_path}  ({zip_path.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-stage DynUNet per-anatomy ensemble inference for SynthRAD2025"
    )
    parser.add_argument("--ckpts_HN", nargs="+", default=None,
                        help="Stage-2 HN checkpoints (one per fold)")
    parser.add_argument("--ckpts_TH", nargs="+", default=None,
                        help="Stage-2 TH checkpoints (one per fold)")
    parser.add_argument("--ckpts_AB", nargs="+", default=None,
                        help="Stage-2 AB checkpoints (one per fold)")
    parser.add_argument("--fallback", nargs="+", default=None,
                        help="Stage-1 multi-anatomy checkpoints used for any missing anatomy")
    parser.add_argument("--input_dirs", nargs="+", required=True,
                        help="Val input root dirs (Task1/)")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write sct_*.mha files and the zip")
    parser.add_argument("--sw_batch_size", type=int, default=None,
                        help="Override sliding-window batch size (default: from checkpoint config)")
    args = parser.parse_args()

    if not any([args.ckpts_HN, args.ckpts_TH, args.ckpts_AB]) and not args.fallback:
        parser.error("Provide at least one of --ckpts_HN/TH/AB or --fallback")

    main(args)
