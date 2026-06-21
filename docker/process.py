#!/usr/bin/env python3
"""
SynthRAD2025 Task 1 — Grand Challenge submission entrypoint.

Extends BaseSynthradAlgorithm from the official template:
  https://github.com/SynthRAD2025/algorithm-template

Routing strategy (auto-detected from checkpoint filenames in /opt/ml/model/):

  Per-anatomy mode  — checkpoint filename contains "HN", "TH", or "AB"
    e.g. HN.pth, dynunet_TH.pth, AB_best.pth
    → each anatomy gets its own model; no cross-anatomy averaging
    → any anatomy without a tagged checkpoint falls back to untagged models

  Ensemble mode (legacy / fallback) — checkpoints have no anatomy tag
    e.g. fold0_best.pth, fold1_best.pth …
    → all models averaged for every anatomy (original behaviour)

  Mixed mode — some anatomies have tagged checkpoints, others don't
    → tagged anatomy uses its own model; untagged anatomy uses untagged ensemble

  Within each anatomy, multiple checkpoints are averaged (fold ensemble).

I/O (handled by base class):
  Input MR   : /input/images/mri/<uuid>.mha
  Input mask : /input/images/body/<uuid>.mha
  Region     : /input/region.json
  Output sCT : /output/images/synthetic-ct/<uuid>.mha
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

sys.path.insert(0, "/opt/app")

from src.dataset import ANATOMY_TO_IDX, denormalise_ct, normalise_mr
from base_algorithm import BaseSynthradAlgorithm

MODEL_DIR = Path("/opt/ml/model")

# Sliding-window inference settings (matched to training configs)
SW_ROI_SIZE   = (64, 128, 128)   # (D, H, W) patch
SW_BATCH_SIZE = 2
SW_OVERLAP    = 0.75


def _build_model(cfg: dict):
    """Instantiate the correct model class from a checkpoint config dict."""
    name = cfg["name"]

    if name in ("unet2d", "attention_unet2d"):
        from src.models.unet2d import AttentionUNet2D, UNet2D
        kwargs = dict(
            in_channels   = cfg["in_channels"],
            out_channels  = cfg.get("out_channels", 1),
            base_features = cfg["base_features"],
            depth         = cfg["depth"],
            n_anatomy     = cfg["n_anatomy"],
            use_anatomy   = cfg["use_anatomy"],
        )
        return AttentionUNet2D(**kwargs) if name == "attention_unet2d" else UNet2D(**kwargs)

    if name == "swin_unetr":
        from src.models.swin_unetr import SwinUNETR3D
        return SwinUNETR3D(
            in_channels        = cfg.get("in_channels", 1),
            feature_size       = cfg.get("feature_size", 48),
            n_anatomy          = cfg.get("n_anatomy", 3),
            use_anatomy        = cfg.get("use_anatomy", True),
            film_hidden        = cfg.get("film_hidden", 128),
            use_checkpoint     = False,   # never checkpoint during inference
            drop_rate          = 0.0,
            attn_drop_rate     = 0.0,
            dropout_path_rate  = 0.0,
        )

    if name == "dynunet":
        from src.models.dynunet import DynUNETR3D
        return DynUNETR3D(
            in_channels        = cfg.get("in_channels", 1),
            n_anatomy          = cfg.get("n_anatomy", 3),
            use_anatomy        = cfg.get("use_anatomy", True),
            film_hidden        = cfg.get("film_hidden", 128),
            deep_supervision   = False,   # always off at inference
            res_block          = cfg.get("res_block", True),
            filters            = cfg.get("filters", None),
            dropout            = 0.0,
        )

    raise ValueError(f"Unknown model name in checkpoint config: {name!r}")


class SynthradAlgorithm(BaseSynthradAlgorithm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SynthradAlgorithm] Device: {self.device}")
        self._load_models()

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_one(self, path: Path) -> tuple:
        """Load a single checkpoint; return (model, model_type_str)."""
        print(f"[SynthradAlgorithm] Loading {path.name} …")
        ckpt  = torch.load(str(path), map_location=self.device, weights_only=False)
        cfg   = ckpt["config"]["model"]
        model = _build_model(cfg)
        model.load_state_dict(ckpt["model"])
        model.eval().to(self.device)
        mtype = "2d" if cfg["name"] in ("unet2d", "attention_unet2d") else "3d"
        print(f"  → {cfg['name']} ({mtype}) loaded OK")
        return model, mtype

    def _load_models(self):
        """
        Build per-anatomy model lists from MODEL_DIR/*.pth.

        Filenames containing "HN", "TH", or "AB" (case-insensitive) are
        assigned only to that anatomy.  Untagged checkpoints are assigned to
        all three anatomies as a fallback ensemble.

        self.anatomy_models["HN"] → list of models used for Head & Neck
        self.anatomy_types["HN"]  → corresponding "2d"/"3d" strings
        """
        ckpt_paths = sorted(MODEL_DIR.glob("*.pth"))
        if not ckpt_paths:
            raise FileNotFoundError(
                f"No *.pth checkpoints found in {MODEL_DIR}. "
                "Did you forget to bake weights into the Docker image?"
            )

        self.anatomy_models: Dict[str, List[torch.nn.Module]] = {"HN": [], "TH": [], "AB": []}
        self.anatomy_types:  Dict[str, List[str]]             = {"HN": [], "TH": [], "AB": []}

        for path in ckpt_paths:
            tagged = [a for a in ("HN", "TH", "AB") if a in path.stem.upper()]
            targets = tagged if tagged else ["HN", "TH", "AB"]
            model, mtype = self._load_one(path)
            for anat in targets:
                self.anatomy_models[anat].append(model)
                self.anatomy_types[anat].append(mtype)

        for anat in ("HN", "TH", "AB"):
            n = len(self.anatomy_models[anat])
            tag = "per-anatomy" if any(anat in p.stem.upper() for p in ckpt_paths) else "fallback"
            print(f"[SynthradAlgorithm] {anat}: {n} model(s) [{tag}]")

    # ── Anatomy detection ──────────────────────────────────────────────────────

    def _detect_anatomy(self, region) -> str:
        region_str = str(region).upper()
        for anat in ["HN", "TH", "AB"]:
            if anat in region_str:
                return anat
        raise ValueError(f"Cannot determine anatomy from region.json: {region}")

    # ── Per-model inference helpers ────────────────────────────────────────────

    @torch.no_grad()
    def _infer_2d(self, model, mr_np: np.ndarray, anat_idx: torch.Tensor) -> np.ndarray:
        """Slice-by-slice inference for 2D models. Returns (D,H,W) in [-1,1]."""
        D = mr_np.shape[0]
        batch_size = 16
        pred_slices = []
        for start in range(0, D, batch_size):
            end   = min(start + batch_size, D)
            batch = torch.from_numpy(mr_np[start:end]).unsqueeze(1).to(self.device)
            ai    = anat_idx.expand(batch.size(0))
            out   = model(batch, ai)          # (B, 1, H, W)
            pred_slices.append(out.squeeze(1).cpu().numpy())
        return np.concatenate(pred_slices, axis=0)   # (D, H, W)

    @torch.no_grad()
    def _infer_3d(self, model, mr_np: np.ndarray, anat_idx: torch.Tensor) -> np.ndarray:
        """Sliding-window 3D inference. Returns (D,H,W) in [-1,1]."""
        from monai.inferers import sliding_window_inference
        from functools import partial

        mr_t = torch.from_numpy(mr_np).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,D,H,W)

        def _predictor(patches, model, anatomy_t):
            B = patches.shape[0]
            ai = anatomy_t.expand(B)
            return model(patches, ai)

        predictor = partial(_predictor, model=model, anatomy_t=anat_idx)

        pred = sliding_window_inference(
            inputs        = mr_t,
            roi_size      = SW_ROI_SIZE,
            sw_batch_size = SW_BATCH_SIZE,
            predictor     = predictor,
            overlap       = SW_OVERLAP,
            mode          = "gaussian",
        )
        return pred.squeeze(0).squeeze(0).cpu().numpy()   # (D, H, W)

    # ── Main prediction entry point ────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, input_dict: Dict[str, sitk.Image]) -> sitk.Image:
        assert list(input_dict.keys()) == ["image", "mask", "region"]

        mr_sitk   = input_dict["image"]
        mask_sitk = input_dict["mask"]
        region    = input_dict["region"]
        anatomy   = self._detect_anatomy(region)
        print(f"[predict] anatomy={anatomy}")

        # Body mask — used to suppress background after inference
        mask_np = sitk.GetArrayFromImage(mask_sitk).astype(bool)

        # Normalise MR — per-case z-score (matches training, handles 0.35T–3T variation)
        mr_np = sitk.GetArrayFromImage(mr_sitk).astype(np.float32)
        mr_np = normalise_mr(mr_np, anatomy)

        anat_idx = torch.tensor([ANATOMY_TO_IDX[anatomy]], dtype=torch.long,
                                device=self.device)

        # Run anatomy-specific model(s) with left-right TTA.
        # For each model: infer on original + LR-flipped MR, flip prediction back,
        # then average all 2×N results.  LR is axis=2 (W) in (D,H,W) layout.
        mr_lr = np.flip(mr_np, axis=2).copy()
        preds = []
        for mr_in, flip_back in ((mr_np, False), (mr_lr, True)):
            for model, mtype in zip(self.anatomy_models[anatomy], self.anatomy_types[anatomy]):
                if mtype == "2d":
                    p = self._infer_2d(model, mr_in, anat_idx)
                else:
                    p = self._infer_3d(model, mr_in, anat_idx)
                if flip_back:
                    p = np.flip(p, axis=2).copy()
                preds.append(p)

        pred_norm = np.mean(preds, axis=0)   # (D, H, W)

        # Denormalise to HU and suppress background (every top team does this)
        pred_hu = denormalise_ct(pred_norm)
        pred_hu = np.clip(pred_hu, -1024, 3000).astype(np.float32)
        pred_hu[~mask_np] = -1024.0

        out_sitk = sitk.GetImageFromArray(pred_hu)
        out_sitk.CopyInformation(mr_sitk)
        return out_sitk


if __name__ == "__main__":
    SynthradAlgorithm().process()
