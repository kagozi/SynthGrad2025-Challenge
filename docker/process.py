#!/usr/bin/env python3
"""
SynthRAD2025 Task 1 — Grand Challenge submission entrypoint.

Extends BaseSynthradAlgorithm from the official template:
  https://github.com/SynthRAD2025/algorithm-template

Ensemble inference:
  Loads all fold checkpoints found under /opt/ml/model/ and averages their
  predictions at the logit level (before Tanh denormalisation).  The number
  of folds and model architecture are auto-detected from each checkpoint's
  saved config — so a mixed Swin+DynUNet ensemble works without code changes.

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

from src.dataset import ANATOMY_TO_IDX, denormalise_ct
from base_algorithm import BaseSynthradAlgorithm

MODEL_DIR = Path("/opt/ml/model")

# MR normalisation constants (must match training — src/dataset.py)
MR_CLIP_LO = 0.0
MR_CLIP_HI = 2500.0

# Sliding-window inference settings (matched to training configs)
SW_ROI_SIZE   = (64, 128, 128)   # (D, H, W) patch
SW_BATCH_SIZE = 2
SW_OVERLAP    = 0.5


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

    def _load_models(self):
        """Load every *.pth checkpoint in MODEL_DIR as one ensemble member."""
        ckpt_paths = sorted(MODEL_DIR.glob("*.pth"))
        if not ckpt_paths:
            raise FileNotFoundError(
                f"No *.pth checkpoints found in {MODEL_DIR}. "
                "Did you forget to bake weights into the Docker image?"
            )

        self.models: List[torch.nn.Module] = []
        self.model_types: List[str] = []   # "2d" or "3d" per model

        for path in ckpt_paths:
            print(f"[SynthradAlgorithm] Loading {path.name} …")
            ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
            cfg  = ckpt["config"]["model"]

            model = _build_model(cfg)
            model.load_state_dict(ckpt["model"])
            model.eval()
            model.to(self.device)

            self.models.append(model)
            self.model_types.append("2d" if cfg["name"] in ("unet2d", "attention_unet2d") else "3d")
            print(f"  → {cfg['name']} loaded OK")

        print(f"[SynthradAlgorithm] Ensemble of {len(self.models)} model(s) ready.")

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

        mr_sitk = input_dict["image"]
        region  = input_dict["region"]
        anatomy = self._detect_anatomy(region)
        print(f"[predict] anatomy={anatomy}")

        # Normalise MR
        mr_np = sitk.GetArrayFromImage(mr_sitk).astype(np.float32)
        mr_np = np.clip(mr_np, MR_CLIP_LO, MR_CLIP_HI)
        mr_np = (mr_np - MR_CLIP_LO) / (MR_CLIP_HI - MR_CLIP_LO + 1e-8)

        anat_idx = torch.tensor([ANATOMY_TO_IDX[anatomy]], dtype=torch.long,
                                device=self.device)

        # Run each model and collect predictions (all in normalised [-1,1] space)
        preds = []
        for model, mtype in zip(self.models, self.model_types):
            if mtype == "2d":
                p = self._infer_2d(model, mr_np, anat_idx)
            else:
                p = self._infer_3d(model, mr_np, anat_idx)
            preds.append(p)

        # Average ensemble predictions (in normalised space, before HU conversion)
        pred_norm = np.mean(preds, axis=0)   # (D, H, W)

        # Denormalise to HU
        pred_hu = denormalise_ct(pred_norm)
        pred_hu = np.clip(pred_hu, -1024, 3000).astype(np.float32)

        out_sitk = sitk.GetImageFromArray(pred_hu)
        out_sitk.CopyInformation(mr_sitk)
        return out_sitk


if __name__ == "__main__":
    SynthradAlgorithm().process()
