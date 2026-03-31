#!/usr/bin/env python3
"""
SynthRAD2025 Task 1 — Grand Challenge submission entrypoint.

Extends BaseSynthradAlgorithm from the official template:
  https://github.com/SynthRAD2025/algorithm-template

I/O (handled by base class):
  Input MR   : /input/images/mri/<uuid>.mha
  Input mask : /input/images/body/<uuid>.mha
  Region     : /input/region.json
  Output sCT : /output/images/synthetic-ct/<uuid>.mha
"""

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import SimpleITK as sitk
import torch

sys.path.insert(0, "/opt/app")

from src.dataset import ANATOMY_TO_IDX, denormalise_ct
from src.models.unet2d import AttentionUNet2D, UNet2D
from base_algorithm import BaseSynthradAlgorithm

MODEL_PATH = Path("/opt/ml/model/model.pth")

# MR normalisation constants (must match training — src/dataset.py)
MR_CLIP_LO = 0.0
MR_CLIP_HI = 2500.0


class SynthradAlgorithm(BaseSynthradAlgorithm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SynthradAlgorithm] Device: {self.device}")
        self._load_model()

    def _load_model(self):
        ckpt = torch.load(str(MODEL_PATH), map_location=self.device, weights_only=False)
        cfg  = ckpt["config"]["model"]
        kwargs = dict(
            in_channels   = cfg["in_channels"],
            out_channels  = cfg["out_channels"],
            base_features = cfg["base_features"],
            depth         = cfg["depth"],
            n_anatomy     = cfg["n_anatomy"],
            use_anatomy   = cfg["use_anatomy"],
        )
        self.model = (
            AttentionUNet2D(**kwargs) if cfg["name"] == "attention_unet2d"
            else UNet2D(**kwargs)
        )
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.model.to(self.device)
        print("[SynthradAlgorithm] Model loaded.")

    def _detect_anatomy(self, region) -> str:
        """Extract HN / TH / AB from region.json content (string or dict)."""
        region_str = str(region).upper()
        for anat in ["HN", "TH", "AB"]:
            if anat in region_str:
                return anat
        raise ValueError(f"Cannot determine anatomy from region.json: {region}")

    @torch.no_grad()
    def predict(self, input_dict: Dict[str, sitk.Image]) -> sitk.Image:
        assert list(input_dict.keys()) == ["image", "mask", "region"]

        mr_sitk = input_dict["image"]
        region  = input_dict["region"]
        anatomy = self._detect_anatomy(region)
        print(f"[predict] anatomy={anatomy}")

        # ── Normalise MR (clip → [0,1]) ────────────────────────────────────────
        mr_np = sitk.GetArrayFromImage(mr_sitk).astype(np.float32)
        mr_np = np.clip(mr_np, MR_CLIP_LO, MR_CLIP_HI)
        mr_np = (mr_np - MR_CLIP_LO) / (MR_CLIP_HI - MR_CLIP_LO + 1e-8)

        # ── Slice-by-slice inference ────────────────────────────────────────────
        anat_idx = torch.tensor([ANATOMY_TO_IDX[anatomy]], dtype=torch.long,
                                device=self.device)
        D = mr_np.shape[0]
        batch_size = 16
        pred_slices = []

        for start in range(0, D, batch_size):
            end   = min(start + batch_size, D)
            batch = torch.from_numpy(mr_np[start:end]).unsqueeze(1).to(self.device)
            ai    = anat_idx.expand(batch.size(0))
            out   = self.model(batch, ai)          # (B, 1, H, W)
            pred_slices.append(out.squeeze(1).cpu().numpy())

        pred_norm = np.concatenate(pred_slices, axis=0)   # (D, H, W)
        pred_hu   = denormalise_ct(pred_norm)
        pred_hu   = np.clip(pred_hu, -1024, 3000).astype(np.float32)

        # ── Wrap in SimpleITK, copy metadata ───────────────────────────────────
        out_sitk = sitk.GetImageFromArray(pred_hu)
        out_sitk.CopyInformation(mr_sitk)
        return out_sitk


if __name__ == "__main__":
    SynthradAlgorithm().process()
