"""
SynthRAD2025 Task 1 — MONAI DynUNet (nnU-Net) wrapper

Key advantage over isotropic models:
  Anisotropic kernels [3,3,1] / strides [2,2,1] at early encoder stages
  directly respect the 1×1×3 mm dataset spacing, avoiding in-plane vs.
  through-plane confusion that hurts bone Dice and HD95 in isotropic models.

Architecture:
  - Backbone: MONAI DynUNet (nnU-Net-style 3D UNet) with anisotropic kernels
  - Deep supervision: 2 auxiliary decoder outputs during training
  - Anatomy conditioning: same FiLM residual head as SwinUNETR3D
  - No pretrained weights (trained from scratch)

Input:  (B, 1, D, H, W) — normalised MR in [0, 1]
Output training:  (main: (B,1,D,H,W), aux: [(B,1,D',H',W'), ...])
Output inference: (B, 1, D, H, W) — normalised sCT in [-1, 1]

Kernel / stride design for 1×1×3 mm spacing (1:1:3 anisotropy):
  Stage 0 (input):   kernel [3,3,1], stride [1,1,1]  — no through-plane pool
  Stage 1:           kernel [3,3,3], stride [2,2,1]  — pool in-plane only
  Stage 2:           kernel [3,3,3], stride [2,2,2]  — isotropic from here
  Stage 3:           kernel [3,3,3], stride [2,2,2]
  Stage 4 (bottom):  kernel [3,3,3], stride [2,2,2]

Usage:
    model = DynUNETR3D(n_anatomy=3, use_anatomy=True)
    # Training
    main, aux = model(mr, anatomy_idx)
    # Inference (eval mode)
    pred = model(mr, anatomy_idx)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from monai.networks.nets import DynUNet as _MonaiDynUNet
except ImportError as e:
    raise ImportError(
        "MONAI is required for DynUNet. Install with: pip install monai>=1.3"
    ) from e

from src.models.swin_unetr import AnatomyFiLMHead


# ── Anisotropic kernel / stride schedule ───────────────────────────────────────
# Designed for 1×1×3 mm spacing (axial depth is 3× coarser than in-plane).
# Stage 0: no through-plane pooling (kernel D=1, stride D=1)
# Stage 1: pool only in-plane (stride [2,2,1]) until voxels are roughly isotropic
# Stage 2+: isotropic pooling
_KERNEL_SIZE = [
    [3, 3, 1],   # stage 0 — input, no depth pooling
    [3, 3, 3],   # stage 1
    [3, 3, 3],   # stage 2
    [3, 3, 3],   # stage 3
    [3, 3, 3],   # stage 4 — bottleneck
]
_STRIDES = [
    [1, 1, 1],   # stage 0 — no downsampling
    [2, 2, 1],   # stage 1 — in-plane only
    [2, 2, 2],   # stage 2
    [2, 2, 2],   # stage 3
    [2, 2, 2],   # stage 4
]
# upsample_kernel_size = strides[1:] (MONAI convention)
_UP_KERNEL = _STRIDES[1:]


# ── Main model ─────────────────────────────────────────────────────────────────

class DynUNETR3D(nn.Module):
    """
    MONAI DynUNet with anisotropic kernels + anatomy FiLM conditioning.

    Args:
        in_channels:        Input channels (1 for single-echo MR).
        n_anatomy:          Number of anatomy classes (3: HN, TH, AB).
        use_anatomy:        Whether to apply FiLM conditioning.
        film_hidden:        Hidden channels in the FiLM residual head.
        deep_supervision:   Use auxiliary decoder outputs during training.
        deep_supr_num:      Number of auxiliary outputs (default 2).
        res_block:          Use residual blocks in DynUNet (slightly better,
                            slightly more memory).
        filters:            Override per-stage filter counts. None = MONAI
                            auto (32, 64, 128, 256, 320, …).
        dropout:            Dropout probability (0 = disabled).
    """

    def __init__(
        self,
        in_channels:      int   = 1,
        n_anatomy:        int   = 3,
        use_anatomy:      bool  = True,
        film_hidden:      int   = 128,
        deep_supervision: bool  = True,
        deep_supr_num:    int   = 2,
        res_block:        bool  = True,
        filters:          list  = None,
        dropout:          float = 0.0,
    ):
        super().__init__()
        self.use_anatomy      = use_anatomy
        self._deep_supervision = deep_supervision

        self.dynunet = _MonaiDynUNet(
            spatial_dims         = 3,
            in_channels          = in_channels,
            out_channels         = 1,
            kernel_size          = _KERNEL_SIZE,
            strides              = _STRIDES,
            upsample_kernel_size = _UP_KERNEL,
            filters              = filters,
            dropout              = dropout,
            norm_name            = ("INSTANCE", {"affine": True}),
            act_name             = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            deep_supervision     = deep_supervision,
            deep_supr_num        = deep_supr_num,
            res_block            = res_block,
        )

        if use_anatomy:
            self.film_head = AnatomyFiLMHead(hidden=film_hidden, n_anatomy=n_anatomy)
        else:
            self.out_act = nn.Tanh()

    def forward(
        self,
        x:            torch.Tensor,
        anatomy_idx:  torch.Tensor | None = None,
    ):
        """
        Args:
            x:            (B, 1, D, H, W) normalised MR in [0, 1]
            anatomy_idx:  (B,) int64 anatomy class index

        Returns (training + deep supervision):
            main:  (B, 1, D, H, W) normalised sCT in [-1, 1]
            aux:   list of (B, 1, D', H', W') at lower resolutions, [-1, 1]

        Returns (eval or no deep supervision):
            (B, 1, D, H, W) normalised sCT in [-1, 1]
        """
        raw = self.dynunet(x)

        if self._deep_supervision and self.training:
            # raw: (B, deep_supr_num+1, 1, D, H, W)
            main_raw = raw[:, 0]                                   # (B, 1, D, H, W)
            aux_raws = [raw[:, i] for i in range(1, raw.shape[1])] # lower-res outputs

            if self.use_anatomy and anatomy_idx is not None:
                main_out = self.film_head(main_raw, anatomy_idx)
            else:
                main_out = self.out_act(main_raw)

            # Apply tanh to aux outputs (no FiLM — too small to benefit)
            aux_outs = [torch.tanh(a) for a in aux_raws]
            return main_out, aux_outs

        else:
            # Eval mode or deep_supervision disabled: single output.
            # NOTE: MONAI DynUNet returns [B, N, C, D, H, W] only during training.
            # In eval mode it already returns [B, C, D, H, W] (main output only),
            # so we must NOT slice [:, 0] here — that would drop the channel dim.
            if self.use_anatomy and anatomy_idx is not None:
                return self.film_head(raw, anatomy_idx)
            return self.out_act(raw)


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DynUNETR3D(use_anatomy=True, deep_supervision=True).to(device)
    model.train()

    B  = 1
    x  = torch.randn(B, 1, 64, 128, 128, device=device)
    ai = torch.tensor([0], device=device, dtype=torch.long)

    main, aux = model(x, ai)
    n_params  = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"DynUNETR3D  main: {main.shape}  aux: {[a.shape for a in aux]}")
    print(f"Params: {n_params:.1f}M")
    assert main.min() >= -1.01 and main.max() <= 1.01

    model.eval()
    with torch.no_grad():
        pred = model(x, ai)
    print(f"Eval output: {pred.shape}")
    print("OK")
