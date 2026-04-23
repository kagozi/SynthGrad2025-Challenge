"""
SynthRAD2025 Task 1 — MONAI Swin-UNETR wrapper

Architecture:
  - Backbone: MONAI SwinUNETR (3D Swin Transformer encoder + UNet decoder)
  - Anatomy conditioning: lightweight FiLM residual head after backbone output
  - Pretrained: SSL encoder weights from MONAI (~5050 CT/MRI volumes, masked inpainting)
  - Gradient checkpointing: enabled by default for memory efficiency

Input:  (B, 1, D, H, W)  — normalised MR in [0, 1]
Output: (B, 1, D, H, W)  — normalised sCT in [-1, 1]

Constraints:
  Each spatial dimension of img_size must be divisible by 32.
  Recommended training patch sizes: (64, 128, 128) or (96, 192, 192).

Usage:
    model = SwinUNETR3D(
        img_size=(64, 128, 128),
        feature_size=48,
        n_anatomy=3,
        use_anatomy=True,
        pretrained=True,        # download & load SSL encoder weights
        pretrained_cache_dir="/pvc/pretrained",
    )
    out = model(mr_patch, anatomy_idx)  # (B, 1, 64, 128, 128)
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

try:
    from monai.networks.nets import SwinUNETR as _MonaiSwinUNETR
except ImportError as e:
    raise ImportError(
        "MONAI is required for SwinUNETR. Install with: pip install monai>=1.3"
    ) from e


# ── Pretrained weight URL ──────────────────────────────────────────────────────
# SSL pretrained Swin ViT encoder from MONAI Model Zoo.
# Trained on ~5,050 CT/MRI volumes using masked volume inpainting.
# Must be used with feature_size=48 (the default).
_PRETRAINED_URL = (
    "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/"
    "download/0.8.1/model_swinvit.pt"
)


def _download_pretrained(cache_dir: str | Path) -> Path:
    """Download SSL pretrained weights if not already cached."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / "model_swinvit.pt"
    if out.exists():
        print(f"[SwinUNETR3D] Using cached pretrained weights: {out}")
        return out
    print(f"[SwinUNETR3D] Downloading pretrained weights → {out}")
    import urllib.request
    urllib.request.urlretrieve(_PRETRAINED_URL, str(out))
    print(f"[SwinUNETR3D] Download complete.")
    return out


# ── Anatomy FiLM residual head ─────────────────────────────────────────────────

class AnatomyFiLMHead(nn.Module):
    """
    Lightweight Feature-wise Linear Modulation head applied after SwinUNETR output.

    Takes the raw 1-channel prediction and an anatomy index, then applies a
    learned affine transform (scale + shift) on a hidden feature space, with a
    residual connection back to the raw prediction.

    This allows the model to learn anatomy-specific HU offsets and scales
    (e.g. HN soft-tissue vs. TH lung/bone) without touching the MONAI backbone.

    Architecture:
        x → Conv3d(1→H) → IN → LeakyReLU
          → FiLM(H, anatomy) → LeakyReLU
          → Conv3d(H→1)
          + x  (residual)
        → Tanh → [-1, 1]

    Parameters: ~2×H² + 2×n_anatomy×H  ≈ 35K for H=128
    """

    def __init__(self, hidden: int = 128, n_anatomy: int = 3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(1, hidden, 3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # FiLM parameters: one (scale, shift) vector per anatomy class
        self.film_scale = nn.Embedding(n_anatomy, hidden)
        self.film_shift = nn.Embedding(n_anatomy, hidden)
        self.head       = nn.Conv3d(hidden, 1, 1)
        self.act        = nn.Tanh()

        # Init FiLM scale to ~1, shift to ~0 so initial output ≈ identity
        nn.init.ones_(self.film_scale.weight)
        nn.init.zeros_(self.film_shift.weight)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, anatomy_idx: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, D, H, W), anatomy_idx: (B,)
        h = self.body(x)

        scale = self.film_scale(anatomy_idx).view(-1, h.shape[1], 1, 1, 1)
        shift = self.film_shift(anatomy_idx).view(-1, h.shape[1], 1, 1, 1)
        h = h * scale + shift
        h = torch.nn.functional.leaky_relu(h, 0.2, inplace=True)

        return self.act(self.head(h) + x)   # residual + Tanh


# ── Main model ─────────────────────────────────────────────────────────────────

class SwinUNETR3D(nn.Module):
    """
    MONAI SwinUNETR for MR → sCT synthesis with anatomy FiLM conditioning.

    Args:
        img_size:           Spatial size of training patches. Each dim must be
                            divisible by 32. E.g. (64, 128, 128).
        in_channels:        Input channels (1 for single-echo MR).
        feature_size:       Swin feature dimension. Must be 48 to use pretrained
                            SSL weights. Larger values (e.g. 96) train from scratch.
        use_checkpoint:     Gradient checkpointing in Swin blocks (saves ~30% VRAM).
        n_anatomy:          Number of anatomy classes (3: HN, TH, AB).
        use_anatomy:        Whether to apply FiLM conditioning.
        film_hidden:        Hidden channels in the FiLM head.
        pretrained:         Load MONAI SSL pretrained encoder weights.
        pretrained_cache_dir: Directory to cache the downloaded checkpoint.
        drop_rate:          Dropout in Swin transformer blocks.
        attn_drop_rate:     Attention dropout rate.
        dropout_path_rate:  Stochastic depth drop rate.
    """

    def __init__(
        self,
        img_size:             tuple = (64, 128, 128),
        in_channels:          int   = 1,
        feature_size:         int   = 48,
        use_checkpoint:       bool  = True,
        n_anatomy:            int   = 3,
        use_anatomy:          bool  = True,
        film_hidden:          int   = 128,
        pretrained:           bool  = True,
        pretrained_cache_dir: str   = "/pvc/pretrained",
        drop_rate:            float = 0.0,
        attn_drop_rate:       float = 0.0,
        dropout_path_rate:    float = 0.1,
    ):
        super().__init__()
        self.use_anatomy = use_anatomy

        self.swin_unetr = _MonaiSwinUNETR(
            in_channels         = in_channels,
            out_channels        = 1,
            feature_size        = feature_size,
            use_checkpoint      = use_checkpoint,
            drop_rate           = drop_rate,
            attn_drop_rate      = attn_drop_rate,
            dropout_path_rate   = dropout_path_rate,
            spatial_dims        = 3,
            norm_name           = "instance",
        )
        # SwinUNETR has a linear head at the end but no Tanh — we add ours.
        # Replace the default output conv with Identity; our FiLM head handles it.
        # Actually SwinUNETR already ends with out_channels=1 via a regular conv.
        # We keep it and treat its raw output as the pre-activation signal.

        if use_anatomy:
            self.film_head = AnatomyFiLMHead(hidden=film_hidden, n_anatomy=n_anatomy)
        else:
            self.out_act = nn.Tanh()

        if pretrained and feature_size == 48:
            self._load_pretrained(pretrained_cache_dir)
        elif pretrained and feature_size != 48:
            print(
                "[SwinUNETR3D] WARNING: Pretrained weights require feature_size=48. "
                f"Got {feature_size}. Training from scratch."
            )

    def _load_pretrained(self, cache_dir: str):
        """
        Load MONAI SSL pretrained Swin ViT encoder weights.

        The checkpoint contains only the encoder (swinViT) weights — the UNet
        decoder is randomly initialised. This is the standard transfer-learning
        recipe for MONAI SwinUNETR.
        """
        try:
            ckpt_path = _download_pretrained(cache_dir)
            weights   = torch.load(str(ckpt_path), map_location="cpu")
            # MONAI's load_from() copies matching keys from the SSL checkpoint
            # into the swinViT sub-module with verbose reporting.
            self.swin_unetr.load_from(weights)
            print("[SwinUNETR3D] SSL pretrained encoder weights loaded successfully.")
        except Exception as e:
            print(f"[SwinUNETR3D] WARNING: Could not load pretrained weights: {e}")
            print("[SwinUNETR3D] Continuing with random initialisation.")

    def encoder_parameters(self):
        """Returns only the SwinViT encoder parameters."""
        return self.swin_unetr.swinViT.parameters()

    def decoder_parameters(self):
        """Returns decoder + FiLM head parameters (everything except encoder)."""
        enc_ids = {id(p) for p in self.swin_unetr.swinViT.parameters()}
        return [p for p in self.parameters() if id(p) not in enc_ids]

    def forward(
        self,
        x: torch.Tensor,
        anatomy_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:            (B, 1, D, H, W) normalised MR in [0, 1]
            anatomy_idx:  (B,) int64 anatomy class index

        Returns:
            (B, 1, D, H, W) normalised sCT in [-1, 1]
        """
        raw = self.swin_unetr(x)   # (B, 1, D, H, W) — no activation yet

        if self.use_anatomy and anatomy_idx is not None:
            return self.film_head(raw, anatomy_idx)
        return self.out_act(raw)


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Small test patch — skip pretrained download
    model = SwinUNETR3D(
        img_size=(64, 128, 128),
        feature_size=48,
        pretrained=False,
        use_anatomy=True,
    ).to(device)

    B  = 1
    x  = torch.randn(B, 1, 64, 128, 128, device=device)
    ai = torch.tensor([0], device=device, dtype=torch.long)

    with torch.no_grad():
        y = model(x, ai)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"SwinUNETR3D output: {y.shape}  |  params: {n_params:.1f}M")
    assert y.shape == (B, 1, 64, 128, 128), f"Unexpected shape: {y.shape}"
    assert y.min() >= -1.01 and y.max() <= 1.01, "Output out of [-1, 1] range"
    print("OK")
