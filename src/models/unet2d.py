"""
2D U-Net for Synthetic CT generation (SynthRAD2025 Task 1 baseline)

Architecture:
  - Encoder: 4 downsampling stages with double conv blocks
  - Bottleneck with optional anatomy conditioning
  - Decoder: 4 upsampling stages with skip connections
  - Output: single-channel sCT in [-1, 1] (normalised HU)

Anatomy conditioning:
  - Anatomy embedding added to bottleneck feature map
  - Allows single model to handle HN / TH / AB

Input:  (B, 1, H, W)   — normalised MR
Output: (B, 1, H, W)   — normalised sCT
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2"""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """MaxPool → DoubleConv"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool  = nn.MaxPool2d(2)
        self.conv  = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Bilinear upsample → concat skip → DoubleConv"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if spatial dims don't match (edge case)
        dh = skip.size(-2) - x.size(-2)
        dw = skip.size(-1) - x.size(-1)
        if dh > 0 or dw > 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class AnatomyEmbedding(nn.Module):
    """
    Converts anatomy index (0/1/2) to a spatial conditioning bias
    added to the bottleneck feature map.
    """

    def __init__(self, n_classes: int, embed_dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_classes, embed_dim)

    def forward(self, x: torch.Tensor, anatomy_idx: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W), anatomy_idx: (B,)
        e = self.emb(anatomy_idx)          # (B, C)
        e = e[:, :, None, None]            # (B, C, 1, 1)
        return x + e


# ── U-Net ──────────────────────────────────────────────────────────────────────

class UNet2D(nn.Module):
    """
    2D U-Net for MR → sCT synthesis.

    Args:
        in_channels:    1 (MR)
        out_channels:   1 (sCT)
        base_features:  number of filters in first encoder stage (default 64)
        depth:          number of encoder/decoder stages (default 4)
        n_anatomy:      number of anatomy classes for conditioning (3)
        use_anatomy:    whether to use anatomy embedding in bottleneck
    """

    def __init__(
        self,
        in_channels:  int = 1,
        out_channels: int = 1,
        base_features: int = 64,
        depth: int = 4,
        n_anatomy: int = 3,
        use_anatomy: bool = True,
    ):
        super().__init__()
        self.depth       = depth
        self.use_anatomy = use_anatomy

        feats = [base_features * (2 ** i) for i in range(depth + 1)]
        # e.g. depth=4, base=64 → [64, 128, 256, 512, 1024]

        # Encoder
        self.inc    = DoubleConv(in_channels, feats[0])
        self.downs  = nn.ModuleList([Down(feats[i], feats[i+1]) for i in range(depth)])

        # Bottleneck anatomy conditioning
        if use_anatomy:
            self.anat_emb = AnatomyEmbedding(n_anatomy, feats[-1])

        # Decoder
        self.ups = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.ups.append(Up(feats[i+1], feats[i], feats[i]))

        # Output
        self.out_conv = nn.Conv2d(feats[0], out_channels, 1)
        self.out_act  = nn.Tanh()   # → [-1, 1]

    def forward(
        self,
        x: torch.Tensor,
        anatomy_idx: torch.Tensor = None,
    ) -> torch.Tensor:
        # Encoder
        skips  = []
        h      = self.inc(x)
        skips.append(h)
        for down in self.downs:
            h = down(h)
            skips.append(h)

        # Bottleneck = last skip (not needed in decoder)
        bottleneck = skips.pop()

        # Anatomy conditioning
        if self.use_anatomy and anatomy_idx is not None:
            bottleneck = self.anat_emb(bottleneck, anatomy_idx)

        # Decoder
        h = bottleneck
        for up in self.ups:
            skip = skips.pop()
            h    = up(h, skip)

        return self.out_act(self.out_conv(h))


# ── Attention U-Net variant ────────────────────────────────────────────────────

class AttentionGate(nn.Module):
    """Soft attention gate on skip connections."""

    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
        super().__init__()
        self.Wg   = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.Wx   = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.psi  = nn.Conv2d(inter_ch, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sig  = nn.Sigmoid()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # g from decoder, x is skip connection
        g_up = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
        psi  = self.relu(self.Wg(g_up) + self.Wx(x))
        attn = self.sig(self.psi(psi))
        return x * attn


class AttentionUp(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.attn = AttentionGate(in_ch, skip_ch, skip_ch // 2)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x    = self.up(x)
        skip = self.attn(x, skip)
        dh   = skip.size(-2) - x.size(-2)
        dw   = skip.size(-1) - x.size(-1)
        if dh > 0 or dw > 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class AttentionUNet2D(UNet2D):
    """U-Net with attention gates on all skip connections."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        depth = self.depth
        feats = [kwargs.get("base_features", 64) * (2 ** i) for i in range(depth + 1)]
        self.ups = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.ups.append(AttentionUp(feats[i+1], feats[i], feats[i]))


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = UNet2D(base_features=64, depth=4, use_anatomy=True).to(device)

    B, H, W = 2, 256, 256
    x        = torch.randn(B, 1, H, W, device=device)
    anat_idx = torch.tensor([0, 2], device=device)
    y        = model(x, anat_idx)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"UNet2D output: {y.shape}  |  params: {n_params:.1f}M")
    assert y.shape == (B, 1, H, W), "Shape mismatch!"
    print("OK")
