"""
Loss functions for SynthRAD2025 Task 1 sCT synthesis.

Provides:
  - MAELoss              — direct optimisation target (eval metric)
  - MSSSIMLoss           — perceptual structural loss
  - CombinedLoss         — weighted MAE + MS-SSIM (default training loss)
  - GradientDifferenceLoss — edge sharpness (optional regulariser)
  - PerceptualLoss       — VGG feature matching (optional)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Gaussian kernel for MS-SSIM ────────────────────────────────────────────────

def _gaussian_kernel(size: int, sigma: float, device=None) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g).unsqueeze(0).unsqueeze(0)   # (1, 1, size, size)


# ── SSIM ───────────────────────────────────────────────────────────────────────

def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor = None,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 2.0,    # our CT is in [-1, 1]
    C1: float = None,
    C2: float = None,
) -> torch.Tensor:
    K1, K2 = 0.01, 0.03
    C1 = C1 or (K1 * data_range) ** 2
    C2 = C2 or (K2 * data_range) ** 2

    kernel  = _gaussian_kernel(window_size, sigma, device=x.device)
    kernel  = kernel.expand(x.size(1), -1, -1, -1)

    pad = window_size // 2
    mu1 = F.conv2d(x, kernel, padding=pad, groups=x.size(1))
    mu2 = F.conv2d(y, kernel, padding=pad, groups=y.size(1))

    mu1_sq  = mu1 ** 2
    mu2_sq  = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sig1_sq = F.conv2d(x * x, kernel, padding=pad, groups=x.size(1)) - mu1_sq
    sig2_sq = F.conv2d(y * y, kernel, padding=pad, groups=y.size(1)) - mu2_sq
    sig12   = F.conv2d(x * y, kernel, padding=pad, groups=x.size(1)) - mu1_mu2

    num = (2 * mu1_mu2 + C1) * (2 * sig12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2)
    ssim_map = num / (den + 1e-8)

    if mask is not None:
        ssim_map = ssim_map * mask
        return ssim_map.sum() / (mask.sum() + 1e-8)
    return ssim_map.mean()


# ── MS-SSIM ────────────────────────────────────────────────────────────────────

_MS_SSIM_WEIGHTS = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]


class MSSSIMLoss(nn.Module):
    """
    Multi-Scale SSIM loss (1 - MS-SSIM).
    Loss = 0 when predictions are perfect.
    """

    def __init__(
        self,
        levels: int = 5,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 2.0,
    ):
        super().__init__()
        self.levels      = levels
        self.window_size = window_size
        self.sigma       = sigma
        self.data_range  = data_range
        weights          = torch.tensor(_MS_SSIM_WEIGHTS[:levels])
        self.register_buffer("weights", weights / weights.sum())

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        ms_ssim_val = torch.zeros(1, device=pred.device)

        for i in range(self.levels):
            s = ssim(pred, target, mask,
                     window_size=self.window_size,
                     sigma=self.sigma,
                     data_range=self.data_range)
            ms_ssim_val = ms_ssim_val + self.weights[i] * s

            if i < self.levels - 1:
                pred   = F.avg_pool2d(pred,   2)
                target = F.avg_pool2d(target, 2)
                if mask is not None:
                    mask = F.avg_pool2d(mask, 2)

        return 1.0 - ms_ssim_val


# ── MAE ────────────────────────────────────────────────────────────────────────

class MAELoss(nn.Module):
    """
    L1 loss optionally restricted to body mask with bone-region upweighting.

    bone_weight    : extra multiplier on bone voxels (1.0 = disabled)
    bone_threshold : CT value in normalised [-1,1] space above which a voxel is
                     considered bone.  HU 200 maps to -0.4 with our clip [-1000,3000].
    """

    def __init__(self, bone_weight: float = 1.0, bone_threshold: float = -0.4):
        super().__init__()
        self.bone_weight    = bone_weight
        self.bone_threshold = bone_threshold

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        diff = torch.abs(pred - target)

        # Build per-voxel weight map: bone voxels get extra emphasis
        if self.bone_weight > 1.0:
            bone_region = (target > self.bone_threshold).float()
            w = 1.0 + (self.bone_weight - 1.0) * bone_region   # (B,1,H,W)
            diff = diff * w

        if mask is not None:
            diff = diff * mask
            if self.bone_weight > 1.0:
                denom = (mask * w).sum() + 1e-8
            else:
                denom = mask.sum() + 1e-8
            return diff.sum() / denom

        return diff.mean()


# ── Gradient Difference Loss ────────────────────────────────────────────────────

class GradientDifferenceLoss(nn.Module):
    """
    Penalises blurry predictions by matching image gradients.
    Helps preserve HU boundaries at bone/soft-tissue interfaces.
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        def grad(t):
            gx = t[:, :, :, 1:] - t[:, :, :, :-1]
            gy = t[:, :, 1:, :] - t[:, :, :-1, :]
            return gx, gy

        pgx, pgy = grad(pred)
        tgx, tgy = grad(target)

        # pgx/tgx: (B,C,H,W-1)  |  pgy/tgy: (B,C,H-1,W) — different shapes, handle separately
        diff_x = torch.abs(pgx - tgx)
        diff_y = torch.abs(pgy - tgy)

        if mask is not None:
            mx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
            my = mask[:, :, 1:, :] * mask[:, :, :-1, :]
            loss_x = (diff_x * mx).sum() / (mx.sum() + 1e-8)
            loss_y = (diff_y * my).sum() / (my.sum() + 1e-8)
            return (loss_x + loss_y) * 0.5

        return (diff_x.mean() + diff_y.mean()) * 0.5


# ── Combined Loss ──────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Training loss:
        L = w_mae * BoneWeightedMAE + w_ssim * (1 - MS-SSIM) + w_gdl * GDL

    All terms are restricted to the body mask when mask is provided.
    bone_weight > 1.0 upweights voxels above bone_threshold in normalised CT space
    (HU 200 ≈ -0.4 with our [-1000, 3000] clip).
    """

    def __init__(
        self,
        w_mae:          float = 1.0,
        w_ssim:         float = 1.0,
        w_gdl:          float = 0.0,
        ms_ssim_levels: int   = 5,
        bone_weight:    float = 1.0,
        bone_threshold: float = -0.4,
    ):
        super().__init__()
        self.w_mae  = w_mae
        self.w_ssim = w_ssim
        self.w_gdl  = w_gdl

        self.mae_loss  = MAELoss(bone_weight=bone_weight, bone_threshold=bone_threshold)
        self.ssim_loss = MSSSIMLoss(levels=ms_ssim_levels)
        self.gdl_loss  = GradientDifferenceLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> dict:
        losses = {}

        if self.w_mae > 0:
            losses["mae"]  = self.w_mae  * self.mae_loss(pred, target, mask)
        if self.w_ssim > 0:
            losses["ssim"] = self.w_ssim * self.ssim_loss(pred, target, mask)
        if self.w_gdl > 0:
            losses["gdl"]  = self.w_gdl  * self.gdl_loss(pred, target, mask)

        losses["total"] = sum(losses.values())
        return losses
