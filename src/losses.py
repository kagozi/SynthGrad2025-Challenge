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

    Supports both 2D inputs (B, C, H, W) and 3D inputs (B, C, D, H, W).
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        ndim = pred.ndim  # 4 for 2D, 5 for 3D

        if ndim == 4:
            # 2D: gradients along H and W axes
            axes = [(-2, slice(None), slice(1, None)),   # W: axis -1
                    (-1, slice(1, None), slice(None))]   # H: axis -2
            def _grad_pair(t):
                gx = t[:, :, :, 1:] - t[:, :, :, :-1]
                gy = t[:, :, 1:, :] - t[:, :, :-1, :]
                return [gx, gy]
            def _mask_pair(m):
                mx = m[:, :, :, 1:] * m[:, :, :, :-1]
                my = m[:, :, 1:, :] * m[:, :, :-1, :]
                return [mx, my]
        else:
            # 3D: gradients along D, H, W axes
            def _grad_pair(t):
                gz = t[:, :, 1:, :, :] - t[:, :, :-1, :, :]  # depth
                gy = t[:, :, :, 1:, :] - t[:, :, :, :-1, :]  # height
                gx = t[:, :, :, :, 1:] - t[:, :, :, :, :-1]  # width
                return [gz, gy, gx]
            def _mask_pair(m):
                mz = m[:, :, 1:, :, :] * m[:, :, :-1, :, :]
                my = m[:, :, :, 1:, :] * m[:, :, :, :-1, :]
                mx = m[:, :, :, :, 1:] * m[:, :, :, :, :-1]
                return [mz, my, mx]

        p_grads = _grad_pair(pred)
        t_grads = _grad_pair(target)
        diffs   = [torch.abs(pg - tg) for pg, tg in zip(p_grads, t_grads)]

        if mask is not None:
            masks  = _mask_pair(mask)
            losses = [(d * m).sum() / (m.sum() + 1e-8)
                      for d, m in zip(diffs, masks)]
        else:
            losses = [d.mean() for d in diffs]

        return sum(losses) / len(losses)


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


# ── GAN Loss ────────────────────────────────────────────────────────────────────

class GANLoss(nn.Module):
    """
    Adversarial loss for PatchGAN (BCEWithLogitsLoss).

    Wraps target-tensor creation so the caller only passes is_real=True/False.
    Uses soft real labels (0.9) for training stability.

    Usage:
        criterion_gan = GANLoss()
        L_D = 0.5 * (criterion_gan(D(real), True) + criterion_gan(D(fake), False))
        L_G_adv = criterion_gan(D(fake), True)
    """

    def __init__(self, real_label: float = 0.9, fake_label: float = 0.0):
        super().__init__()
        self.real_label = real_label
        self.fake_label = fake_label

    def _target(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        val = self.real_label if is_real else self.fake_label
        return torch.full_like(pred, val)

    def forward(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(pred, self._target(pred, is_real))
