"""
Loss functions for SynthRAD2025 Task 1 sCT synthesis.

Provides:
  - MAELoss              — direct optimisation target (eval metric)
  - MSSSIMLoss           — perceptual structural loss
  - CombinedLoss         — weighted MAE + MS-SSIM + GDL + Perceptual
  - GradientDifferenceLoss — edge sharpness (optional regulariser)
  - PerceptualLoss       — VGG16 feature matching (AFP-style, optional)
  - TotalSegmentatorAFP  — CT-domain AFP via nnU-Net v2 ResidualEncoder (KoalAI-style)
"""

from __future__ import annotations

import contextlib
import warnings
from pathlib import Path

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
        # Reshape 3D (B,C,D,H,W) → 2D (B*D,C,H,W) so 2D convolutions apply slice-wise
        if pred.ndim == 5:
            B, C, D, H, W = pred.shape
            pred   = pred.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
            target = target.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
            if mask is not None:
                mask = mask.permute(0, 2, 1, 3, 4).reshape(B * D, 1, H, W)

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
        L = w_mae * BoneWeightedMAE
          + w_ssim * (1 - MS-SSIM)
          + w_gdl  * GDL
          + w_perc * AFP (VGG16 or TotalSegmentator, see afp_type)

    All terms are restricted to the body mask when mask is provided.
    bone_weight > 1.0 upweights voxels above bone_threshold in normalised CT space
    (HU 200 ≈ -0.4 with our [-1000, 3000] clip).

    Perceptual loss (w_perc > 0) significantly improves Dice and HD95 by
    enforcing anatomically correct structures — the key fix for 3D MONAI models.
    afp_type="totalseg" uses CT-domain features (KoalAI-style); "vgg" uses VGG16.
    """

    def __init__(
        self,
        w_mae:           float = 1.0,
        w_ssim:          float = 1.0,
        w_gdl:           float = 0.0,
        w_perc:          float = 0.0,
        ms_ssim_levels:  int   = 5,
        bone_weight:     float = 1.0,
        bone_threshold:  float = -0.4,
        perc_max_slices: int   = 16,
        afp_type:        str   = "vgg",   # "vgg" | "totalseg"
        afp_weights_dir: str   = None,
    ):
        super().__init__()
        self.w_mae  = w_mae
        self.w_ssim = w_ssim
        self.w_gdl  = w_gdl
        self.w_perc = w_perc

        self.mae_loss  = MAELoss(bone_weight=bone_weight, bone_threshold=bone_threshold)
        self.ssim_loss = MSSSIMLoss(levels=ms_ssim_levels)
        self.gdl_loss  = GradientDifferenceLoss()

        if w_perc > 0:
            if afp_type == "totalseg":
                self.perc_loss = TotalSegmentatorAFP(
                    weights_dir   = afp_weights_dir,
                    max_2d_slices = perc_max_slices,
                )
            else:
                self.perc_loss = PerceptualLoss(max_2d_slices=perc_max_slices)
        else:
            self.perc_loss = None

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
        if self.w_perc > 0 and self.perc_loss is not None:
            losses["perc"] = self.w_perc * self.perc_loss(pred, target, mask)

        losses["total"] = sum(losses.values())
        return losses


# ── Perceptual Loss (VGG16 AFP-style) ─────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """
    VGG16 perceptual / feature-matching loss (AFP-style).

    Extracts multi-scale features from relu1_2, relu2_2, relu3_3, relu4_3
    and computes L1 distance between prediction and target feature maps.

    Inputs are expected in [-1, 1] (normalised CT space); they are
    re-scaled to [0, 1] and replicated to 3 channels for VGG.

    Works on both 2D (B, 1, H, W) and 3D (B, 1, D, H, W) inputs.
    For 3D, up to `max_2d_slices` axial slices are randomly sampled to
    limit GPU cost (VGG is 2-D).

    All VGG parameters are frozen; gradients only flow through `pred`.
    """

    # VGG16 relu1_2, relu2_2, relu3_3, relu4_3 (0-based indices in .features)
    _SLICE_ENDS = [3, 8, 15, 22]
    # ImageNet statistics (after mapping [-1,1] → [0,1] and 1-ch → 3-ch)
    _MEAN = (0.485, 0.456, 0.406)
    _STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        layer_weights: tuple = (0.25, 0.25, 0.25, 0.25),
        max_2d_slices: int   = 16,
    ):
        super().__init__()
        try:
            from torchvision.models import vgg16, VGG16_Weights
            feat = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        except Exception as exc:
            raise RuntimeError(
                "PerceptualLoss requires torchvision with VGG16 weights. "
                f"Original error: {exc}"
            )

        # Split VGG features into sequential slices between target layers
        self.slices = nn.ModuleList()
        prev = 0
        for end in self._SLICE_ENDS:
            self.slices.append(nn.Sequential(*list(feat.children())[prev : end + 1]))
            prev = end + 1

        for p in self.parameters():
            p.requires_grad_(False)

        assert len(layer_weights) == len(self.slices), "Need 4 layer weights"
        self.layer_weights = layer_weights
        self.max_2d_slices = max_2d_slices

        mean = torch.tensor(self._MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(self._STD).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean)
        self.register_buffer("_std",  std)

    def _to_vgg(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, H, W) in [-1, 1] → (B, 3, H, W) ImageNet-normalised."""
        x = (x.clamp(-1.0, 1.0) + 1.0) * 0.5      # → [0, 1]
        x = x.expand(-1, 3, -1, -1)                 # broadcast to 3ch (no copy)
        return (x - self._mean) / self._std

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
        mask:   torch.Tensor = None,
    ) -> torch.Tensor:
        is_3d = pred.ndim == 5
        if is_3d:
            B, C, D, H, W = pred.shape
            if D > self.max_2d_slices:
                idx    = torch.randperm(D, device=pred.device)[: self.max_2d_slices]
                pred   = pred  [:, :, idx]
                target = target[:, :, idx]
                mask   = mask  [:, :, idx] if mask is not None else None
                D      = self.max_2d_slices
            pred   = pred.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
            target = target.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
            if mask is not None:
                mask = mask.permute(0, 2, 1, 3, 4).reshape(B * D, 1, H, W)

        h_p = self._to_vgg(pred)
        h_t = self._to_vgg(target.detach())

        loss = pred.new_zeros(())
        for w, vgg_slice in zip(self.layer_weights, self.slices):
            h_p = vgg_slice(h_p)
            with torch.no_grad():
                h_t = vgg_slice(h_t)

            if mask is not None:
                m    = F.adaptive_avg_pool2d(mask.float(), h_p.shape[-2:]).gt(0.5).float()
                diff = (h_p - h_t).abs()
                term = (diff * m).sum() / (m.sum() * h_p.shape[1] + 1e-8)
            else:
                term = (h_p - h_t).abs().mean()

            loss = loss + w * term

        return loss


# ── TotalSegmentator AFP Loss ──────────────────────────────────────────────────

class TotalSegmentatorAFP(nn.Module):
    """
    Anatomy Feature Preservation (AFP) loss via TotalSegmentator v2.

    Extracts multi-scale features from a pretrained 104-structure CT
    segmentation network (nnU-Net v2 ResidualEncoder) and penalises L1
    distance between predicted and target feature maps.

    CT-domain features directly target the anatomical accuracy metrics used in
    SynthRAD2025 evaluation (Dice, HD95), outperforming VGG16 which was never
    trained on medical images.  This is the KoalAI 1st-place approach.

    Inputs: [-1, 1] normalised CT (clip −1000…3000 HU).
    Supports 2D (B,1,H,W) and 3D (B,1,D,H,W) inputs.
    Falls back to VGG16 PerceptualLoss if TotalSegmentator / nnunetv2 is
    not installed or weights are not found.

    Weight paths searched in order:
      1. weights_dir constructor argument
      2. /pvc/pretrained/totalsegmentator/nnunet/results  (K8s PVC)
      3. ~/.totalsegmentator/nnunet/results               (local install)
    """

    _ROOTS = [
        "/pvc/pretrained/totalsegmentator/nnunet/results",
        "~/.totalsegmentator/nnunet/results",
    ]
    _DATASETS = [
        "Dataset291_TotalSegmentator_part1_organs_1559subj",
        "Dataset291_TotalSegmentator_part1_organs",
        "Dataset291",
    ]
    _TRAINER = "nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"

    def __init__(
        self,
        weights_dir:   "str | None" = None,
        n_stages:      int          = 4,
        layer_weights: tuple        = (0.25, 0.25, 0.25, 0.25),
        max_2d_slices: int          = 16,
    ):
        super().__init__()
        self.n_stages      = n_stages
        self.layer_weights = tuple(layer_weights)
        self.max_2d_slices = max_2d_slices

        enc = self._try_load(weights_dir)
        if enc is None:
            warnings.warn(
                "TotalSegmentatorAFP: weights not found or nnunetv2 not installed — "
                "falling back to VGG16 PerceptualLoss.",
                RuntimeWarning, stacklevel=2,
            )
            self._fallback = PerceptualLoss(max_2d_slices=max_2d_slices)
            self.encoder   = None
        else:
            self._fallback = None
            self.encoder   = enc
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()

    # ── weight loading ──────────────────────────────────────────────────────────

    def _try_load(self, extra_dir: "str | None") -> "nn.Module | None":
        try:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        except ImportError:
            return None

        roots = ([Path(extra_dir).expanduser()] if extra_dir else []) + \
                [Path(r).expanduser() for r in self._ROOTS]

        for root in roots:
            for ds in self._DATASETS:
                model_dir = root / ds / self._TRAINER
                if not model_dir.is_dir():
                    continue
                with contextlib.suppress(Exception):
                    pred = nnUNetPredictor(
                        tile_step_size=0.5,
                        use_gaussian=False,
                        use_mirroring=False,
                        perform_everything_on_device=False,
                        device=torch.device("cpu"),
                        verbose=False,
                    )
                    pred.initialize_from_trained_model_folder(
                        str(model_dir),
                        use_folds=(0,),
                        checkpoint_name="checkpoint_final.pth",
                    )
                    enc = pred.network.encoder
                    enc.eval()
                    return enc
        return None

    # ── feature extraction ──────────────────────────────────────────────────────

    def _get_stages(self) -> "list[nn.Module]":
        for attr in ("stages", "encoders", "encoder_stages"):
            if hasattr(self.encoder, attr):
                return list(getattr(self.encoder, attr))
        return list(self.encoder.children())

    def _extract(self, x: torch.Tensor) -> "list[torch.Tensor]":
        feats: list = []
        handles = []
        for stage in self._get_stages()[: self.n_stages]:
            def _hook(_, _i, out, _f=feats):
                _f.append(out[0] if isinstance(out, (list, tuple)) else out)
            handles.append(stage.register_forward_hook(_hook))
        try:
            self.encoder(x)
        finally:
            for h in handles:
                h.remove()
        return feats

    # ── normalisation ───────────────────────────────────────────────────────────

    @staticmethod
    def _to_nnunet(x: torch.Tensor) -> torch.Tensor:
        """[-1, 1] → approximate nnU-Net z-score (CT mean ≈ 100 HU, std ≈ 350 HU)."""
        hu = (x.clamp(-1.0, 1.0) + 1.0) * 2000.0 - 1000.0   # → HU  (clip -1000…3000)
        return (hu - 100.0) / 350.0

    # ── forward ─────────────────────────────────────────────────────────────────

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
        mask:   torch.Tensor = None,
    ) -> torch.Tensor:
        if self._fallback is not None:
            return self._fallback(pred, target, mask)

        # Promote 2D → 3D (treat H×W as 1×H×W depth)
        if pred.ndim == 4:
            pred   = pred.unsqueeze(2)
            target = target.unsqueeze(2)
            mask   = mask.unsqueeze(2) if mask is not None else None

        xp = self._to_nnunet(pred)
        xt = self._to_nnunet(target)

        fp = self._extract(xp)
        with torch.no_grad():
            ft = self._extract(xt)

        if not fp:
            warnings.warn("TotalSegmentatorAFP: encoder returned no features; using raw L1.",
                          RuntimeWarning, stacklevel=2)
            return (pred - target.detach()).abs().mean()

        loss = pred.new_zeros(())
        for i, (a, b) in enumerate(zip(fp, ft)):
            w = self.layer_weights[i] if i < len(self.layer_weights) else 1.0 / len(fp)
            if mask is not None:
                m    = F.adaptive_avg_pool3d(mask.float(), a.shape[-3:]).gt(0.5).float()
                term = ((a - b.detach()).abs() * m).sum() / (m.sum() * a.shape[1] + 1e-8)
            else:
                term = (a - b.detach()).abs().mean()
            loss = loss + w * term

        return loss


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
