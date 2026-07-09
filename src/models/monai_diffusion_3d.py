"""
3D Conditional DDPM for MR → sCT synthesis using MONAI's DiffusionModelUNet.

Architecture (v2 — closest to Faking_it, rank 7 in SynthRAD2025):
  - Backbone: MONAI DiffusionModelUNet (3D), channels [64,128,256,512],
              attention at 3 levels, 3 residual blocks, flash attention
  - Anatomy conditioning: class embedding (HN/TH/AB) via num_class_embeds=3
  - Noise schedule: cosine (Nichol & Dhariwal 2021), T=1000
  - Parameterisation: predict both noise ε and log-variance v (out_channels=2)
  - VS-DDPM: Variable-Step training — sample T_eff from {50,100,250,500,1000}
             per batch so the model works well at any inference step count
  - Training loss: noise MSE + Min-SNR MAE + SSIM + VLB (improved DDPM)
  - Inference: DDIM deterministic sampler (50 steps, eta=0)
  - EMA: exponential moving average (decay=0.9999) for stable inference

Usage in training:
    t    = diffusion.sample_t(B, device)   # VS-DDPM aware
    loss, parts = diffusion.p_loss(ct_patch, mr_patch, t, anatomy_idx)

Usage in inference:
    pred_ct = diffusion.ddim_sample(mr_patch, anatomy_idx, steps=50)
"""

from __future__ import annotations

import math
import random
import warnings
from copy import deepcopy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from monai.networks.nets import DiffusionModelUNet
    _MONAI_DIFFUSION_AVAILABLE = True
except ImportError:
    _MONAI_DIFFUSION_AVAILABLE = False
    warnings.warn(
        "MONAI DiffusionModelUNet not found. Install MONAI >= 1.4. "
        "MonaiDiffusion3D will raise on instantiation.",
        ImportWarning, stacklevel=1,
    )

try:
    from monai.losses import SSIMLoss as MonaiSSIMLoss
    _MONAI_SSIM_AVAILABLE = True
except ImportError:
    _MONAI_SSIM_AVAILABLE = False


# ── Cosine noise schedule ──────────────────────────────────────────────────────

def _cosine_betas(T: int, s: float = 0.008) -> torch.Tensor:
    t_arr     = torch.arange(T + 1, dtype=torch.float64)
    f         = torch.cos(((t_arr / T) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alpha_bar = f / f[0]
    betas     = (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999)
    return betas.float()


# ── EMA ────────────────────────────────────────────────────────────────────────

class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay   = decay
        self.shadow  = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v.detach())

    def apply_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state: dict):
        self.shadow = {k: v.clone() for k, v in state.items()}


# ── MONAI backbone wrapper ─────────────────────────────────────────────────────

class MonaiDiffusion3D(nn.Module):
    """
    Thin wrapper around MONAI's DiffusionModelUNet for 3D MR→CT synthesis.

    in_channels  = 2  (MR + noisy CT, channel-concatenated)
    out_channels = 1  → predict noise ε only  (original)
    out_channels = 2  → predict noise ε + log-variance v  (improved DDPM / VLB)

    Anatomy conditioning: class embedding (HN/TH/AB) added to timestep embedding.
    """

    def __init__(
        self,
        in_channels:       int   = 2,
        out_channels:      int   = 2,
        channels:          tuple = (64, 128, 256, 512),
        attention_levels:  tuple = (False, True, True, True),
        num_res_blocks:    int   = 3,
        num_head_channels: int   = 64,
        norm_num_groups:   int   = 32,
        norm_eps:          float = 1e-6,
        n_anatomy:         int   = 3,
        dropout_cattn:     float = 0.1,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        if not _MONAI_DIFFUSION_AVAILABLE:
            raise ImportError(
                "MONAI DiffusionModelUNet is required. "
                "Install with: pip install 'monai[all]>=1.4'"
            )

        self.out_channels = out_channels

        self.net = DiffusionModelUNet(
            spatial_dims         = 3,
            in_channels          = in_channels,
            out_channels         = out_channels,
            channels             = channels,
            attention_levels     = attention_levels,
            num_res_blocks       = num_res_blocks,
            norm_num_groups      = norm_num_groups,
            norm_eps             = norm_eps,
            resblock_updown      = False,
            num_head_channels    = num_head_channels,
            with_conditioning    = False,
            transformer_num_layers = 1,
            cross_attention_dim  = None,
            num_class_embeds     = n_anatomy,
            upcast_attention     = False,
            dropout_cattn        = dropout_cattn,
            include_fc           = True,
            use_combined_linear  = True,
            use_flash_attention  = use_flash_attention,
        )

    def forward(
        self,
        x:           torch.Tensor,
        t:           torch.Tensor,
        anatomy_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Returns (B, out_channels, D, H, W)
        return self.net(x, timesteps=t, class_labels=anatomy_idx)


# ── Composite loss ─────────────────────────────────────────────────────────────

class DiffusionLoss3D:
    """
    Composite diffusion training loss.

    Components:
      1. Noise MSE  (primary — always full weight)
      2. Min-SNR weighted MAE on predicted x0
      3. Min-SNR weighted SSIM on predicted x0

    Min-SNR weighting (Hang et al. 2023): clamp SNR to snr_gamma=5 to prevent
    gradient explosion at t≈0.
    """

    def __init__(
        self,
        noise_mse_weight: float = 1.0,
        mae_weight:       float = 0.25,
        ssim_weight:      float = 0.25,
        snr_gamma:        float = 5.0,
    ):
        self.noise_mse_weight = noise_mse_weight
        self.mae_weight       = mae_weight
        self.ssim_weight      = ssim_weight
        self.snr_gamma        = snr_gamma

        if _MONAI_SSIM_AVAILABLE and ssim_weight > 0:
            self.ssim_fn = MonaiSSIMLoss(spatial_dims=3, data_range=1.0, reduction="none")
        else:
            self.ssim_fn = None
            if ssim_weight > 0:
                warnings.warn(
                    "MONAI SSIMLoss not available; SSIM term disabled.",
                    RuntimeWarning, stacklevel=1,
                )

    def __call__(
        self,
        pred_noise: torch.Tensor,
        true_noise: torch.Tensor,
        pred_x0:    torch.Tensor,
        target_x0:  torch.Tensor,
        snr:        Optional[torch.Tensor] = None,
        mask:       Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        total = torch.zeros((), device=pred_noise.device)
        parts = {}

        # 1. Noise MSE
        if self.noise_mse_weight > 0:
            if mask is not None:
                diff = (pred_noise - true_noise) ** 2
                mse  = (diff * mask).sum() / (mask.sum() + 1e-8)
            else:
                mse = F.mse_loss(pred_noise, true_noise)
            total = total + self.noise_mse_weight * mse
            parts["noise_mse"] = mse.item()

        if self.mae_weight > 0 or (self.ssim_weight > 0 and self.ssim_fn is not None):
            pred_01   = ((pred_x0  + 1.0) * 0.5).clamp(0.0, 1.0)
            target_01 = ((target_x0 + 1.0) * 0.5).clamp(0.0, 1.0)

            if snr is not None:
                w = snr.clamp(max=self.snr_gamma).view(-1, 1, 1, 1, 1)
            else:
                w = torch.ones(pred_x0.shape[0], 1, 1, 1, 1, device=pred_x0.device)

            # 2. Min-SNR weighted MAE
            if self.mae_weight > 0:
                mae_raw = (pred_01 - target_01).abs()
                if mask is not None:
                    mae_map = (mae_raw * mask * w).sum() / ((mask * w).sum() + 1e-8)
                else:
                    mae_map = (mae_raw * w).mean()
                total = total + self.mae_weight * mae_map
                parts["mae"] = mae_map.item()

            # 3. Min-SNR weighted SSIM
            if self.ssim_weight > 0 and self.ssim_fn is not None:
                ssim_raw = self.ssim_fn(pred_01, target_01)
                ssim_raw = ssim_raw.view(pred_x0.shape[0], -1).mean(dim=1)
                w_b = snr.clamp(max=self.snr_gamma) if snr is not None else \
                      torch.ones(pred_x0.shape[0], device=pred_x0.device)
                ssim_weighted = (ssim_raw * w_b).mean()
                total = total + self.ssim_weight * ssim_weighted
                parts["ssim"] = ssim_weighted.item()

        parts["total"] = total.item()
        return total, parts


# ── Gaussian diffusion wrapper ─────────────────────────────────────────────────

class GaussianDiffusion3D(nn.Module):
    """
    3D Gaussian diffusion for MR → CT synthesis.

    Key upgrades vs. original:
      - VS-DDPM: sample_t() draws T_eff from vs_step_buckets per batch,
                 making the model robust at any inference step count.
      - VLB: when the backbone outputs 2 channels, the second channel is
             a log-variance prediction. A VLB term (Nichol & Dhariwal 2021)
             is added to the loss, computed in stable log-space.
    """

    def __init__(
        self,
        model:            MonaiDiffusion3D,
        T:                int   = 1000,
        cosine_s:         float = 0.008,
        noise_mse_weight: float = 1.0,
        mae_weight:       float = 0.25,
        ssim_weight:      float = 0.25,
        snr_gamma:        float = 5.0,
        vlb_weight:       float = 0.001,
        vs_step_buckets:  Optional[List[int]] = None,
    ):
        super().__init__()
        self.model           = model
        self.T               = T
        self.vlb_weight      = vlb_weight
        self.vs_step_buckets = list(vs_step_buckets) if vs_step_buckets else None

        betas     = _cosine_betas(T, s=cosine_s)
        alphas    = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas",      betas)
        self.register_buffer("alphas",     alphas)
        self.register_buffer("alpha_bar",  alpha_bar)
        self.register_buffer("sqrt_ab",    alpha_bar.sqrt())
        self.register_buffer("sqrt_1mab",  (1.0 - alpha_bar).sqrt())

        self.loss_fn = DiffusionLoss3D(
            noise_mse_weight = noise_mse_weight,
            mae_weight       = mae_weight,
            ssim_weight      = ssim_weight,
            snr_gamma        = snr_gamma,
        )

    # ── VS-DDPM timestep sampling ─────────────────────────────────────────────

    def sample_t(self, batch_size: int, device) -> torch.Tensor:
        """
        Sample timesteps for training.
        VS-DDPM: pick T_eff from vs_step_buckets at random, then
                 sample t ~ Uniform[0, T_eff).  Training with mixed
                 step budgets makes the model work at any inference count.
        """
        T_eff = random.choice(self.vs_step_buckets) if self.vs_step_buckets else self.T
        return torch.randint(0, T_eff, (batch_size,), device=device)

    # ── Forward (noising) ────────────────────────────────────────────────────

    def q_sample(
        self,
        x0:    torch.Tensor,
        t:     torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        ab  = self.sqrt_ab[t].view(-1, 1, 1, 1, 1)
        mab = self.sqrt_1mab[t].view(-1, 1, 1, 1, 1)
        return ab * x0 + mab * noise, noise

    # ── VLB term (Improved DDPM) ──────────────────────────────────────────────

    def _vlb_loss(
        self,
        x0:         torch.Tensor,   # (B, 1, D, H, W) clean CT
        xt:         torch.Tensor,   # (B, 1, D, H, W) noisy CT
        t:          torch.Tensor,   # (B,) long
        pred_noise: torch.Tensor,   # (B, 1, D, H, W) — detached
        logvar_v:   torch.Tensor,   # (B, 1, D, H, W) in [-1, 1]
    ) -> torch.Tensor:
        """
        KL(q(x_{t-1}|x_t,x0) || p_θ(x_{t-1}|x_t)) in log space.
        For t=0: Gaussian NLL of the decoder.
        Numerically stable — everything in log space.
        """
        beta_t  = self.betas[t].view(-1, 1, 1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1, 1, 1)
        ab_t    = self.alpha_bar[t].view(-1, 1, 1, 1, 1)

        # ᾱ_{t-1}: 1.0 at t=0
        t_prev  = (t - 1).clamp(min=0)
        ab_tm1  = self.alpha_bar[t_prev].view(-1, 1, 1, 1, 1)
        ab_tm1  = torch.where(
            t.view(-1, 1, 1, 1, 1) == 0,
            torch.ones_like(ab_tm1), ab_tm1,
        )

        # Posterior variance β̃_t
        beta_tilde = ((1 - ab_tm1) / (1 - ab_t).clamp(min=1e-8) * beta_t).clamp(min=1e-20)

        # Learned log-variance: interpolate log(β̃_t) → log(β_t) based on logvar_v ∈ [-1,1]
        log_bv_min = torch.log(beta_tilde)
        log_bv_max = torch.log(beta_t.clamp(min=1e-20))
        frac       = (logvar_v + 1.0) * 0.5           # [-1,1] → [0,1]
        logvar_p   = log_bv_min + frac * (log_bv_max - log_bv_min)

        # Posterior mean μ̃_t(x_t, x_0)
        c_x0  = ab_tm1.sqrt() * beta_t / (1 - ab_t).clamp(min=1e-8)
        c_xt  = alpha_t.sqrt() * (1 - ab_tm1) / (1 - ab_t).clamp(min=1e-8)
        mu_q  = c_x0 * x0 + c_xt * xt

        # Model mean from noise prediction
        pred_x0 = ((xt - (1 - ab_t).sqrt() * pred_noise) /
                   ab_t.sqrt().clamp(min=1e-8)).clamp(-1.0, 1.0)
        mu_p    = c_x0 * pred_x0 + c_xt * xt

        # KL(q||p) in log space: numerically stable form
        logvar_q = torch.log(beta_tilde)
        kl = 0.5 * (
            logvar_p - logvar_q - 1.0
            + torch.exp(logvar_q - logvar_p)
            + (mu_q - mu_p).pow(2) * torch.exp(-logvar_p)
        )

        # For t=0: replace KL with Gaussian NLL of decoder
        nll = (0.5 * (x0 - pred_x0).pow(2) * torch.exp(-logvar_p)
               + 0.5 * logvar_p)

        is_t0 = (t == 0).view(-1, 1, 1, 1, 1)
        vlb   = torch.where(is_t0, nll, kl)

        return vlb.mean()

    # ── Training loss ─────────────────────────────────────────────────────────

    def p_loss(
        self,
        x0:          torch.Tensor,
        mr:          torch.Tensor,
        t:           torch.Tensor,
        anatomy_idx: Optional[torch.Tensor] = None,
        mask:        Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        noise = torch.randn_like(x0)
        x_t, _ = self.q_sample(x0, t, noise)
        x_in   = torch.cat([mr, x_t], dim=1)   # (B, 2, D, H, W)

        model_out = self.model(x_in, t, anatomy_idx)

        # Split noise and log-variance predictions
        if model_out.shape[1] == 2:
            pred_noise = model_out[:, :1]
            logvar_v   = model_out[:, 1:].tanh()   # clamp to [-1, 1]
        else:
            pred_noise = model_out
            logvar_v   = None

        # Predicted x0 for auxiliary losses
        ab      = self.sqrt_ab[t].view(-1, 1, 1, 1, 1)
        mab     = self.sqrt_1mab[t].view(-1, 1, 1, 1, 1)
        pred_x0 = ((x_t - mab * pred_noise) / ab).clamp(-1.0, 1.0)

        ab_t = self.alpha_bar[t]
        snr  = ab_t / (1.0 - ab_t + 1e-8)

        total, parts = self.loss_fn(pred_noise, noise, pred_x0, x0, snr=snr, mask=mask)

        # VLB term — only when backbone predicts log-variance
        if self.vlb_weight > 0 and logvar_v is not None:
            vlb   = self._vlb_loss(x0, x_t, t, pred_noise.detach(), logvar_v)
            total = total + self.vlb_weight * vlb
            parts["vlb"] = vlb.item()

        parts["total"] = total.item()
        return total, parts

    # ── DDIM inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(
        self,
        mr:          torch.Tensor,
        anatomy_idx: Optional[torch.Tensor] = None,
        steps:       int   = 50,
        eta:         float = 0.0,
    ) -> torch.Tensor:
        B, _, D, H, W = mr.shape
        device = mr.device

        t_seq = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=device)
        x     = torch.randn(B, 1, D, H, W, device=device)

        for i, t_val in enumerate(t_seq):
            t_prev_val = t_seq[i + 1] if i + 1 < steps else torch.tensor(-1, device=device)
            t_batch    = t_val.expand(B)

            x_in      = torch.cat([mr, x], dim=1)
            model_out = self.model(x_in, t_batch, anatomy_idx)
            # Use only the noise channel regardless of out_channels
            pred_noise = model_out[:, :1]

            ab_t  = self.alpha_bar[t_val]
            ab_tp = self.alpha_bar[t_prev_val] if t_prev_val >= 0 else \
                    torch.ones(1, device=device)

            x0_pred = ((x - (1.0 - ab_t).sqrt() * pred_noise) /
                       ab_t.sqrt()).clamp(-1.0, 1.0)
            sigma   = eta * ((1 - ab_tp) / (1 - ab_t) * (1 - ab_t / ab_tp)).clamp(min=0).sqrt()
            dir_xt  = (1.0 - ab_tp - sigma ** 2).clamp(min=0).sqrt() * pred_noise
            noise   = sigma * torch.randn_like(x) if eta > 0 else 0.0
            x       = ab_tp.sqrt() * x0_pred + dir_xt + noise

        return x.clamp(-1.0, 1.0)


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = MonaiDiffusion3D(
        in_channels=2, out_channels=2,
        channels=(32, 64, 128, 128),
        attention_levels=(False, True, True, True),
        num_res_blocks=2,
        n_anatomy=3,
    ).to(device)

    diffusion = GaussianDiffusion3D(
        backbone, T=1000,
        vlb_weight=0.001,
        vs_step_buckets=[50, 100, 250, 500, 1000],
    ).to(device)

    B, D, H, W = 1, 16, 64, 64
    mr   = torch.randn(B, 1, D, H, W, device=device)
    ct   = torch.randn(B, 1, D, H, W, device=device)
    anat = torch.zeros(B, dtype=torch.long, device=device)

    t = diffusion.sample_t(B, device)
    loss, parts = diffusion.p_loss(ct, mr, t, anat)
    print(f"Training loss: {loss.item():.4f}  parts: {parts}")

    sample = diffusion.ddim_sample(mr, anat, steps=5)
    print(f"DDIM sample shape: {sample.shape}")

    n_params = sum(p.numel() for p in backbone.parameters()) / 1e6
    print(f"MonaiDiffusion3D v2 params: {n_params:.1f}M")
