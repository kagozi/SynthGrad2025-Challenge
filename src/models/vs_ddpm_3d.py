"""
3D Variable-Step DDPM (VS-DDPM) for MR → sCT synthesis — SynthRAD2025 Task 1.

Inspired by the Faking_it submission (SynthRAD2025 challenge winner):
  - 3D patch-based denoising (128×128×32 patches)
  - Swin-ViT-style 3D window attention in the U-Net backbone
  - Epsilon + learned variance prediction (2-channel output)
  - Loss: L_simple + 0.001 * L_VLB + L_MAE + L_SSIM
  - Cosine noise schedule, DDIM deterministic sampler

References:
  Ho et al. 2020       — DDPM          (https://arxiv.org/abs/2006.11239)
  Nichol & Dhariwal 2021 — Improved DDPM (https://arxiv.org/abs/2102.09672)
  Song et al. 2020     — DDIM          (https://arxiv.org/abs/2010.02502)
  Liu et al. 2021      — Swin Transformer (https://arxiv.org/abs/2103.14030)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import reusable losses from the project
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.losses import MAELoss, MSSSIMLoss


# ── Time (sinusoidal) embedding ────────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps (copied from ddpm.py)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device   = t.device
        half_dim = self.dim // 2
        emb      = math.log(10000) / (half_dim - 1)
        emb      = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb      = t.float()[:, None] * emb[None, :]   # (B, half_dim)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _num_groups(channels: int, max_groups: int = 32) -> int:
    """Return the largest divisor of `channels` that is ≤ max_groups."""
    g = min(max_groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return max(g, 1)


# ── 3D ResBlock ────────────────────────────────────────────────────────────────

class ResBlock3D(nn.Module):
    """
    3D ResBlock with GroupNorm, SiLU, and FiLM conditioning from time embedding.

    Layout: norm → silu → conv3d → FiLM(time) → norm → silu → dropout → conv3d
            + residual projection if in_ch ≠ out_ch
    """

    def __init__(
        self,
        in_ch:        int,
        out_ch:       int,
        time_emb_dim: int,
        groups:       int   = 32,
        dropout:      float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_ch, groups), in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)

        # FiLM: time embedding → scale + shift for second norm
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch * 2),
        )

        self.norm2    = nn.GroupNorm(_num_groups(out_ch, groups), out_ch)
        self.drop     = nn.Dropout(dropout)
        self.conv2    = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.residual = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        # FiLM conditioning — broadcast over D, H, W
        scale, shift = self.time_proj(temb).chunk(2, dim=-1)
        h = self.norm2(h) * (1.0 + scale[:, :, None, None, None]) \
                          + shift[:, :, None, None, None]

        h = self.conv2(self.drop(F.silu(h)))
        return h + self.residual(x)


# ── 3D Window Attention ────────────────────────────────────────────────────────

class WindowAttention3D(nn.Module):
    """
    3D window-based multi-head self-attention (Swin-style).

    Partitions the (D, H, W) volume into non-overlapping windows and runs
    self-attention within each window.  Handles non-divisible spatial sizes
    by symmetric padding before partitioning and cropping after.

    Args:
        dim:         number of input channels
        window_size: (wd, wh, ww) window dimensions in voxels
        num_heads:   number of attention heads
        groups:      GroupNorm groups for pre-norm
    """

    def __init__(
        self,
        dim:         int,
        window_size: tuple = (4, 4, 4),
        num_heads:   int   = 8,
        groups:      int   = 32,
    ):
        super().__init__()
        self.window_size = window_size
        self.norm        = nn.GroupNorm(_num_groups(dim, groups), dim)
        self.qkv         = nn.Linear(dim, dim * 3)
        self.proj        = nn.Linear(dim, dim)
        self.num_heads   = num_heads
        self.head_dim    = dim // num_heads
        self.scale       = self.head_dim ** -0.5

    @staticmethod
    def _pad_to_multiple(
        x: torch.Tensor,
        window_size: tuple,
    ) -> tuple[torch.Tensor, tuple]:
        """Pad spatial dims to be divisible by window_size.  Returns (padded, padding)."""
        _, _, D, H, W = x.shape
        wd, wh, ww    = window_size

        pd = (wd - D % wd) % wd
        ph = (wh - H % wh) % wh
        pw = (ww - W % ww) % ww

        if pd > 0 or ph > 0 or pw > 0:
            # F.pad pads the last dims in reverse order: (W_before, W_after, H_before, ...)
            x = F.pad(x, (0, pw, 0, ph, 0, pd))

        return x, (pd, ph, pw)

    @staticmethod
    def _partition_windows(
        x: torch.Tensor,
        window_size: tuple,
    ) -> tuple[torch.Tensor, tuple]:
        """
        x: (B, C, D, H, W) → windows: (B*nW, wd*wh*ww, C)

        Returns (windows, (B, D, H, W, nD, nH, nW)) for reversing.
        """
        B, C, D, H, W = x.shape
        wd, wh, ww     = window_size

        nD = D // wd
        nH = H // wh
        nW = W // ww

        # (B, C, nD, wd, nH, wh, nW, ww)
        x = x.reshape(B, C, nD, wd, nH, wh, nW, ww)
        # → (B, nD, nH, nW, wd, wh, ww, C)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        # → (B*nD*nH*nW, wd*wh*ww, C)
        x = x.reshape(B * nD * nH * nW, wd * wh * ww, C)

        return x, (B, D, H, W, nD, nH, nW)

    @staticmethod
    def _reverse_windows(
        windows: torch.Tensor,
        meta: tuple,
        window_size: tuple,
    ) -> torch.Tensor:
        """Reverse _partition_windows. Returns (B, C, D, H, W)."""
        B, D, H, W, nD, nH, nW = meta
        wd, wh, ww              = window_size
        C                       = windows.shape[-1]

        # (B*nD*nH*nW, wd*wh*ww, C) → (B, nD, nH, nW, wd, wh, ww, C)
        x = windows.reshape(B, nD, nH, nW, wd, wh, ww, C)
        # → (B, C, nD, wd, nH, wh, nW, ww)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        # → (B, C, D, H, W)
        x = x.reshape(B, C, D, H, W)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, D, H, W) → (B, C, D, H, W)"""
        B, C, D, H, W = x.shape

        # 1. Pre-norm (GroupNorm on channel dim)
        h = self.norm(x)

        # 2. Pad to multiples of window_size
        h, (pd, ph, pw) = self._pad_to_multiple(h, self.window_size)
        _, _, Dp, Hp, Wp = h.shape

        # 3. Partition into windows: (B*nW, L, C)
        windows, meta = self._partition_windows(h, self.window_size)

        # 4. Multi-head attention
        L = windows.shape[1]
        qkv = self.qkv(windows)                          # (B*nW, L, 3*C)
        qkv = qkv.reshape(windows.shape[0], L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                # (3, B*nW, heads, L, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale    # (B*nW, heads, L, L)
        attn = attn.softmax(dim=-1)

        out = (attn @ v)                                  # (B*nW, heads, L, head_dim)
        out = out.transpose(1, 2).reshape(windows.shape[0], L, C)
        out = self.proj(out)                              # (B*nW, L, C)

        # 5. Reverse windows
        out = self._reverse_windows(out, meta, self.window_size)  # (B, C, Dp, Hp, Wp)

        # 6. Unpad
        if pd > 0 or ph > 0 or pw > 0:
            out = out[:, :, :D, :H, :W].contiguous()

        # 7. Residual add
        return x + out


# ── 3D U-Net backbone ──────────────────────────────────────────────────────────

class DDPMUNet3D(nn.Module):
    """
    3D Denoising U-Net for VS-DDPM.

    Architecture:
      channels = [32, 64, 128, 256]

      Encoder:
        Level 0: 2×ResBlock3D(32→32)  → skip → DownConv3d(32,  64,  stride=(1,2,2))
        Level 1: 2×ResBlock3D(64→64)  → skip → DownConv3d(64,  128, stride=(2,2,2))
        Level 2: 2×ResBlock3D(128→128) + WindowAttn3D(128, w=4, h=4) → skip
                 → DownConv3d(128, 256, stride=(2,2,2))

      Bottleneck:
        2×ResBlock3D(256,256) + WindowAttn3D(256, w=4, h=4)

      Decoder (symmetric, trilinear upsample + concat skip):
        Level 2: concat(256+128) → 2×ResBlock3D(384→128) + WindowAttn3D(128)
        Level 1: concat(128+64)  → 2×ResBlock3D(192→64)
        Level 0: concat(64+32)   → 2×ResBlock3D(96→32)

      Output: GroupNorm(32,32) → SiLU → Conv3d(32, 2, 3, p=1)
              2 channels: [eps_pred, v_pred]  (epsilon + learned variance)

    Args:
        in_channels:   2  (MR + noisy CT concatenated channel-wise)
        out_channels:  2  (eps + v for learned variance)
        base_ch:       32
        time_emb_dim:  256
        n_anatomy:     3  (HN / TH / AB)
        dropout:       0.0
    """

    def __init__(
        self,
        in_channels:  int   = 2,
        out_channels: int   = 2,
        base_ch:      int   = 32,
        time_emb_dim: int   = 256,
        n_anatomy:    int   = 3,
        dropout:      float = 0.0,
    ):
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]  # [32, 64, 128, 256]

        # Time + anatomy embedding
        self.time_emb = nn.Sequential(
            SinusoidalEmbedding(base_ch),
            nn.Linear(base_ch, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.anat_emb = nn.Embedding(n_anatomy, time_emb_dim)

        # Input projection
        self.in_conv = nn.Conv3d(in_channels, ch[0], 3, padding=1)

        # ── Encoder ──────────────────────────────────────────────────────────
        # Level 0: 32 → 32, then downsample to 64
        self.enc0_res = nn.ModuleList([
            ResBlock3D(ch[0], ch[0], time_emb_dim, dropout=dropout),
            ResBlock3D(ch[0], ch[0], time_emb_dim, dropout=dropout),
        ])
        self.down0 = nn.Conv3d(ch[0], ch[1], 3, stride=(1, 2, 2), padding=1)

        # Level 1: 64 → 64, then downsample to 128
        self.enc1_res = nn.ModuleList([
            ResBlock3D(ch[1], ch[1], time_emb_dim, dropout=dropout),
            ResBlock3D(ch[1], ch[1], time_emb_dim, dropout=dropout),
        ])
        self.down1 = nn.Conv3d(ch[1], ch[2], 3, stride=(2, 2, 2), padding=1)

        # Level 2: 128 → 128 + window attention, then downsample to 256
        self.enc2_res = nn.ModuleList([
            ResBlock3D(ch[2], ch[2], time_emb_dim, dropout=dropout),
            ResBlock3D(ch[2], ch[2], time_emb_dim, dropout=dropout),
        ])
        self.enc2_attn = WindowAttention3D(ch[2], window_size=(4, 4, 4), num_heads=4)
        self.down2 = nn.Conv3d(ch[2], ch[3], 3, stride=(2, 2, 2), padding=1)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.mid_res1 = ResBlock3D(ch[3], ch[3], time_emb_dim, dropout=dropout)
        self.mid_res2 = ResBlock3D(ch[3], ch[3], time_emb_dim, dropout=dropout)
        self.mid_attn = WindowAttention3D(ch[3], window_size=(4, 4, 4), num_heads=8)

        # ── Decoder ───────────────────────────────────────────────────────────
        # Level 2: upsample 256 → concat(256+128)=384 → 128 + window attention
        self.dec2_res = nn.ModuleList([
            ResBlock3D(ch[3] + ch[2], ch[2], time_emb_dim, dropout=dropout),
            ResBlock3D(ch[2],         ch[2], time_emb_dim, dropout=dropout),
        ])
        self.dec2_attn = WindowAttention3D(ch[2], window_size=(4, 4, 4), num_heads=4)

        # Level 1: upsample 128 → concat(128+64)=192 → 64
        self.dec1_res = nn.ModuleList([
            ResBlock3D(ch[2] + ch[1], ch[1], time_emb_dim, dropout=dropout),
            ResBlock3D(ch[1],         ch[1], time_emb_dim, dropout=dropout),
        ])

        # Level 0: upsample 64 → concat(64+32)=96 → 32
        self.dec0_res = nn.ModuleList([
            ResBlock3D(ch[1] + ch[0], ch[0], time_emb_dim, dropout=dropout),
            ResBlock3D(ch[0],         ch[0], time_emb_dim, dropout=dropout),
        ])

        # Output head
        self.out_norm = nn.GroupNorm(_num_groups(ch[0], 32), ch[0])
        self.out_conv = nn.Conv3d(ch[0], out_channels, 3, padding=1)

    def forward(
        self,
        x:           torch.Tensor,            # (B, 2, D, H, W)
        t:           torch.Tensor,            # (B,) integer timestep
        anatomy_idx: Optional[torch.Tensor] = None,   # (B,)
    ) -> torch.Tensor:                        # (B, 2, D, H, W)
        # Time + anatomy embedding
        temb = self.time_emb(t)
        if anatomy_idx is not None:
            temb = temb + self.anat_emb(anatomy_idx)

        h = self.in_conv(x)                  # (B, 32, D, H, W)

        # ── Encoder ──────────────────────────────────────────────────────────
        # Level 0
        for r in self.enc0_res:
            h = r(h, temb)
        skip0 = h                             # (B, 32, D, H, W)
        h = self.down0(h)                     # (B, 64, D, H/2, W/2)

        # Level 1
        for r in self.enc1_res:
            h = r(h, temb)
        skip1 = h                             # (B, 64, D, H/2, W/2)
        h = self.down1(h)                     # (B, 128, D/2, H/4, W/4)

        # Level 2
        for r in self.enc2_res:
            h = r(h, temb)
        h = self.enc2_attn(h)
        skip2 = h                             # (B, 128, D/2, H/4, W/4)
        h = self.down2(h)                     # (B, 256, D/4, H/8, W/8)

        # ── Bottleneck ────────────────────────────────────────────────────────
        h = self.mid_res1(h, temb)
        h = self.mid_res2(h, temb)
        h = self.mid_attn(h)

        # ── Decoder ───────────────────────────────────────────────────────────
        # Level 2
        h = F.interpolate(h, size=skip2.shape[2:], mode="trilinear", align_corners=False)
        h = torch.cat([h, skip2], dim=1)      # (B, 384, ...)
        for r in self.dec2_res:
            h = r(h, temb)
        h = self.dec2_attn(h)

        # Level 1
        h = F.interpolate(h, size=skip1.shape[2:], mode="trilinear", align_corners=False)
        h = torch.cat([h, skip1], dim=1)      # (B, 192, ...)
        for r in self.dec1_res:
            h = r(h, temb)

        # Level 0
        h = F.interpolate(h, size=skip0.shape[2:], mode="trilinear", align_corners=False)
        h = torch.cat([h, skip0], dim=1)      # (B, 96, ...)
        for r in self.dec0_res:
            h = r(h, temb)

        return self.out_conv(F.silu(self.out_norm(h)))   # (B, 2, D, H, W)


# ── Gaussian Diffusion (3D, VLB) ──────────────────────────────────────────────

class GaussianDiffusion3D(nn.Module):
    """
    3D Gaussian diffusion with cosine noise schedule and learned variance (VLB).

    Training loss (p_loss) returns a dict:
      'simple'  — MSE between predicted noise and true noise
      'vlb'     — Improved DDPM VLB (KL divergence term)
      'mae'     — L1 on x0_pred vs x0 in normalised [-1,1] space
      'ssim'    — 1-MS-SSIM on x0_pred vs x0
      'total'   — L_simple + 0.001*L_vlb + 1.0*L_mae + 1.0*L_ssim

    Inference: deterministic DDIM (ddim_sample).
    """

    def __init__(
        self,
        model:          DDPMUNet3D,
        T:              int   = 1000,
        s:              float = 0.008,   # cosine schedule offset
        lambda_vlb:     float = 0.001,
        lambda_mae:     float = 1.0,
        lambda_ssim:    float = 1.0,
    ):
        super().__init__()
        self.model       = model
        self.T           = T
        self.lambda_vlb  = lambda_vlb
        self.lambda_mae  = lambda_mae
        self.lambda_ssim = lambda_ssim

        # ── Cosine noise schedule ──────────────────────────────────────────────
        t_arr     = torch.arange(T + 1, dtype=torch.float64)
        f         = torch.cos((t_arr / T + s) / (1.0 + s) * math.pi * 0.5) ** 2
        alpha_bar = f / f[0]
        betas     = (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999)
        alphas    = 1.0 - betas
        alpha_bar = alpha_bar[1:].float()   # length T (ᾱ_1 … ᾱ_T)

        # ᾱ_{t-1}: shift right by 1, with ᾱ_0 = 1.0  (needed for posterior)
        alpha_bar_prev = torch.cat([
            torch.ones(1, dtype=torch.float32),
            alpha_bar[:-1],
        ])  # length T

        # β̃_t = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) * β_t  (posterior variance)
        beta_tilde = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar) * betas.float()
        # For t=0 the posterior degenerates to β_1, so clamp ≥ β_1
        beta_tilde = beta_tilde.clamp(min=betas[0].item())

        self.register_buffer("betas",           betas.float())
        self.register_buffer("alphas",          alphas.float())
        self.register_buffer("alpha_bar",       alpha_bar)
        self.register_buffer("alpha_bar_prev",  alpha_bar_prev)
        self.register_buffer("sqrt_ab",         alpha_bar.sqrt())
        self.register_buffer("sqrt_1mab",       (1.0 - alpha_bar).sqrt())
        self.register_buffer("sqrt_alphas",     alphas.float().sqrt())
        # For VLB
        self.register_buffer("log_betas",       betas.float().log())
        self.register_buffer("log_beta_tilde",  beta_tilde.log())

        # Precomputed loss helpers
        self._mae_loss  = MAELoss()
        self._ssim_loss = MSSSIMLoss()

    # ── Forward (noising) ─────────────────────────────────────────────────────

    def q_sample(
        self,
        x0:    torch.Tensor,   # (B, 1, D, H, W) clean CT in [-1, 1]
        t:     torch.Tensor,   # (B,) long
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab   = self.sqrt_ab[t][:, None, None, None, None]
        sqrt_1mab = self.sqrt_1mab[t][:, None, None, None, None]
        return sqrt_ab * x0 + sqrt_1mab * noise, noise

    # ── Training loss ─────────────────────────────────────────────────────────

    def p_loss(
        self,
        x0:          torch.Tensor,                   # (B, 1, D, H, W) clean CT
        mr:          torch.Tensor,                   # (B, 1, D, H, W) MR conditioning
        t:           torch.Tensor,                   # (B,) long
        anatomy_idx: Optional[torch.Tensor] = None,  # (B,)
        mask:        Optional[torch.Tensor] = None,  # (B, 1, D, H, W)
    ) -> dict:
        noise  = torch.randn_like(x0)
        x_t, _ = self.q_sample(x0, t, noise)

        x_in = torch.cat([mr, x_t], dim=1)           # (B, 2, D, H, W)
        out  = self.model(x_in, t, anatomy_idx)       # (B, 2, D, H, W)

        eps_pred = out[:, :1]    # predicted noise
        v_pred   = out[:, 1:]    # predicted log-variance interpolant

        # ── L_simple ─────────────────────────────────────────────────────────
        diff = (eps_pred - noise) ** 2
        if mask is not None:
            L_simple = (diff * mask).sum() / (mask.sum() + 1e-8)
        else:
            L_simple = diff.mean()

        # ── VLB (Nichol & Dhariwal 2021) ─────────────────────────────────────
        # Posterior log-variance: log β̃_t
        log_var_q = self.log_beta_tilde[t][:, None, None, None, None]  # (B,1,1,1,1)

        # Model log-variance: interpolate between log_beta and log_beta_tilde
        # using the learned v_pred in [-1, 1]
        frac      = (v_pred.clamp(-1.0, 1.0) + 1.0) / 2.0             # → [0, 1]
        log_beta  = self.log_betas[t][:, None, None, None, None]
        log_var_p = frac * log_beta + (1.0 - frac) * log_var_q

        # Predicted x0 (stop gradient through eps_pred for VLB branch)
        eps_pred_sg = eps_pred.detach()
        sqrt_ab_t   = self.sqrt_ab[t][:, None, None, None, None]
        sqrt_1mab_t = self.sqrt_1mab[t][:, None, None, None, None]

        x0_recon = (x_t - sqrt_1mab_t * eps_pred_sg) / sqrt_ab_t.clamp(min=1e-8)
        x0_recon = x0_recon.clamp(-1.0, 1.0)

        # Posterior mean: μ̃_t(x_t, x0)
        betas_t        = self.betas[t][:, None, None, None, None]
        alphas_t       = self.alphas[t][:, None, None, None, None]
        sqrt_alphas_t  = self.sqrt_alphas[t][:, None, None, None, None]
        ab_prev_t      = self.alpha_bar_prev[t][:, None, None, None, None]
        ab_t           = self.alpha_bar[t][:, None, None, None, None]

        coef1  = ab_prev_t.sqrt() * betas_t / (1.0 - ab_t)
        coef2  = alphas_t.sqrt() * (1.0 - ab_prev_t) / (1.0 - ab_t)
        mu_q   = coef1 * x0 + coef2 * x_t

        # Model mean: μ_θ from eps prediction (with stop grad)
        mu_p   = (x_t - betas_t / sqrt_1mab_t.clamp(min=1e-8) * eps_pred_sg) \
                 / sqrt_alphas_t.clamp(min=1e-8)

        # KL divergence (Gaussian, per element)
        var_q  = log_var_q.exp()
        var_p  = log_var_p.exp()
        kl     = (log_var_p - log_var_q
                  + var_q / var_p.clamp(min=1e-8)
                  + (mu_q - mu_p).pow(2) / var_p.clamp(min=1e-8)
                  - 1.0) * 0.5
        L_vlb  = kl.mean()

        # ── x0_pred for MAE / SSIM ────────────────────────────────────────────
        # Use the non-stop-grad eps_pred for reconstruction.
        # Restrict to low-noise timesteps (SNR > 1 ≈ t < T/2) to avoid
        # unbounded gradients ∝ sqrt(1-ᾱ)/sqrt(ᾱ) at high t.
        x0_pred = (x_t - sqrt_1mab_t * eps_pred) / sqrt_ab_t.clamp(min=1e-8)
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        snr     = self.alpha_bar[t] / (1.0 - self.alpha_bar[t] + 1e-8)  # (B,)
        low_idx = (snr > 1.0).nonzero(as_tuple=True)[0]

        if low_idx.numel() > 0:
            m_low   = mask[low_idx] if mask is not None else None
            L_mae   = self._mae_loss (x0_pred[low_idx], x0[low_idx], m_low)
            L_ssim  = self._ssim_loss(x0_pred[low_idx], x0[low_idx], m_low)
        else:
            L_mae  = x0_pred.new_zeros(())
            L_ssim = x0_pred.new_zeros(())

        total = (L_simple
                 + self.lambda_vlb  * L_vlb
                 + self.lambda_mae  * L_mae
                 + self.lambda_ssim * L_ssim)

        return {
            "simple": L_simple,
            "vlb":    L_vlb,
            "mae":    L_mae,
            "ssim":   L_ssim,
            "total":  total,
        }

    # ── DDIM inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(
        self,
        mr:          torch.Tensor,           # (B, 1, D, H, W)
        anatomy_idx: Optional[torch.Tensor] = None,
        steps:       int   = 20,
        eta:         float = 0.0,            # 0 = deterministic DDIM
    ) -> torch.Tensor:                       # (B, 1, D, H, W)
        B, _, D, H, W = mr.shape
        device        = mr.device

        # Evenly spaced timestep sequence (T-1 → 0)
        t_seq      = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=device)
        t_prev_seq = torch.cat([t_seq[1:], torch.tensor([-1], device=device)])

        x = torch.randn(B, 1, D, H, W, device=device)

        for t_val, t_prev_val in zip(t_seq, t_prev_seq):
            t_batch = t_val.expand(B)
            x_in    = torch.cat([mr, x], dim=1)         # (B, 2, D, H, W)
            out     = self.model(x_in, t_batch, anatomy_idx)
            eps     = out[:, :1]                         # only epsilon channel

            ab_t  = self.alpha_bar[t_val]
            ab_tp = (self.alpha_bar[t_prev_val]
                     if t_prev_val >= 0
                     else torch.ones(1, device=device))

            # Predicted x0
            x0_pred = (x - (1.0 - ab_t).sqrt() * eps) / ab_t.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            # Direction pointing to x_t
            sigma = eta * ((1 - ab_tp) / (1 - ab_t) * (1 - ab_t / ab_tp)).sqrt()
            dir_  = (1.0 - ab_tp - sigma ** 2).clamp(min=0.0).sqrt() * eps

            noise = sigma * torch.randn_like(x) if eta > 0 else 0.0
            x     = ab_tp.sqrt() * x0_pred + dir_ + noise

        return x.clamp(-1.0, 1.0)


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet3d = DDPMUNet3D(
        in_channels=2, out_channels=2, base_ch=32,
        time_emb_dim=256, n_anatomy=3,
    ).to(device)

    diffusion3d = GaussianDiffusion3D(unet3d, T=1000).to(device)

    B, D, H, W = 2, 32, 128, 128
    mr   = torch.randn(B, 1, D, H, W, device=device)
    ct   = torch.randn(B, 1, D, H, W, device=device)
    t    = torch.randint(0, 1000, (B,), device=device)
    anat = torch.tensor([0, 2], device=device)

    loss_dict = diffusion3d.p_loss(ct, mr, t, anat)
    print("Loss dict:", {k: f"{v.item():.4f}" for k, v in loss_dict.items()})

    sample = diffusion3d.ddim_sample(mr, anat, steps=5)
    print(f"DDIM sample shape: {sample.shape}")

    n_params = sum(p.numel() for p in unet3d.parameters()) / 1e6
    print(f"DDPMUNet3D params: {n_params:.1f}M")
