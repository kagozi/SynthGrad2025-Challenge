"""
2D Conditional DDPM for MR → sCT synthesis (SynthRAD2025 Task 1).

Architecture:
  - Conditioning: MR slice concatenated channel-wise with noisy CT
  - Denoising U-Net: GroupNorm + SiLU + ResBlocks + self-attention at bottleneck
  - Anatomy conditioning: learned embedding summed into time embedding
  - Parameterisation: epsilon (predict noise)
  - Noise schedule: cosine (Nichol & Dhariwal 2021)
  - Inference: DDIM deterministic sampler (50 steps, eta=0)

Input:   MR (B, 1, H, W) + noisy CT (B, 1, H, W) → cat → (B, 2, H, W)
Output:  predicted noise (B, 1, H, W)

References:
  Ho et al. 2020 — DDPM (https://arxiv.org/abs/2006.11239)
  Nichol & Dhariwal 2021 — improved DDPM (https://arxiv.org/abs/2102.09672)
  Song et al. 2020 — DDIM (https://arxiv.org/abs/2010.02502)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Time (sinusoidal) embedding ────────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

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


# ── Building blocks ────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    ResBlock with GroupNorm, SiLU, and FiLM conditioning from time embedding.

    Layout: norm → silu → conv → FiLM(time) → norm → silu → dropout → conv
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
        self.norm1 = nn.GroupNorm(min(groups, in_ch),  in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        # FiLM: time embedding → scale + shift for second norm
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch * 2),
        )

        self.norm2   = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.drop    = nn.Dropout(dropout)
        self.conv2   = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        # FiLM conditioning
        scale, shift = self.time_proj(temb).chunk(2, dim=-1)
        h = self.norm2(h) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]

        h = self.conv2(self.drop(F.silu(h)))
        return h + self.residual(x)


class SelfAttentionBlock(nn.Module):
    """Multi-head self-attention (spatial), GroupNorm pre-norm."""

    def __init__(self, ch: int, num_heads: int = 8, groups: int = 32):
        super().__init__()
        self.norm  = nn.GroupNorm(min(groups, ch), ch)
        self.attn  = nn.MultiheadAttention(ch, num_heads, batch_first=True)
        self.proj  = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(h)


class DownBlock(nn.Module):
    """ResBlock(s) + optional attention + strided downsample."""

    def __init__(
        self,
        in_ch:        int,
        out_ch:       int,
        time_emb_dim: int,
        num_res:      int   = 2,
        use_attn:     bool  = False,
        groups:       int   = 32,
        dropout:      float = 0.0,
    ):
        super().__init__()
        self.resnets = nn.ModuleList(
            [ResBlock(
                in_ch if i == 0 else out_ch,
                out_ch, time_emb_dim, groups, dropout,
            ) for i in range(num_res)]
        )
        self.attn     = SelfAttentionBlock(out_ch) if use_attn else None
        self.downsamp = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, temb: torch.Tensor):
        for r in self.resnets:
            x = r(x, temb)
        if self.attn is not None:
            x = self.attn(x)
        skip = x
        x    = self.downsamp(x)
        return x, skip


class UpBlock(nn.Module):
    """Upsample → concat skip → ResBlock(s) + optional attention."""

    def __init__(
        self,
        in_ch:        int,
        skip_ch:      int,
        out_ch:       int,
        time_emb_dim: int,
        num_res:      int   = 2,
        use_attn:     bool  = False,
        groups:       int   = 32,
        dropout:      float = 0.0,
    ):
        super().__init__()
        # First ResBlock fuses upsampled features + skip connection
        self.resnets = nn.ModuleList(
            [ResBlock(
                (in_ch + skip_ch) if i == 0 else out_ch,
                out_ch, time_emb_dim, groups, dropout,
            ) for i in range(num_res)]
        )
        self.attn = SelfAttentionBlock(out_ch) if use_attn else None

    def forward(self, x: torch.Tensor, skip: torch.Tensor, temb: torch.Tensor):
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        for r in self.resnets:
            x = r(x, temb)
        if self.attn is not None:
            x = self.attn(x)
        return x


# ── Denoising U-Net ────────────────────────────────────────────────────────────

class DDPMUNet(nn.Module):
    """
    Denoising U-Net for 2D conditional DDPM.

    Input:  (B, in_channels, H, W)    — cat(mr_slice, noisy_ct)
    Output: (B, out_channels, H, W)   — predicted noise ε

    Args:
        in_channels:    2 (MR + noisy CT concatenated)
        out_channels:   1 (sCT noise)
        base_ch:        base feature count (default 64)
        ch_mult:        channel multipliers per level (default (1,2,4,8))
        num_res_blocks: ResBlocks per encoder/decoder level (default 2)
        attn_levels:    which depth levels get self-attention (0-indexed from bottom)
        time_emb_dim:   time embedding MLP output size (default 256)
        n_anatomy:      number of anatomy classes for conditioning (3)
        dropout:        dropout in ResBlocks (default 0.0)
    """

    def __init__(
        self,
        in_channels:  int   = 2,
        out_channels: int   = 1,
        base_ch:      int   = 64,
        ch_mult:      tuple = (1, 2, 4, 8),
        num_res:      int   = 2,
        attn_levels:  tuple = (3,),   # attention only at deepest encoder level
        time_emb_dim: int   = 256,
        n_anatomy:    int   = 3,
        dropout:      float = 0.0,
    ):
        super().__init__()
        channels = [base_ch * m for m in ch_mult]

        # Time embedding: sinusoidal → MLP
        self.time_emb = nn.Sequential(
            SinusoidalEmbedding(base_ch),
            nn.Linear(base_ch, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        # Anatomy conditioning added to time embedding
        self.anat_emb = nn.Embedding(n_anatomy, time_emb_dim)

        # Input projection
        self.in_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            use_attn = (i in attn_levels)
            self.down_blocks.append(
                DownBlock(in_ch, out_ch, time_emb_dim, num_res, use_attn, dropout=dropout)
            )
            in_ch = out_ch

        # Bottleneck (two ResBlocks + attention)
        self.mid_res1  = ResBlock(in_ch, in_ch, time_emb_dim)
        self.mid_attn  = SelfAttentionBlock(in_ch)
        self.mid_res2  = ResBlock(in_ch, in_ch, time_emb_dim)

        # Decoder
        self.up_blocks = nn.ModuleList()
        for i, skip_ch in enumerate(reversed(channels)):
            out_ch   = skip_ch
            use_attn = ((len(channels) - 1 - i) in attn_levels)
            self.up_blocks.append(
                UpBlock(in_ch, skip_ch, out_ch, time_emb_dim, num_res, use_attn, dropout=dropout)
            )
            in_ch = out_ch

        # Output
        self.out_norm = nn.GroupNorm(min(32, channels[0]), channels[0])
        self.out_conv = nn.Conv2d(channels[0], out_channels, 3, padding=1)

    def forward(
        self,
        x:           torch.Tensor,            # (B, 2, H, W) = cat(mr, noisy_ct)
        t:           torch.Tensor,            # (B,) integer timestep
        anatomy_idx: Optional[torch.Tensor] = None,  # (B,)
    ) -> torch.Tensor:
        # Build time + anatomy embedding
        temb = self.time_emb(t)
        if anatomy_idx is not None:
            temb = temb + self.anat_emb(anatomy_idx)

        h = self.in_conv(x)

        # Encoder — collect skips
        skips = []
        for down in self.down_blocks:
            h, skip = down(h, temb)
            skips.append(skip)

        # Bottleneck
        h = self.mid_res1(h, temb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, temb)

        # Decoder
        for up, skip in zip(self.up_blocks, reversed(skips)):
            h = up(h, skip, temb)

        return self.out_conv(F.silu(self.out_norm(h)))


# ── Gaussian Diffusion ─────────────────────────────────────────────────────────

class GaussianDiffusion(nn.Module):
    """
    Gaussian diffusion process with cosine noise schedule.

    Provides:
      p_loss(x0, mr, t, anatomy_idx) — training loss (MSE on noise)
      ddim_sample(mr, anatomy_idx, steps, eta) — inference via DDIM

    Conditioning: MR slice concatenated to noisy CT before denoising U-Net.
    x0 / predictions are in [-1, 1] (normalised CT space).
    """

    def __init__(
        self,
        model:   DDPMUNet,
        T:       int   = 1000,
        s:       float = 0.008,    # cosine schedule offset
    ):
        super().__init__()
        self.model = model
        self.T     = T

        # Cosine schedule
        t_arr        = torch.arange(T + 1, dtype=torch.float64)
        f            = torch.cos((t_arr / T + s) / (1.0 + s) * math.pi * 0.5) ** 2
        alpha_bar    = f / f[0]
        betas        = (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999)
        alphas       = 1.0 - betas
        alpha_bar    = alpha_bar[1:].float()  # length T

        self.register_buffer("betas",      betas.float())
        self.register_buffer("alphas",     alphas.float())
        self.register_buffer("alpha_bar",  alpha_bar)
        self.register_buffer("sqrt_ab",    alpha_bar.sqrt())
        self.register_buffer("sqrt_1mab",  (1.0 - alpha_bar).sqrt())

    # ── forward (noising) ──────────────────────────────────────────────────────

    def q_sample(
        self,
        x0:      torch.Tensor,   # (B, 1, H, W) clean CT in [-1, 1]
        t:       torch.Tensor,   # (B,) long timestep indices
        noise:   torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab   = self.sqrt_ab[t][:, None, None, None]
        sqrt_1mab = self.sqrt_1mab[t][:, None, None, None]
        return sqrt_ab * x0 + sqrt_1mab * noise, noise

    # ── training loss ──────────────────────────────────────────────────────────

    def p_loss(
        self,
        x0:          torch.Tensor,           # (B, 1, H, W) clean CT
        mr:          torch.Tensor,           # (B, 1, H, W) MR conditioning
        t:           torch.Tensor,           # (B,) long
        anatomy_idx: Optional[torch.Tensor] = None,
        mask:        Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        noise = torch.randn_like(x0)
        x_t, _  = self.q_sample(x0, t, noise)
        x_in    = torch.cat([mr, x_t], dim=1)    # (B, 2, H, W)
        eps_hat = self.model(x_in, t, anatomy_idx)

        diff = (eps_hat - noise) ** 2
        if mask is not None:
            return (diff * mask).sum() / (mask.sum() + 1e-8)
        return diff.mean()

    # ── DDIM inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(
        self,
        mr:          torch.Tensor,           # (B, 1, H, W)
        anatomy_idx: Optional[torch.Tensor] = None,
        steps:       int   = 50,
        eta:         float = 0.0,            # 0 = deterministic DDIM
    ) -> torch.Tensor:
        B, _, H, W = mr.shape
        device     = mr.device

        # Evenly spaced timestep sequence (T-1 → 0)
        t_seq      = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=device)
        t_prev_seq = torch.cat([t_seq[1:], torch.tensor([-1], device=device)])

        x = torch.randn(B, 1, H, W, device=device)   # start from pure noise

        for t_val, t_prev_val in zip(t_seq, t_prev_seq):
            t_batch      = t_val.expand(B)
            x_in         = torch.cat([mr, x], dim=1)
            eps          = self.model(x_in, t_batch, anatomy_idx)

            ab_t  = self.alpha_bar[t_val]
            ab_tp = self.alpha_bar[t_prev_val] if t_prev_val >= 0 else torch.ones(1, device=device)

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

    unet = DDPMUNet(
        in_channels=2, out_channels=1,
        base_ch=64, ch_mult=(1, 2, 4, 8),
        num_res=2, attn_levels=(3,),
        time_emb_dim=256, n_anatomy=3,
    ).to(device)

    diffusion = GaussianDiffusion(unet, T=1000).to(device)

    B, H, W = 2, 256, 256
    mr   = torch.randn(B, 1, H, W, device=device)
    ct   = torch.randn(B, 1, H, W, device=device)
    t    = torch.randint(0, 1000, (B,), device=device)
    anat = torch.tensor([0, 2], device=device)

    loss = diffusion.p_loss(ct, mr, t, anat)
    print(f"Training loss: {loss.item():.4f}")

    sample = diffusion.ddim_sample(mr, anat, steps=10)
    print(f"DDIM sample shape: {sample.shape}")

    n_params = sum(p.numel() for p in unet.parameters()) / 1e6
    print(f"DDPMUNet params: {n_params:.1f}M")
