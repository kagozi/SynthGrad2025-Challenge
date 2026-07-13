"""
3D CWDM — Conditional Wavelet Diffusion Model (ablation vs VS-DDPM).

Architecture: Same 5-level 3D UNet as VS-DDPM but replaces all nearest-neighbour
down/upsample ops with 3D Haar DWT/IDWT.  HF subbands from each encoder DWT are
stored as wavelet skip connections and injected back at the corresponding decoder
IDWT, enabling lossless frequency-band reconstruction.

Diffusion stack (GaussianDiffusion / SpacedDiffusion / Variable-T) is identical
to VS-DDPM and imported directly from vs_ddpm_3d.py.
"""

from __future__ import annotations

import math
from typing import List

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from src.models.vs_ddpm_3d import (
    normalization, zero_module, conv_nd, linear, timestep_embedding,
    TimestepBlock, TimestepEmbedSequential, ResBlock, AttentionBlock,
    create_spaced_diffusion, VARIABLE_T_VALUES,
)


# ── 3D Haar DWT / IDWT ────────────────────────────────────────────────────────

def _haar_kernels() -> th.Tensor:
    """8 Haar 3D convolution kernels, shape (8, 1, 2, 2, 2).  float32."""
    s = 1.0 / math.sqrt(2)
    lo = th.tensor([s,  s], dtype=th.float32)
    hi = th.tensor([-s, s], dtype=th.float32)
    bases = [
        (lo, lo, lo), (lo, lo, hi), (lo, hi, lo), (lo, hi, hi),
        (hi, lo, lo), (hi, lo, hi), (hi, hi, lo), (hi, hi, hi),
    ]
    ks = []
    for f0, f1, f2 in bases:
        k = th.einsum("i,j,k->ijk", f0, f1, f2).unsqueeze(0).unsqueeze(0)
        ks.append(k)
    return th.cat(ks, dim=0)  # (8, 1, 2, 2, 2)


class WavDown3D(nn.Module):
    """3D Haar DWT downsample.

    forward(x) → (LLL, hf_bands)
      LLL:      (B, C, D//2, H//2, W//2)  low-frequency residual
      hf_bands: list of 7 tensors of same shape (LLH … HHH)
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("kernels", _haar_kernels())   # (8,1,2,2,2) float32

    def forward(self, x: th.Tensor):
        B, C, D, H, W = x.shape
        k = self.kernels.to(x.dtype)
        x_flat = x.reshape(B * C, 1, D, H, W)
        out = F.conv3d(x_flat, k, stride=2)               # (B*C, 8, D2, H2, W2)
        out = out.reshape(B, C, 8, D // 2, H // 2, W // 2)
        return out[:, :, 0].contiguous(), [out[:, :, i].contiguous() for i in range(1, 8)]


class WavUp3D(nn.Module):
    """3D Haar IDWT upsample with optional HF channel projection.

    The encoder stores HF subbands with enc_ch channels, but the decoder feature
    at the matching spatial scale may have dec_ch channels (different when the
    encoder/decoder cross a channel-multiplier boundary, e.g. 128→256).
    A learned 1×1 conv projects the 7 HF subbands when enc_ch ≠ dec_ch.

    forward(x, hf_bands) → reconstructed tensor (B, dec_ch, D*2, H*2, W*2)
      x:        LLL from decoder, shape (B, dec_ch, D2, H2, W2)
      hf_bands: list of 7 HF tensors from the matching WavDown3D, each (B, enc_ch, …)
    """

    def __init__(self, enc_ch: int, dec_ch: int):
        super().__init__()
        self.register_buffer("kernels", _haar_kernels())   # (8,1,2,2,2) float32
        self.hf_proj = (
            conv_nd(3, enc_ch, dec_ch, 1) if enc_ch != dec_ch else nn.Identity()
        )

    def forward(self, x: th.Tensor, hf_bands: List[th.Tensor]) -> th.Tensor:
        hf = [self.hf_proj(sb) for sb in hf_bands]        # project to dec_ch
        subbands = [x] + hf                               # 8 × (B, dec_ch, D2, H2, W2)
        B, C, D2, H2, W2 = subbands[0].shape
        k = self.kernels.to(subbands[0].dtype)
        result: th.Tensor | None = None
        for i, sb in enumerate(subbands):
            sb_flat = sb.reshape(B * C, 1, D2, H2, W2)
            contrib = F.conv_transpose3d(sb_flat, k[i : i + 1], stride=2)
            contrib = contrib.reshape(B, C, D2 * 2, H2 * 2, W2 * 2)
            result = contrib if result is None else result + contrib
        return result


# ── Wavelet U-Net ─────────────────────────────────────────────────────────────

class WavUNetModel3D(nn.Module):
    """
    5-level 3D UNet with Haar DWT/IDWT instead of nearest-neighbour sampling.

    Key differences from UNetModel3D (VS-DDPM):
      - WavDown3D at every encoder transition (3D DWT, isotropic ×½ per axis)
      - WavUp3D at every decoder transition (3D IDWT using stored HF subbands)
      - HF wavelet subbands form additional skip connections from encoder to decoder

    Everything else is identical: model_channels=64, channel_mult=(1,2,2,4,4),
    2 ResBlocks per level, attention at H-dim ∈ {16,8}, GroupNorm32, FiLM.
    """

    CHANNEL_MULT   = (1, 2, 2, 4, 4)
    NUM_RES_BLOCKS = 2
    ATTENTION_DS   = {16, 8}

    def __init__(
        self,
        in_channels:    int   = 2,
        out_channels:   int   = 2,
        model_channels: int   = 64,
        dropout:        float = 0.2,
        image_size:     int   = 128,
    ):
        super().__init__()
        self.model_channels = model_channels
        ch_mult = self.CHANNEL_MULT
        n_res   = self.NUM_RES_BLOCKS
        dims    = 3

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch       = int(ch_mult[0] * model_channels)   # 64
        input_ch = ch
        ds       = image_size                          # tracks H-dim for attn placement

        # ── Encoder ──────────────────────────────────────────────────────────
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))
        ])
        input_block_chans = [ch]
        enc_wav_channels: list = []   # channel count at each WavDown3D (for WavUp3D projection)

        for level, mult in enumerate(ch_mult):
            for _ in range(n_res):
                layers: list = [
                    ResBlock(ch, time_embed_dim, dropout,
                             out_channels=int(mult * model_channels),
                             use_scale_shift_norm=True, dims=dims)
                ]
                ch = int(mult * model_channels)
                if ds in self.ATTENTION_DS:
                    layers.append(AttentionBlock(ch, num_heads=4))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            if level != len(ch_mult) - 1:
                enc_wav_channels.append(ch)   # record enc_ch before wavelet down
                self.input_blocks.append(WavDown3D())
                input_block_chans.append(ch)
                ds //= 2

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, use_scale_shift_norm=True, dims=dims),
            AttentionBlock(ch, num_heads=4),
            ResBlock(ch, time_embed_dim, dropout, use_scale_shift_norm=True, dims=dims),
        )

        # ── Decoder ───────────────────────────────────────────────────────────
        # Layout per level: [n_res+1 × TimestepEmbedSequential] then optional WavUp3D.
        # WavUp3D must project HF subbands from enc_ch → dec_ch when they differ.
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(ch_mult))[::-1]:
            for i in range(n_res + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(ch + ich, time_embed_dim, dropout,
                             out_channels=int(model_channels * mult),
                             use_scale_shift_norm=True, dims=dims)
                ]
                ch = int(model_channels * mult)
                if ds in self.ATTENTION_DS:
                    layers.append(AttentionBlock(ch, num_heads=4))
                self.output_blocks.append(TimestepEmbedSequential(*layers))

            if level:
                enc_ch = enc_wav_channels.pop()   # reverse order matches decoder
                self.output_blocks.append(WavUp3D(enc_ch=enc_ch, dec_ch=ch))
                ds *= 2

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def forward(self, x: th.Tensor, timesteps: th.Tensor) -> th.Tensor:
        """
        x         : (B, 2, D, H, W)  [noisy_ct | mr]
        timesteps : (B,) scaled timestep indices
        returns   : (B, 2, D, H, W)  [x0_pred | log_var_interp]
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        hs: list = []
        wav_stack: list = []    # LIFO stack of HF subband lists from encoder DWTs
        h = x

        for module in self.input_blocks:
            if isinstance(module, WavDown3D):
                lll, hf = module(h)
                h = lll
                wav_stack.append(hf)
            else:
                h = module(h, emb)
            hs.append(h)

        h = self.middle_block(h, emb)

        for module in self.output_blocks:
            if isinstance(module, WavUp3D):
                hf = wav_stack.pop()
                h  = module(h, hf)
            else:
                h = th.cat([h, hs.pop()], dim=1)
                h = module(h, emb)

        return self.out(h)


# ── Factory ────────────────────────────────────────────────────────────────────

def build_cwdm_and_diffusions(
    model_channels: int   = 64,
    dropout:        float = 0.2,
    image_size:     int   = 128,
    noise_schedule: str   = "linear",
) -> tuple:
    """Returns (WavUNetModel3D, dict[T → SpacedDiffusion]) — same interface as VS-DDPM."""
    model = WavUNetModel3D(
        in_channels    = 2,
        out_channels   = 2,
        model_channels = model_channels,
        dropout        = dropout,
        image_size     = image_size,
    )
    diffusions = {T: create_spaced_diffusion(T, noise_schedule) for T in VARIABLE_T_VALUES}
    return model, diffusions


# ── Sanity check ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model, diffusions = build_cwdm_and_diffusions()
    model = model.to(device)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"WavUNetModel3D parameters: {n:.1f}M")

    B, D, H, W = 1, 32, 128, 128
    mr = th.randn(B, 1, D, H, W, device=device)
    ct = th.randn(B, 1, D, H, W, device=device)

    diff = diffusions[100]
    t    = th.randint(0, 100, (B,), device=device)
    terms, _, x0_pred = diff.training_losses(model, ct, mr, t)
    print(f"Loss={terms['loss'].item():.4f}  MSE={terms['mse'].item():.4f}  VB={terms['vb'].item():.4f}")

    sample = diffusions[50].p_sample_loop(model, (B, 1, D, H, W), mr, device=device)
    print(f"Sample shape={sample.shape}  range=[{sample.min():.2f}, {sample.max():.2f}]")
