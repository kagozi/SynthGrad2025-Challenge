"""
3D Variable-Step DDPM — faithful port of Faking_it (SynthRAD2025, Task 1, rank 7).

Key differences from the previous implementation:
  - x_0 prediction   (was: ε-prediction)
  - Linear β schedule (was: cosine)
  - Stochastic DDPM   (was: DDIM)
  - 5-level UNet, model_channels=64, channel_mult=(1,2,2,4,4)  (was: 4-level, 32ch)
  - GroupNorm32, scale-shift FiLM, QKV attention at ds=16 & ds=8
  - Anisotropic downsampling: (1,2,2) for first two levels, (2,2,2) for rest
  - SpacedDiffusion for variable-T training
  - Variance regularisation: 0.0001 * mean(model_var²)
"""

from __future__ import annotations

import enum
import math
from typing import Optional, Sequence

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


# ── Utilities (from Faking_it util_network.py) ─────────────────────────────────

class GroupNorm32(nn.GroupNorm):
    def forward(self, x: th.Tensor) -> th.Tensor:
        return super().forward(x.float()).type(x.dtype)


def normalization(channels: int) -> nn.Module:
    return GroupNorm32(32, channels)


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.detach().zero_()
    return module


def conv_nd(dims: int, *args, **kwargs) -> nn.Module:
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dims: {dims}")


def linear(*args, **kwargs) -> nn.Module:
    return nn.Linear(*args, **kwargs)


def timestep_embedding(timesteps: th.Tensor, dim: int, max_period: int = 10000) -> th.Tensor:
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def mean_flat(tensor: th.Tensor) -> th.Tensor:
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def _checkpoint(func, inputs, params, flag):
    if flag:
        from torch.utils.checkpoint import checkpoint as th_ckpt
        return th_ckpt(func, *inputs, use_reentrant=False)
    return func(*inputs)


# ── Diffusion math utilities ────────────────────────────────────────────────────

def normal_kl(mean1, logvar1, mean2, logvar2) -> th.Tensor:
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]
    return 0.5 * (
        -1.0 + logvar2 - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x: th.Tensor) -> th.Tensor:
    return 0.5 * (
        1.0 + th.tanh(
            th.sqrt(th.tensor(2.0 / math.pi, device=x.device))
            * (x + 0.044715 * th.pow(x, 3))
        )
    )


def discretized_gaussian_log_likelihood(x, *, means, log_scales) -> th.Tensor:
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    return log_probs


def _extract_into_tensor(arr, timesteps: th.Tensor, broadcast_shape: tuple) -> th.Tensor:
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


# ── Enum types ─────────────────────────────────────────────────────────────────

class ModelMeanType(enum.Enum):
    START_X = enum.auto()
    EPSILON = enum.auto()
    PREVIOUS_X = enum.auto()


class ModelVarType(enum.Enum):
    LEARNED_RANGE = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()


# ── U-Net building blocks ───────────────────────────────────────────────────────

class TimestepBlock(nn.Module):
    def forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Downsample(nn.Module):
    """Nearest-neighbour fractional downsample via th.nn.Upsample(scale<1)."""

    def __init__(self, channels: int, sample_kernel: Sequence[int], dims: int = 3):
        super().__init__()
        self.channels = channels
        scale = tuple(1.0 / k for k in sample_kernel)
        self.op = th.nn.Upsample(scale_factor=scale, mode="nearest")

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[1] == self.channels
        return self.op(x)


class Upsample(nn.Module):
    """Nearest-neighbour upsample — channel count unchanged."""

    def __init__(self, channels: int, sample_kernel: Sequence[int], dims: int = 3):
        super().__init__()
        self.channels = channels
        self.op = th.nn.Upsample(scale_factor=tuple(sample_kernel), mode="nearest")

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """Residual block with optional FiLM scale-shift norm conditioning."""

    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_scale_shift_norm: bool = False,
        dims: int = 3,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
        sample_kernel: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.channels = channels
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        out_channels = out_channels or channels
        self.out_channels = out_channels

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            assert sample_kernel is not None
            self.h_upd = Upsample(channels, sample_kernel, dims)
            self.x_upd = Upsample(channels, sample_kernel, dims)
        elif down:
            assert sample_kernel is not None
            self.h_upd = Downsample(channels, sample_kernel, dims)
            self.x_upd = Downsample(channels, sample_kernel, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2 * out_channels if use_scale_shift_norm else out_channels),
        )
        self.out_layers = nn.Sequential(
            normalization(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, out_channels, out_channels, 3, padding=1)),
        )

        if out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, channels, out_channels, 1)

    def forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        return _checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h


class QKVAttentionLegacy(nn.Module):
    """Multi-head QKV attention (legacy head-splitting order)."""

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: th.Tensor) -> th.Tensor:
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    """Full spatial self-attention with GroupNorm32 pre-norm.
    Always uses gradient checkpointing to bound memory."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttentionLegacy(num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x: th.Tensor) -> th.Tensor:
        return _checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x: th.Tensor) -> th.Tensor:
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x).float()).type(x.dtype)
        h = self.attention(qkv)
        h = self.proj_out(h.float()).type(x.dtype)
        return (x + h).reshape(b, c, *spatial)


# ── 3D UNet (Faking_it get_Unet architecture) ──────────────────────────────────

class UNetModel3D(nn.Module):
    """
    5-level 3D U-Net matching Faking_it 'get_Unet':
      model_channels=64, channel_mult=(1,2,2,4,4),
      2 res-blocks per level, attention at ds=16 & ds=8,
      anisotropic downsampling (D×H×W): (1,2,2) for levels 0-1, (2,2,2) for 2-4.

    image_size tracks the H dimension to know when to add attention.
    """

    SAMPLE_KERNELS = [
        [1, 2, 2],  # level 0 enc/dec
        [1, 2, 2],  # level 1
        [2, 2, 2],  # level 2
        [2, 2, 2],  # level 3
        [2, 2, 2],  # level 4  (unused for encoder — last level)
    ]
    CHANNEL_MULT   = (1, 2, 2, 4, 4)
    NUM_RES_BLOCKS = 2
    ATTENTION_DS   = {16, 8}   # add attention when ds ∈ this set

    def __init__(
        self,
        in_channels:  int   = 2,    # MR + noisy CT
        out_channels: int   = 2,    # x0_pred + log_var_interp
        model_channels: int = 64,
        dropout:      float = 0.2,
        image_size:   int   = 128,  # H dimension of the patch
    ):
        super().__init__()
        ch_mult = self.CHANNEL_MULT
        n_res   = self.NUM_RES_BLOCKS
        dims    = 3

        time_embed_dim = model_channels * 4  # 256
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # ── Encoder ──────────────────────────────────────────────────────────
        ch        = int(ch_mult[0] * model_channels)    # 64
        input_ch  = ch
        ds        = image_size

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))
        ])
        input_block_chans = [ch]
        self._feature_size = ch

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
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(ch_mult) - 1:      # not last level → downsample
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(ch, self.SAMPLE_KERNELS[level], dims=dims)
                    )
                )
                input_block_chans.append(ch)
                ds //= 2
                self._feature_size += ch

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, use_scale_shift_norm=True, dims=dims),
            AttentionBlock(ch, num_heads=4),
            ResBlock(ch, time_embed_dim, dropout, use_scale_shift_norm=True, dims=dims),
        )
        self._feature_size += ch

        # ── Decoder ───────────────────────────────────────────────────────────
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
                if level and i == n_res:        # upsample at last block of each dec level
                    layers.append(Upsample(ch, self.SAMPLE_KERNELS[level - 1], dims=dims))
                    ds *= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def forward(self, x: th.Tensor, timesteps: th.Tensor) -> th.Tensor:
        """
        x         : (B, 2, D, H, W) — [noisy_ct | mr]
        timesteps : (B,) — scaled to [0, 1000)
        returns   : (B, 2, D, H, W) — [x0_pred | log_var_interp]
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.out[0].num_groups * 2))
        # time_embed input must match model_channels; GroupNorm32 groups == 32
        # so model_channels = 64 → we need timestep_embedding(t, 64)
        hs = []
        h = x.type(th.float32)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


# Fix the embedding dimension bug: time_embed takes model_channels not GroupNorm groups
# Override to pass correct dim
class UNetModel3D(nn.Module):
    """
    5-level 3D U-Net matching Faking_it 'get_Unet':
      model_channels=64, channel_mult=(1,2,2,4,4),
      2 res-blocks per level, attention at ds=16 & ds=8,
      anisotropic downsampling (D×H×W): (1,2,2) for levels 0-1, (2,2,2) for 2-4.
    """

    SAMPLE_KERNELS = [
        [1, 2, 2],
        [1, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
    ]
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

        time_embed_dim = model_channels * 4   # 256

        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch       = int(ch_mult[0] * model_channels)
        input_ch = ch
        ds       = image_size

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))
        ])
        input_block_chans = [ch]

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
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(ch, self.SAMPLE_KERNELS[level], dims=dims)
                    )
                )
                input_block_chans.append(ch)
                ds //= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, use_scale_shift_norm=True, dims=dims),
            AttentionBlock(ch, num_heads=4),
            ResBlock(ch, time_embed_dim, dropout, use_scale_shift_norm=True, dims=dims),
        )

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
                if level and i == n_res:
                    layers.append(Upsample(ch, self.SAMPLE_KERNELS[level - 1], dims=dims))
                    ds *= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def forward(self, x: th.Tensor, timesteps: th.Tensor) -> th.Tensor:
        emb = self.time_embed(
            timestep_embedding(timesteps, self.model_channels)
        )
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        return self.out(h)


# ── Gaussian Diffusion ──────────────────────────────────────────────────────────

class GaussianDiffusion:
    """
    IDDPM-style Gaussian diffusion.
    - predict_xstart=True  (x_0 prediction)
    - learn_sigma=True     (LEARNED_RANGE variance)
    - loss_type=RESCALED_MSE
    - rescale_timesteps=True (SpacedDiffusion remaps t to [0,1000))
    """

    def __init__(
        self,
        *,
        betas:           np.ndarray,
        model_mean_type: ModelMeanType   = ModelMeanType.START_X,
        model_var_type:  ModelVarType    = ModelVarType.LEARNED_RANGE,
        loss_type:       LossType        = LossType.RESCALED_MSE,
        rescale_timesteps: bool          = True,
    ):
        self.model_mean_type   = model_mean_type
        self.model_var_type    = model_var_type
        self.loss_type         = loss_type
        self.rescale_timesteps = rescale_timesteps

        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        self.num_timesteps = int(betas.shape[0])

        alphas                = 1.0 - betas
        self.alphas_cumprod   = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        self.sqrt_alphas_cumprod        = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod  = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod  = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def _scale_timesteps(self, t: th.Tensor) -> th.Tensor:
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_sample(self, x_start: th.Tensor, t: th.Tensor,
                 noise: Optional[th.Tensor] = None) -> th.Tensor:
        if noise is None:
            noise = th.randn_like(x_start)
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start: th.Tensor, x_t: th.Tensor, t: th.Tensor):
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance         = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_var_clipped  = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_var_clipped

    def p_mean_variance(
        self,
        model:         UNetModel3D,
        x:             th.Tensor,          # (B,1,D,H,W) noisy CT
        t:             th.Tensor,          # (B,) spaced-t indices
        condition:     th.Tensor,          # (B,1,D,H,W) MR in [-1,1]
        clip_denoised: bool = True,
    ) -> dict:
        B, C = x.shape[:2]
        x_input = th.cat([x, condition], dim=1)
        model_output = model(x_input, self._scale_timesteps(t))

        assert model_output.shape == (B, C * 2, *x.shape[2:])
        model_output, model_var_values = th.split(model_output, C, dim=1)

        min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
        frac    = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        model_variance     = th.exp(model_log_variance)

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = model_output.clamp(-1, 1) if clip_denoised else model_output
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x, t, model_output)
            if clip_denoised:
                pred_xstart = pred_xstart.clamp(-1, 1)
        else:
            raise NotImplementedError(self.model_mean_type)

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )
        return {
            "mean":        model_mean,
            "variance":    model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t: th.Tensor, t: th.Tensor, eps: th.Tensor) -> th.Tensor:
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _vb_terms_bpd(
        self,
        model,
        x_start:  th.Tensor,
        x_t:      th.Tensor,
        t:        th.Tensor,
        condition: th.Tensor,
        clip_denoised: bool = True,
    ) -> dict:
        true_mean, _, true_log_var = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(model, x_t, t, condition, clip_denoised=clip_denoised)
        kl  = normal_kl(true_mean, true_log_var, out["mean"], out["log_variance"])
        kl  = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(
        self,
        model:     UNetModel3D,
        x_start:   th.Tensor,     # (B,1,D,H,W) clean CT in [-1,1]
        condition: th.Tensor,     # (B,1,D,H,W) MR in [-1,1]
        t:         th.Tensor,     # (B,) spaced timestep indices
        penalize_high_variance: bool = True,
    ) -> dict:
        noise = th.randn_like(x_start)
        x_t   = self.q_sample(x_start, t, noise=noise)

        x_input      = th.cat([x_t, condition], dim=1)
        model_output = model(x_input, self._scale_timesteps(t))

        B, C = x_t.shape[:2]
        assert model_output.shape == (B, C * 2, *x_t.shape[2:])
        model_output, model_var_values = th.split(model_output, C, dim=1)

        # VLB via frozen mean branch
        frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
        vb = self._vb_terms_bpd(
            model=lambda *args, r=frozen_out: r,
            x_start=x_start,
            x_t=x_t,
            t=t,
            condition=condition,
            clip_denoised=False,
        )["output"]
        if self.loss_type == LossType.RESCALED_MSE:
            vb = vb * (self.num_timesteps / 1000.0)

        # MAE on x_0 prediction
        target = x_start   # ModelMeanType.START_X
        mse    = th.nn.L1Loss()(target, model_output)

        loss = mse + vb.mean()

        var_reg = 0.0001 * th.mean(model_var_values ** 2)
        if penalize_high_variance:
            loss = loss + var_reg

        return {
            "loss":    loss,
            "mse":     mse,
            "vb":      vb.mean(),
            "var_reg": var_reg,
        }, target, model_output

    # ── Stochastic DDPM sampling ──────────────────────────────────────────────

    @th.no_grad()
    def p_sample(
        self,
        model:     UNetModel3D,
        x:         th.Tensor,
        t:         th.Tensor,
        condition: th.Tensor,
        clip_denoised: bool = True,
    ) -> dict:
        out = self.p_mean_variance(model, x, t, condition, clip_denoised=clip_denoised)
        noise        = th.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    @th.no_grad()
    def p_sample_loop(
        self,
        model:     UNetModel3D,
        shape:     tuple,
        condition: th.Tensor,
        clip_denoised: bool  = True,
        device:    Optional[th.device] = None,
        progress:  bool = False,
    ) -> th.Tensor:
        if device is None:
            device = next(model.parameters()).device
        img     = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        for i in indices:
            t_batch = th.tensor([i] * shape[0], device=device)
            out  = self.p_sample(model, img, t_batch, condition, clip_denoised)
            img  = out["sample"]
        return img

    @th.no_grad()
    def p_sample_loop_mask(
        self,
        model:         UNetModel3D,
        shape:         tuple,
        condition:     th.Tensor,
        mask:          th.Tensor,         # (B,1,D,H,W) body mask
        clip_denoised: bool  = True,
        device:        Optional[th.device] = None,
    ) -> th.Tensor:
        """Like p_sample_loop but forces background to -1 after every step."""
        if device is None:
            device = next(model.parameters()).device
        img     = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]
        for i in indices:
            t_batch = th.tensor([i] * shape[0], device=device)
            out     = self.p_sample(model, img, t_batch, condition, clip_denoised)
            img     = out["sample"]
            # background masking: keep only foreground, set background to -1
            if mask is not None:
                img = img * mask + (-1.0) * (1 - mask)
        return img


# ── Beta schedule ───────────────────────────────────────────────────────────────

def get_named_beta_schedule(schedule_name: str, num_diffusion_timesteps: int) -> np.ndarray:
    if schedule_name == "linear":
        scale      = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end   = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            a1 = math.cos((t1 + 0.008) / 1.008 * math.pi / 2) ** 2
            a2 = math.cos((t2 + 0.008) / 1.008 * math.pi / 2) ** 2
            betas.append(min(1 - a2 / a1, 0.999))
        return np.array(betas)
    else:
        raise NotImplementedError(f"unknown schedule: {schedule_name}")


# ── Spaced Diffusion (for Variable-Step training) ──────────────────────────────

def space_timesteps(num_timesteps: int, section_counts) -> set:
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired = int(section_counts[4:])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired:
                    return set(range(0, num_timesteps, i))
            raise ValueError(f"cannot create exactly {desired} ddim steps")
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra    = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < count:
            raise ValueError(f"cannot divide {size} steps into {count}")
        frac_stride = 1 if count <= 1 else (size - 1) / (count - 1)
        cur_idx     = 0.0
        taken       = []
        for _ in range(count):
            taken.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    GaussianDiffusion that remaps a subset of T timesteps.
    Used for Variable-Step training (random T ∈ [5,10,...,300] per batch).
    The wrapped model maps spaced t-indices back to original 1000-step scale.
    """

    def __init__(self, use_timesteps: set, **kwargs):
        self.use_timesteps       = set(use_timesteps)
        self.timestep_map        = []
        self.original_num_steps  = len(kwargs["betas"])

        base = GaussianDiffusion(**kwargs)
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def _wrap_model(self, model: UNetModel3D) -> "_WrappedModel":
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map,
            self.rescale_timesteps, self.original_num_steps
        )

    def p_mean_variance(self, model, *args, **kwargs):
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def _scale_timesteps(self, t: th.Tensor) -> th.Tensor:
        return t     # wrapping is done in _WrappedModel


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model              = model
        self.timestep_map       = timestep_map
        self.rescale_timesteps  = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x: th.Tensor, ts: th.Tensor, **kwargs) -> th.Tensor:
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts     = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)

    # Expose parameters() so _checkpoint works correctly
    def parameters(self):
        return self.model.parameters()


# ── Factory ────────────────────────────────────────────────────────────────────

VARIABLE_T_VALUES = [5, 10, 15, 20, 25, 35, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]


def create_spaced_diffusion(T: int, noise_schedule: str = "linear") -> SpacedDiffusion:
    """Create a SpacedDiffusion with T evenly-spaced steps from the 1000-step base."""
    betas = get_named_beta_schedule(noise_schedule, 1000)
    return SpacedDiffusion(
        use_timesteps  = space_timesteps(1000, str(T)),
        betas          = betas,
        model_mean_type  = ModelMeanType.START_X,
        model_var_type   = ModelVarType.LEARNED_RANGE,
        loss_type        = LossType.RESCALED_MSE,
        rescale_timesteps = True,
    )


def build_model_and_diffusions(
    model_channels: int   = 64,
    dropout:        float = 0.2,
    image_size:     int   = 128,
    noise_schedule: str   = "linear",
) -> tuple:
    """
    Returns:
        model       : UNetModel3D
        diffusions  : dict mapping T → SpacedDiffusion (one per VS-T value)
    """
    model = UNetModel3D(
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
    model, diffusions = build_model_and_diffusions()
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"UNetModel3D parameters: {n_params:.1f}M")

    B, D, H, W = 1, 32, 128, 128
    mr = th.randn(B, 1, D, H, W, device=device)
    ct = th.randn(B, 1, D, H, W, device=device)

    diff = diffusions[100]
    t    = th.randint(0, 100, (B,), device=device)
    with th.cuda.amp.autocast():
        terms, target, x0_pred = diff.training_losses(model, ct, mr, t)
    print(f"Loss: {terms['loss'].item():.4f} | MSE: {terms['mse'].item():.4f} | VB: {terms['vb'].item():.4f}")

    diff50 = diffusions[50]
    sample = diff50.p_sample_loop(model, (B, 1, D, H, W), mr, device=device)
    print(f"Sample shape: {sample.shape}, range: [{sample.min():.2f}, {sample.max():.2f}]")
