"""Conditional velocity network for wavelet flow matching.

Models v_theta(detail_t, t | coarse, scale_coord): the velocity of the detail triple at
one octave, conditioned on the coarse field at that octave and (arm B) a low-dimensional
scale coordinate. The SAME network is used at every octave (weights shared across scales)
-- the RG-fixed-point prior. Arm A passes ``cond_dim=0`` (no scale input); arm B passes the
2-D running-coupling coordinate from stage-0.

Architecture: a small U-Net. The coarse field is concatenated to the (noised) detail as an
input channel; time and the scale coordinate are embedded and injected additively at every
resolution level (FiLM-free, matches jax_flows' SimpleUNet pattern).
"""
from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from jax_flows.utils import sinusoidal_time_embedding


class _ConvBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        ng = min(8, self.features)
        for _ in range(2):
            x = nn.Conv(self.features, (3, 3), padding="SAME")(x)
            x = nn.GroupNorm(num_groups=ng)(x)
            x = nn.silu(x)
        return x


class ConditionalUNet(nn.Module):
    """v(detail_t, t | coarse, cond_vec) -> velocity with ``out_channels`` channels.

    detail: (B,H,W,out_channels); coarse: (B,H,W,1); cond_vec: (B,cond_dim) or None.
    """

    out_channels: int = 3
    channels: Sequence[int] = (32, 64, 128)
    bottleneck: int = 256
    embed_dim: int = 128
    cond_dim: int = 0
    cond_mode: str = "add"      # "add" (embedding sum) or "film" (per-channel modulation)
    variance_head: bool = False  # phase-1c NLL head: also output per-coefficient log-sigma

    @nn.compact
    def __call__(self, detail, t, coarse, cond_vec=None):
        # time embedding (always additive)
        emb = sinusoidal_time_embedding(t, self.embed_dim)
        cemb = None
        if self.cond_dim > 0:
            if cond_vec is None:
                cond_vec = jnp.zeros((detail.shape[0], self.cond_dim), detail.dtype)
            cemb = nn.Dense(self.embed_dim)(nn.silu(nn.Dense(self.embed_dim)(cond_vec)))
            if self.cond_mode == "add":
                emb = emb + cemb        # coordinate folded into the additive embedding

        def inject(h, ch):
            p = nn.Dense(ch)(nn.silu(nn.Dense(ch)(emb)))
            h = h + p[:, None, None, :]
            if self.cond_mode == "film" and cemb is not None:
                # FiLM: the scale coordinate modulates features multiplicatively, so it
                # cannot be substituted by the coarse field the way additive bias can.
                gb = nn.Dense(2 * ch)(nn.silu(nn.Dense(ch)(cemb)))
                g, b = jnp.split(gb, 2, axis=-1)
                h = h * (1.0 + g[:, None, None, :]) + b[:, None, None, :]
            return h

        h = jnp.concatenate([detail, coarse], axis=-1)   # condition on coarse via channels
        skips = []
        for ch in self.channels:
            h = _ConvBlock(ch)(h)
            h = inject(h, ch)
            skips.append(h)
            h = nn.max_pool(h, (2, 2), strides=(2, 2))

        h = _ConvBlock(self.bottleneck)(h)
        h = inject(h, self.bottleneck)

        for ch, skip in zip(reversed(self.channels), reversed(skips)):
            B, H, W, C = h.shape
            h = jax.image.resize(h, (B, H * 2, W * 2, C), method="nearest")
            h = jnp.concatenate([h, skip], axis=-1)
            h = _ConvBlock(ch)(h)
            h = inject(h, ch)

        v = nn.Conv(self.out_channels, (1, 1))(h)
        if not self.variance_head:
            return v
        # log-sigma head off the shared trunk; zeros-init so sigma starts at 1 (the
        # standardized-detail scale). Output is [v, g] -- consumers slice.
        g = nn.Conv(self.out_channels, (1, 1),
                    kernel_init=nn.initializers.zeros)(h)
        return jnp.concatenate([v, g], axis=-1)
