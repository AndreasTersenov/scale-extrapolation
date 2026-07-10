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

    @nn.compact
    def __call__(self, detail, t, coarse, cond_vec=None):
        # global conditioning embedding: time (+ optional scale coordinate)
        emb = sinusoidal_time_embedding(t, self.embed_dim)
        if self.cond_dim > 0:
            if cond_vec is None:
                cond_vec = jnp.zeros((detail.shape[0], self.cond_dim), detail.dtype)
            emb = emb + nn.Dense(self.embed_dim)(nn.silu(nn.Dense(self.embed_dim)(cond_vec)))

        def inject(h, ch):
            p = nn.Dense(ch)(nn.silu(nn.Dense(ch)(emb)))
            return h + p[:, None, None, :]

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

        return nn.Conv(self.out_channels, (1, 1))(h)
