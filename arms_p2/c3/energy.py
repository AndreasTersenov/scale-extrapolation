"""Patched energy score (beta=1) — the C3 training objective.

ES(P, y) = E||X - y||^beta - 1/2 E||X - X'||^beta summed over local patches, beta=1
fixed: per patch this is the multivariate-CRPS generalization of the energy score,
strictly proper for beta in (0, 2) (Gneiting & Raftery 2007; Pacchiardi et al. line
per the Gate-0 reading). Estimated with the FAIR (unbiased) m-sample form

    ES_m = (1/m) sum_i ||X_i - y||  -  1/(2 m (m-1)) sum_{i!=j} ||X_i - X_j||.

Patch 8x8 / stride 4 over all three detail channels jointly (dim 192) is licensed by
B1: conditional predictability saturates at r* ~ 1 coarse pixel, so small local
patches carry essentially all the conditional structure. Octave-4 maps (8x8) are
exactly one patch.
"""
from __future__ import annotations

import jax.numpy as jnp

PATCH = 8
STRIDE = 4
M_SAMPLES = 8


def extract_patches(x, patch=PATCH, stride=STRIDE):
    """(..., H, W, C) -> (..., P, patch*patch*C): flattened overlapping patches.

    Static Python loop over offsets (P is small: 1 at 8x8, 9 at 16x16, 49 at 32x32),
    jit-friendly. Requires H, W >= patch.
    """
    H, W, C = x.shape[-3], x.shape[-2], x.shape[-1]
    assert H >= patch and W >= patch, (x.shape, patch)
    lead = x.shape[:-3]
    ps = []
    for i in range(0, H - patch + 1, stride):
        for j in range(0, W - patch + 1, stride):
            ps.append(x[..., i:i + patch, j:j + patch, :].reshape(
                lead + (patch * patch * C,)))
    return jnp.stack(ps, axis=-2)


def _norm(v, eps=1e-12):
    """Euclidean norm over the last axis, eps-smoothed at zero for differentiability."""
    return jnp.sqrt(jnp.sum(v * v, axis=-1) + eps)


def energy_score_fair(samples_p, target_p):
    """Fair m-sample energy score per (batch, patch).

    samples_p: (m, B, P, D) patched model samples; target_p: (B, P, D) patched data.
    Returns (B, P). The i==j diagonal contributes only sqrt(eps)*m to the pair sum
    (~1e-6, negligible), so no masking is needed.
    """
    m = samples_p.shape[0]
    term1 = _norm(samples_p - target_p[None]).mean(axis=0)              # (B, P)
    pair = _norm(samples_p[:, None] - samples_p[None])                  # (m, m, B, P)
    term2 = pair.sum(axis=(0, 1)) / (m * (m - 1))
    return term1 - 0.5 * term2


def patched_energy_score(samples, target, patch=PATCH, stride=STRIDE):
    """Scalar loss: mean fair ES over batch and patches.

    samples: (m, B, H, W, C) model samples; target: (B, H, W, C) data detail.
    """
    return jnp.mean(energy_score_fair(extract_patches(samples, patch, stride),
                                      extract_patches(target, patch, stride)))
