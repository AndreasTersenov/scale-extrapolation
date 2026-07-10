"""JAX-native orthonormal Haar 2D wavelet transform (the generator's own transform).

The generator is wavelet-factorized: it models p(detail_j | coarse_j) with weights shared
across octaves, and generates coarse-to-fine by sampling a detail triple and inverting one
level. That recursion needs an on-device, exactly-invertible transform — this module.

It is deliberately SEPARATE from the `scaledrift` instrument (which uses pywt on the CPU
to *measure* fields). Both are orthonormal Haar, so they define the same octaves / detail
subspace; the generated full field is re-analyzed independently by `scaledrift`. Only the
round-trip exactness matters here, and it is gate-tested (`tests_wfm/test_haar_roundtrip`).

Arrays are channel-last ``(B, H, W, C)``; H and W must be even at each decomposed level.
Convention: octave j is 1-based, j=1 the finest; a j-level transform yields ``cA_j`` and
the level-j detail triple on the same (H/2**j, W/2**j) grid — matching
``scaledrift.wavelet.octave_pair`` semantics.
"""
from __future__ import annotations

import jax.numpy as jnp

_S = jnp.sqrt(2.0)


def _interleave(a, b, axis):
    """Interleave even (a) and odd (b) halves back along ``axis`` (2x length)."""
    stacked = jnp.stack([a, b], axis=axis + 1)
    shape = list(a.shape)
    shape[axis] = shape[axis] * 2
    return stacked.reshape(shape)


def dwt2(x):
    """One-level Haar DWT. ``x`` (B,H,W,C) -> (cA, (cH, cV, cD)), each (B,H/2,W/2,C)."""
    e = x[:, :, 0::2, :]
    o = x[:, :, 1::2, :]
    lo = (e + o) / _S            # low  along width
    hi = (e - o) / _S            # high along width
    le, lo_ = lo[:, 0::2], lo[:, 1::2]
    he, ho = hi[:, 0::2], hi[:, 1::2]
    cA = (le + lo_) / _S
    cV = (le - lo_) / _S         # low-width, high-height
    cH = (he + ho) / _S          # high-width, low-height
    cD = (he - ho) / _S
    return cA, (cH, cV, cD)


def idwt2(cA, details):
    """Inverse of :func:`dwt2`. Exact to machine precision."""
    cH, cV, cD = details
    le = (cA + cV) / _S
    lo_ = (cA - cV) / _S
    he = (cH + cD) / _S
    ho = (cH - cD) / _S
    lo = _interleave(le, lo_, axis=1)
    hi = _interleave(he, ho, axis=1)
    e = (lo + hi) / _S
    o = (lo - hi) / _S
    return _interleave(e, o, axis=2)


def octave_pair(field, j):
    """Detail triple and coarse field at octave ``j`` (matches scaledrift semantics).

    ``field`` (B,H,W,1). Returns ``(detail, coarse)`` with detail = stacked
    (cH, cV, cD) as (B, H/2**j, W/2**j, 3) and coarse = cA_j as (B, ..., 1).
    """
    cA = field
    det = None
    for _ in range(j):
        cA, det = dwt2(cA)
    cH, cV, cD = det
    return jnp.concatenate([cH, cV, cD], axis=-1), cA


def wavedec2(field, levels):
    """Full multi-level Haar decomposition: ``[cA_L, det_L, ..., det_1]`` (det = 3-tuple)."""
    coeffs = []
    cA = field
    for _ in range(levels):
        cA, det = dwt2(cA)
        coeffs.append(det)
    return [cA] + coeffs[::-1]      # coarsest approx first, then det_L..det_1


def waverec2(coeffs):
    """Inverse of :func:`wavedec2`."""
    cA = coeffs[0]
    for det in coeffs[1:]:           # coeffs = [cA_L, det_L, ..., det_1]: coarsest first
        cA = idwt2(cA, det)
    return cA


def upsample_coarse(coarse, detail_zero=None):
    """Reconstruct the next finer field from a coarse map with zero detail (helper for
    tests / a trivial coarsen-refine identity check)."""
    z = jnp.zeros_like(coarse) if detail_zero is None else detail_zero
    return idwt2(coarse, (z, z, z))
