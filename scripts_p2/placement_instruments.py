"""Position-pure placement instruments (Phase A, log/2026-07-24-prereg-placement.md,
approved R30). Validated by tests_p2/test_placement_instruments.py BEFORE any model
scoring (tests-first).

Design rules (from the prereg):
- COUNT-MATCHED peak sets everywhere: the top-K strict local maxima by standardized
  height, K frozen from the truth reference — so no instrument can see marginal or
  count miscalibration, only WHERE structure sits.
- Every instrument returns a per-field SHARE PROFILE over fixed classes; sets are
  compared by z per class (SEs over the exchangeable units) and T = max|z|.
- Per-field standardization + rank/quantile constructions make every profile exactly
  invariant under per-field affine maps (and the peak sets under monotone maps).

Instruments:
  env_rate_profile  8 classes: share of peaks per own-map coarse octile (level-lvl
                    Haar approximation; per-map rank octiles) — "do extremes sit in
                    the environments that demand them"
  nn_profile        10 classes: share of peak nearest-neighbour distances per decile
                    (edges frozen from the truth reference)
  pk2pt_profile     8 classes: share of peak pairs at separation r in (0,8], unit
                    bins, normalized by ALL peak pairs — small-scale clustering
  parity_profile    4 classes: share of peaks per (y%2, x%2) parity — octave-seam /
                    upsampling-artifact catch (truth is uniform by symmetry)
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ peak machinery --

def _local_maxima(field):
    """Strict 8-neighbour local maxima on the interior (score_stageD convention).

    Returns (ys, xs, heights) in FULL-grid coordinates, heights standardized."""
    f = (field - field.mean()) / field.std()
    c = f[1:-1, 1:-1]
    is_max = np.ones(c.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            is_max &= c > f[1 + dy:f.shape[0] - 1 + dy, 1 + dx:f.shape[1] - 1 + dx]
    ys, xs = np.nonzero(is_max)
    return ys + 1, xs + 1, c[is_max]


def topk_peaks_xy(field, k):
    """Count-matched peak set: the K highest local maxima (rank-stable under
    monotone maps). If fewer than K maxima exist, returns all of them."""
    ys, xs, h = _local_maxima(field)
    if len(h) > k:
        idx = np.argsort(h)[::-1][:k]
        ys, xs, h = ys[idx], xs[idx], h[idx]
    return ys, xs, h


# --------------------------------------------------------------------- instruments --

def env_rate_profile(field, k, lvl=2, nbins=8):
    """Share of top-K peaks per own-map coarse-octile class."""
    b = 2 ** lvl
    H, W = field.shape
    coarse = field.reshape(H // b, b, W // b, b).mean(axis=(1, 3))
    ranks = np.argsort(np.argsort(coarse.ravel())).reshape(coarse.shape)
    bins = (ranks * nbins) // coarse.size          # 0..nbins-1, equal occupancy
    ys, xs, _ = topk_peaks_xy(field, k)
    if len(ys) == 0:
        return None
    peak_bins = bins[ys // b, xs // b]
    return np.bincount(peak_bins, minlength=nbins) / len(ys)


def nn_profile(field, k, edges):
    """Share of nearest-neighbour distances per class; classes = (-inf,e1], ...,
    (e9, inf) for the 9 frozen edges (10 shares)."""
    ys, xs, _ = topk_peaks_xy(field, k)
    if len(ys) < 2:
        return None
    pts = np.stack([ys, xs], axis=1).astype(np.float64)
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(d2.min(axis=1))
    cls = np.searchsorted(edges, nn, side="left")
    return np.bincount(cls, minlength=len(edges) + 1) / len(nn)


def pk2pt_profile(field, k, rmax=8):
    """Share of ALL peak pairs at separation r in unit bins (1,2],...,(rmax-1,rmax].

    The (0,1] bin is omitted: strict 8-neighbour maxima are never closer than
    separation 2, so that class is structurally empty (validation finding)."""
    ys, xs, _ = topk_peaks_xy(field, k)
    if len(ys) < 2:
        return None
    pts = np.stack([ys, xs], axis=1).astype(np.float64)
    iu = np.triu_indices(len(pts), 1)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))[iu]
    npairs = len(d)
    counts = np.array([((d > r) & (d <= r + 1)).sum() for r in range(1, rmax)],
                      float)
    return counts / npairs


def parity_profile(field, k):
    """Share of top-K peaks per (y%2, x%2) parity class (4 shares)."""
    ys, xs, _ = topk_peaks_xy(field, k)
    if len(ys) == 0:
        return None
    cls = (ys % 2) * 2 + (xs % 2)
    return np.bincount(cls, minlength=4) / len(ys)


# ----------------------------------------------------------------- set comparisons --

def stack_profiles(profiles):
    """Mean and SE over fields (None profiles skipped)."""
    arr = np.array([p for p in profiles if p is not None], float)
    return arr.mean(axis=0), arr.std(axis=0, ddof=1) / np.sqrt(len(arr))


def tstat(mean_a, se_a, mean_b, se_b):
    """z per class and T = max|z|. Classes with zero variance on BOTH sides
    (structurally constant) contribute z=0 when the means agree; a mean
    difference with zero measured variance is reported as z=99 (validation
    finding: discrete geometry can create constant classes)."""
    num = np.asarray(mean_a) - np.asarray(mean_b)
    denom = np.sqrt(np.asarray(se_a) ** 2 + np.asarray(se_b) ** 2)
    z = np.where(denom > 0, num / np.where(denom == 0, 1.0, denom),
                 np.where(np.abs(num) < 1e-12, 0.0, 99.0))
    return float(np.max(np.abs(z))), z


def effect_l2(mean_a, mean_b):
    """Effect size: L2 distance between mean profiles."""
    return float(np.sqrt(((mean_a - mean_b) ** 2).sum()))


# ------------------------------------------------------------------- the surrogate --

def surrogate(field, rng):
    """Joint-structure null: FFT phase randomization, then EXACT rank remapping to
    the original marginal (sorted values transplanted onto the surrogate's ranks).
    Marginals identical; joint/positional structure destroyed."""
    F = np.fft.fft2(field)
    phase = np.exp(2j * np.pi * rng.random(field.shape))
    g = np.fft.ifft2(np.abs(F) * phase).real
    order = np.argsort(g.ravel())
    out = np.empty(field.size)
    out[order] = np.sort(field.ravel())
    return out.reshape(field.shape)
