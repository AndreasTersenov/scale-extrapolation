"""Scale-drift of conditional wavelet statistics (Measurement M1, headline metric).

Object measured
---------------
Per octave ``j``, the conditional PDF ``p_j(w | c-bin)`` of the orientation-pooled,
per-octave-standardized detail coefficient ``w`` given the standardized coarse field
``c`` (quantile-binned). DRIFT between two octaves = mean over coarse bins of the
1-D Wasserstein-1 distance between their conditional PDFs.

Why "excess" drift
------------------
Finite samples give a POSITIVE W1 even between two draws of the *same* distribution.
So the honest null is not zero but the within-octave, same-sample-size W1 floor. We
report ``excess = measured - floor``:

    measured : W1 between octave-j and octave-k conditional PDFs (sizes matched to M)
    floor    : W1 between two disjoint size-M subsamples of the finer octave (same law)
    excess   : measured - floor   (~0 under scale-invariance, > 0 under real drift)

Bootstrap resamples MAPS (the independent unit), so the error bar shrinks ~1/sqrt(N_maps)
and inherits the field's map-to-map variance. Sizes are matched per resample so the
finite-sample bias cancels in ``excess``.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

from .wavelet import DEFAULT_MODE, DEFAULT_WAVELET, octave_wc

_MIN_BIN = 8   # skip conditional bins with fewer than this many samples on a side


def _standardize(w, c):
    return (w - w.mean()) / w.std(), (c - c.mean()) / c.std()


def binned_w1(w_a, c_a, w_b, c_b, n_bins, edges=None):
    """Mean W1 between p_a(w|c-bin) and p_b(w|c-bin) over coarse quantile bins.

    Bin edges come from the pooled ``c`` of both sides (comparable conditioning).
    Each bin is weighted by the smaller of its two occupancies.
    """
    if edges is None:
        edges = np.quantile(np.concatenate([c_a, c_b]), np.linspace(0, 1, n_bins + 1))
    edges = np.array(edges, dtype=float)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    num = 0.0
    den = 0
    for b in range(n_bins):
        ma = (c_a >= edges[b]) & (c_a < edges[b + 1])
        mb = (c_b >= edges[b]) & (c_b < edges[b + 1])
        na, nb = int(ma.sum()), int(mb.sum())
        if na < _MIN_BIN or nb < _MIN_BIN:
            continue
        d = wasserstein_distance(w_a[ma], w_b[mb])
        n = min(na, nb)
        num += d * n
        den += n
    if den == 0:
        return np.nan
    return num / den


def collect_wc(maps, octaves, wavelet=DEFAULT_WAVELET, mode=DEFAULT_MODE):
    """Per-map (w, c) arrays at each octave: ``{j: [(w, c), ... one per map]}``."""
    data = {j: [] for j in octaves}
    for f in maps:
        for j in octaves:
            data[j].append(octave_wc(f, j, wavelet, mode))
    return data


def collect_wc_grouped(parents, octaves, wavelet=DEFAULT_WAVELET, mode=DEFAULT_MODE):
    """Like :func:`collect_wc`, but each entry is a PARENT map given as a list of tiles.

    ``parents`` is a list (one per parent map) of lists of 2-D tiles. For each parent
    and octave the tile (w, c) samples are concatenated into a single entry, so the
    bootstrap in :func:`drift_estimate` resamples parent maps (the independent unit)
    rather than the spatially-correlated tiles cut from one parent.
    """
    data = {j: [] for j in octaves}
    for tiles in parents:
        for j in octaves:
            ws, cs = [], []
            for t in tiles:
                w, c = octave_wc(t, j, wavelet, mode)
                ws.append(w)
                cs.append(c)
            data[j].append((np.concatenate(ws), np.concatenate(cs)))
    return data


def _pool(sel):
    w = np.concatenate([wc[0] for wc in sel])
    c = np.concatenate([wc[1] for wc in sel])
    return w, c


def _measured_once(list_j, list_k, idx, n_bins, rng):
    """Cross-octave conditional W1 for a map resample ``idx``, sizes matched to M.

    ``list_j``/``list_k`` are the two octaves' per-map (w, c). The larger pool is
    subsampled to the smaller pool size M so the two sides carry equal sample counts.
    """
    wj, cj = _standardize(*_pool([list_j[i] for i in idx]))
    wk, ck = _standardize(*_pool([list_k[i] for i in idx]))
    if wj.size < wk.size:            # "fine" = larger (higher-frequency) pool
        wj, cj, wk, ck = wk, ck, wj, cj
    M = wk.size
    perm = rng.permutation(wj.size)
    fw, fc = wj[perm[:M]], cj[perm[:M]]
    edges = np.quantile(np.concatenate([fc, ck]), np.linspace(0, 1, n_bins + 1))
    return binned_w1(fw, fc, wk, ck, n_bins, edges)


def _floor(list_j, list_k, n_bins, rng, n_rep=8):
    """Finite-sample W1 baseline: two disjoint size-M subsamples of the SAME octave.

    Computed once on the FULL data (no bootstrap duplicates) and averaged over
    ``n_rep`` random partitions -- a fixed bias correction. Using the full,
    duplicate-free pool is essential: split-halves drawn from a bootstrap-resampled
    pool would share identical maps and give an artificially low floor. Uses the
    finer (larger) octave, which always holds >= 2M samples for separation >= 1;
    M matches :func:`_measured_once` because every map contributes the same count.
    """
    pj_w, pj_c = _pool(list_j)
    pk_w, _pk_c = _pool(list_k)
    M = min(pj_w.size, pk_w.size)
    # standardize the finer (larger) octave's full pool
    if pj_w.size >= pk_w.size:
        wj, cj = _standardize(pj_w, pj_c)
    else:
        wj, cj = _standardize(pk_w, _pk_c)
    if wj.size < 2 * M:
        return np.nan, np.nan
    vals = []
    for _ in range(n_rep):
        perm = rng.permutation(wj.size)
        a, b = perm[:M], perm[M:2 * M]
        edges = np.quantile(np.concatenate([cj[a], cj[b]]),
                            np.linspace(0, 1, n_bins + 1))
        vals.append(binned_w1(wj[a], cj[a], wj[b], cj[b], n_bins, edges))
    vals = np.array(vals)
    return float(np.nanmean(vals)), float(np.nanstd(vals, ddof=1))


def drift_estimate(data, j_fine, j_coarse, n_bins=8, n_boot=200, seed=0):
    """Excess conditional-W1 drift between two octaves, with map-bootstrap error bars.

    Parameters
    ----------
    data : dict from :func:`collect_wc` (must contain both octaves).
    j_fine, j_coarse : octave indices (order irrelevant; the larger pool is treated
        as "fine" internally).
    Returns a dict with point estimates (``measured``, ``floor``, ``excess``),
    bootstrap SEs, and the 95% percentile CI of ``excess`` and ``measured``.
    """
    list_j, list_k = data[j_fine], data[j_coarse]
    n = len(list_j)
    rng = np.random.default_rng(seed)

    # Floor: fixed finite-sample bias correction from the full (duplicate-free) data.
    floor0, floor_se = _floor(list_j, list_k, n_bins, rng)
    # Measured: point estimate on the full data, then map-bootstrap the sampling dist.
    meas0 = _measured_once(list_j, list_k, np.arange(n), n_bins, rng)
    meas_b = np.array([_measured_once(list_j, list_k, rng.integers(0, n, n), n_bins, rng)
                       for _ in range(n_boot)])

    exc0 = meas0 - floor0
    exc_b = meas_b - floor0            # floor is a fixed offset
    se = float(np.nanstd(exc_b, ddof=1))
    return {
        "j_fine": j_fine, "j_coarse": j_coarse,
        "separation": abs(j_coarse - j_fine),
        "measured": meas0, "floor": floor0, "excess": exc0,
        "measured_se": float(np.nanstd(meas_b, ddof=1)),
        "floor_se": floor_se,
        "excess_se": se,
        "excess_ci": [float(np.nanpercentile(exc_b, 2.5)),
                      float(np.nanpercentile(exc_b, 97.5))],
        "measured_ci": [float(np.nanpercentile(meas_b, 2.5)),
                        float(np.nanpercentile(meas_b, 97.5))],
        "z": float(exc0 / se) if se > 0 else np.nan,
        "n_maps": n,
    }


def drift_curve(maps, octaves, ref_octave=None, n_bins=8, n_boot=200, seed=0,
                wavelet=DEFAULT_WAVELET, mode=DEFAULT_MODE):
    """Excess drift vs octave separation.

    If ``ref_octave`` is given, measures drift of every other octave against it
    (drift as a function of separation from a fixed finest train scale). Otherwise
    measures adjacent-octave drift for each consecutive pair. Returns a list of
    :func:`drift_estimate` dicts.
    """
    data = collect_wc(maps, octaves, wavelet, mode)
    out = []
    if ref_octave is not None:
        for j in octaves:
            if j == ref_octave:
                continue
            out.append(drift_estimate(data, ref_octave, j, n_bins, n_boot, seed))
    else:
        for a, b in zip(octaves[:-1], octaves[1:]):
            out.append(drift_estimate(data, a, b, n_bins, n_boot, seed))
    return out
