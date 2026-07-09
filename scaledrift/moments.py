"""Conditional moment profiles, marginal PDFs, cross-octave couplings, and the
P9b "running couplings" dimensionality analysis.

The conditional moments E[w|c-bin], Var[w|c-bin], Skew[w|c-bin] as functions of the
coarse quantile bin ARE the objects WC-RG models per scale. Their variation across
octaves is the "running coupling" the scale-conditioning must absorb; P9b asks whether
that variation is low-dimensional (1-3 smooth functions of scale).
"""
from __future__ import annotations

import numpy as np

from .wavelet import DEFAULT_MODE, DEFAULT_WAVELET, octave_pair, octave_wc


def _standardize(x):
    return (x - x.mean()) / x.std()


def conditional_moments(w, c, n_bins=10):
    """Per-quantile-bin conditional moments of w given c (both standardized here).

    Equal-count (quantile) bins in c. Returns dict of length-``n_bins`` arrays:
    ``c_center`` (mean c in bin), ``count``, ``mean``, ``var``, ``skew``.
    """
    w = _standardize(np.asarray(w, float))
    c = _standardize(np.asarray(c, float))
    edges = np.quantile(c, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(c, edges) - 1, 0, n_bins - 1)
    cc = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins)
    mean = np.full(n_bins, np.nan)
    var = np.full(n_bins, np.nan)
    skew = np.full(n_bins, np.nan)
    for b in range(n_bins):
        wb = w[idx == b]
        cb = c[idx == b]
        cnt[b] = wb.size
        if wb.size < 8:
            continue
        cc[b] = cb.mean()
        mean[b] = wb.mean()
        v = wb.var()
        var[b] = v
        if v > 0:
            skew[b] = float(np.mean(((wb - wb.mean()) / np.sqrt(v)) ** 3))
    return {"c_center": cc, "count": cnt, "mean": mean, "var": var, "skew": skew}


def octave_conditional_moments(maps, j, n_bins=10, wavelet=DEFAULT_WAVELET,
                               mode=DEFAULT_MODE):
    """Conditional moments at octave ``j`` pooled over a set of maps."""
    ws, cs = [], []
    for f in maps:
        w, c = octave_wc(f, j, wavelet, mode)
        ws.append(w)
        cs.append(c)
    return conditional_moments(np.concatenate(ws), np.concatenate(cs), n_bins)


def coupling_scalars(w, c, n_bins=10):
    """Interpretable per-octave "running couplings" from pooled, standardized (w, c).

    Returns three scalars that summarize the conditional structure a WC-RG model would
    carry per scale:

    * ``var_slope`` -- OLS slope of conditional Var(w | coarse bin) against the bin's
      mean coarse value. The conditional-variance modulation: 0 for a Gaussian field,
      positive when detail power grows with the local coarse amplitude (the physical
      non-Gaussianity). This is the primary running coupling.
    * ``var_hi_lo`` -- Var in the top coarse bin divided by Var in the bottom bin
      (modulation amplitude, dimensionless).
    * ``kurtosis`` -- marginal excess kurtosis of the detail coefficients (scale-
      dependent non-Gaussianity; 0 for a GRF).
    """
    ws = (np.asarray(w, float) - np.mean(w)) / np.std(w)
    m = conditional_moments(ws, c, n_bins)
    cc, var = m["c_center"], m["var"]
    ok = ~np.isnan(cc) & ~np.isnan(var)
    var_slope = float(np.polyfit(cc[ok], var[ok], 1)[0]) if ok.sum() >= 2 else np.nan
    v = var[ok]
    var_hi_lo = float(v[-1] / v[0]) if v.size >= 2 and v[0] > 0 else np.nan
    kurtosis = float(np.mean(ws ** 4) - 3.0)
    return {"var_slope": var_slope, "var_hi_lo": var_hi_lo, "kurtosis": kurtosis}


def marginal_pdf(maps, j, bins=None, wavelet=DEFAULT_WAVELET, mode=DEFAULT_MODE):
    """Histogram of standardized detail coefficients at octave ``j`` (measurement a)."""
    ws = [octave_wc(f, j, wavelet, mode)[0] for f in maps]
    w = _standardize(np.concatenate(ws))
    if bins is None:
        bins = np.linspace(-6, 6, 121)
    hist, edges = np.histogram(w, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def cross_octave_coupling(maps, j, wavelet=DEFAULT_WAVELET, mode=DEFAULT_MODE,
                          n_boot=200, seed=0):
    """Correlation of |w_j| (block-averaged 2x) with |w_{j+1}| at aligned positions
    (measurement c), with a map-bootstrap error bar.
    """
    per_map = []
    for f in maps:
        dj, _ = octave_pair(f, j, wavelet, mode)
        dk, _ = octave_pair(f, j + 1, wavelet, mode)
        mag_j = np.abs(dj).mean(axis=0)          # (Hj, Wj)  orientation-averaged
        mag_k = np.abs(dk).mean(axis=0)          # (Hj/2, Wj/2)
        h, w_ = mag_k.shape
        block = mag_j[:2 * h, :2 * w_].reshape(h, 2, w_, 2).mean(axis=(1, 3))
        per_map.append((block.reshape(-1), mag_k.reshape(-1)))

    def corr(sel):
        a = np.concatenate([p[0] for p in sel])
        b = np.concatenate([p[1] for p in sel])
        return float(np.corrcoef(a, b)[0, 1])

    point = corr(per_map)
    rng = np.random.default_rng(seed)
    n = len(per_map)
    boot = np.array([corr([per_map[i] for i in rng.integers(0, n, n)])
                     for _ in range(n_boot)])
    return {"j": j, "rho": point, "se": float(boot.std(ddof=1)),
            "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def running_coupling_pca(maps, octaves, n_bins=10, stats=("var", "skew"),
                         wavelet=DEFAULT_WAVELET, mode=DEFAULT_MODE):
    """Effective dimensionality of the cross-octave drift of conditional moments (P9b).

    Builds one feature vector per octave by concatenating the chosen conditional-moment
    profiles (default variance + skewness vs coarse quantile bin), stacks them into an
    (n_octaves x n_features) matrix, centers across octaves, and does a PCA (SVD). The
    number of components needed for >= 80% of the across-octave variance is the drift's
    effective dimensionality; <= 3 => P9b holds ("few running couplings").
    """
    profiles = []
    for j in octaves:
        m = octave_conditional_moments(maps, j, n_bins, wavelet, mode)
        vec = np.concatenate([m[s] for s in stats])
        profiles.append(vec)
    P = np.array(profiles)                        # (n_oct, n_feat)
    good = ~np.any(np.isnan(P), axis=0)           # drop bins missing in some octave
    P = P[:, good]
    # per-feature standardization so var and skew profiles weigh comparably
    scale = P.std(axis=0)
    scale[scale == 0] = 1.0
    Pz = (P - P.mean(axis=0)) / scale
    # SVD of the octave-centered matrix
    _, s, _ = np.linalg.svd(Pz, full_matrices=False)
    var_ratio = (s ** 2) / np.sum(s ** 2)
    cum = np.cumsum(var_ratio)
    eff_dim = int(np.searchsorted(cum, 0.80) + 1)
    return {"octaves": list(octaves), "explained_var_ratio": var_ratio,
            "cumulative": cum, "eff_dim_80": eff_dim,
            "profiles": P, "profiles_z": Pz}
