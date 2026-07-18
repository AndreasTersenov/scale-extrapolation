"""Starlet l1-norm behind a numpy boundary (SPEC-starlet-l1).

Wraps wl_stats_torch.WLStatistics: numpy stacks in -> torch float64 CPU
transform -> numpy curves out. Only the forward transform and
compute_wavelet_l1_norms are used (never reconstruct()).

Conventions (fixed here, declared in the prereg):
  N_SCALES = 5  -> 4 detail scales (~2,4,8,16 px <-> octaves 1..4) + coarse;
                   the coarse plane is excluded from all curves.
  N_BINS   = 31 SNR bins per scale.
  SNR ranges are per-scale [min, max] over the COMBINED map sets of a leg
  (combined_ranges), passed explicitly to every call so real and generated
  share identical bins and no coefficient falls outside them.
  Noise normalization: per-scale SCALAR = sigma * interior plateau of the
  package's own impulse-response noise propagation (the plane's max). The
  shipped spatially-varying noise plane carries zero-pad border artifacts
  inconsistent with the periodic forward transform (breaks D4 of the l1 on
  sim tiles); the plateau value equals the analytic sigma*||h_s||_2 for
  interior pixels and is exactly D4-invariant. snr_coeffs is overwritten
  with w / plateau_s before binning.
"""
import numpy as np
import torch

from wl_stats_torch import WLStatistics

N_SCALES = 5
N_BINS = 31
N_DETAIL = N_SCALES - 1

torch.set_num_threads(4)


def _snr(maps, sigma):
    """WLStatistics with plateau-normalized SNR for a stack under scalar sigma.

    Returns (stats, snr) where snr = wavelet_coeffs / (per-scale plateau
    noise scalar); stats.snr_coeffs is overwritten with it so that
    compute_wavelet_l1_norms bins this convention.
    """
    stats = WLStatistics(n_scales=N_SCALES, device=torch.device("cpu"))
    t = torch.from_numpy(np.ascontiguousarray(maps, dtype=np.float64))
    out = stats.compute_wavelet_transform(t, noise_sigma=float(sigma))
    nl = out["noise_levels"]  # (B, n_scales, H, W), constant across batch
    plateau = nl[0].amax(dim=(-2, -1))  # (n_scales,) interior plateau values
    snr = out["wavelet_coeffs"] / plateau[None, :, None, None]
    stats.snr_coeffs = snr
    return stats, snr


def snr_planes(maps, sigma):
    """Plateau-normalized SNR planes as numpy (B, N_DETAIL, H, W)."""
    _, snr = _snr(maps, sigma)
    return snr.numpy()[:, :N_DETAIL]


def combined_ranges(map_sets, sigma):
    """Per-detail-scale (lo, hi) of SNR over the union of the given stacks."""
    lo = np.full(N_DETAIL, np.inf)
    hi = np.full(N_DETAIL, -np.inf)
    for maps in map_sets:
        _, snr = _snr(maps, sigma)
        snr = snr.numpy()
        for s in range(N_DETAIL):
            lo[s] = min(lo[s], snr[:, s].min())
            hi[s] = max(hi[s], snr[:, s].max())
    # The package builds thresholds with torch.linspace at default (float32)
    # precision and searchsorted(right=False) drops values at/outside the end
    # thresholds: pad both ends by 1e-6 of the span (>> float32 eps) so every
    # coefficient lands in a bin.
    return [(float(lo[s] - 1e-6 * (hi[s] - lo[s])),
             float(hi[s] + 1e-6 * (hi[s] - lo[s])))
            for s in range(N_DETAIL)]


def l1_curves(maps, sigma, ranges):
    """Per-detail-scale binned l1 curves.

    Returns (bins, l1): bins[s] is (N_BINS,) centers, l1[s] is (B, N_BINS)
    per-map sums of |w| in each SNR bin, scale s in 0..3 (~2,4,8,16 px).
    """
    stats, _ = _snr(maps, sigma)
    bins_out, l1_out = [], []
    for s in range(N_DETAIL):
        lo, hi = ranges[s]
        bins_list, l1_list = stats.compute_wavelet_l1_norms(
            n_bins=N_BINS, min_snr=lo, max_snr=hi)
        b = bins_list[s].numpy().astype(np.float64)
        v = l1_list[s].numpy().astype(np.float64)
        if v.ndim == 1:
            v = v[None, :]
        bins_out.append(b)
        l1_out.append(v)
    return bins_out, l1_out


def totals_boot(l1_permap, n_boot=400, seed=0):
    """Mean per-map total l1 (sum over bins) and bootstrap SE over maps."""
    totals = l1_permap.sum(axis=1)
    rng = np.random.default_rng(seed)
    n = len(totals)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = totals[idx].mean(axis=1)
    return float(totals.mean()), float(boots.std(ddof=1))


def curve_boot(l1_permap, n_boot=400, seed=0):
    """Per-bin mean curve and bootstrap SE over maps."""
    rng = np.random.default_rng(seed)
    n = l1_permap.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = l1_permap[idx].mean(axis=1)
    return l1_permap.mean(axis=0), boots.std(axis=0, ddof=1)


def tail_share_boot(l1_permap, bins, thresh=3.0, n_boot=400, seed=0):
    """Share of l1 carried by |SNR| >= thresh bins; bootstrap SE over maps."""
    mask = np.abs(bins) >= thresh
    tails = l1_permap[:, mask].sum(axis=1)
    totals = l1_permap.sum(axis=1)
    rng = np.random.default_rng(seed)
    n = len(totals)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = tails[idx].mean(axis=1) / totals[idx].mean(axis=1)
    return float(tails.mean() / totals.mean()), float(boots.std(ddof=1))


def add_noise(maps, sigma_n, seed):
    """Fixed-seed additive white Gaussian noise (secondary convention)."""
    rng = np.random.default_rng(seed)
    return maps + sigma_n * rng.standard_normal(maps.shape)
