"""Validation gates for the starlet-l1 instrument (SPEC-starlet-l1).

Instrument = Andreas's wl_stats_torch behind the numpy boundary in
scripts_p2/starlet_l1_lib.py. Gates required by the SPEC:
  (i)  GRF sanity: l1 of a synthetic power-law GRF ensemble consistent
       between two independent halves within bootstrap;
  (ii) batch-vs-loop identity;
  (iii) D4-symmetry invariance.
Plus the transform identity our statistics rely on: the forward starlet is
first-generation (sum of scales == input). The package's gen2 reconstruct()
path does NOT satisfy the round trip (verify_installation.py fails on it);
that path is never used here — pinned by test_gen1_sum_identity.
"""
import numpy as np
import pytest

from scripts_p2.starlet_l1_lib import (
    N_BINS,
    combined_ranges,
    l1_curves,
    totals_boot,
)


def make_grf(n_maps, n=128, slope=-2.0, seed=0):
    """Power-law GRF ensemble, P(k) ~ k^slope, unit variance, periodic."""
    rng = np.random.default_rng(seed)
    kx = np.fft.fftfreq(n)[:, None]
    ky = np.fft.fftfreq(n)[None, :]
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1.0
    amp = k ** (slope / 2.0)
    amp[0, 0] = 0.0
    maps = np.empty((n_maps, n, n))
    for i in range(n_maps):
        white = rng.standard_normal((n, n))
        f = np.fft.fft2(white) * amp
        m = np.fft.ifft2(f).real
        maps[i] = m / m.std()
    return maps


def test_gen1_sum_identity():
    import torch
    from wl_stats_torch.starlet import Starlet2D

    maps = make_grf(2, seed=5)
    st = Starlet2D(n_scales=5, device=torch.device("cpu"), dtype=torch.float64)
    t = torch.from_numpy(maps).unsqueeze(1)
    coeffs = st(t)
    recon = coeffs.sum(dim=1).numpy()
    assert np.max(np.abs(recon - maps)) < 1e-10


def test_grf_half_consistency():
    maps = make_grf(48, seed=1)
    sigma = float(maps.std())
    ranges = combined_ranges([maps], sigma)
    half_a, half_b = maps[:24], maps[24:]
    _, l1_a = l1_curves(half_a, sigma, ranges)
    _, l1_b = l1_curves(half_b, sigma, ranges)
    for s in range(4):
        ta, sea = totals_boot(l1_a[s], seed=10 + s)
        tb, seb = totals_boot(l1_b[s], seed=20 + s)
        assert abs(ta - tb) <= 3.0 * np.hypot(sea, seb), (
            f"scale {s}: halves differ {ta:.4g} vs {tb:.4g} "
            f"(se {sea:.3g}/{seb:.3g})"
        )


def test_batch_vs_loop_identity():
    maps = make_grf(8, seed=2)
    sigma = float(maps.std())
    ranges = combined_ranges([maps], sigma)
    bins_batch, l1_batch = l1_curves(maps, sigma, ranges)
    for i in range(len(maps)):
        bins_one, l1_one = l1_curves(maps[i : i + 1], sigma, ranges)
        for s in range(4):
            assert np.allclose(bins_batch[s], bins_one[s], rtol=0, atol=1e-12)
            assert np.allclose(l1_batch[s][i], l1_one[s][0], rtol=1e-10,
                               atol=1e-12), f"map {i} scale {s}"


def test_d4_symmetry():
    maps = make_grf(8, seed=3)
    sigma = float(maps.std())
    ranges = combined_ranges([maps], sigma)
    _, l1_ref = l1_curves(maps, sigma, ranges)
    transforms = {
        "rot90": np.rot90(maps, k=1, axes=(1, 2)),
        "rot180": np.rot90(maps, k=2, axes=(1, 2)),
        "fliplr": maps[:, :, ::-1],
        "flipud": maps[:, ::-1, :],
    }
    for name, tmaps in transforms.items():
        _, l1_t = l1_curves(np.ascontiguousarray(tmaps), sigma, ranges)
        for s in range(4):
            scale = max(1.0, np.abs(l1_ref[s]).max())
            assert np.allclose(l1_t[s], l1_ref[s], rtol=0,
                               atol=1e-8 * scale), f"{name} scale {s}"


def test_bins_cover_everything():
    # With combined ranges nothing falls outside the bins: the per-map total
    # over bins equals the plain sum of |SNR| at that scale (the package's
    # l1 accumulates |SNR|, the cosmostat convention).
    from scripts_p2.starlet_l1_lib import snr_planes

    maps = make_grf(4, seed=4)
    sigma = float(maps.std())
    ranges = combined_ranges([maps], sigma)
    _, l1 = l1_curves(maps, sigma, ranges)
    snr = snr_planes(maps, sigma)
    for s in range(4):
        direct = np.abs(snr[:, s]).sum(axis=(1, 2))
        binned = l1[s].sum(axis=1)
        assert np.allclose(binned, direct, rtol=1e-9), f"scale {s}"
