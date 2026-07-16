"""Validation gates for the pilot's held-out statistics (wavelet-L1, scattering).

Per CLAUDE.md: estimators exist only after their gates. Gates: (1) identical stacks
give z = 0 exactly; (2) two independent GRF stacks are consistent (null); (3) a gross
texture difference is detected. GRF generated in-test with numpy FFT (no scaledrift
dependency in this env).
"""
import numpy as np
import pytest

from pilotstats import (peak_counts, scattering_logmeans, scattering_summary,
                        wavelet_l1, z_stack)


def powerlaw_grf(n, seed, size=128, alpha=1.5):
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(size)[:, None] ** 2 + np.fft.fftfreq(size)[None, :] ** 2
    amp = np.where(k > 0, k ** (-alpha / 2 / 2), 0.0)          # sqrt of P ~ k^-alpha
    out = []
    for _ in range(n):
        ph = np.fft.fft2(rng.normal(size=(size, size)))
        f = np.real(np.fft.ifft2(ph * amp))
        out.append((f - f.mean()) / f.std())
    return np.stack(out)


@pytest.fixture(scope="module")
def grf_pair():
    return powerlaw_grf(20, 1), powerlaw_grf(20, 2)


def test_identical_stack_zero(grf_pair):
    a, _ = grf_pair
    assert z_stack(wavelet_l1(a, 1), wavelet_l1(a, 1)) == 0.0
    z = z_stack(scattering_logmeans(a), scattering_logmeans(a))
    assert np.all(z == 0.0)


def test_grf_null(grf_pair):
    a, b = grf_pair
    for j in (1, 2):
        assert abs(z_stack(wavelet_l1(a, j), wavelet_l1(b, j))) < 3.0
    z = z_stack(scattering_logmeans(a), scattering_logmeans(b))
    s = scattering_summary(z)
    assert s["frac_flagged"] <= 0.10, s
    assert s["median_absz"] < 2.0, s


def test_peak_counts_gates(grf_pair):
    a, b = grf_pair
    # identical stack -> exact zero
    assert z_stack(peak_counts(a, 2.0), peak_counts(a, 2.0)) == 0.0
    # GRF-vs-GRF null
    for nu in (1.0, 2.0):
        assert abs(z_stack(peak_counts(a, nu), peak_counts(b, nu))) < 3.0
    # analytic sanity: higher threshold -> strictly fewer peaks, and some exist
    n1, n2 = peak_counts(a, 1.0).mean(), peak_counts(a, 2.0).mean()
    assert n1 > n2 > 0
    # gross difference: smoothing suppresses high peaks
    from scipy.signal import convolve2d
    k = np.ones((3, 3)) / 9.0
    sm = np.stack([convolve2d(x, k, mode="same", boundary="wrap") for x in b])
    sm = (sm - sm.mean(axis=(1, 2), keepdims=True)) / sm.std(axis=(1, 2), keepdims=True)
    assert z_stack(peak_counts(sm, 2.0), peak_counts(a, 2.0)) < -3.0


def test_gross_difference_detected(grf_pair):
    a, b = grf_pair
    k = np.ones((3, 3)) / 9.0
    from scipy.signal import convolve2d
    smoothed = np.stack([convolve2d(x, k, mode="same", boundary="wrap") for x in b])
    smoothed = (smoothed - smoothed.mean(axis=(1, 2), keepdims=True)) / \
        smoothed.std(axis=(1, 2), keepdims=True)
    assert abs(z_stack(wavelet_l1(smoothed, 1), wavelet_l1(a, 1))) > 3.0
    s = scattering_summary(z_stack(scattering_logmeans(smoothed),
                                   scattering_logmeans(a)))
    assert s["frac_flagged"] >= 0.25, s
