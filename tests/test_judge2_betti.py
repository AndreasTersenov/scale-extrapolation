"""Tests-first validation of JUDGE-2 (Betti curves; prereg
2026-08-05-prereg-night3 §N1f). Synthetic in-test GRFs ONLY — no real
or generated field stack is loaded here, per the quarantine."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_p2"))

from judge2_betti import (  # noqa: E402
    Z_SENTINEL,
    betti_curves,
    judge2_T,
    stack_betti,
)


def _grf(rng, size=128, alpha=1.0):
    k = np.fft.fftfreq(size) * size
    kk = np.hypot(k[:, None], k[None, :])
    filt = np.where(kk > 0, np.maximum(kk, 1e-9) ** -alpha, 0.0)
    f = np.fft.ifft2(np.fft.fft2(rng.standard_normal((size, size)))
                     * filt).real
    return f / f.std()


def _lognormal(rng):
    """Standardized exp(GRF): asymmetric component/hole structure."""
    x = np.exp(_grf(rng))
    return (x - x.mean()) / x.std()


def test_null_grf_vs_grf():
    rng = np.random.default_rng(20260805)
    a = [_grf(rng) for _ in range(24)]
    b = [_grf(rng) for _ in range(24)]
    T, detail = judge2_T(a, b, n_boot=400, seed=1)
    assert np.isfinite(detail["z"]).all()
    assert T < 3.5, f"null T {T:.2f} >= 3.5"


def test_discriminates_lognormal():
    rng = np.random.default_rng(7)
    grf = [_grf(rng) for _ in range(24)]
    logn = [_lognormal(rng) for _ in range(24)]
    T, _ = judge2_T(logn, grf, n_boot=400, seed=2)
    assert T > 5, f"lognormal T {T:.1f} <= 5"


def test_d4_invariance():
    rng = np.random.default_rng(11)
    f = _grf(rng)
    c0 = betti_curves(f)
    for g in (lambda x: np.rot90(x, 1), lambda x: x[::-1],
              lambda x: np.rot90(x, 2)[:, ::-1]):
        assert np.array_equal(betti_curves(g(f)), c0)


def test_se_scaling():
    rng = np.random.default_rng(13)
    fields = [_grf(rng) for _ in range(48)]
    _, s24 = stack_betti(fields[:24], n_boot=2000, seed=5)
    _, s48 = stack_betti(fields, n_boot=2000, seed=6)
    ok = (s24 > 0) & (s48 > 0)
    ratio = np.median(s24[ok] / s48[ok])
    assert 1.15 < ratio < 1.75, f"SE ratio {ratio:.2f}"


def test_sanity_anchor_low_nu():
    rng = np.random.default_rng(3)
    f = _grf(rng, alpha=1.5)
    b = betti_curves(f)
    assert b[0, 0] == 1, f"b0(-3) = {b[0, 0]:.0f} != 1"
    assert b[1, 0] == 0, f"b1(-3) = {b[1, 0]:.0f} != 0"


def test_zero_variance_guard():
    """Degenerate stacks (identical copies -> SE 0): z = 0 where means
    agree, +-Z_SENTINEL where they differ, never NaN."""
    rng = np.random.default_rng(17)
    a = [_grf(rng)] * 24
    b = [_grf(rng)] * 24
    T, detail = judge2_T(a, b, n_boot=400, seed=4)
    z = detail["z"]
    assert np.isfinite(z).all()
    diff = detail["gen_mean"] - detail["real_mean"]
    assert np.all(z[diff == 0] == 0)
    assert np.all(np.abs(z[diff != 0]) == Z_SENTINEL)
    assert T == Z_SENTINEL, f"T {T} != sentinel"
