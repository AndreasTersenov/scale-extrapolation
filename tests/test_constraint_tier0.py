"""Tests-first validation of the tier-0 empty-beam-floor violation
instrument (prereg 2026-08-06-d1; BRIEF-foundations tier 0)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_p2"))

from constraint_tier0 import (  # noqa: E402
    adjudicate,
    floor,
    real_loo_null,
    stack_violation,
    violation_rate,
)


def _grf(rng, size=128, alpha=1.0):
    k = np.fft.fftfreq(size) * size
    kk = np.hypot(k[:, None], k[None, :])
    filt = np.where(kk > 0, np.maximum(kk, 1e-9) ** -alpha, 0.0)
    f = np.fft.ifft2(np.fft.fft2(rng.standard_normal((size, size)))
                     * filt).real
    return f / f.std()


def test_null_no_false_violation():
    """Two independent same-law stacks -> generator does not undershoot."""
    rng = np.random.default_rng(1)
    real = [_grf(rng) for _ in range(32)]
    gen = [_grf(rng) for _ in range(32)]
    v = adjudicate(gen, real, seed=10)
    assert not v["violation"], v
    assert v["z"] < 3, v


def test_discriminates_inflated_lower_tail():
    """A stack with a heavier LOWER tail crosses the real floor -> flagged."""
    rng = np.random.default_rng(2)
    real = [_grf(rng) for _ in range(32)]
    # push the lower tail down: subtract a one-sided exponential bump
    gen = []
    for _ in range(32):
        f = _grf(rng)
        e = rng.standard_exponential(f.shape)
        gen.append(f - 0.8 * e)
    v = adjudicate(gen, real, seed=11)
    assert v["violation"] and v["z"] > 5, v


def test_d4_invariance():
    rng = np.random.default_rng(3)
    real = [_grf(rng) for _ in range(8)]
    flr = floor(real)
    f = _grf(rng)
    base = violation_rate([f], flr)[0]
    for g in (lambda x: np.rot90(x, 1), lambda x: x[::-1],
              lambda x: np.rot90(x, 2)[:, ::-1]):
        assert abs(violation_rate([g(f)], flr)[0] - base) < 1e-12


def test_se_scaling():
    rng = np.random.default_rng(4)
    real = [_grf(rng) for _ in range(16)]
    flr = floor(real)
    gen = [_grf(rng) - 0.5 * rng.standard_exponential((128, 128))
           for _ in range(48)]
    _, s24 = stack_violation(gen[:24], flr, n_boot=3000, seed=5)
    _, s48 = stack_violation(gen, flr, n_boot=3000, seed=6)
    assert 1.15 < s24 / s48 < 1.75, s24 / s48


def test_loo_null_positive_and_finite():
    rng = np.random.default_rng(7)
    real = [_grf(rng) for _ in range(32)]
    m, s = real_loo_null(real, seed=8)
    assert m >= 0 and np.isfinite(m) and s > 0
