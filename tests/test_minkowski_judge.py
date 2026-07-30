"""Validation of the FROZEN Stage-3 Minkowski judge (R38 order 5).

Synthetic GRFs generated in-test ONLY — the judge touches no committed or
future generation until Stage 3. Gaussian references: V0 = 1 - Phi(nu);
V1 shape prop. exp(-nu^2/2); V2 shape prop. nu*exp(-nu^2/2) (Tomita) —
shape tests use threshold RATIOS so lattice constants cancel.
"""
import os
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_p2"))

from minkowski_judge import judge_T, minkowski_vector, stack_minkowski  # noqa: E402

NUS = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)


def _grf(rng, n, size=128, alpha=1.0):
    k = np.fft.fftfreq(size) * size
    kk = np.hypot(k[:, None], k[None, :])
    filt = np.where(kk > 0, np.maximum(kk, 1e-9) ** -alpha, 0.0)
    out = []
    for _ in range(n):
        f = np.fft.ifft2(np.fft.fft2(rng.standard_normal((size, size)))
                         * filt).real
        out.append(f / f.std())
    return np.array(out)


def test_v0_matches_gaussian_area():
    rng = np.random.default_rng(20260730)
    fields = _grf(rng, 32)
    mean, se = stack_minkowski(fields, NUS, n_boot=400, seed=1)
    for i, nu in enumerate(NUS):
        expect = 1.0 - norm.cdf(nu)
        assert abs(mean[i] - expect) < max(5 * se[i], 0.01), \
            f"V0({nu}) {mean[i]:.4f} vs Gaussian {expect:.4f}"


def test_v1_v2_gaussian_shapes():
    rng = np.random.default_rng(4)
    fields = _grf(rng, 48)
    mean, _ = stack_minkowski(fields, NUS, n_boot=10, seed=2)
    n = len(NUS)
    v1 = {nu: mean[n + i] for i, nu in enumerate(NUS)}
    v2 = {nu: mean[2 * n + i] for i, nu in enumerate(NUS)}
    # V1(1)/V1(0) = exp(-1/2); V1(2)/V1(0) = exp(-2)
    assert abs(v1[1.0] / v1[0.0] - np.exp(-0.5)) < 0.05
    assert abs(v1[2.0] / v1[0.0] - np.exp(-2.0)) < 0.05
    # V2: odd in nu (sign flip) and |V2(2)/V2(1)| = 2*exp(-3/2)
    assert v2[1.0] * v2[-1.0] < 0, "V2 not odd in nu"
    assert abs(abs(v2[2.0] / v2[1.0]) - 2 * np.exp(-1.5)) < 0.08
    assert abs(v2[0.0]) < 0.2 * abs(v2[1.0]), "V2(0) not ~0"


def test_d4_invariance():
    rng = np.random.default_rng(9)
    fields = _grf(rng, 3)
    for f in fields:
        v0 = minkowski_vector(f)
        for g in (lambda x: np.rot90(x, 1), lambda x: np.rot90(x, 2),
                  lambda x: x[::-1], lambda x: x[:, ::-1],
                  lambda x: np.rot90(x, 1)[:, ::-1]):
            assert np.allclose(minkowski_vector(g(f)), v0, atol=1e-12), \
                "Minkowski vector not D4-invariant"


def test_judge_discriminates_and_nulls():
    rng = np.random.default_rng(17)
    a = _grf(rng, 32, alpha=1.0)
    b = _grf(rng, 32, alpha=1.0)
    c = _grf(rng, 32, alpha=1.6)  # different morphology
    t_null, _ = judge_T(a, b, n_boot=400, seed=5)
    t_disc, _ = judge_T(a, c, n_boot=400, seed=6)
    assert t_null < 3.5, f"null T {t_null:.2f} too high (same-law stacks)"
    assert t_disc > 5, f"discrimination T {t_disc:.2f} too low"


def test_bootstrap_se_scaling():
    rng = np.random.default_rng(23)
    fields = _grf(rng, 64)
    _, se32 = stack_minkowski(fields[:32], NUS, n_boot=1000, seed=7)
    _, se64 = stack_minkowski(fields, NUS, n_boot=1000, seed=8)
    med = np.median(se32 / np.maximum(se64, 1e-30))
    assert 1.15 < med < 1.75, f"SE scaling {med:.2f} not ~sqrt(2)"
