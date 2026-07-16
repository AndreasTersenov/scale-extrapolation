"""Stage-A validation gates (tests-first, per NIGHT-ORDERS): the exact conditional
sampler is validated against dense linear algebra on a small grid BEFORE any ensemble
is generated.

Gates:
  1. operator identities  — P/P^T adjointness; Sigma symmetric PSD action; my Haar
     coarse == pywt cA (pywt cross-check auto-skips where pywt is absent — the hook's
     JAX env; it runs under the env.sh stack, recorded in the Stage-A log).
  2. dense conditional mean — Hoffman–Ribak deterministic path == dense
     Sigma P^T M^{-1} c* at machine tolerance (16^2 grid, level 2).
  3. dense conditional covariance — empirical covariance of redraws matches the dense
     conditional covariance (statistical tolerance, seeded).
  4. constraint reproduction — haar_coarse(redraw) == c* at machine tolerance,
     including at the production geometry (128^2, level 4).
  5. marginal law — pooling conditional redraws over prior-drawn parents reproduces
     the unconditional pixel variance (statistical tolerance).
  6. estimand nulls — truth_stats on a pure GRF gives var_slope ~ 0 and kurtosis ~ 0;
     independent-Haar octave_wc_pooled matches scaledrift's octave_wc where available.
"""
import numpy as np
import pytest

from sandbox.lognormal import (
    GRFSpec,
    coarse_spectrum,
    conditional_mean_map,
    conditional_redraw,
    haar_coarse,
    haar_coarse_adjoint,
    lognormal_map,
    sample_grf,
    sigma_apply,
)
from sandbox.haar import haar_level, octave_wc_pooled
from sandbox.truth_stats import estimand_scalars

SPEC16 = GRFSpec(shape=(16, 16), alpha=2.0)
LVL = 2  # 16^2 -> coarse 4^2


def _dense_sigma(spec):
    n = spec.shape[0] * spec.shape[1]
    S = np.empty((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        S[:, i] = sigma_apply(e.reshape(spec.shape), spec).reshape(-1)
    return S


def _dense_P(spec, lvl):
    n = spec.shape[0] * spec.shape[1]
    nc = n // 4 ** lvl
    P = np.empty((nc, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        P[:, i] = haar_coarse(e.reshape(spec.shape), lvl).reshape(-1)
    return P


def test_adjointness():
    rng = np.random.default_rng(0)
    v = rng.standard_normal(SPEC16.shape)
    u = rng.standard_normal((4, 4))
    lhs = np.sum(haar_coarse(v, LVL) * u)
    rhs = np.sum(v * haar_coarse_adjoint(u, LVL))
    assert abs(lhs - rhs) < 1e-12


def test_haar_coarse_matches_pywt():
    pywt = pytest.importorskip("pywt")
    rng = np.random.default_rng(1)
    f = rng.standard_normal((32, 32))
    for lvl in (1, 2, 3):
        cA = pywt.wavedec2(f, "haar", mode="periodization", level=lvl)[0]
        assert np.allclose(haar_coarse(f, lvl), cA, atol=1e-12)
    # single-level detail conventions too (truth-side Haar vs pywt)
    cA, (cH, cV, cD) = pywt.wavedec2(f, "haar", mode="periodization", level=1)[0], \
        pywt.wavedec2(f, "haar", mode="periodization", level=1)[1]
    a2, (h2, v2, d2) = haar_level(f)
    assert np.allclose(a2, cA, atol=1e-12)
    assert np.allclose(h2, cH, atol=1e-12)
    assert np.allclose(v2, cV, atol=1e-12)
    assert np.allclose(d2, cD, atol=1e-12)


def test_octave_wc_matches_scaledrift():
    pytest.importorskip("pywt")
    from scaledrift import octave_wc
    rng = np.random.default_rng(2)
    f = rng.standard_normal((64, 64))
    for j in (1, 2, 3):
        w_ref, c_ref = octave_wc(f, j)
        w_my, c_my = octave_wc_pooled(f, j)
        assert np.allclose(np.sort(np.abs(w_my)), np.sort(np.abs(w_ref)), atol=1e-12)
        assert np.allclose(c_my, c_ref, atol=1e-12)
        # pooled-statistic equality is what the estimand needs
        assert np.allclose(w_my.std(), w_ref.std(), atol=1e-12)


def test_dense_conditional_mean_machine_tolerance():
    Sd = _dense_sigma(SPEC16)
    Pd = _dense_P(SPEC16, LVL)
    M = Pd @ Sd @ Pd.T
    rng = np.random.default_rng(3)
    g_parent = sample_grf(SPEC16, rng)
    c_star = haar_coarse(g_parent, LVL)
    # dense conditional mean via pseudo-inverse (M has the null DC mode)
    mu_dense = Sd @ Pd.T @ np.linalg.pinv(M, rcond=1e-10) @ c_star.reshape(-1)
    mu_op = conditional_mean_map(c_star, SPEC16, LVL)
    assert np.max(np.abs(mu_op.reshape(-1) - mu_dense)) < 1e-8


def test_dense_conditional_covariance_statistical():
    Sd = _dense_sigma(SPEC16)
    Pd = _dense_P(SPEC16, LVL)
    M = Pd @ Sd @ Pd.T
    Minv = np.linalg.pinv(M, rcond=1e-10)
    cov_dense = Sd - Sd @ Pd.T @ Minv @ Pd @ Sd
    rng = np.random.default_rng(4)
    g_parent = sample_grf(SPEC16, rng)
    c_star = haar_coarse(g_parent, LVL)
    lam = coarse_spectrum(SPEC16, LVL)
    n_draw = 6000
    draws = np.stack([conditional_redraw(c_star, SPEC16, LVL, rng, lam=lam).reshape(-1)
                      for _ in range(n_draw)])
    cov_emp = np.cov(draws.T)
    scale = np.max(np.abs(cov_dense))
    err = np.max(np.abs(cov_emp - cov_dense)) / scale
    # empirical covariance from 6000 draws: relative max-error tolerance ~ 4/sqrt(N)
    assert err < 0.08, f"conditional covariance mismatch: rel max err {err:.3f}"


def test_constraint_reproduction_machine_tolerance():
    for spec, lvl in ((SPEC16, 2), (GRFSpec(shape=(128, 128), alpha=2.0), 4)):
        rng = np.random.default_rng(5)
        g_parent = sample_grf(spec, rng)
        c_star = haar_coarse(g_parent, lvl)
        lam = coarse_spectrum(spec, lvl)
        g = conditional_redraw(c_star, spec, lvl, rng, lam=lam)
        assert np.max(np.abs(haar_coarse(g, lvl) - c_star)) < 1e-8


def test_marginal_law_preserved():
    rng = np.random.default_rng(6)
    lam = coarse_spectrum(SPEC16, LVL)
    vals = []
    for _ in range(400):
        c_star = haar_coarse(sample_grf(SPEC16, rng), LVL)
        vals.append(conditional_redraw(c_star, SPEC16, LVL, rng, lam=lam))
    pixvar = np.var(np.stack(vals))
    assert abs(pixvar - 1.0) < 0.05, f"marginal pixel variance {pixvar:.3f} != 1"


def test_estimand_null_on_grf():
    """GRF null: var_slope is exactly 0 in population. The pooled KURTOSIS is NOT
    exactly 0 even on a GRF — the three orientation bands have unequal variances, so
    the orientation-pooled marginal is a Gaussian scale MIXTURE (excess kurtosis
    3*(E[v^2]/E[v]^2 - 1) > 0). This is a property of the (frozen) estimand's
    definition shared by instrument and truth; assert the analytic mixture value.
    """
    rng = np.random.default_rng(7)
    spec = GRFSpec(shape=(64, 64), alpha=2.0)
    fields = [sample_grf(spec, rng) for _ in range(200)]
    ws, cs = [], []
    per_band = [[], [], []]
    for f in fields:
        w, c = octave_wc_pooled(f, 1)
        ws.append(w)
        cs.append(c)
        nb = w.size // 3
        for k in range(3):
            per_band[k].append(w[k * nb:(k + 1) * nb])
    s = estimand_scalars(np.concatenate(ws), np.concatenate(cs))
    assert abs(s["var_slope"]) < 0.03, s
    v = np.array([np.var(np.concatenate(b)) for b in per_band])
    kurt_mix = 3.0 * (np.mean(v ** 2) / np.mean(v) ** 2 - 1.0)
    assert abs(s["kurtosis"] - kurt_mix) < 0.10, (s, kurt_mix)


def test_lognormal_map_moments():
    rng = np.random.default_rng(8)
    g = rng.standard_normal(500000)
    for sig in (0.4, 0.8):
        L = lognormal_map(g, sig)
        assert abs(L.mean()) < 5e-3
        assert abs(L.var() - (np.exp(sig ** 2) - 1.0)) < 5e-3 * np.exp(2 * sig ** 2)


def test_redraw_determinism():
    rng1 = np.random.default_rng(9)
    rng2 = np.random.default_rng(9)
    c = haar_coarse(sample_grf(SPEC16, np.random.default_rng(10)), LVL)
    a = conditional_redraw(c, SPEC16, LVL, rng1)
    b = conditional_redraw(c, SPEC16, LVL, rng2)
    assert np.array_equal(a, b)
