"""Validation gates for the R13 bake-off candidate objectives/samplers
(log/2026-07-17-reconvene-c3blocker.md; spec log/2026-07-17-c3-bakeoff.md).

Tests-first per CLAUDE.md: each new estimator piece is validated before the
bake-off toys run.

  1. chain_tail geometry: identity in the bulk, slope 1+a beyond tau, odd.
  2. tw identity: the chained fair-m estimator is unbiased for the
     threshold-weighted CRPS computed by DIRECT numerical quadrature of
     integral w(z) (F(z) - 1{y<=z})^2 dz with w = v' — the Gneiting-Ranjan /
     Allen et al. kernel-score identity, on a Gaussian predictive.
  3. beta=0.5 fair ES: propriety direction (paired design) — the true law beats
     shifted / under- / over-dispersed candidates.
  4. t-base flow pieces: unit variance + heavy tails of the base draws;
     deterministic sampler given a key; tcfm step runs with finite loss and
     nonzero gradients; beta=1/chain=None defaults leave the frozen C3 objective
     bit-identical (behavior preservation).
"""
import numpy as np
import pytest
import scipy.stats as st

import jax
import jax.numpy as jnp

from arms_p2.c3.energy import (CHAIN_A, CHAIN_TAU, chain_tail, energy_score_fair,
                               extract_patches, patched_energy_score)
from arms_p2.c1t.flow import make_tcfm_step, sample_tbase, t_base
from arms_p2.toys import make_data
from wfm.cfm import make_train_state
from wfm.model import ConditionalUNet


def test_chain_tail_geometry():
    z = jnp.array([-5.0, -2.0, -1.0, 0.0, 0.7, 2.0, 3.5])
    v = np.asarray(chain_tail(z))
    # bulk identity
    assert np.allclose(v[np.abs(np.asarray(z)) <= CHAIN_TAU],
                       np.asarray(z)[np.abs(np.asarray(z)) <= CHAIN_TAU])
    # slope 1+a beyond tau: v(3.5) = 3.5 + 4*1.5 = 9.5; odd symmetry
    assert np.isclose(v[-1], 3.5 + CHAIN_A * 1.5)
    assert np.isclose(v[0], -(5.0 + CHAIN_A * 3.0))
    assert np.allclose(np.asarray(chain_tail(-z)), -v)


def _tw_crps_quadrature(y, mu, sigma, tau=CHAIN_TAU, a=CHAIN_A):
    """twCRPS(N(mu,sigma^2), y; w) = integral w(z) (F(z) - 1{y<=z})^2 dz by direct
    quadrature, with w = v' = 1 + a*1{|z|>tau}. Identity: substituting u = v(z)
    (v strictly increasing) gives integral (F_v(u) - 1{v(y)<=u})^2 du =
    CRPS(F_v, v(y)) = E|v(X)-v(y)| - 1/2 E|v(X)-v(X')| — the chained kernel
    score. So the chained fair estimator must match this integral."""
    z = np.linspace(mu - 12 * sigma - 3 * a, mu + 12 * sigma + 3 * a, 400_001)
    w = 1.0 + a * (np.abs(z) > tau)
    F = st.norm.cdf(z, mu, sigma)
    ind = (z >= y).astype(float)
    return np.trapezoid(w * (F - ind) ** 2, z)


def test_tw_identity_gaussian():
    """Chained fair-m estimator == numerically integrated twCRPS (statistical)."""
    rng = np.random.default_rng(11)
    mu, sigma, m, reps = 0.4, 1.3, 8, 6000
    for y in (-2.5, 0.0, 1.5, 3.0):
        X = rng.normal(mu, sigma, (m, reps, 1, 1, 1)).astype(np.float64)
        tgt = np.full((reps, 1, 1, 1), y, np.float64)
        per = np.asarray(energy_score_fair(
            extract_patches(chain_tail(jnp.asarray(X)), patch=1, stride=1),
            extract_patches(chain_tail(jnp.asarray(tgt)), patch=1, stride=1))).ravel()
        se = per.std(ddof=1) / np.sqrt(reps)
        want = _tw_crps_quadrature(y, mu, sigma)
        assert abs(per.mean() - want) < 5 * se + 2e-3, (y, per.mean(), want, se)


def test_beta05_propriety_direction():
    rng = np.random.default_rng(7)
    m, reps = 8, 20000
    y = rng.normal(0, 1, (reps, 1, 1, 1)).astype(np.float32)
    tgt = extract_patches(jnp.asarray(y), patch=1, stride=1)

    def es_per_rep(mu, sigma):
        X = rng.normal(mu, sigma, (m, reps, 1, 1, 1)).astype(np.float32)
        return np.asarray(energy_score_fair(
            extract_patches(jnp.asarray(X), patch=1, stride=1), tgt,
            beta=0.5)).ravel()

    per_true = es_per_rep(0.0, 1.0)
    for mu, sigma in ((0.5, 1.0), (0.0, 0.6), (0.0, 1.6)):
        d = es_per_rep(mu, sigma) - per_true
        se = d.std(ddof=1) / np.sqrt(reps)
        assert d.mean() > 5 * se, ((mu, sigma), d.mean(), se)


def test_defaults_preserve_frozen_objective():
    """beta=1/chain=None must be bit-identical to the pre-bake-off code path."""
    key = jax.random.PRNGKey(0)
    samples = jax.random.normal(key, (4, 2, 16, 16, 3))
    target = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 16, 3))
    a = patched_energy_score(samples, target)
    b = jnp.mean(energy_score_fair(extract_patches(samples, 8, 4),
                                   extract_patches(target, 8, 4)))
    assert jnp.array_equal(a, b)


def test_t_base_draws():
    x = np.asarray(t_base(jax.random.PRNGKey(5), (2_000_000,), nu=5.0)).astype(np.float64)
    assert abs(x.std() - 1.0) < 0.02, x.std()          # unit variance
    r = (x - x.mean()) / x.std()
    assert np.mean(r ** 4) - 3 > 3.0, "base must be heavy-tailed (t5 excess ~6)"
    assert abs(np.mean(r ** 3)) < 0.15, "base must be symmetric"


def test_tcfm_step_and_sampler():
    detail, coarse = make_data(jax.random.PRNGKey(3), "t5", flat_sigma=True, n=8)
    model = ConditionalUNet(out_channels=3, channels=(8, 16), bottleneck=32,
                            embed_dim=32, cond_dim=0)
    state = make_train_state(model, jax.random.PRNGKey(0), (4,) + detail.shape[1:],
                             (4,) + coarse.shape[1:], 0, 1e-3,
                             total_steps=4, warmup=1)
    step = make_tcfm_step(None, nu=5.0)
    for _ in range(3):
        state, loss = step(state, detail[:4], coarse[:4])
    assert np.isfinite(float(loss))
    g1 = sample_tbase(state.apply_fn, state.params, jax.random.PRNGKey(2),
                      coarse[:2], 3, n_steps=8)
    g2 = sample_tbase(state.apply_fn, state.params, jax.random.PRNGKey(2),
                      coarse[:2], 3, n_steps=8)
    g3 = sample_tbase(state.apply_fn, state.params, jax.random.PRNGKey(4),
                      coarse[:2], 3, n_steps=8)
    assert jnp.array_equal(g1, g2) and not jnp.array_equal(g1, g3)
    assert g1.shape == (2, 16, 16, 3) and bool(jnp.all(jnp.isfinite(g1)))


def test_binned_sigma_maxrel():
    """Dispersion estimator (R15 probe 2): near-zero on an independent same-law
    realization; detects a 2x under-dispersion at the right magnitude."""
    from arms_p2.toys import binned_sigma_maxrel, make_data, true_mean
    d1, c1 = make_data(jax.random.PRNGKey(0), "gauss", flat_sigma=False, n=256)
    d2, c2 = make_data(jax.random.PRNGKey(1), "gauss", flat_sigma=False, n=256)
    same = binned_sigma_maxrel(d1, c1, d2, c2)
    assert same < 0.05, same
    under = true_mean(c1) + 0.5 * (d1 - true_mean(c1))
    det = binned_sigma_maxrel(under, c1, d2, c2)
    assert 0.4 < det < 0.6, det
