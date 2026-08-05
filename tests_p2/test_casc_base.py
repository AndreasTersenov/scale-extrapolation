"""CASC base validation (NIGHT-3 prereg §CASC; JAX gate, tests-first
BEFORE the inference-only probe touches the committed chain).

  (a) whiteness   : mean ring spectrum of many casc_seed maps flat —
                    low-k vs high-k band-mean ratio within [0.9, 1.1];
  (b) determinism : same key -> identical; different key -> different;
  (c) isotropy    : kx-marginal vs ky-marginal band powers agree
                    (isotropic-by-design; no axis bias);
  (d) multifractal: 1-px eps^2 clustering separates casc_seed(LAM) from
                    plain white (lam=0 control, same machinery) at z > 5
                    over 24-map stacks; M dynamic range stays sane;
  (e) splice smoke: casc_colored_base with an identity filter applies the
                    committed filter+quantile steps to the casc seed
                    EXACTLY as colored_t_base applies them to its white
                    draw (structural identities, not statistics).
"""
import os
import sys

import numpy as np

import jax

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts_p2"))

from casc_base import (LAM, casc_colored_base, casc_seed,  # noqa: E402
                       log_corr_field)
from colored_base import colored_t_base, ring_spectrum  # noqa: E402

ART = np.load(os.path.join(REPO, "results_p2", "l1pp_filter.npz"))
Z, X = ART["z_grid"], ART["x_grid"]
N = 64
LOW, HIGH = (1, N // 8), (N // 4 + 1, N // 2)  # coloring_index band rule


def _seed(n, key, lam=LAM):
    x = casc_seed(jax.random.PRNGKey(key), (n, N, N, 1), lam)
    return np.asarray(x)[..., 0]


def test_a_whiteness_flat_ring_spectrum():
    spec = ring_spectrum(_seed(256, key=1))
    lo = spec[LOW[0]:LOW[1] + 1].mean()
    hi = spec[HIGH[0]:HIGH[1] + 1].mean()
    assert 0.9 < lo / hi < 1.1, f"band ratio {lo / hi:.3f}"


def test_b_seeded_determinism():
    a = _seed(4, key=7)
    b = _seed(4, key=7)
    c = _seed(4, key=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_c_isotropy_axis_marginals():
    x = _seed(128, key=2)
    P = (np.abs(np.fft.fft2(x)) ** 2 / N**2).mean(axis=0)
    k = np.abs(np.fft.fftfreq(N) * N)
    for name, (b0, b1) in (("low", LOW), ("high", HIGH)):
        band = (k >= b0) & (k <= b1)
        p_ky = P.mean(axis=1)[band].mean()
        p_kx = P.mean(axis=0)[band].mean()
        r = p_ky / p_kx
        assert 0.95 < r < 1.05, f"{name}-band axis ratio {r:.3f}"


def _clustering(maps):
    """Per-map corr(eps^2, 1-px torus-shifted eps^2), axes averaged."""
    vals = []
    for m in maps:
        e2 = m * m
        e2 = e2 - e2.mean()
        v = (e2 * e2).mean()
        vals.append(0.5 * ((e2 * np.roll(e2, 1, 0)).mean()
                           + (e2 * np.roll(e2, 1, 1)).mean()) / v)
    return np.array(vals)


def test_d_multifractality_separates_from_white():
    cc = _clustering(_seed(24, key=3, lam=LAM))
    cw = _clustering(_seed(24, key=4, lam=0.0))
    se = np.hypot(cc.std(ddof=1) / np.sqrt(24), cw.std(ddof=1) / np.sqrt(24))
    z = (cc.mean() - cw.mean()) / se
    assert z > 5, f"clustering z {z:.1f}"
    assert abs(cw.mean()) < 0.02, f"white control {cw.mean():.4f}"
    # M dynamic range sane at the documented LAM
    om = np.asarray(log_corr_field(jax.random.PRNGKey(5), (8, N, N, 1)))
    M = np.exp(LAM * om)
    ratios = M.max(axis=(1, 2, 3)) / M.min(axis=(1, 2, 3))
    assert ratios.mean() < 300, f"M range {ratios.mean():.0f}"


def test_e_copula_splice_structural():
    ones = np.ones((N, N))  # identity filter (already unit-normalized)
    shape = (4, N, N, 1)
    key = jax.random.PRNGKey(11)
    # colored_t_base under a white seed == table applied to its white draw
    y_ref = np.asarray(colored_t_base(key, shape, ones, Z, X))
    g = np.asarray(jax.random.normal(key, shape))
    assert np.allclose(y_ref, np.interp(g, Z, X), rtol=1e-3, atol=1e-3)
    # casc_colored_base == the SAME steps applied to the casc seed
    y = np.asarray(casc_colored_base(ones, Z, X, LAM)(key, shape))
    eps = np.asarray(casc_seed(key, shape, LAM))
    assert np.allclose(y, np.interp(eps, Z, X), rtol=1e-3, atol=1e-3)
    # finite, t-tailed-by-table (interp clips at the frozen table range)
    assert np.isfinite(y).all()
    assert np.abs(y).max() <= float(X.max()) * 1.001
