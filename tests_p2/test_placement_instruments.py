"""Tests-first validation of the placement instruments (Phase A of
log/2026-07-24-prereg-placement.md, approved R30).

Gates encoded here (green before ANY model scoring):
- position-purity: profiles invariant under per-field affine / monotone maps;
- count-matching: top-K peak sets have exactly K members, rank-stable;
- truth-null: independent iid ensembles give T = max|z| < 3;
- surrogate: rank-remapped phase randomization preserves marginals EXACTLY
  and is detected by the environment instrument on an env-associated field;
- parity: iid ensembles give ~uniform parity shares.
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts_p2"))

from placement_instruments import (
    env_rate_profile, nn_profile, parity_profile, pk2pt_profile, stack_profiles,
    surrogate, topk_peaks_xy, tstat,
)

RNG = np.random.default_rng(20260724)


def _iid_fields(n, size=64, rng=RNG):
    return rng.standard_normal((n, size, size))


def test_topk_count_matching_and_rank_stability():
    f = RNG.standard_normal((64, 64))
    ys, xs, h = topk_peaks_xy(f, 20)
    assert len(ys) == len(xs) == len(h) == 20
    # monotone transform preserves ranks -> identical peak set
    ys2, xs2, h2 = topk_peaks_xy(np.tanh(f) * 3.0 + 1.0, 20)
    assert np.array_equal(ys, ys2) and np.array_equal(xs, xs2)


def test_affine_invariance_of_profiles():
    f = RNG.standard_normal((64, 64))
    for fn in (lambda g: env_rate_profile(g, k=20, lvl=2),
               lambda g: parity_profile(g, k=20),
               lambda g: pk2pt_profile(g, k=20)):
        a, b = fn(f), fn(5.0 * f - 3.0)
        assert np.allclose(a, b), fn


def test_surrogate_preserves_marginals_exactly():
    f = RNG.standard_normal((64, 64))
    s = surrogate(f, np.random.default_rng(0))
    assert np.allclose(np.sort(f.ravel()), np.sort(s.ravel()))
    assert not np.allclose(f, s)


def test_truth_null_iid():
    a = [env_rate_profile(f, k=20, lvl=2) for f in _iid_fields(48)]
    b = [env_rate_profile(f, k=20, lvl=2) for f in _iid_fields(48)]
    T, _ = tstat(*stack_profiles(a), *stack_profiles(b))
    assert T < 3.0, T


def test_parity_uniform_on_iid():
    profs = [parity_profile(f, k=20) for f in _iid_fields(64)]
    m, se = stack_profiles(profs)
    assert np.all(np.abs(m - 0.25) < 4 * se + 0.02)


def _env_associated_field(rng, size=64, lvl=2):
    """Coarse-modulated noise: peaks preferentially in high-coarse regions."""
    b = 2 ** lvl
    coarse = rng.standard_normal((size // b, size // b))
    up = np.repeat(np.repeat(coarse, b, 0), b, 1)
    return rng.standard_normal((size, size)) * np.exp(0.9 * up)


def test_env_instrument_detects_association_and_surrogate_destroys_it():
    rng = np.random.default_rng(7)
    real = [_env_associated_field(rng) for _ in range(48)]
    prof_real = [env_rate_profile(f, k=20, lvl=2) for f in real]
    # association: high-env bins carry far more than 1/8 of the peaks
    m, _ = stack_profiles(prof_real)
    assert m[-1] > 0.25
    # the surrogate destroys the association while keeping marginals
    surr = [surrogate(f, np.random.default_rng(i)) for i, f in enumerate(real)]
    prof_surr = [env_rate_profile(f, k=20, lvl=2) for f in surr]
    T, _ = tstat(*stack_profiles(prof_real), *stack_profiles(prof_surr))
    assert T >= 5.0, T


def test_nn_profile_shares_sum_to_one():
    f = RNG.standard_normal((64, 64))
    edges = np.array([1.5, 2.5, 3.5, 4.5, 6.0, 8.0, 10.0, 13.0, 17.0])
    p = nn_profile(f, k=20, edges=edges)
    assert p.shape == (10,) and np.isclose(p.sum(), 1.0)
