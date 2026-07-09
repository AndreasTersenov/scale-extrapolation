"""Correctness guards for the measurement pieces that produce RESULTS numbers
(conditional moments, cross-octave couplings, and the P9b running-coupling PCA).
Not one of the 4 formal gates, but RESULTS quotes these, so they must be covered.
"""
import numpy as np

from scaledrift import (conditional_moments, coupling_scalars, cross_octave_coupling,
                        octave_conditional_moments, octave_wc, running_coupling_pca)
from conftest import make_grf, make_lognormal


def test_conditional_variance_modulation_is_the_signal():
    """Physical non-Gaussianity of a lognormal shows up as coarse-dependent detail
    variance: |detail| is larger where the coarse field is larger."""
    maps = make_lognormal(20, seed=4)
    m = octave_conditional_moments(maps, j=2, n_bins=10)
    var = m["var"]
    valid = ~np.isnan(var)
    # variance in the top coarse bin exceeds the bottom bin by a clear margin
    assert var[valid][-1] > 1.5 * var[valid][0]
    # signed conditional mean stays near zero (details are ~symmetric given coarse)
    assert np.nanmax(np.abs(m["mean"])) < 0.5


def test_conditional_moments_shapes_and_gaussian_limit():
    """A GRF (Gaussian) has near-constant conditional variance and ~zero skew."""
    rng = np.random.default_rng(0)
    w = rng.standard_normal(60000)
    c = rng.standard_normal(60000)
    m = conditional_moments(w, c, n_bins=8)
    for key in ("c_center", "count", "mean", "var", "skew"):
        assert m[key].shape == (8,)
    assert np.nanstd(m["var"]) < 0.15          # ~flat for independent Gaussians
    assert np.nanmax(np.abs(m["skew"])) < 0.2


def test_cross_octave_coupling_positive_for_real_field():
    maps = make_lognormal(12, seed=6)
    r = cross_octave_coupling(maps, j=2, n_boot=60, seed=0)
    assert -1.0 <= r["rho"] <= 1.0
    assert r["rho"] > 0.0                       # structure correlated across octaves
    assert r["se"] > 0.0


def test_coupling_scalars_separate_gaussian_from_nongaussian():
    """The running-coupling scalars vanish for a GRF and turn on for a lognormal."""
    def pooled(maps, j):
        ws, cs = zip(*(octave_wc(m, j) for m in maps))
        return np.concatenate(ws), np.concatenate(cs)

    g = coupling_scalars(*pooled(make_grf(20, seed=8), j=2), n_bins=10)
    n = coupling_scalars(*pooled(make_lognormal(20, seed=8), j=2), n_bins=10)
    # GRF: ~flat conditional variance, ~Gaussian kurtosis
    assert abs(g["var_slope"]) < 0.1 and abs(g["kurtosis"]) < 0.3
    # lognormal: positive variance modulation, heavier tails, clear separation
    assert n["var_slope"] > g["var_slope"] + 0.1
    assert n["kurtosis"] > 0.3
    assert n["var_hi_lo"] > 1.3


def test_running_coupling_pca_invariants_and_low_dim():
    maps = make_lognormal(20, seed=7)
    pca = running_coupling_pca(maps, [1, 2, 3, 4], n_bins=8)
    vr = pca["explained_var_ratio"]
    assert np.all(vr >= -1e-9)
    assert abs(pca["cumulative"][-1] - 1.0) < 1e-6
    assert np.all(np.diff(pca["cumulative"]) >= -1e-9)   # monotone non-decreasing
    assert 1 <= pca["eff_dim_80"] <= 3                   # drift is low-dimensional
