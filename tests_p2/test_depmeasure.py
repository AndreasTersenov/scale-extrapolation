"""B1 estimator validation gates (tests-first, per NIGHT-ORDERS Stage B1):
both predictability estimators must recover the EXACT analytic conditional variance
on a GRF before any real-field measurement.

Gates:
  1. Haar synthesis roundtrip (atoms well-defined) — machine tolerance.
  2. Atom covariance identity: <atom_a, Sigma atom_b> reproduces the empirical
     covariance of the corresponding coefficients (statistical).
  3. Ridge V(r) on a GRF == analytic linear-class truth within 10% at every r.
  4. k-NN V(r) on a GRF == analytic annulus-class truth within 10% at every r.
  5. White-noise null: V(r) == Var(w) for all r (no predictability to find).
  6. Mask machinery: exact area matching; D4 rotation exactness; orientation classes
     recover the orientation of a synthetic oriented field.
"""
import numpy as np
import pytest

from sandbox.haar import haar_atom, haar_level, haar_synth_level
from sandbox.lognormal import GRFSpec, sample_grf, sigma_apply
from depmeasure.predictability import (
    analytic_grf_vr,
    analytic_grf_vr_annulus,
    disk_offsets,
    predictability_curve,
)
from depmeasure.masks import (
    elongated_offsets, orientation_class, rotate_offsets_90,
)

SPEC = GRFSpec(shape=(64, 64), alpha=2.0)
J = 2
R_TEST = (0, 1, 2, 4)


def test_haar_synth_roundtrip():
    rng = np.random.default_rng(0)
    f = rng.standard_normal((32, 32))
    cA, bands = haar_level(f)
    assert np.allclose(haar_synth_level(cA, bands), f, atol=1e-12)


def test_atom_covariance_identity():
    rng = np.random.default_rng(1)
    n = 3000
    Hj = 64 // 2 ** J
    p0 = (Hj // 2, Hj // 2)
    wa = haar_atom(SPEC.shape, J, "w", 0, p0)
    ca = haar_atom(SPEC.shape, J, "c", 0, (p0[0], p0[1] + 1))
    analytic = float(np.sum(wa * sigma_apply(ca, SPEC)))
    vals_w, vals_c = [], []
    for _ in range(n):
        g = sample_grf(SPEC, rng)
        vals_w.append(float(np.sum(wa * g)))
        vals_c.append(float(np.sum(ca * g)))
    emp = np.cov(vals_w, vals_c)[0, 1]
    se = np.std(np.array(vals_w) * np.array(vals_c)) / np.sqrt(n)
    assert abs(emp - analytic) < 5 * se + 1e-12


def test_ridge_recovers_analytic_grf():
    rng = np.random.default_rng(2)
    fields = [sample_grf(SPEC, rng) for _ in range(96)]
    truth = analytic_grf_vr(SPEC, J, R_TEST)
    curve = predictability_curve(fields, J, R_TEST, periodic=True, seed=0,
                                 max_pos_per_field=512, estimators=("ridge",))
    for r in R_TEST:
        rel = abs(curve[r]["ridge"] - truth[r]) / truth[r]
        assert rel < 0.10, f"r={r}: ridge {curve[r]['ridge']:.4f} vs truth {truth[r]:.4f} ({rel:.1%})"


def test_knn_recovers_analytic_grf():
    rng = np.random.default_rng(3)
    fields = [sample_grf(SPEC, rng) for _ in range(96)]
    truth = analytic_grf_vr_annulus(SPEC, J, R_TEST)
    curve = predictability_curve(fields, J, R_TEST, periodic=True, seed=0,
                                 max_pos_per_field=512, estimators=("knn",))
    for r in R_TEST:
        rel = abs(curve[r]["knn"] - truth[r]) / truth[r]
        assert rel < 0.10, f"r={r}: knn {curve[r]['knn']:.4f} vs truth {truth[r]:.4f} ({rel:.1%})"


def test_white_noise_null():
    rng = np.random.default_rng(4)
    fields = [rng.standard_normal((64, 64)) for _ in range(64)]
    curve = predictability_curve(fields, 1, (0, 2, 4), periodic=True, seed=0,
                                 max_pos_per_field=512)
    var_w = 1.0  # orthonormal Haar of unit white noise: details unit variance
    for r in (0, 2, 4):
        for est in ("ridge", "knn"):
            assert abs(curve[r][est] - var_w) < 0.05, (r, est, curve[r][est])


def test_mask_area_matching_and_rotation():
    disk = disk_offsets(4)
    ell_m = elongated_offsets(len(disk), aspect=4.0, angle_deg=0.0)
    assert len(ell_m) == len(disk)
    rot = rotate_offsets_90(ell_m)
    assert len(set(rot)) == len(ell_m)
    d = np.asarray(ell_m, dtype=float)
    r = np.asarray(rot, dtype=float)
    # exact rotation: the multiset of radii is preserved
    assert np.allclose(np.sort(np.hypot(d[:, 0], d[:, 1])),
                       np.sort(np.hypot(r[:, 0], r[:, 1])), atol=1e-12)


def test_orientation_class_on_synthetic_stripes():
    ys, xs = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    for angle, expect_cls in ((0, 0), (90, 2), (45, 1), (135, 3)):
        # stripes ALONG direction `angle` (array coords: x=col, y=row): intensity
        # varies along the perpendicular direction angle+90
        phi = np.deg2rad(angle + 90.0)
        phase = np.cos(phi) * xs + np.sin(phi) * ys
        stripes = np.sin(2 * np.pi * phase / 8.0)
        cls, coh = orientation_class(stripes)
        core = coh > 0.5
        assert core.mean() > 0.5
        frac = (cls[core] == expect_cls).mean()
        assert frac > 0.8, (angle, expect_cls, frac)
