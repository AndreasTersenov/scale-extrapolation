"""FROZEN 2026-08-05 (JUDGE-2, NIGHT-ORDERS-3 N1f). Do not modify.
QUARANTINED: not to be applied to any real or generated field stack until
the next blind-shot prereg; validated on synthetic GRFs only. Its
adjudication bar is NOT set here — it will be computed from a real
split-half null inside the next blind-shot prereg (A4 pattern).

Held-out audit tier replacing the Minkowski judge (reclassified as a
development metric; prereg log/2026-08-05-prereg-night3.md §N1f). Betti
curves of superlevel sets {f_std > nu}, per-field standardization, nu on
NUS (13 thresholds):
  b0(nu): connected components of the mask, 4-connectivity
          (scipy.ndimage.label, cross structure).
  b1(nu): holes — connected components of the COMPLEMENT under
          8-connectivity, minus those touching the image border
          (digital-topology (4,8) duality). Deliberately NOT the
          Minkowski judge's 4/8-average Euler convention, so the two
          tiers are not the same statistic.
Stack scoring mirrors minkowski_judge verbatim: per-field vectors ->
stack mean +- bootstrap SE per (curve, nu) entry; z per entry between
gen and real stacks; T = max |z| over all 26 entries.
Zero-variance guard: entries where the combined bootstrap SE is 0 give
z = 0 when the stack means agree exactly (e.g. b1 = 0 everywhere at
high nu) and z = +-Z_SENTINEL (large FINITE sentinel, sign of the mean
difference) when they differ — never NaN.
Conventions: native = fields as given; declared = declared(fields) =
0.5-px Gaussian smoothing (scipy gaussian_filter, sigma=0.5, default
mode), byte-identical to the stage3_a_score.py declared convention.
Future callers apply the declared convention through declared() only.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter

NUS = tuple(np.arange(-3.0, 3.01, 0.5))  # 13 thresholds, exact halves
STRUCT4 = ndimage.generate_binary_structure(2, 1)  # cross: components
STRUCT8 = ndimage.generate_binary_structure(2, 2)  # full: complement holes
Z_SENTINEL = 1e6  # zero-SE unequal-means entries; large, finite, never NaN


def _standardize(field):
    f = np.asarray(field, np.float64)
    return (f - f.mean()) / f.std()


def betti_curves(field, nus=NUS):
    """(2, 13) array — b0 row, b1 row — over the nu grid, one field."""
    f = _standardize(field)
    b0, b1 = [], []
    for nu in nus:
        m = f > nu
        _, ncomp = ndimage.label(m, structure=STRUCT4)
        lab, nhole = ndimage.label(~m, structure=STRUCT8)
        border = np.unique(np.concatenate(
            [lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))
        b0.append(ncomp)
        b1.append(nhole - np.count_nonzero(border))
    return np.array([b0, b1], np.float64)


def stack_betti(fields, nus=NUS, n_boot=2000, seed=0):
    """Stack mean and bootstrap-SE vectors (flattened: b0 entries, b1)."""
    vecs = np.array([betti_curves(f, nus).ravel() for f in fields])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vecs), (n_boot, len(vecs)))
    boot = vecs[idx].mean(axis=1)
    return vecs.mean(axis=0), boot.std(axis=0, ddof=1)


def judge2_T(gen_fields, real_fields, nus=NUS, n_boot=2000, seed=0):
    """The frozen comparison: guarded z per entry and T = max |z|."""
    mg, sg = stack_betti(gen_fields, nus, n_boot, seed)
    mr, sr = stack_betti(real_fields, nus, n_boot, seed + 1)
    denom = np.hypot(sg, sr)
    diff = mg - mr
    z = np.zeros_like(diff)
    ok = denom > 0
    z[ok] = diff[ok] / denom[ok]
    frozen = (~ok) & (diff != 0)
    z[frozen] = np.sign(diff[frozen]) * Z_SENTINEL
    detail = {"nus": np.asarray(nus), "gen_mean": mg, "gen_se": sg,
              "real_mean": mr, "real_se": sr, "z": z}
    return float(np.max(np.abs(z))), detail


def declared(fields):
    """Declared-resolution convention: 0.5-px Gaussian smoothing per
    field (identical to stage3_a_score.py). Native = fields as given."""
    return [gaussian_filter(np.asarray(f, np.float64), 0.5)
            for f in fields]
