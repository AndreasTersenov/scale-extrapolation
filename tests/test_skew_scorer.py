"""Tests-first validation of the T3 skewness scorer (synthetics only)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_p2"))

from parity_localization import idwt_levels  # noqa: E402
from skew_scorer import map_skew, stack_detail_skew, stack_map_skew  # noqa: E402


def _fields_with_skewed_details(rng, n, gamma, size=128, levels=2):
    """Planted-detail fields; gamma>0 -> right-skewed detail coefficients
    (shifted-exponential construction with analytic-skew ~2 scaled)."""
    out = []
    for _ in range(n):
        cs = size >> levels
        coeffs = [rng.standard_normal((cs, cs))]
        for lv in range(levels, 0, -1):
            ps = size >> lv
            tri = []
            for _c in range(3):
                if gamma == 0:
                    tri.append(rng.standard_normal((ps, ps)))
                else:
                    e = rng.exponential(1.0, (ps, ps)) - 1.0
                    tri.append(np.sign(gamma) * e)
            coeffs.append(tuple(tri))
        out.append(idwt_levels(coeffs))
    return np.array(out)


def test_recovers_planted_detail_skew():
    rng = np.random.default_rng(20260805)
    sym = _fields_with_skewed_details(rng, 24, 0.0)
    pos = _fields_with_skewed_details(rng, 24, +1.0)
    neg = _fields_with_skewed_details(rng, 24, -1.0)
    m0, s0 = stack_detail_skew(sym, 1, n_boot=400, seed=1)
    mp, sp = stack_detail_skew(pos, 1, n_boot=400, seed=2)
    mn, sn = stack_detail_skew(neg, 1, n_boot=400, seed=3)
    assert abs(m0) < 4 * s0, f"symmetric details read skew {m0:.3f}"
    assert abs(mp - 2.0) < 0.15, f"exponential skew {mp:.3f} != 2"
    assert abs(mn + 2.0) < 0.15
    assert (mp - m0) / np.hypot(sp, s0) > 10


def test_mirror_flips_sign():
    rng = np.random.default_rng(7)
    pos = _fields_with_skewed_details(rng, 6, +1.0)
    for f in pos:
        a = map_skew(f)
        assert abs(map_skew(-f) + a) < 1e-12


def test_map_skew_matches_scipy():
    from scipy.stats import skew as sp_skew
    rng = np.random.default_rng(11)
    f = rng.exponential(1.0, (128, 128))
    assert abs(map_skew(f) - sp_skew(f.ravel())) < 1e-9


def test_bootstrap_se_scaling():
    rng = np.random.default_rng(13)
    fields = _fields_with_skewed_details(rng, 48, +1.0)
    _, se24 = stack_map_skew(fields[:24], n_boot=2000, seed=5)
    _, se48 = stack_map_skew(fields, n_boot=2000, seed=6)
    assert 1.15 < se24 / se48 < 1.75
