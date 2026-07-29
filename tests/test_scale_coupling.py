"""Tests-first validation of the cross-scale coupling estimator (N1
diagnostic D2; runs before any real data). Synthetics built by inverse DWT
(the generator's synthesis direction), matching the coloring-index lesson."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_p2"))

from parity_localization import idwt_levels  # noqa: E402
from scale_coupling import stack_coupling  # noqa: E402


def _fields(rng, n, coupled, size=128, levels=3):
    """Details independent (coupled=False) or amplitude-modulated by the
    upsampled coarser-octave energy (coupled=True)."""
    out = []
    for _ in range(n):
        cs = size >> levels
        coeffs = [rng.standard_normal((cs, cs))]
        prev_env = None
        for lv in range(levels, 0, -1):
            ps = size >> lv
            tri = [rng.standard_normal((ps, ps)) for _ in range(3)]
            if coupled and prev_env is not None:
                mod = np.repeat(np.repeat(prev_env, 2, axis=0), 2, axis=1)
                mod = 0.3 + mod / mod.mean()
                tri = [t * mod for t in tri]
            prev_env = sum(t * t for t in tri)
            coeffs.append(tuple(tri))
        out.append(idwt_levels(coeffs))
    return np.array(out)


def test_discriminates_coupled_vs_independent():
    rng = np.random.default_rng(20260729)
    indep = _fields(rng, 32, coupled=False)
    coup = _fields(rng, 32, coupled=True)
    for j in (1, 2):
        ki, si = stack_coupling(indep, j, n_boot=500, seed=1)
        kc, sc = stack_coupling(coup, j, n_boot=500, seed=2)
        z = (kc - ki) / np.hypot(si, sc)
        assert abs(ki) < 0.05, f"j={j}: independent K {ki:.3f} not ~0"
        assert z > 10, f"j={j}: discrimination z {z:.1f} <= 10"


def test_d4_invariance():
    rng = np.random.default_rng(3)
    fields = _fields(rng, 4, coupled=True)
    for j in (1, 2):
        k0, _ = stack_coupling(fields, j, n_boot=10, seed=0)
        for g in (lambda f: np.rot90(f, 1), lambda f: f[::-1],
                  lambda f: np.rot90(f, 3)[:, ::-1]):
            kg, _ = stack_coupling([g(f) for f in fields], j,
                                   n_boot=10, seed=0)
            assert abs(kg - k0) < 1e-9, f"j={j}: not D4-invariant"
