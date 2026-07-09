"""Validation gate 4 -- symmetry (isotropy invariance).

The drift metric must be invariant under flips and 90-degree rotations of the maps,
within estimator noise: the fields are statistically isotropic and parity-symmetric, so
any orientation-specific dependence would be a pipeline artifact. Pooling the three
orientation sub-bands plus the exactly-symmetric Haar wavelet makes the transform
commute with the dihedral group; we verify the end-to-end metric respects it.

Tested on a lognormal (non-Gaussian) field so the metric carries real signal that could,
in principle, be broken by a non-isotropic bug. Tolerance = 3 * bootstrap SE.
"""
import numpy as np

from scaledrift import collect_wc, drift_estimate
from conftest import make_lognormal

N_BINS = 6
N_BOOT = 200
K_SIGMA = 3.0


def _excess(maps):
    d = drift_estimate(collect_wc(maps, [2, 3]), 2, 3, n_bins=N_BINS, n_boot=N_BOOT, seed=0)
    return d["excess"], d["excess_se"]


def test_drift_invariant_under_flips_and_rotations():
    base = make_lognormal(30, seed=3)
    e0, se0 = _excess(base)
    assert e0 > 0 and se0 > 0

    transforms = {
        "flip_x": [np.flip(m, axis=1) for m in base],
        "flip_y": [np.flip(m, axis=0) for m in base],
        "rot90": [np.rot90(m, 1) for m in base],
        "rot180": [np.rot90(m, 2) for m in base],
        "rot270": [np.rot90(m, 3) for m in base],
    }
    tol = K_SIGMA * se0
    for name, maps in transforms.items():
        e, _ = _excess(maps)
        assert abs(e - e0) < tol, (
            f"{name}: excess {e:.4f} differs from {e0:.4f} by > {K_SIGMA}sigma ({tol:.4f})")
