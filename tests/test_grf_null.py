"""Validation gate 2 -- the GRF null (executable).

For a power-law GRF the wavelet statistics are self-similar: after per-octave
normalization the conditional PDFs are scale-invariant, so the measured *excess* drift
(cross-octave W1 minus the within-octave finite-sample floor) is consistent with zero.
If the pipeline reports significant drift on a GRF, the pipeline is buggy (K-M1a).

The gate is only meaningful if the estimator can also SEE drift when it exists, so we
pair it with a positive control: a lognormal field must show highly significant drift
at the same setting. Null = |z| < 3 (the same 3-sigma bar P9a uses for a real detection).
"""
import numpy as np

from scaledrift import collect_wc, drift_estimate
from conftest import make_grf, make_lognormal

N_BINS = 6
N_BOOT = 120
NULL_Z = 3.0          # GRF drift must be BELOW the 3-sigma detection threshold


def test_grf_null_no_significant_drift():
    maps = make_grf(24, seed=1)
    data = collect_wc(maps, [2, 3, 4])
    zs = {}
    for a, b in [(2, 3), (3, 4)]:
        d = drift_estimate(data, a, b, n_bins=N_BINS, n_boot=N_BOOT, seed=0)
        zs[(a, b)] = d["z"]
        # sanity: the raw distance and its floor are positive and finite
        assert d["measured"] > 0 and d["floor"] > 0
        assert np.isfinite(d["excess_se"]) and d["excess_se"] > 0
    assert all(abs(z) < NULL_Z for z in zs.values()), \
        f"GRF shows significant drift (null FAILS): z = {zs}"


def test_estimator_has_power_on_lognormal():
    """Positive control: a non-Gaussian field must yield significant drift, so the
    null above is not passing merely because the estimator is blind."""
    maps = make_lognormal(24, seed=2)
    data = collect_wc(maps, [2, 3])
    d = drift_estimate(data, 2, 3, n_bins=N_BINS, n_boot=N_BOOT, seed=0)
    assert d["z"] > 4.0, f"estimator failed to detect lognormal drift: z={d['z']:.2f}"
    assert d["excess"] > 0
