"""Validation gate 3 -- estimator consistency (sqrt(N) error scaling).

The bootstrap error bar must behave like a real sampling error: doubling the number of
maps (the independent unit that the bootstrap resamples) shrinks it by ~sqrt(2). A gate
against a mis-scaled or degenerate bootstrap.

The bootstrap SE of a single realization is itself noisy and varies with the drawn
field, so we average measured-SE over several independent GRF sets at each N and over
two octave pairs before forming the ratio. Target sqrt(2)=1.414; accept [1.2, 1.7]
("~sqrt(2)").
"""
import numpy as np

from scaledrift import collect_wc, drift_estimate
from conftest import make_grf

N = 12
N_BOOT = 90
PAIRS = [(2, 3), (3, 4)]
SEEDS = [21, 22, 23]


def _mean_se(n_maps, seed_offset):
    """Mean measured-SE over SEEDS independent GRF sets, per octave pair."""
    acc = {p: [] for p in PAIRS}
    for s in SEEDS:
        data = collect_wc(make_grf(n_maps, seed=s + seed_offset), [2, 3, 4])
        for a, b in PAIRS:
            acc[(a, b)].append(
                drift_estimate(data, a, b, n_bins=6, n_boot=N_BOOT, seed=0)["measured_se"])
    return {p: float(np.mean(v)) for p, v in acc.items()}


def test_bootstrap_error_scales_as_sqrt_n():
    se_N = _mean_se(N, seed_offset=0)
    se_2N = _mean_se(2 * N, seed_offset=1000)   # disjoint seeds => independent fields
    ratios = [se_N[p] / se_2N[p] for p in PAIRS]
    mean_ratio = float(np.mean(ratios))
    assert 1.2 < mean_ratio < 1.7, (
        f"SE ratio {mean_ratio:.3f} not ~sqrt(2); per-pair {dict(zip(PAIRS, ratios))}")
