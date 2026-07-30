"""L1 filter fit + freeze (env.sh; descriptive measurement of TRAINING data
only — no generation, no training, no scoring of any model output).

Measures the oct-1 target ring spectrum from the TRAINING tiles' Haar
detail planes (channel-pooled — one shared filter for H/V/D keeps the base
exactly D4-invariant) and calibrates the Gaussian-stage filter through the
t(5) quantile map by fixed-point iteration until the FINAL base spectrum
matches the target (mode-weighted rel dev <= 3% on rings with >= 16 modes).

TWO filter sources fitted per field (the reconvene picks one — L1 delta
memo): 'oct1' = measured directly on octave-1 planes (uses target-octave
information; strongest correction; weakens the deployment claim);
'oct2rescaled' = octave-2 ring shape mapped to octave-1's grid in k/N units
(deployment-pure; licensed by the measured near-scale-invariance of C:
real 0.786/0.774/0.785 at octaves 1/2/3).

Writes results_p2/l1_filter_<field>.npz with BOTH calibrated filters, the
target spectra, the z->x table, and the calibration residuals. Frozen on
commit.
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts_p2"))

from colored_base import (
    colored_t_base_np,
    filter_from_ring_amps,
    make_z_table,
    ring_spectrum,
    ring_table,
)
from parity_localization import dwt_levels

CAL_BATCH = 512
CAL_ITERS = 6
TOL = 0.03
MIN_MODES = 16


def detail_planes(tiles, octave):
    """Standardized-per-plane-set detail planes of a tile stack (channel-
    pooled: all three channels as separate planes, matching the shared
    filter and the global std_j convention of the training pipeline)."""
    planes = []
    for f in tiles:
        for c in dwt_levels(np.asarray(f, np.float64), octave)[1]:
            planes.append(np.asarray(c, np.float64))
    planes = np.array(planes)
    return planes / planes.std()


def target_ring_spec(tiles, octave, N_target):
    """Per-ring target spectrum at grid size N_target from planes at
    `octave` (rescaled in k/N units if the plane size differs)."""
    planes = detail_planes(tiles, octave)
    spec = ring_spectrum(planes)
    N_meas = planes.shape[-1]
    if N_meas == N_target:
        return spec
    rings_t = np.arange(ring_table(N_target).max() + 1, dtype=float)
    k_meas = np.arange(len(spec), dtype=float)
    scale = N_meas / N_target
    return np.interp(np.maximum(rings_t * scale, 1.0 * scale),
                     k_meas, spec)


def calibrate(target_spec, N, z_grid, x_grid, seed):
    """Fixed-point: adjust Gaussian-stage ring amps until the post-map base
    spectrum matches target_spec (shape) within TOL on well-sampled rings."""
    rings = ring_table(N)
    counts = np.array([(rings == r).sum() for r in range(rings.max() + 1)])
    tgt = np.asarray(target_spec, np.float64).copy()
    tgt[0] = max(tgt[0], 1e-12)
    tgt_n = tgt / tgt[counts > 0].mean()
    amps = np.sqrt(tgt_n)
    rng = np.random.default_rng(seed)
    history = []
    for it in range(CAL_ITERS):
        filt = filter_from_ring_amps(amps, N)
        x = colored_t_base_np(rng, (CAL_BATCH, N, N, 1), filt,
                              z_grid, x_grid)[..., 0]
        got = ring_spectrum(x)
        got_n = got / got[counts > 0].mean()
        sel = counts >= MIN_MODES
        dev = np.abs(got_n[sel] - tgt_n[sel]) / np.maximum(tgt_n[sel], 1e-12)
        history.append(float(dev.max()))
        if dev.max() <= TOL:
            break
        ratio = np.sqrt(np.maximum(tgt_n, 1e-12) / np.maximum(got_n, 1e-12))
        amps = amps * np.clip(ratio, 0.5, 2.0)
    return filt, amps, history


def fit_field(field, tiles, N1=64):
    z_grid, x_grid = make_z_table()
    out = {"z_grid": z_grid, "x_grid": x_grid}
    for src, octave in (("oct1", 1), ("oct2rescaled", 2)):
        tgt = target_ring_spec(tiles, octave, N1)
        filt, amps, hist = calibrate(tgt, N1, z_grid, x_grid,
                                     seed=20260730 + octave)
        out[f"filt_{src}"] = filt
        out[f"amps_{src}"] = amps
        out[f"target_{src}"] = tgt
        out[f"cal_history_{src}"] = np.array(hist)
        print(f"{field} {src}: calibration dev history "
              f"{[f'{h:.3f}' for h in hist]} -> "
              f"{'OK' if hist[-1] <= TOL else 'NOT CONVERGED'}")
    path = os.path.join(RES, f"l1_filter_{field}.npz")
    np.savez(path, **out)
    print(f"wrote {path}")
    return out


if __name__ == "__main__":
    gow = np.load(os.path.join(REPO, "data_cache",
                               "tiles_pnull.npz"))["gowerstreet"][:-64]
    fit_field("gowerstreet", gow)
    sb = np.load(os.path.join(REPO, "data_cache",
                              "tiles_sandbox.npz"))["sandbox"][:-64]
    fit_field("sandbox", sb)
