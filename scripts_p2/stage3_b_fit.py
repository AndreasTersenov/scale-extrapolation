"""Stage-3 (b) phase 2 (env.sh): measure T from the run's white stream,
deconvolve the BLIND target (octave-3 ring shape of TRAINING tiles rescaled
two octaves in k/N — training-octave information only), calibrate, and
compute the run's P-T band (multiplicative model; sigma_lnT transferred
from the committed L1'' two-point spread, per the prereg).
Usage: stage3_b_fit.py <tag> <target_octave> (3 = blind two-octave
rescale; 2 = the (a) seed legs' deployment target). Writes results_p2/stage3_<tag>_filter.npz +
stage3_<tag>_pt.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts_p2"))

from colored_base import make_z_table, ring_spectrum, ring_table
from l1_fit_filter import calibrate, target_ring_spec
from parity_localization import dwt_levels

SIGMA_INSTR = 0.0049
N1 = 64


def oct1_ring_spec(fields):
    planes = []
    for f in fields:
        for c in dwt_levels(np.asarray(f, np.float64), 1)[1]:
            planes.append(np.asarray(c, np.float64))
    planes = np.array(planes)
    return ring_spectrum(planes / planes.std())


def instrument_C_of_ring_spec(spec, N=N1):
    # verbatim from l1pp_fit_filter.py (that script executes on import)
    k = np.fft.fftfreq(N) * N
    kk = np.hypot(k[:, None], k[None, :])
    rings = np.rint(kk).astype(int)
    P = np.asarray(spec)[np.minimum(rings, len(spec) - 1)]
    low = (kk >= 1) & (kk <= N / 8)
    high = (kk > N / 4) & (kk <= N / 2)
    return float(P[low].mean() / P[high].mean())


tag, target_oct = sys.argv[1], int(sys.argv[2])
rings = ring_table(N1)
counts = np.array([(rings == r).sum() for r in range(rings.max() + 1)])
pos = counts > 0

white = np.load(os.path.join(RES, f"stage3_{tag}_white.npz"))["white"]
T_run = np.maximum(oct1_ring_spec(white), 1e-9)
T_run = T_run / T_run[pos].mean()
gtrain = np.load(os.path.join(REPO, "data_cache",
                              "tiles_pnull.npz"))["gowerstreet"][:-64]
tgt = target_ring_spec(gtrain, target_oct, N1)
tgt = tgt / tgt[pos].mean()
z_grid, x_grid = make_z_table()
base_spec = tgt / T_run
filt, _, hist = calibrate(base_spec, N1, z_grid, x_grid, seed=20260834)
np.savez(os.path.join(RES, f"stage3_{tag}_filter.npz"),
         filt=filt, T_run=T_run, target=tgt, base_spec=base_spec,
         z_grid=z_grid, x_grid=x_grid)

sigma_lnT = np.asarray(np.load(os.path.join(
    RES, "l1pp_filter.npz"))["sigma_lnT"])
c_point = instrument_C_of_ring_spec(np.maximum(T_run, 1e-9) * base_spec)
rng = np.random.default_rng(20260835)
draws = [instrument_C_of_ring_spec(
    T_run * np.exp(rng.standard_normal(len(T_run)) * sigma_lnT) * base_spec)
    for _ in range(4000)]
sig_mc = float(np.std(draws))
band = 3.0 * float(np.hypot(sig_mc, SIGMA_INSTR))
pt = {"C_pred": c_point, "sigma_mc": sig_mc, "band_3sigma": band,
      "interval": [c_point - band, c_point + band],
      "cal_history": hist,
      "note": "sigma_lnT transferred from committed L1'' spread (prereg)"}
with open(os.path.join(RES, f"stage3_{tag}_pt.json"), "w") as f:
    json.dump(pt, f, indent=1)
print(f"{tag}: C_pred={c_point:.4f} ±{band:.4f} cal={hist}")
