"""L1'' deconvolved-filter fit + the P-T point prediction (env.sh; R40).

Runs BEFORE the prereg commit; everything here is derived from COMMITTED
artifacts (R12): the two-point transfer measurement
(l1p_transfer_analysis.json), the frozen L1 targets
(l1_filter_gowerstreet.npz), and committed generated maps (for the sandbox
transfer, deployment-available: generated maps + known base only).

Deconvolution: S_base = S_target / T_hat, with T_hat(k) the geometric mean
of the two committed transfer estimates (white-input and colored-input) and
per-ring spread sigma_lnT = |ln(T_white/T_col)|/2 as the T uncertainty.
Adjudicating: S_target = the oct2rescaled target (deployment-pure).
Oracle ablation: S_target = the measured oct-1 real spectrum.
Sandbox canary: S_target = sandbox oct2rescaled, T from the committed
white-base sandbox F2 maps (F2_A_e2e).

P-T (the new first-scored gate): the multiplicative model predicts the
scored C (frozen stack_coloring instrument convention) of the L1'' output.
Computed mode-exactly from the ring model + Monte-Carlo over the T
uncertainty; band = 3*sqrt(sigma_MC^2 + sigma_instr^2), sigma_instr =
0.0049 (the committed L1' pooled-96-map instrument SE).

Writes results_p2/l1pp_filter.npz + results_p2/l1pp_pt_prediction.json.
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
from l1_fit_filter import calibrate
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
    """The frozen stack_coloring band convention applied to a ring model:
    every mode carries its ring's power; C = mean(low)/mean(high)."""
    k = np.fft.fftfreq(N) * N
    kk = np.hypot(k[:, None], k[None, :])
    rings = np.rint(kk).astype(int)
    P = np.asarray(spec)[np.minimum(rings, len(spec) - 1)]
    low = (kk >= 1) & (kk <= N / 8)
    high = (kk > N / 4) & (kk <= N / 2)
    return float(P[low].mean() / P[high].mean())


rings = ring_table(N1)
counts = np.array([(rings == r).sum() for r in range(rings.max() + 1)])
pos = counts > 0

TA = json.load(open(os.path.join(RES, "l1p_transfer_analysis.json")))
s_real = np.array(TA["spectra"]["s_real"])
s_wout = np.array(TA["spectra"]["s_white_out"])
s_cin = np.array(TA["spectra"]["s_colored_in"])
s_cout = np.array(TA["spectra"]["s_colored_out"])
T_white = np.maximum(s_wout, 1e-9)
T_col = np.maximum(s_cout / np.maximum(s_cin, 1e-9), 1e-9)
T_hat = np.sqrt(T_white * T_col)
sigma_lnT = np.abs(np.log(T_white / T_col)) / 2.0

F = np.load(os.path.join(RES, "l1_filter_gowerstreet.npz"))
z_grid, x_grid = make_z_table()
out_npz = {"z_grid": z_grid, "x_grid": x_grid, "T_hat": T_hat,
           "sigma_lnT": sigma_lnT}
pt = {"convention": "instrument C on the ring model; band = 3*sqrt(MC^2 + "
                    f"{SIGMA_INSTR}^2); MC = 4000 draws of ln T per ring",
      "T_hat_range": [float(T_hat[pos][1]), float(T_hat[pos][-1])]}

rng = np.random.default_rng(20260812)
for name, tgt in (("adj", np.array(F["target_oct2rescaled"], np.float64)),
                  ("oracle", np.array(F["target_oct1"], np.float64))):
    tgt_n = tgt / tgt[pos].mean()
    base_spec = tgt_n / np.maximum(T_hat, 1e-9)
    filt, amps, hist = calibrate(base_spec, N1, z_grid, x_grid,
                                 seed=20260813 + (0 if name == "adj" else 1))
    out_npz[f"filt_{name}"] = filt
    out_npz[f"base_spec_{name}"] = base_spec
    # P-T: predicted output spectrum = T * base = tgt (model-exact) with MC
    c_point = instrument_C_of_ring_spec(np.maximum(T_hat, 1e-9) * base_spec)
    draws = []
    for _ in range(4000):
        T_i = T_hat * np.exp(rng.standard_normal(len(T_hat)) * sigma_lnT)
        draws.append(instrument_C_of_ring_spec(T_i * base_spec))
    sig_mc = float(np.std(draws))
    band = 3.0 * float(np.hypot(sig_mc, SIGMA_INSTR))
    pt[name] = {"C_pred": c_point, "sigma_mc": sig_mc,
                "band_3sigma": band,
                "interval": [c_point - band, c_point + band],
                "base_C_instrument": instrument_C_of_ring_spec(base_spec),
                "cal_history": hist}
    print(f"{name}: C_pred={c_point:.4f} ±{band:.4f} (3σ) "
          f"base_C={pt[name]['base_C_instrument']:.3f} cal={hist}")

# sandbox canary: T from committed white-base F2 sandbox maps
sb_gen = np.load(os.path.join(RES, "f2_test_gen.npz"))["F2_A_e2e"]
T_sb = np.maximum(oct1_ring_spec(sb_gen), 1e-9)
T_sb = T_sb / T_sb[pos].mean()
FS = np.load(os.path.join(RES, "l1_filter_sandbox.npz"))
tgt_sb = np.array(FS["target_oct2rescaled"], np.float64)
tgt_sb_n = tgt_sb / tgt_sb[pos].mean()
base_sb = tgt_sb_n / T_sb
filt_sb, _, hist_sb = calibrate(base_sb, N1, z_grid, x_grid, seed=20260815)
out_npz["filt_canary_sandbox"] = filt_sb
pt["canary_note"] = {"T_source": "committed F2_A_e2e (white-base) maps",
                     "cal_history": hist_sb}
print(f"canary filter: cal={hist_sb}")

np.savez(os.path.join(RES, "l1pp_filter.npz"), **out_npz)
with open(os.path.join(RES, "l1pp_pt_prediction.json"), "w") as f:
    json.dump(pt, f, indent=1)
print("wrote l1pp_filter.npz + l1pp_pt_prediction.json")
