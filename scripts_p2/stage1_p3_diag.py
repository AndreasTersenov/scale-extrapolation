"""N1 diagnostics D1-D3 (JOBS.md ledger 2026-07-29; DESCRIPTIVE, committed
artifacts only, no verdicts fed back).

D1: full annular spectrum of octave-1 detail planes, real vs gen (trained
    leg) — the SHAPE of the whiteness defect (per-annulus gen/real ratio,
    absolute band powers).
D2: cross-scale detail-energy coupling K_12/K_23 (tests-first estimator),
    trained + edge legs.
D3: sandbox contrast — C (oct 1-3) and K_12 on the committed sandbox trained
    legs (c1t gen_A and F2_A_e2e vs the sandbox test tiles), where peak
    counts show DEFICITS.

Writes results_p2/stage1_p3_diag.json.
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

from coloring_index import stack_coloring
from parity_localization import dwt_levels
from scale_coupling import stack_coupling

GEN = [np.asarray(f, np.float64) for f in
       np.load(os.path.join(RES, "f2_test_gen.npz"))["F2_gowA_e2e"]]
REAL = [np.asarray(f, np.float64) for f in
        np.load(os.path.join(RES, "arms_c1t_gowerstreet.npz"),
                allow_pickle=True)["real"]]
E = np.load(os.path.join(RES, "stage0_p3_gen.npz"), allow_pickle=True)
EDGE_GEN = [np.asarray(f, np.float64) for f in E["gen_A"]]
EDGE_REAL = [np.asarray(f, np.float64) for f in E["real"]]

out = {}

# ---- D1: annular spectrum shape at octave 1 (trained leg) --------------------
def annular_mean(fields, octave):
    """Mean channel-summed power per integer-|k| annulus over a stack."""
    planes = []
    for f in fields:
        tri = dwt_levels(f, octave)[1]
        planes.append(np.stack([np.asarray(c, np.float64) for c in tri]))
    N = planes[0].shape[-1]
    k = np.fft.fftfreq(N) * N
    kk = np.hypot(k[:, None], k[None, :])
    rings = np.rint(kk).astype(int)
    nmax = N // 2
    power = np.zeros(nmax + 1)
    counts = np.zeros(nmax + 1)
    for p in planes:
        P = np.abs(np.fft.fft2(p)) ** 2 / N**2  # (3, N, N)
        Psum = P.sum(axis=0)
        for r in range(1, nmax + 1):
            m = rings == r
            power[r] += Psum[m].sum()
            counts[r] += m.sum()
    spec = power[1:] / np.maximum(counts[1:], 1)
    return spec / len(fields)


spec_r = annular_mean(REAL, 1)
spec_g = annular_mean(GEN, 1)
out["d1_annular_oct1"] = {
    "k": list(range(1, len(spec_r) + 1)),
    "real": spec_r.tolist(), "gen": spec_g.tolist(),
    "ratio_gen_over_real": (spec_g / spec_r).tolist()}

# ---- D2: cross-scale coupling ------------------------------------------------
d2 = {}
for leg, gen, real in (("trained", GEN, REAL), ("edge", EDGE_GEN, EDGE_REAL)):
    d2[leg] = {}
    for j in (1, 2):
        kr, sr = stack_coupling(real, j, seed=30 + j)
        kg, sg = stack_coupling(gen, j, seed=40 + j)
        d2[leg][f"K_{j}{j+1}"] = {
            "real": kr, "SE_real": sr, "gen": kg, "SE_gen": sg,
            "z": (kr - kg) / float(np.hypot(sr, sg))}
out["d2_coupling"] = d2

# ---- D3: sandbox contrast ------------------------------------------------------
tiles = np.load(os.path.join(REPO, "data_cache", "tiles_sandbox.npz"))["sandbox"]
SB_REAL = [np.asarray(f, np.float64) for f in tiles[-32:]]
SB = {"c1t_A": np.load(os.path.join(RES, "arms_c1t_sandbox.npz"),
                       allow_pickle=True)["gen_A"],
      "F2_A": np.load(os.path.join(RES, "f2_test_gen.npz"))["F2_A_e2e"]}
d3 = {"real": {}}
for j in (1, 2, 3):
    c, s = stack_coloring(SB_REAL, j, seed=50 + j)
    d3["real"][f"C_oct{j}"] = {"C": c, "SE": s}
kr, sr = stack_coupling(SB_REAL, 1, seed=60)
d3["real"]["K_12"] = {"K": kr, "SE": sr}
for name, gen in SB.items():
    gen = [np.asarray(f, np.float64) for f in gen]
    d3[name] = {}
    for j in (1, 2, 3):
        c, s = stack_coloring(gen, j, seed=70 + j)
        zr = (d3["real"][f"C_oct{j}"]["C"] - c) / float(
            np.hypot(d3["real"][f"C_oct{j}"]["SE"], s))
        d3[name][f"C_oct{j}"] = {"C": c, "SE": s, "z_real_minus_gen": zr}
    kg, sg = stack_coupling(gen, 1, seed=80)
    d3[name]["K_12"] = {"K": kg, "SE": sg,
                        "z_real_minus_gen": (kr - kg) / float(np.hypot(sr, sg))}
out["d3_sandbox"] = d3

with open(os.path.join(RES, "stage1_p3_diag.json"), "w") as f:
    json.dump(out, f, indent=1)

ratio = spec_g / spec_r
print("D1 oct1 gen/real power ratio: k=1-4:",
      [f"{r:.3f}" for r in ratio[:4]],
      " k=8,16,24,32:", [f"{ratio[k-1]:.3f}" for k in (8, 16, 24, 32)])
for leg in d2:
    for kname, v in d2[leg].items():
        print(f"D2 {leg} {kname}: real {v['real']:.4f}±{v['SE_real']:.4f} "
              f"gen {v['gen']:.4f}±{v['SE_gen']:.4f} z={v['z']:+.2f}")
for name in d3:
    row = "  ".join(f"{k}: {v.get('C', v.get('K')):.4f}"
                    + (f" (z={v['z_real_minus_gen']:+.1f})"
                       if "z_real_minus_gen" in v else "")
                    for k, v in d3[name].items())
    print(f"D3 sandbox {name}: {row}")
print("wrote stage1_p3_diag.json")
