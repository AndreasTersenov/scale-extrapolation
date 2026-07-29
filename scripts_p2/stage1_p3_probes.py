"""N1 probes 1-4 (env.sh; prereg log/2026-07-29-prereg-n1-mechanism.md,
commit 5e955b4) + the mechanical GATE-N1 verdict. Committed artifacts only,
zero training, frozen instruments (peaks/nn/transplant) + the tests-first
coloring index.

Substrate: TRAINED leg gen = f2_test_gen.npz F2_gowA_e2e vs
arms_c1t_gowerstreet.npz real (paired by coarse conditioning); EDGE
continuity leg = stage0_p3_gen.npz gen_A vs its real.

GATE-N1 (verbatim from NIGHT-ORDERS-2): CONFIRMED iff at >=2 octaves of
{1,2,3}: C_real - C_gen >= 3*hypot(SE_real, SE_gen) with C_real > C_gen,
evaluated on the TRAINED leg. An octave whose margin lies within 0.5*hypot
of the 3-sigma line is AT-BAR and does NOT count (ambiguity = negative).

Writes results_p2/stage1_p3_probes.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts_p2"))

from audit_peak_ci import bootstrap_excess
from coloring_index import stack_coloring
from parity_localization import hybrid_with_gen_octave
from placement_instruments import nn_profile, stack_profiles, tstat

K_REAL = 174
NN_EDGES = np.array(json.load(open(os.path.join(
    RES, "placement_phase_a.json")))["convention"]["nn_edges"])

GEN = [np.asarray(f, np.float64) for f in
       np.load(os.path.join(RES, "f2_test_gen.npz"))["F2_gowA_e2e"]]
REAL = [np.asarray(f, np.float64) for f in
        np.load(os.path.join(RES, "arms_c1t_gowerstreet.npz"),
                allow_pickle=True)["real"]]
E = np.load(os.path.join(RES, "stage0_p3_gen.npz"), allow_pickle=True)
EDGE_GEN = [np.asarray(f, np.float64) for f in E["gen_A"]]
EDGE_REAL = [np.asarray(f, np.float64) for f in E["real"]]

out = {"substrate": "trained: F2_gowA_e2e vs c1t gowerstreet real; "
                    "edge: stage0_p3_gen gen_A vs its real"}

# ---- probe 1: coloring index + GATE-N1 --------------------------------------
col = {}
for leg, gen, real in (("trained", GEN, REAL), ("edge", EDGE_GEN, EDGE_REAL)):
    col[leg] = {}
    for j in (1, 2, 3):
        cr, sr = stack_coloring(real, j, seed=10 + j)
        cg, sg = stack_coloring(gen, j, seed=20 + j)
        hyp = float(np.hypot(sr, sg))
        margin = (cr - cg) - 3 * hyp
        col[leg][str(j)] = {
            "C_real": cr, "SE_real": sr, "C_gen": cg, "SE_gen": sg,
            "diff": cr - cg, "z": (cr - cg) / hyp,
            "margin_over_3sigma": margin,
            "at_bar": bool(abs(margin) <= 0.5 * hyp),
            "clear_confirm": bool(cr > cg and margin > 0.5 * hyp)}
out["coloring"] = col
clear = [j for j in ("1", "2", "3") if col["trained"][j]["clear_confirm"]]
at_bar = [j for j in ("1", "2", "3") if col["trained"][j]["at_bar"]]
gate = "CONFIRMED" if len(clear) >= 2 else "NOT-CONFIRMED"
out["gate_n1"] = {"clear_octaves": clear, "at_bar_octaves": at_bar,
                  "verdict": gate,
                  "rule": "CONFIRMED iff >=2 clear octaves; at-bar "
                          "(|margin|<=0.5*hypot) never counts"}

# ---- probe 2: morphology -----------------------------------------------------
morph = {"height": {}, "smoothing": {}}
for i, nu in enumerate((2.5, 3.0, 3.5, 4.0)):
    morph["height"][str(nu)] = bootstrap_excess(GEN, REAL, nu, seed=500 + i)
for i, sig in enumerate((0.5, 1.0, 2.0)):
    gs = [gaussian_filter(f, sig) for f in GEN]
    rs = [gaussian_filter(f, sig) for f in REAL]
    morph["smoothing"][str(sig)] = {
        str(nu): bootstrap_excess(gs, rs, nu, seed=600 + 10 * i + k)
        for k, nu in enumerate((2.5, 3.0))}
out["morphology"] = morph

# ---- probes 3+4: octave transplant + nn co-location --------------------------
trans = {}
for j in (1, 2, 3, 4):
    hyb = [hybrid_with_gen_octave(r, g, j) for r, g in zip(REAL, GEN)]
    ex = {str(nu): bootstrap_excess(hyb, REAL, nu, seed=700 + 10 * j + k)
          for k, nu in enumerate((2.5, 3.0))}
    a = [nn_profile(h, k=K_REAL, edges=NN_EDGES) for h in hyb]
    b = [nn_profile(r, k=K_REAL, edges=NN_EDGES) for r in REAL]
    T, _ = tstat(*stack_profiles(a), *stack_profiles(b))
    trans[str(j)] = {"peak_excess": ex, "nn_T": T}
out["transplant"] = trans
# full-gen nn_T re-read (instrument identity check vs committed 4.045)
a = [nn_profile(g, k=K_REAL, edges=NN_EDGES) for g in GEN]
b = [nn_profile(r, k=K_REAL, edges=NN_EDGES) for r in REAL]
T_full, _ = tstat(*stack_profiles(a), *stack_profiles(b))
out["nn_T_full_gen"] = T_full

with open(os.path.join(RES, "stage1_p3_probes.json"), "w") as f:
    json.dump(out, f, indent=1)

for leg in ("trained", "edge"):
    for j in ("1", "2", "3"):
        c = col[leg][j]
        print(f"coloring {leg} oct{j}: C_real={c['C_real']:.4f}"
              f"±{c['SE_real']:.4f} C_gen={c['C_gen']:.4f}±{c['SE_gen']:.4f} "
              f"z={c['z']:+.2f} clear={c['clear_confirm']} atbar={c['at_bar']}")
print(f"GATE-N1: {gate} (clear octaves {clear}, at-bar {at_bar})")
for nu, r in morph["height"].items():
    print(f"height nu={nu}: {r['excess']:+.1%}±{r['se']:.1%}")
for sig, d in morph["smoothing"].items():
    print(f"smooth {sig}px: " + "  ".join(
        f"nu={nu}: {d[nu]['excess']:+.1%}±{d[nu]['se']:.1%}" for nu in d))
for j, t in trans.items():
    print(f"transplant oct{j}: " + "  ".join(
        f"nu={nu}: {t['peak_excess'][nu]['excess']:+.1%}"
        f"±{t['peak_excess'][nu]['se']:.1%}" for nu in t["peak_excess"])
        + f"  nn_T={t['nn_T']:.2f}")
print(f"nn_T full gen (committed 4.045): {T_full:.3f}")
print("wrote stage1_p3_probes.json")
