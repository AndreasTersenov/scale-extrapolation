"""Phase-3b topology-diagnosis scorer (env.sh; prereg 2026-08-05, R46).

T1: frozen MF judge on hc stacks (declared res adjudicating vs 3.5, #11
    band 0.25; native descriptive); e2e T re-emitted from the same code
    path; mechanical branch (pass < 3.25, fail > 3.75, else at-bar;
    hc-pass & e2e-fail -> CONDITIONED-PASS; both fail -> BOTH-FAIL; else
    MIXED, worse governs).
T2: transplant chi: single-octave + cumulative hybrids (blind final1 vs
    real, paired), judge T + chi(0) shift fraction vs the full-gen shift.
T3: detail skew per octave (real vs blind-e2e / blind-hc / trained-l1pp
    pooled sets) + map-level skew at declared resolution.

Writes results_p2/topo_diag_verdict.json.
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

from minkowski_judge import NUS, judge_T, minkowski_vector
from parity_localization import dwt_levels, idwt_levels
from skew_scorer import stack_detail_skew, stack_map_skew

BAR, BAND = 3.5, 0.25


def cat(T):
    return "pass" if T < BAR - BAND else ("fail" if T > BAR + BAND
                                          else "at-bar")


def sm(stack):
    return [gaussian_filter(np.asarray(f, np.float64), 0.5) for f in stack]


real = [np.asarray(f, np.float64) for f in
        np.load(os.path.join(RES, "arms_stageD.npz"),
                allow_pickle=True)["real"]]
B = np.load(os.path.join(RES, "stage3_blind_final.npz"))
e2e = [np.asarray(f, np.float64) for k in ("final1", "final2", "final3")
       for f in B[k]]
HC = np.load(os.path.join(RES, "topo_diag_hc_gen.npz"))
hc_blind = [np.asarray(f, np.float64) for k in ("blind_hc1", "blind_hc2",
                                                "blind_hc3") for f in HC[k]]
hc_sd = [np.asarray(f, np.float64) for f in HC["stageD_context_hc1"]]
rs = sm(real)
out = {}

# ---- T1 ---------------------------------------------------------------------
T_e2e, _ = judge_T(sm(e2e), rs, seed=20260845)
T_hc, _ = judge_T(sm(hc_blind), rs, seed=20260860)
T_hc_nat, _ = judge_T(hc_blind, real, seed=20260861)
T_sdctx, _ = judge_T(sm(hc_sd), rs, seed=20260862)
c_hc, c_e2e = cat(T_hc), cat(T_e2e)
if c_hc == "pass" and c_e2e == "fail":
    branch = "T1-CONDITIONED-PASS"
elif c_hc == "fail" and c_e2e == "fail":
    branch = "T1-BOTH-FAIL"
else:
    branch = "T1-MIXED/AT-BAR (worse governs; one fresh-stream " \
             "disambiguation licensed)"
out["T1"] = {"T_hc_declared": T_hc, "T_hc_native_desc": T_hc_nat,
             "T_e2e_reemitted": T_e2e, "T_stageD_context_hc": T_sdctx,
             "categories": {"hc": c_hc, "e2e": c_e2e}, "branch": branch}
print(f"T1: hc={T_hc:.2f} ({c_hc}) e2e={T_e2e:.2f} ({c_e2e}) "
      f"hc_native={T_hc_nat:.2f} sdctx={T_sdctx:.2f} -> {branch}")

# ---- T2 ---------------------------------------------------------------------
N_GRID = tuple(NUS)
i0 = list(NUS).index(0.0)
sc = real[0].size


def chi0(stack):
    v = np.array([minkowski_vector(f, NUS) for f in sm(stack)])
    return float(v[:, 2 * len(NUS) + i0].mean() * sc)


gen1 = [np.asarray(f, np.float64) for f in B["final1"]]
chi_real = chi0(real)
chi_full = chi0(gen1)
full_shift = chi_full - chi_real
t2 = {"chi0_real": chi_real, "chi0_fullgen": chi_full,
      "full_shift": full_shift, "single": {}, "cumulative": {}}


def hybrid(r, g, octs):
    cr = list(dwt_levels(r, 4))
    cg = dwt_levels(g, 4)
    for j in octs:
        cr[4 - j + 1] = cg[4 - j + 1]
    return idwt_levels(cr)


for j in (1, 2, 3, 4):
    for name, octs in (("single", (j,)), ("cumulative", tuple(range(1, j + 1)))):
        hyb = [hybrid(r, g, octs) for r, g in zip(real, gen1)]
        Th, _ = judge_T(sm(hyb), rs, seed=20260863 + 10 * j
                        + (0 if name == "single" else 1))
        ch = chi0(hyb)
        t2[name][str(j)] = {"T_MF": Th, "chi0": ch,
                            "shift_fraction": (ch - chi_real) / full_shift}
    print(f"T2 oct{j}: single T={t2['single'][str(j)]['T_MF']:.2f} "
          f"frac={t2['single'][str(j)]['shift_fraction']:+.2f}  "
          f"cumul T={t2['cumulative'][str(j)]['T_MF']:.2f} "
          f"frac={t2['cumulative'][str(j)]['shift_fraction']:+.2f}")
t2["line_single_half"] = bool(any(
    t2["single"][str(j)]["shift_fraction"] >= 0.5 for j in (1, 2, 3, 4)))
t2["line_cumul2_80"] = bool(
    t2["cumulative"]["2"]["shift_fraction"] >= 0.8)
out["T2"] = t2

# ---- T3 ---------------------------------------------------------------------
SETS = {"blind_e2e": e2e, "blind_hc": hc_blind,
        "trained_l1pp": [np.asarray(f, np.float64) for k in
                         ("adj1", "adj2", "adj3")
                         for f in np.load(os.path.join(
                             RES, "l1pp_main_gen.npz"))[k]]}
t3 = {"detail": {}, "map_declared": {}}
for j in (1, 2, 3, 4):
    mr_, sr_ = stack_detail_skew(real, j, seed=100 + j)
    row = {"real": [mr_, sr_]}
    for name, st in SETS.items():
        mg_, sg_ = stack_detail_skew(st, j, seed=200 + j + hash(name) % 50)
        row[name] = [mg_, sg_, (mg_ - mr_) / float(np.hypot(sr_, sg_))]
    t3["detail"][str(j)] = row
    print(f"T3 detail oct{j}: real {mr_:+.4f}±{sr_:.4f}  " + "  ".join(
        f"{n} {row[n][0]:+.4f} (z={row[n][2]:+.1f})" for n in SETS))
mr_, sr_ = stack_map_skew(rs, seed=300)
row = {"real": [mr_, sr_]}
for name, st in SETS.items():
    mg_, sg_ = stack_map_skew(sm(st), seed=400 + hash(name) % 50)
    row[name] = [mg_, sg_, (mg_ - mr_) / float(np.hypot(sr_, sg_))]
t3["map_declared"] = row
print(f"T3 map(0.5px): real {mr_:+.4f}±{sr_:.4f}  " + "  ".join(
    f"{n} {row[n][0]:+.4f} (z={row[n][2]:+.1f})" for n in SETS))
gen_e2e_under = [j for j in ("2", "3", "4")
                 if t3["detail"][j]["blind_e2e"][2] <= -3]
t3["rec_line_fires"] = bool(len(gen_e2e_under) >= 2)
t3["exec_map_line_fires"] = bool(row["blind_e2e"][2] <= -3)
out["T3"] = t3

with open(os.path.join(RES, "topo_diag_verdict.json"), "w") as f:
    json.dump(out, f, indent=1)
print("lines: T2 single>=half:", t2["line_single_half"],
      "| T2 cumul2>=80%:", t2["line_cumul2_80"],
      "| T3 rec (detail under-skew >=2 oct):", t3["rec_line_fires"],
      "| T3 exec (map deficit 3sig):", t3["exec_map_line_fires"])
print("wrote topo_diag_verdict.json")
