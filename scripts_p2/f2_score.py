"""Arm F2 SCORE phase (env.sh stack) + the corr(H,V) M5 curve rider (R33).

F2 adjudication (sandbox arm A, frozen prereg rules): all three defect stats
< 3 (T_coef oct-1, T_coef oct-2, level-1 parity) AND all marginal bars pass
-> F2-CLEAN; any defect stat >= 3 with marginals passing -> F2-PARTIAL; any
marginal bar fails -> F2-BREAKS. Delta-vs-F texture reported per entry.
Echo: gowerstreet arm A group-averaged vs before/F — nn WATCHED.
Rider: full 9-component z-vector (esp. corrHV) across parity_ckpt_gen.npz.

Writes results_p2/f2_verdict.json and extends results_p2/parity_ckpt_curve.json
-> results_p2/parity_ckpt_curve_full.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts_p2"))

from measure_generated import couplings
from parity_localization import COEF_STAT_NAMES, coef_stats_profile, peak_parity_profile
from placement_instruments import nn_profile, stack_profiles, tstat

TRUTH = json.load(open(os.path.join(RES, "sandbox_truth_normconv.json")))["truth"]
BARS = {"var_slope": 0.10, "kurtosis": 0.15}
TRAINED = [2, 3, 4]
K_SANDBOX, K_REAL = 160, 174

GEN = np.load(os.path.join(RES, "f2_test_gen.npz"), allow_pickle=True)
HC = json.load(open(os.path.join(RES, "f2_test_hc.json")))
FV = json.load(open(os.path.join(RES, "fj_verdict.json")))
tiles = np.load(os.path.join(REPO, "data_cache", "tiles_sandbox.npz"))["sandbox"]
test = tiles[-32:].astype(np.float64)


def check(metric, val, se, j):
    t, tse = TRUTH[str(j)][metric], TRUTH[str(j)][metric + "_se"]
    rel = abs(val - t) / abs(t)
    bar = max(BARS[metric], 3 * float(np.hypot(se, tse) / abs(t)))
    return {"value": val, "se": se, "truth": t, "rel_err": rel, "bar": bar,
            "pass": bool(rel <= bar)}


def marginal_suite(e2e_fields, hc_stats):
    rows, all_pass = {}, True
    e2e = couplings([np.asarray(f, np.float64) for f in e2e_fields],
                    [1, 2, 3, 4], n_boot=200, seed=0)
    for level, src in (("head-conditional", hc_stats), ("end-to-end", e2e)):
        for j in TRAINED:
            s = src[str(j)] if str(j) in src else src[j]
            for metric in ("var_slope", "kurtosis"):
                r = check(metric, s[metric], s[metric + "_se"], j)
                rows.setdefault(level, {}).setdefault(str(j), {})[metric] = r
                all_pass &= r["pass"]
    return rows, all_pass


def T_of(gen, ref, fn, with_z=False):
    a = [fn(np.asarray(f, np.float64)) for f in gen]
    b = [fn(np.asarray(f, np.float64)) for f in ref]
    T, z = tstat(*stack_profiles(a), *stack_profiles(b))
    return (T, z) if with_z else T


out = {"identity_gate": HC["identity_gate"], "F2": {}, "echo": {}}

for arm in ("A", "B"):
    fields = GEN[f"F2_{arm}_e2e"]
    rows, all_pass = marginal_suite(fields, HC["hc"][f"F2_{arm}"])
    t1, z1 = T_of(fields, test, lambda f: coef_stats_profile(f, octave=1),
                  with_z=True)
    t2 = T_of(fields, test, lambda f: coef_stats_profile(f, octave=2))
    tp = T_of(fields, test, lambda f: peak_parity_profile(f, k=K_SANDBOX,
                                                          level=1))
    entry = {"marginals": rows, "marginals_all_pass": all_pass,
             "T_coef_oct1": t1, "T_coef_oct2": t2, "parity_T": tp,
             "surviving_components": {COEF_STAT_NAMES[i]: float(z1[i])
                                      for i in range(9) if abs(z1[i]) >= 3}}
    if arm == "A":
        if not all_pass:
            entry["branch"] = "F2-BREAKS"
        elif t1 < 3.0 and t2 < 3.0 and tp < 3.0:
            entry["branch"] = "F2-CLEAN"
        else:
            entry["branch"] = "F2-PARTIAL"
    out["F2"][arm] = entry
    print(f"F2 {arm}: marginals_all_pass={all_pass} T_coef1={t1:.1f} "
          f"T_coef2={t2:.1f} parity={tp:.1f} "
          f"surv={entry['surviving_components']}"
          + (f" -> {entry.get('branch')}" if arm == "A" else " (descriptive)"))

# echo: before (committed) / F (constant) / F2 (group-averaged)
gow = np.load(os.path.join(RES, "arms_c1t_gowerstreet.npz"), allow_pickle=True)
FJGEN = np.load(os.path.join(RES, "fj_test_gen.npz"), allow_pickle=True)
greal = [np.asarray(f, np.float64) for f in gow["real"]]
NN_EDGES = np.array(json.load(open(os.path.join(
    RES, "placement_phase_a.json")))["convention"]["nn_edges"])
for label, fields in (("before", gow["gen_A"]), ("F_const", FJGEN["F_gowA_e2e"]),
                      ("F2_groupavg", GEN["F2_gowA_e2e"])):
    e2e = couplings([np.asarray(f, np.float64) for f in fields],
                    [1, 2, 3, 4], n_boot=200, seed=0)
    out["echo"][label] = {
        "T_coef_oct1": T_of(fields, greal,
                            lambda f: coef_stats_profile(f, octave=1)),
        "parity_T": T_of(fields, greal,
                         lambda f: peak_parity_profile(f, k=K_REAL, level=1)),
        "nn_T": T_of(fields, greal,
                     lambda f: nn_profile(f, k=K_REAL, edges=NN_EDGES)),
        "e2e_kurt_oct2": e2e[2]["kurtosis"], "e2e_vs_oct2": e2e[2]["var_slope"]}
    e = out["echo"][label]
    print(f"echo {label}: T_coef1={e['T_coef_oct1']:.1f} "
          f"parity={e['parity_T']:.1f} nn={e['nn_T']:.1f} "
          f"oct2 kurt={e['e2e_kurt_oct2']:.2f} vs={e['e2e_vs_oct2']:.3f}")

with open(os.path.join(RES, "f2_verdict.json"), "w") as f:
    json.dump(out, f, indent=1)

# ---- rider: full z-vector across the M5 checkpoint curve --------------------
CK = np.load(os.path.join(RES, "parity_ckpt_gen.npz"), allow_pickle=True)
curve = {}
for key in sorted(CK.files, key=lambda s: (s.split("_s")[0],
                                           int(s.split("_s")[1]))):
    T, z = T_of(CK[key], test, lambda f: coef_stats_profile(f, octave=1),
                with_z=True)
    curve[key] = {"T": T, "z": {COEF_STAT_NAMES[i]: float(z[i])
                                for i in range(9)}}
    print(f"rider {key}: T={T:.1f} corrHV={z[6]:+.1f}")
with open(os.path.join(RES, "parity_ckpt_curve_full.json"), "w") as f:
    json.dump({"K": K_SANDBOX, "curve": curve}, f, indent=1)
print("wrote f2_verdict.json + parity_ckpt_curve_full.json")
