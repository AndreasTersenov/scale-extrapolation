"""Arms F/J, SCORE phase 2 (env.sh stack): the verdicts, per the frozen
branch rules of log/2026-07-25-prereg-parity-cure.md (approved R32).

F (sandbox arm A adjudicating; B descriptive): corrected test e2e scored with
the frozen production scorer vs the sandbox truth bars (c1t convention,
octaves 2-4, hc + e2e); T_coef (octave-1 D4 statistic vs the 32 test tiles)
and level-1 peak parity on the corrected fields. Mechanical order: any
marginal bar fails -> F-BREAKS; else T_coef<3 AND parity<3 -> F-CLEAN; else
T_coef dropped >=2x from the committed 15.3 (or parity residual) -> F-PARTIAL;
a <2x drop with clean marginals would be an UNLISTED outcome, reported as
such.

J (arm A at the joint pick): the same marginal bars + T_coef<3 on the
UNcorrected joint-pick generations -> J-WINDOW else J-TRADEOFF (the
exists-quantifier witnessed by the criterion's own pick; documented).

Echo (gowerstreet arm A, descriptive, load-bearing): T_coef/parity/nn
before vs after correction; marginal e2e couplings descriptive.

Writes results_p2/fj_verdict.json.
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
from parity_localization import coef_stats_profile, peak_parity_profile
from placement_instruments import nn_profile, stack_profiles, tstat

TRUTH = json.load(open(os.path.join(RES, "sandbox_truth_normconv.json")))["truth"]
BARS = {"var_slope": 0.10, "kurtosis": 0.15}
TRAINED = [2, 3, 4]
T_COEF_REF_A = 15.3          # A' committed reference (marginal pick, test side)
K_SANDBOX, K_REAL = 160, 174

GEN = np.load(os.path.join(RES, "fj_test_gen.npz"), allow_pickle=True)
HC = json.load(open(os.path.join(RES, "fj_test_hc.json")))
JP = json.load(open(os.path.join(RES, "fj_joint_pick.json")))
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
    return rows, all_pass, {"oct1_kurt": e2e[1]["kurtosis"],
                            "oct1_vs": e2e[1]["var_slope"]}


def T_of(gen, ref, fn):
    a = [fn(np.asarray(f, np.float64)) for f in gen]
    b = [fn(np.asarray(f, np.float64)) for f in ref]
    T, z = tstat(*stack_profiles(a), *stack_profiles(b))
    return T


out = {"F": {}, "J": {}, "echo": {}}

# ------------------------------- F ------------------------------------------
for arm in ("A", "B"):
    fields = GEN[f"F_{arm}_e2e"]
    rows, all_pass, oct1 = marginal_suite(fields, HC["hc"][f"F_{arm}"])
    t1 = T_of(fields, test, lambda f: coef_stats_profile(f, octave=1))
    t2 = T_of(fields, test, lambda f: coef_stats_profile(f, octave=2))
    tp = T_of(fields, test, lambda f: peak_parity_profile(f, k=K_SANDBOX,
                                                          level=1))
    entry = {"marginals": rows, "marginals_all_pass": all_pass,
             "e2e_oct1_descriptive": oct1,
             "T_coef_oct1": t1, "T_coef_oct2": t2, "parity_T": tp}
    if arm == "A":
        if not all_pass:
            branch = "F-BREAKS"
        elif t1 < 3.0 and tp < 3.0:
            branch = "F-CLEAN"
        elif t1 < T_COEF_REF_A / 2 or tp >= 3.0:
            branch = "F-PARTIAL"
        else:
            branch = "UNLISTED (T_coef >= 3 with <2x drop, marginals clean)"
        entry["branch"] = branch
    out["F"][arm] = entry
    print(f"F {arm}: marginals_all_pass={all_pass} T_coef1={t1:.1f} "
          f"T_coef2={t2:.1f} parity={tp:.1f}"
          + (f" -> {entry.get('branch')}" if arm == "A" else " (descriptive)"))

# ------------------------------- J ------------------------------------------
jstep = HC["J_A_step"]
fields = GEN["J_A_e2e"]
rows, all_pass, oct1 = marginal_suite(fields, HC["hc"]["J_A"])
t1 = T_of(fields, test, lambda f: coef_stats_profile(f, octave=1))
tp = T_of(fields, test, lambda f: peak_parity_profile(f, k=K_SANDBOX, level=1))
branch = "J-WINDOW" if (all_pass and t1 < 3.0) else "J-TRADEOFF"
out["J"] = {"step": jstep, "marginals": rows, "marginals_all_pass": all_pass,
            "T_coef_oct1": t1, "parity_T": tp, "branch": branch,
            "reading_note": "exists-quantifier witnessed by the criterion's "
                            "own pick (val-side); documented in the readout",
            "e2e_oct1_descriptive": oct1}
print(f"J A@{jstep}: marginals_all_pass={all_pass} T_coef1={t1:.1f} "
      f"parity={tp:.1f} -> {branch}")

# ------------------------------- echo ---------------------------------------
gow = np.load(os.path.join(RES, "arms_c1t_gowerstreet.npz"), allow_pickle=True)
greal = [np.asarray(f, np.float64) for f in gow["real"]]
NN_EDGES = np.array(json.load(open(os.path.join(
    RES, "placement_phase_a.json")))["convention"]["nn_edges"])
for label, fields in (("before", gow["gen_A"]), ("after", GEN["F_gowA_e2e"])):
    e2e = couplings([np.asarray(f, np.float64) for f in fields],
                    [1, 2, 3, 4], n_boot=200, seed=0)
    out["echo"][label] = {
        "T_coef_oct1": T_of(fields, greal,
                            lambda f: coef_stats_profile(f, octave=1)),
        "T_coef_oct2": T_of(fields, greal,
                            lambda f: coef_stats_profile(f, octave=2)),
        "parity_T": T_of(fields, greal,
                         lambda f: peak_parity_profile(f, k=K_REAL, level=1)),
        "nn_T": T_of(fields, greal,
                     lambda f: nn_profile(f, k=K_REAL, edges=NN_EDGES)),
        "e2e_kurt_oct2": e2e[2]["kurtosis"], "e2e_vs_oct2": e2e[2]["var_slope"]}
    e = out["echo"][label]
    print(f"echo {label}: T_coef1={e['T_coef_oct1']:.1f} "
          f"T_coef2={e['T_coef_oct2']:.1f} parity={e['parity_T']:.1f} "
          f"nn={e['nn_T']:.1f} e2e_oct2 kurt={e['e2e_kurt_oct2']:.2f} "
          f"vs={e['e2e_vs_oct2']:.3f}")

with open(os.path.join(RES, "fj_verdict.json"), "w") as f:
    json.dump(out, f, indent=1)
print("wrote results_p2/fj_verdict.json")
