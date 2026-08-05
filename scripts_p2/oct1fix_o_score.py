"""O-arm scorer (env.sh; prereg 2026-08-05-oct1fix, R47). A5 order:
gates -> C/P-T context -> entries -> mechanical branch.

Targets (O-FIXES needs BOTH, no regression): MF declared <= 3.5 (frozen
judge, #11 band 0.25, at-bar = worse) AND native peak excesses both < half
the committed A3 references (+15.290/+12.474% -> halves 7.645/6.237%).
Must-not-regress: marginals hc+e2e octaves 1-4 (standing bars; octave 1
NOTED as a new trained octave), starlet leg 15, parity_T < 3, determinism.
Descriptive: MF native, I-instrument on the O maps, C/P-T (6th test).
Writes results_p2/oct1fix_o_verdict.json.
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
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts_p2"))

from audit_peak_ci import bootstrap_excess
from coloring_index import stack_coloring
from measure_generated import couplings
from minkowski_judge import judge_T
from oct1_texture import stack_alignment, stack_fragmentation
from parity_localization import peak_parity_profile
from placement_instruments import nn_profile, stack_profiles, tstat

A3 = {"2.5": 0.15290, "3.0": 0.12474}
BARS = {"var_slope": 0.10, "kurtosis": 0.15}
BAR_MF, BAND_MF = 3.5, 0.25
K_REAL = 174

GEN = np.load(os.path.join(RES, "oct1fix_oracle_final.npz"))
streams = [[np.asarray(f, np.float64) for f in GEN[k]]
           for k in ("final1", "final2", "final3")]
pooled = [f for s in streams for f in s]
real = [np.asarray(f, np.float64) for f in
        np.load(os.path.join(RES, "arms_c1t_gowerstreet.npz"),
                allow_pickle=True)["real"]]
out = {}

W = json.load(open(os.path.join(RES, "stage3_oracle_white.json")))
PT = json.load(open(os.path.join(RES, "stage3_oracle_pt.json")))
gates_ok = (W["determinism_gate_maxabs"] == 0.0
            and PT["cal_history"][-1] <= 0.03)
out["gates"] = {"determinism": W["determinism_gate_maxabs"],
                "fit": PT["cal_history"], "ok": bool(gates_ok)}

c_pool, se_pool = stack_coloring(pooled, 1, seed=7000)
pt_lands = bool(PT["interval"][0] <= c_pool <= PT["interval"][1])
out["coloring"] = {"C_pooled": c_pool, "SE": se_pool, "pt": PT,
                   "PT_LANDS": pt_lands}
with open(os.path.join(RES, "oct1fix_o_c.json"), "w") as f:
    json.dump(out["coloring"], f, indent=1)
print(f"C={c_pool:.4f}±{se_pool:.4f} vs pred {PT['C_pred']:.4f} "
      f"[{PT['interval'][0]:.4f},{PT['interval'][1]:.4f}] -> "
      f"{'LANDS' if pt_lands else 'FAILS'} (6th consecutive test)")

# entries
sm = lambda st: [gaussian_filter(f, 0.5) for f in st]  # noqa: E731
T_dec, _ = judge_T(sm(pooled), sm(real), seed=20260890)
T_nat, _ = judge_T(pooled, real, seed=20260891)
mf_cat = ("pass" if T_dec < BAR_MF - BAND_MF else
          ("fail" if T_dec > BAR_MF + BAND_MF else "at-bar"))
mf_target = mf_cat == "pass"
pk = {nu: bootstrap_excess(pooled, real, float(nu), seed=7100 + i)
      for i, nu in enumerate(("2.5", "3.0"))}
pk_target = all(pk[nu]["excess"] < A3[nu] / 2 for nu in pk)
out["MF"] = {"declared": float(T_dec), "category": mf_cat,
             "native_desc": float(T_nat)}
out["native_peaks"] = {nu: {k: pk[nu][k] for k in ("excess", "se", "ci95")}
                       for nu in pk}
out["targets"] = {"mf": mf_target, "peaks_halved": pk_target}
print(f"MF declared {T_dec:.2f} ({mf_cat}); native {T_nat:.2f}")
print("native peaks: " + "  ".join(
    f"nu={nu}: {pk[nu]['excess']:+.2%}±{pk[nu]['se']:.2%} "
    f"(half-target {A3[nu] / 2:.2%})" for nu in pk))

# must-not-regress
sel = json.load(open(os.path.join(RES, "oct1fix_oracle_selection.json")))
real_ref = couplings(real, [1, 2, 3, 4], n_boot=50, seed=0)
e2e = couplings(pooled, [1, 2, 3, 4], n_boot=200, seed=0)
rows, marg_ok = {}, True
for level, src_by_j in (("head-conditional",
                         {j: sel["A"]["final"][str(j)] for j in (1, 2, 3, 4)}),
                        ("end-to-end", {j: e2e[j] for j in (1, 2, 3, 4)})):
    for j in (1, 2, 3, 4):
        for metric in ("var_slope", "kurtosis"):
            t, tse = real_ref[j][metric], real_ref[j][metric + "_se"]
            v, se = src_by_j[j][metric], src_by_j[j][metric + "_se"]
            rel = abs(v - t) / abs(t)
            bar = max(BARS[metric], 3 * float(np.hypot(se, tse) / abs(t)))
            ok = bool(rel <= bar)
            rows.setdefault(level, {}).setdefault(str(j), {})[metric] = \
                {"rel_err": rel, "bar": bar, "pass": ok}
            marg_ok &= ok


def T_of(gen, fn):
    a = [fn(np.asarray(f, np.float64)) for f in gen]
    b = [fn(f) for f in real]
    T, _ = tstat(*stack_profiles(a), *stack_profiles(b))
    return float(T)


parity = T_of(pooled, lambda f: peak_parity_profile(f, k=K_REAL, level=1))
NN_EDGES = np.array(json.load(open(os.path.join(
    RES, "placement_phase_a.json")))["convention"]["nn_edges"])
nns = [T_of(s, lambda f: nn_profile(f, k=K_REAL, edges=NN_EDGES))
       for s in streams]
st_ok = None
p = os.path.join(RES, "starlet_l1_oct1fix.json")
if os.path.exists(p):
    st_ok = bool(json.load(open(p))["checks"]["gen_A"]["all_scored_pass"])
regression = (not marg_ok) or parity >= 3 or (st_ok is False) or not gates_ok
out["must_not_regress"] = {"marginals": rows, "marg_ok": bool(marg_ok),
                           "parity_T": parity, "starlet": st_ok,
                           "nn_mean_sd": [float(np.mean(nns)),
                                          float(np.std(nns, ddof=1))],
                           "regression": bool(regression)}

# I instrument (descriptive texture verification)
mfz, sfz = stack_fragmentation(pooled, seed=7200)
mr, sr_ = stack_fragmentation(real, seed=7201)
zf = ((mfz - mr) / np.hypot(sfz, sr_)).tolist()
al, sal = stack_alignment(pooled, seed=7202)
alr, salr = stack_alignment(real, seed=7203)
out["instrument"] = {"frag_z_vs_real": zf,
                     "alignment": [al, sal],
                     "alignment_z": (al - alr) / float(np.hypot(sal, salr))}

# branch
n_targets = int(mf_target) + int(pk_target)
if not gates_ok:
    branch = "GATES"
elif regression:
    branch = "O-REGRESSED"
elif n_targets == 2:
    branch = "O-FIXES"
elif n_targets == 1:
    branch = "O-PARTIAL"
else:
    branch = "O-NULL"
if mf_cat == "at-bar":
    branch += " (MF at-bar; worse governs)"
out["adjudication"] = {"branch": branch, "targets": out["targets"]}
with open(os.path.join(RES, "oct1fix_o_verdict.json"), "w") as f:
    json.dump(out, f, indent=1)
print(f"marginals={marg_ok} parity={parity:.2f} starlet={st_ok} "
      f"nn={np.mean(nns):.2f}±{np.std(nns, ddof=1):.2f} "
      f"align_z={out['instrument']['alignment_z']:+.1f} "
      f"max|frag_z|={max(abs(v) for v in zf):.1f}")
print("ADJUDICATION:", branch)
