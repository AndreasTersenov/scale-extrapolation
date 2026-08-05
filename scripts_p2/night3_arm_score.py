"""NIGHT-3 arm scorer (env.sh; prereg 2026-08-05-night3). A5 order:
gates -> C/P-T context (watched) -> entries -> mechanical branch.
Usage: night3_arm_score.py <tag> <sel_json> <family AUG|TIDAL>

Branch rules (mechanical order gates -> REGRESSED -> FIXES -> IMPROVED ->
NULL; ambiguity = negative; #11 band on MF):
  X-FIXES    MF(dev) declared category pass AND both native peak excesses
             < half committed A3 (7.645/6.237%); no regression.
  X-IMPROVED not FIXES; MF(dev) declared <= 2.24 AND max |frag z| over
             ncomp+holes rows <= 2.35; no regression.
  X-NULL     neither; no regression.
  X-REGRESSED any must-not-regress failure (marginals hc+e2e oct 1-4,
             starlet leg, parity < 3, determinism/fit gates).
Arm-level branch = WORSE of the two seeds (applied in the readout, not
here — this scorer emits per-seed categories). MF is a DEVELOPMENT
metric tonight (JUDGE-2 is the quarantined held-out tier).
Writes results_p2/night3_<tag>_verdict.json."""
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
T_IMPROVED, FRAG_IMPROVED = 2.24, 2.35
NCOMP_HOLES = [0, 1, 4, 5, 8, 9]
K_REAL = 174
SEED0 = {"aug11": 20261100, "aug12": 20261140,
         "tidal21": 20261180, "tidal22": 20261220}

tag, sel_path, family = sys.argv[1], sys.argv[2], sys.argv[3]
s0 = SEED0[tag]

GEN = np.load(os.path.join(RES, f"stage3_{tag}_final.npz"))
streams = [[np.asarray(f, np.float64) for f in GEN[k]]
           for k in ("final1", "final2", "final3")]
pooled = [f for s in streams for f in s]
real = [np.asarray(f, np.float64) for f in
        np.load(os.path.join(RES, "arms_c1t_gowerstreet.npz"),
                allow_pickle=True)["real"]]
out = {"tag": tag, "family": family}

W = json.load(open(os.path.join(RES, f"stage3_{tag}_white.json")))
PT = json.load(open(os.path.join(RES, f"stage3_{tag}_pt.json")))
gates_ok = (W["determinism_gate_maxabs"] == 0.0
            and PT["cal_history"][-1] <= 0.03)
out["gates"] = {"determinism": W["determinism_gate_maxabs"],
                "fit": PT["cal_history"], "ok": bool(gates_ok)}

c_pool, se_pool = stack_coloring(pooled, 1, seed=s0)
pt_lands = bool(PT["interval"][0] <= c_pool <= PT["interval"][1])
out["coloring"] = {"C_pooled": c_pool, "SE": se_pool, "pt": PT,
                   "PT_LANDS": pt_lands, "watched_not_branch": True}
print(f"[{tag}] C={c_pool:.4f}±{se_pool:.4f} vs pred {PT['C_pred']:.4f} "
      f"[{PT['interval'][0]:.4f},{PT['interval'][1]:.4f}] -> "
      f"{'LANDS' if pt_lands else 'FAILS'} (watched)")

# entries
sm = lambda st: [gaussian_filter(f, 0.5) for f in st]  # noqa: E731
T_dec, _ = judge_T(sm(pooled), sm(real), seed=s0 + 1)
T_nat, _ = judge_T(pooled, real, seed=s0 + 2)
mf_cat = ("pass" if T_dec < BAR_MF - BAND_MF else
          ("fail" if T_dec > BAR_MF + BAND_MF else "at-bar"))
pk = {nu: bootstrap_excess(pooled, real, float(nu), seed=s0 + 3 + i)
      for i, nu in enumerate(("2.5", "3.0"))}
pk_target = all(pk[nu]["excess"] < A3[nu] / 2 for nu in pk)

mf, sf = stack_fragmentation(pooled, seed=s0 + 5)
mr, sr_ = stack_fragmentation(real, seed=s0 + 6)
fz = (mf - mr) / np.hypot(sf, sr_)
frag_max = float(np.max(np.abs(fz[NCOMP_HOLES])))
al, sal = stack_alignment(pooled, seed=s0 + 7)
alr, salr = stack_alignment(real, seed=s0 + 8)

out["MF_dev"] = {"declared": float(T_dec), "category": mf_cat,
                 "native_desc": float(T_nat)}
out["native_peaks"] = {nu: {k: pk[nu][k] for k in ("excess", "se", "ci95")}
                       for nu in pk}
out["instrument"] = {"frag_z": fz.tolist(), "frag_maxabs_z": frag_max,
                     "alignment_z": (al - alr) / float(np.hypot(sal, salr))}
mf_target = mf_cat == "pass"
improved = (T_dec <= T_IMPROVED) and (frag_max <= FRAG_IMPROVED)
out["targets"] = {"mf": bool(mf_target), "peaks_halved": bool(pk_target),
                  "improved_rule": bool(improved)}
print(f"[{tag}] MF(dev) declared {T_dec:.2f} ({mf_cat}); native {T_nat:.2f}; "
      f"frag_max|z|={frag_max:.2f}")
print(f"[{tag}] native peaks: " + "  ".join(
    f"nu={nu}: {pk[nu]['excess']:+.2%}±{pk[nu]['se']:.2%} "
    f"(half {A3[nu] / 2:.2%})" for nu in pk))

# must-not-regress
sel = json.load(open(sel_path))["A"]
real_ref = couplings(real, [1, 2, 3, 4], n_boot=50, seed=0)
e2e = couplings(pooled, [1, 2, 3, 4], n_boot=200, seed=0)
rows, marg_ok = {}, True
for level, src_by_j in (("head-conditional",
                         {j: sel["final"][str(j)] for j in (1, 2, 3, 4)}),
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
p = os.path.join(RES, f"starlet_l1_{tag}.json")
if os.path.exists(p):
    st_ok = bool(json.load(open(p))["checks"]["gen_A"]["all_scored_pass"])
regression = (not marg_ok) or parity >= 3 or (st_ok is False) or not gates_ok
out["must_not_regress"] = {"marginals": rows, "marg_ok": bool(marg_ok),
                           "parity_T": parity, "starlet": st_ok,
                           "nn_mean_sd": [float(np.mean(nns)),
                                          float(np.std(nns, ddof=1))],
                           "regression": bool(regression)}

# branch (mechanical order)
if not gates_ok:
    branch = "GATES"
elif regression:
    branch = f"{family}-REGRESSED"
elif mf_target and pk_target:
    branch = f"{family}-FIXES"
elif improved:
    branch = f"{family}-IMPROVED"
else:
    branch = f"{family}-NULL"
out["adjudication"] = {"branch": branch,
                       "note": "arm branch = worse of the two seeds "
                               "(applied in the readout)"}
with open(os.path.join(RES, f"night3_{tag}_verdict.json"), "w") as f:
    json.dump(out, f, indent=1)
print(f"[{tag}] marginals={marg_ok} parity={parity:.2f} starlet={st_ok} "
      f"nn={np.mean(nns):.2f}±{np.std(nns, ddof=1):.2f} "
      f"align_z={out['instrument']['alignment_z']:+.1f}")
print(f"[{tag}] PER-SEED CATEGORY:", branch)
