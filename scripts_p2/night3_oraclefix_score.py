"""O-CORRECTED probe scorer (env.sh; prereg 2026-08-05-night3, grant-2;
DESCRIPTIVE — R47's O adjudication is NOT reopened tonight). The oracle
chain at the corrected-cage pick @16000. Entries: MF(dev) declared+native,
native peaks vs half-targets, frag/alignment, C/P-T watched, e2e marginals
octaves 1-4, starlet, parity/nn, determinism. hc marginals unavailable at
@16000 (run script evaluated hc only at the bugged pick and 20k) — noted.
Registered expectations (exec): MF(dev) declared <= 3.5 P 60; peak
excesses REMAIN > half-targets P 80. Writes
results_p2/night3_oraclefix_verdict.json."""
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
NCOMP_HOLES = [0, 1, 4, 5, 8, 9]
K_REAL = 174
S0 = 20261320

GEN = np.load(os.path.join(RES, "stage3_oraclefix_final.npz"))
streams = [[np.asarray(f, np.float64) for f in GEN[k]]
           for k in ("final1", "final2", "final3")]
pooled = [f for s in streams for f in s]
real = [np.asarray(f, np.float64) for f in
        np.load(os.path.join(RES, "arms_c1t_gowerstreet.npz"),
                allow_pickle=True)["real"]]
out = {"probe": "oraclefix @16000 (corrected cage)", "descriptive": True}

W = json.load(open(os.path.join(RES, "stage3_oraclefix_white.json")))
PT = json.load(open(os.path.join(RES, "stage3_oraclefix_pt.json")))
gates_ok = (W["determinism_gate_maxabs"] == 0.0
            and PT["cal_history"][-1] <= 0.03)
out["gates"] = {"determinism": W["determinism_gate_maxabs"],
                "fit": PT["cal_history"], "ok": bool(gates_ok)}

c_pool, se_pool = stack_coloring(pooled, 1, seed=S0)
pt_lands = bool(PT["interval"][0] <= c_pool <= PT["interval"][1])
out["coloring"] = {"C_pooled": c_pool, "SE": se_pool, "pt": PT,
                   "PT_LANDS": pt_lands}
print(f"C={c_pool:.4f}±{se_pool:.4f} vs pred {PT['C_pred']:.4f} "
      f"[{PT['interval'][0]:.4f},{PT['interval'][1]:.4f}] -> "
      f"{'LANDS' if pt_lands else 'FAILS'}")

sm = lambda st: [gaussian_filter(f, 0.5) for f in st]  # noqa: E731
T_dec, _ = judge_T(sm(pooled), sm(real), seed=S0 + 1)
T_nat, _ = judge_T(pooled, real, seed=S0 + 2)
mf_cat = ("pass" if T_dec < BAR_MF - BAND_MF else
          ("fail" if T_dec > BAR_MF + BAND_MF else "at-bar"))
pk = {nu: bootstrap_excess(pooled, real, float(nu), seed=S0 + 3 + i)
      for i, nu in enumerate(("2.5", "3.0"))}
pk_halved = all(pk[nu]["excess"] < A3[nu] / 2 for nu in pk)
mf_, sf = stack_fragmentation(pooled, seed=S0 + 5)
mr_, sr_ = stack_fragmentation(real, seed=S0 + 6)
fz = (mf_ - mr_) / np.hypot(sf, sr_)
al, sal = stack_alignment(pooled, seed=S0 + 7)
alr, salr = stack_alignment(real, seed=S0 + 8)
out["MF_dev"] = {"declared": float(T_dec), "category": mf_cat,
                 "native_desc": float(T_nat)}
out["native_peaks"] = {nu: {k: pk[nu][k] for k in ("excess", "se", "ci95")}
                       for nu in pk}
out["instrument"] = {"frag_z": fz.tolist(),
                     "frag_maxabs_z": float(np.max(np.abs(fz[NCOMP_HOLES]))),
                     "alignment_z": (al - alr) / float(np.hypot(sal, salr))}
out["expectations"] = {"mf_le_bar (exec 60)": bool(T_dec <= BAR_MF),
                       "peaks_persist (exec 80)": bool(not pk_halved)}
print(f"MF(dev) declared {T_dec:.2f} ({mf_cat}); native {T_nat:.2f}; "
      f"frag_max|z|={out['instrument']['frag_maxabs_z']:.2f}")
print("native peaks: " + "  ".join(
    f"nu={nu}: {pk[nu]['excess']:+.2%}±{pk[nu]['se']:.2%} "
    f"(half {A3[nu] / 2:.2%})" for nu in pk))

real_ref = couplings(real, [1, 2, 3, 4], n_boot=50, seed=0)
e2e = couplings(pooled, [1, 2, 3, 4], n_boot=200, seed=0)
rows, marg_ok = {}, True
for j in (1, 2, 3, 4):
    for metric in ("var_slope", "kurtosis"):
        t, tse = real_ref[j][metric], real_ref[j][metric + "_se"]
        v, se = e2e[j][metric], e2e[j][metric + "_se"]
        rel = abs(v - t) / abs(t)
        bar = max(BARS[metric], 3 * float(np.hypot(se, tse) / abs(t)))
        ok = bool(rel <= bar)
        rows.setdefault(str(j), {})[metric] = \
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
p = os.path.join(RES, "starlet_l1_oraclefix.json")
if os.path.exists(p):
    st_ok = bool(json.load(open(p))["checks"]["gen_A"]["all_scored_pass"])
out["battery"] = {"e2e_marginals": rows, "marg_ok": bool(marg_ok),
                  "hc_marginals": "unavailable at @16000 (noted)",
                  "parity_T": parity, "starlet": st_ok,
                  "nn_mean_sd": [float(np.mean(nns)),
                                 float(np.std(nns, ddof=1))]}
with open(os.path.join(RES, "night3_oraclefix_verdict.json"), "w") as f:
    json.dump(out, f, indent=1)
print(f"e2e marginals={marg_ok} parity={parity:.2f} starlet={st_ok} "
      f"nn={np.mean(nns):.2f}±{np.std(nns, ddof=1):.2f} "
      f"align_z={out['instrument']['alignment_z']:+.1f}")
