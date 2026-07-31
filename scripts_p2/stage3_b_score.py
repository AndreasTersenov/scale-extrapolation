"""Stage-3 (b) scorer (env.sh; prereg af34d93 + R42 A5 order, structural):
gates -> coloring C + P-T (written to disk BEFORE any entry number) ->
adjudicating entries E1-E4 -> branch (blind only; dry-run outputs are
QUARANTINED and carry no branch).

Usage: stage3_b_score.py <tag> <sel_json_path>.
Entries (bars frozen in the prereg):
  E1 edge marginals: hc (selection json, arm A final oct 2) + e2e oct 2
     from the pooled final streams; var_slope <= max(10%,3SE), kurtosis <=
     max(15%,3SE) vs the real test reference (stage-D convention).
  E2 smoothed peaks sigma=0.5px, nu {2.5,3}: pooled 96 vs 32 real, both
     stacks smoothed; ci95 include 0 AND per-parent (per-stream panels
     averaged) not sign-consistent.
  E3 Minkowski FIRST APPLICATION at declared resolution: T_MF <= 3.5
     (committed null stage3_mf_null.json: smoothed 1.48/0.55/2.87);
     at-bar band +-0.25 -> AT-THE-BAR (worse governs; one-shot, no rerun).
     Native T_MF DESCRIPTIVE.
  E4 starlet edge leg (reads starlet_l1_stage3_<tag>.json, wl-env script).
Supporting: parity_T, nn per-stream mean+-sd, native peaks (#10).
Writes stage3_<tag>_c.json then stage3_<tag>_verdict.json.
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

from audit_peak_ci import bootstrap_excess, per_parent_excess
from coloring_index import stack_coloring
from measure_generated import couplings
from minkowski_judge import judge_T
from parity_localization import peak_parity_profile
from placement_instruments import nn_profile, stack_profiles, tstat

BARS = {"var_slope": 0.10, "kurtosis": 0.15}
BLOCKS = [(0, 10), (10, 21), (21, 32)]
K_REAL = 174
MF_BAR, MF_BAND = 3.5, 0.25
EDGE = 2

tag, sel_path = sys.argv[1], sys.argv[2]
quarantined = tag == "dryrun"
GEN = np.load(os.path.join(RES, f"stage3_{tag}_final.npz"))
streams = [[np.asarray(f, np.float64) for f in GEN[k]]
           for k in ("final1", "final2", "final3")]
pooled = [f for s in streams for f in s]
real = [np.asarray(f, np.float64) for f in
        np.load(os.path.join(RES, "arms_stageD.npz"),
                allow_pickle=True)["real"]]
out = {"tag": tag, "QUARANTINED": quarantined}

# ---- gates ------------------------------------------------------------------
W = json.load(open(os.path.join(RES, f"stage3_{tag}_white.json")))
PT = json.load(open(os.path.join(RES, f"stage3_{tag}_pt.json")))
gates_ok = (W["determinism_gate_maxabs"] == 0.0
            and PT["cal_history"][-1] <= 0.03)
out["gates"] = {"determinism": W["determinism_gate_maxabs"],
                "fit_convergence": PT["cal_history"], "ok": bool(gates_ok)}

# ---- C + P-T (A5: on disk before any entry number) --------------------------
c_pool, se_pool = stack_coloring(pooled, 1, seed=5000)
pt_lands = bool(PT["interval"][0] <= c_pool <= PT["interval"][1])
cjson = {"C_pooled": c_pool, "SE_pooled": se_pool,
         "per_stream": [stack_coloring(s, 1, seed=5001 + i)
                        for i, s in enumerate(streams)],
         "pt": PT, "PT_LANDS": pt_lands}
with open(os.path.join(RES, f"stage3_{tag}_c.json"), "w") as f:
    json.dump(cjson, f, indent=1)
out["coloring"] = cjson
print(f"[{tag}] C={c_pool:.4f}±{se_pool:.4f} vs pred {PT['C_pred']:.4f} "
      f"[{PT['interval'][0]:.4f},{PT['interval'][1]:.4f}] -> "
      f"{'LANDS' if pt_lands else 'FAILS'}")

# ---- E1 marginals -----------------------------------------------------------
real_ref = couplings(real, [1, 2, 3, 4], n_boot=50, seed=0)
sel = json.load(open(sel_path))
hc = sel["A"]["final"][str(EDGE)]
e2e = couplings(pooled, [1, 2, 3, 4], n_boot=200, seed=0)
e1_rows, e1_pass = {}, True
for level, src in (("head-conditional", hc), ("end-to-end", e2e[EDGE])):
    for metric in ("var_slope", "kurtosis"):
        t, tse = real_ref[EDGE][metric], real_ref[EDGE][metric + "_se"]
        v, se = src[metric], src[metric + "_se"]
        rel = abs(v - t) / abs(t)
        bar = max(BARS[metric], 3 * float(np.hypot(se, tse) / abs(t)))
        r = {"rel_err": rel, "bar": bar, "pass": bool(rel <= bar)}
        e1_rows.setdefault(level, {})[metric] = r
        e1_pass &= r["pass"]
out["E1"] = {"rows": e1_rows, "pass": bool(e1_pass)}

# ---- E2 smoothed peaks ------------------------------------------------------
gs = [gaussian_filter(f, 0.5) for f in pooled]
rs = [gaussian_filter(f, 0.5) for f in real]
e2 = {}
for i, nu in enumerate(("2.5", "3.0")):
    ex = bootstrap_excess(gs, rs, float(nu), seed=5100 + i)
    e2[nu] = {k: ex[k] for k in ("excess", "se", "ci95")}
pps = [per_parent_excess([gaussian_filter(f, 0.5) for f in s], rs, BLOCKS)
       for s in streams]
panel = {nu: [float(np.mean([p[nu][b] for p in pps])) for b in range(3)]
         for nu in ("2.5", "3.0")}
sign_cons = {nu: bool(all(v > 0 for v in panel[nu]) or
                      all(v < 0 for v in panel[nu])) for nu in panel}
ci_ok = all(e2[nu]["ci95"][0] <= 0 <= e2[nu]["ci95"][1]
            for nu in ("2.5", "3.0"))
over_smoothed = [nu for nu in ("2.5", "3.0") if e2[nu]["ci95"][1] < 0]
e2_pass = bool(ci_ok and not any(sign_cons.values()))
out["E2"] = {"excess": e2, "panel": panel, "sign_consistent": sign_cons,
             "over_smoothing_flip": over_smoothed, "pass": e2_pass}

# ---- E3 Minkowski -----------------------------------------------------------
t_mf, _ = judge_T(gs, rs, seed=20260845)
t_mf_native, _ = judge_T(pooled, real, seed=20260846)
mf_at_bar = bool(abs(t_mf - MF_BAR) <= MF_BAND)
e3_pass = bool(t_mf <= MF_BAR and not mf_at_bar)
out["E3"] = {"T_MF_declared": float(t_mf), "at_bar": mf_at_bar,
             "T_MF_native_descriptive": float(t_mf_native),
             "bar": MF_BAR, "band": MF_BAND, "pass": e3_pass}

# ---- E4 starlet (wl-env json) -----------------------------------------------
sp = os.path.join(RES, f"starlet_l1_stage3_{tag}.json")
e4_pass = None
if os.path.exists(sp):
    m = json.load(open(sp))
    e4_pass = bool(m["checks"]["gen_A"]["all_scored_pass"])
out["E4"] = {"pass": e4_pass}

# ---- supporting -------------------------------------------------------------
def T_of(gen, ref, fn):
    a = [fn(np.asarray(f, np.float64)) for f in gen]
    b = [fn(f) for f in ref]
    T, _ = tstat(*stack_profiles(a), *stack_profiles(b))
    return float(T)


NN_EDGES = np.array(json.load(open(os.path.join(
    RES, "placement_phase_a.json")))["convention"]["nn_edges"])
nns = [T_of(s, real, lambda f: nn_profile(f, k=K_REAL, edges=NN_EDGES))
       for s in streams]
out["supporting"] = {
    "parity_T": T_of(pooled, real,
                     lambda f: peak_parity_profile(f, k=K_REAL, level=1)),
    "nn_T_streams": nns, "nn_T_mean": float(np.mean(nns)),
    "nn_T_sd": float(np.std(nns, ddof=1)),
    "native_peaks": {nu: {k: bootstrap_excess(pooled, real, float(nu),
                                              seed=5200 + i)[k]
                          for k in ("excess", "se")}
                     for i, nu in enumerate(("1.0", "2.5", "3.0"))}}

# ---- branch (blind only) ----------------------------------------------------
if not quarantined:
    fails = [e for e, ok in (("E1", e1_pass), ("E2", e2_pass),
                             ("E3", e3_pass), ("E4", e4_pass)) if not ok]
    if not gates_ok:
        branch = "GATES"
    elif not fails:
        branch = "B3-PASS"
    elif e1_pass and e4_pass and len(fails) == 1 and fails[0] in ("E2", "E3"):
        branch = "B3-PARTIAL"
        if over_smoothed:
            out["a1_guard_named"] = f"over-smoothing flip at {over_smoothed}"
    else:
        branch = "B3-FAIL"
    out["adjudication"] = {"branch": branch, "failed_entries": fails,
                           "mf_at_bar": mf_at_bar}
with open(os.path.join(RES, f"stage3_{tag}_verdict.json"), "w") as f:
    json.dump(out, f, indent=1)
print(f"[{tag}] E1={e1_pass} E2={e2_pass} (excess "
      + " ".join(f"{nu}:{e2[nu]['excess']:+.2%}" for nu in e2)
      + f") E3={e3_pass} (T_MF {t_mf:.2f}, native {t_mf_native:.2f}) "
      f"E4={e4_pass} parity={out['supporting']['parity_T']:.2f} "
      f"nn={out['supporting']['nn_T_mean']:.2f}±{out['supporting']['nn_T_sd']:.2f}")
if not quarantined:
    print("ADJUDICATION:", json.dumps(out["adjudication"]))
else:
    print("QUARANTINED dry-run — no branch.")
