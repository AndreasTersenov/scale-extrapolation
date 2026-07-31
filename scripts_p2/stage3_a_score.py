"""Stage-3 (a) robustness-table scorer (env.sh; prereg af34d93; A5 order
per row: gates -> C + P-T band -> entries). ADDITIVE file — no shared
module touched (A6). Rows: seed0 (the SHIPPED config: committed l1pp adj
streams + committed L1'' P-T band; survivorship labeled), seed1, seed2
(fresh, each through its own deployment-recipe artifacts).

Row PASS rule (prereg): marginals (hc+e2e octaves 2-4, standing bars) AND
starlet trained leg AND parity_T < 3 AND declared-resolution smoothed-peak
rule AND C in the seed's own P-T band. Branch: R-ROBUST iff all three rows
pass; else R-SEED-FRAGILE with entries NAMED (#14 conditional meanings).
nn: pooled per-stream mean +- sd, DESCRIPTIVE. Native peaks descriptive.

Writes results_p2/stage3_a_table.json.
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
from parity_localization import peak_parity_profile
from placement_instruments import nn_profile, stack_profiles, tstat

BARS = {"var_slope": 0.10, "kurtosis": 0.15}
BLOCKS = [(0, 10), (10, 21), (21, 32)]
K_REAL = 174
TRAINED = (2, 3, 4)

real = [np.asarray(f, np.float64) for f in
        np.load(os.path.join(RES, "arms_c1t_gowerstreet.npz"),
                allow_pickle=True)["real"]]
real_ref = couplings(real, [1, 2, 3, 4], n_boot=50, seed=0)
NN_EDGES = np.array(json.load(open(os.path.join(
    RES, "placement_phase_a.json")))["convention"]["nn_edges"])


def T_of(gen, fn):
    a = [fn(np.asarray(f, np.float64)) for f in gen]
    b = [fn(f) for f in real]
    T, _ = tstat(*stack_profiles(a), *stack_profiles(b))
    return float(T)


def row(tag, streams, sel_json, pick, pt_interval, gates_ok, starlet_key):
    pooled = [f for s in streams for f in s]
    r = {"pick": pick, "gates_ok": bool(gates_ok)}
    # A5: C + P-T first
    c, se = stack_coloring(pooled, 1, seed=6000 + hash(tag) % 100)
    c_ok = bool(pt_interval[0] <= c <= pt_interval[1])
    r["C"] = {"pooled": c, "SE": se, "band": list(pt_interval),
              "in_band": c_ok}
    # marginals
    sel = json.load(open(sel_json))
    e2e = couplings(pooled, [1, 2, 3, 4], n_boot=200, seed=0)
    marg_ok, rows = True, {}
    for level, src_by_j in (("head-conditional",
                             {j: sel["A"]["final"][str(j)] for j in TRAINED}),
                            ("end-to-end", {j: e2e[j] for j in TRAINED})):
        for j in TRAINED:
            src = src_by_j[j]
            for metric in ("var_slope", "kurtosis"):
                t, tse = real_ref[j][metric], real_ref[j][metric + "_se"]
                v, s_ = src[metric], src[metric + "_se"]
                rel = abs(v - t) / abs(t)
                bar = max(BARS[metric],
                          3 * float(np.hypot(s_, tse) / abs(t)))
                ok = bool(rel <= bar)
                rows.setdefault(level, {}).setdefault(
                    str(j), {})[metric] = {"rel_err": rel, "bar": bar,
                                           "pass": ok}
                marg_ok &= ok
    r["marginals"] = {"rows": rows, "pass": bool(marg_ok)}
    # smoothed peaks (declared-resolution rule)
    gs = [gaussian_filter(np.asarray(f, np.float64), 0.5) for f in pooled]
    rs = [gaussian_filter(f, 0.5) for f in real]
    pk = {nu: bootstrap_excess(gs, rs, float(nu),
                               seed=6100 + hash(tag) % 100 + i)
          for i, nu in enumerate(("2.5", "3.0"))}
    pps = [per_parent_excess([gaussian_filter(np.asarray(f, np.float64),
                                              0.5) for f in s], rs, BLOCKS)
           for s in streams]
    panel = {nu: [float(np.mean([p[nu][b] for p in pps])) for b in range(3)]
             for nu in ("2.5", "3.0")}
    sc = {nu: bool(all(v > 0 for v in panel[nu]) or
                   all(v < 0 for v in panel[nu])) for nu in panel}
    pk_ok = bool(all(pk[nu]["ci95"][0] <= 0 <= pk[nu]["ci95"][1]
                     for nu in pk) and not any(sc.values()))
    r["smoothed_peaks"] = {
        nu: {"excess": pk[nu]["excess"], "se": pk[nu]["se"],
             "ci95": pk[nu]["ci95"]} for nu in pk}
    r["smoothed_peaks"]["panel"] = panel
    r["smoothed_peaks"]["pass"] = pk_ok
    # parity + starlet + descriptive
    par = T_of(pooled, lambda f: peak_parity_profile(f, k=K_REAL, level=1))
    sp = os.path.join(RES, starlet_key)
    st_ok = None
    if os.path.exists(sp):
        st_ok = bool(json.load(open(sp))["checks"]["gen_A"]
                     ["all_scored_pass"])
    nns = [T_of(s, lambda f: nn_profile(f, k=K_REAL, edges=NN_EDGES))
           for s in streams]
    r["parity_T"] = par
    r["starlet_pass"] = st_ok
    r["nn"] = {"streams": nns, "mean": float(np.mean(nns)),
               "sd": float(np.std(nns, ddof=1))}
    r["native_peaks_desc"] = {
        nu: {k: bootstrap_excess(pooled, real, float(nu),
                                 seed=6200 + i)[k] for k in ("excess", "se")}
        for i, nu in enumerate(("2.5", "3.0"))}
    fails = [n for n, ok in (("marginals", marg_ok), ("starlet", st_ok),
                             ("parity", par < 3),
                             ("declared_res_peaks", pk_ok),
                             ("C_in_band", c_ok), ("gates", gates_ok))
             if not ok]
    r["failed_entries"] = fails
    r["row_pass"] = not fails
    return r


table = {}
# seed 0 — the shipped row (committed artifacts; survivorship labeled)
LPP = np.load(os.path.join(RES, "l1pp_main_gen.npz"))
PT0 = json.load(open(os.path.join(RES, "l1pp_pt_prediction.json")))
table["seed0_shipped"] = row(
    "seed0",
    [[np.asarray(f, np.float64) for f in LPP[k]]
     for k in ("adj1", "adj2", "adj3")],
    os.path.join(RES, "c1t_selection_gowerstreet.json"), 16000,
    PT0["adj"]["interval"], True, "starlet_l1_l1pp_trained.json")
table["seed0_shipped"]["survivorship_note"] = (
    "the campaign's pick; fresh seeds 1-2 are the unbiased draws")
for seed, pick in (("seed1", 3500), ("seed2", 5500)):
    G = np.load(os.path.join(RES, f"stage3_{seed}_final.npz"))
    W = json.load(open(os.path.join(RES, f"stage3_{seed}_white.json")))
    PT = json.load(open(os.path.join(RES, f"stage3_{seed}_pt.json")))
    gates_ok = (W["determinism_gate_maxabs"] == 0.0
                and PT["cal_history"][-1] <= 0.03)
    table[seed] = row(
        seed,
        [[np.asarray(f, np.float64) for f in G[k]]
         for k in ("final1", "final2", "final3")],
        os.path.join(RES, f"stage3_{seed}_selection.json"), pick,
        PT["interval"], gates_ok, f"starlet_l1_stage3a_{seed}.json")

fails = {k: v["failed_entries"] for k, v in table.items()
         if v["failed_entries"]}
branch = "R-ROBUST" if not fails else "R-SEED-FRAGILE"
out = {"table": table, "adjudication": {"branch": branch,
                                        "named_failures": fails}}
with open(os.path.join(RES, "stage3_a_table.json"), "w") as f:
    json.dump(out, f, indent=1)
for k, v in table.items():
    print(f"{k}: pick@{v['pick']} C={v['C']['pooled']:.4f}"
          f"{'IN' if v['C']['in_band'] else 'OUT'} "
          f"marg={v['marginals']['pass']} "
          f"peaks0.5={v['smoothed_peaks']['pass']} "
          f"(2.5:{v['smoothed_peaks']['2.5']['excess']:+.2%} "
          f"3.0:{v['smoothed_peaks']['3.0']['excess']:+.2%}) "
          f"parity={v['parity_T']:.2f} starlet={v['starlet_pass']} "
          f"nn={v['nn']['mean']:.2f}±{v['nn']['sd']:.2f} "
          f"-> {'PASS' if v['row_pass'] else 'FAIL ' + str(v['failed_entries'])}")
print("ADJUDICATION:", json.dumps(out["adjudication"]))
