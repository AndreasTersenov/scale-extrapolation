"""L1'' SCORE (env.sh; prereg 2026-07-30-l1doubleprime, R40).

--leg canary : kill check (var_slope rel > 50% at scored octaves 2-4);
               watched context: oct2 kurtosis. Writes l1pp_canary_score.json.
--leg main   : binding order — identity gates recorded; P-T FIRST (pooled
               3-stream C vs the COMMITTED interval [0.7192, 0.7920];
               l1pp_pt.json written before any peak number; directional
               line scored); peaks vs A3; must-not-regress; mechanical
               branch with the NULL split keyed on P-T. Writes
               l1pp_verdict.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts_p2"))

A3 = {"2.5": (0.15290, 0.0312), "3.0": (0.12474, 0.0280)}
BASELINE_C, SIGMA_INSTR = 0.7237, 0.0049
BARS = {"var_slope": 0.10, "kurtosis": 0.15}
K_REAL = 174


def canary():
    from measure_generated import couplings
    TRUTH = json.load(open(os.path.join(
        RES, "sandbox_truth_normconv.json")))["truth"]
    gen = [np.asarray(f, np.float64) for f in
           np.load(os.path.join(RES, "l1pp_canary_gen.npz"))["gen_canary"]]
    e2e = couplings(gen, [1, 2, 3, 4], n_boot=200, seed=0)
    rows, kill = {}, []
    for j in (2, 3, 4):
        rows[str(j)] = {}
        for metric in ("var_slope", "kurtosis"):
            t, tse = TRUTH[str(j)][metric], TRUTH[str(j)][metric + "_se"]
            v, se = e2e[j][metric], e2e[j][metric + "_se"]
            rel = abs(v - t) / abs(t)
            bar = max(BARS[metric], 3 * float(np.hypot(se, tse) / abs(t)))
            rows[str(j)][metric] = {"rel_err": rel, "bar": bar,
                                    "pass": bool(rel <= bar)}
            if metric == "var_slope" and rel > 0.50:
                kill.append(f"oct{j} var_slope rel {rel:.1%}")
    verdict = "CANARY KILL" if kill else "CANARY PASS"
    out = {"rows": rows, "kill": kill, "verdict": verdict,
           "watched_oct2_kurtosis": rows["2"]["kurtosis"],
           "replay_gate": json.load(open(os.path.join(
               RES, "l1pp_canary_sample.json")))["replay_gate"]}
    with open(os.path.join(RES, "l1pp_canary_score.json"), "w") as f:
        json.dump(out, f, indent=1)
    for j in ("2", "3", "4"):
        print(f"canary oct{j}: " + "  ".join(
            f"{m} rel {r['rel_err']:.1%} (bar {r['bar']:.1%}) "
            f"{'PASS' if r['pass'] else 'FAIL'}"
            for m, r in rows[j].items()))
    print(verdict, kill if kill else "")


def main_leg():
    from audit_peak_ci import bootstrap_excess
    from coloring_index import stack_coloring
    from measure_generated import couplings
    from parity_localization import coef_stats_profile, peak_parity_profile
    from placement_instruments import nn_profile, stack_profiles, tstat

    PT = json.load(open(os.path.join(RES, "l1pp_pt_prediction.json")))
    GEN = np.load(os.path.join(RES, "l1pp_main_gen.npz"))
    real = [np.asarray(f, np.float64) for f in
            np.load(os.path.join(RES, "arms_c1t_gowerstreet.npz"),
                    allow_pickle=True)["real"]]
    adj = [np.asarray(f, np.float64)
           for k in ("adj1", "adj2", "adj3") for f in GEN[k]]
    out = {"identity_gates": {
        "canary": json.load(open(os.path.join(
            RES, "l1pp_canary_sample.json")))["replay_gate"],
        "main": json.load(open(os.path.join(
            RES, "l1pp_main_sample.json")))["replay_gate"]}}

    # ---- P-T FIRST ----------------------------------------------------------
    c_pool, se_pool = stack_coloring(adj, 1, seed=2000)
    lo, hi = PT["adj"]["interval"]
    pt_lands = bool(lo <= c_pool <= hi)
    directional = bool((c_pool - BASELINE_C) >= 3 * SIGMA_INSTR)
    per_stream = {k: stack_coloring([np.asarray(f, np.float64)
                                     for f in GEN[k]], 1, seed=2001 + i)
                  for i, k in enumerate(("adj1", "adj2", "adj3",
                                         "oracle_oct1meas"))}
    out["pt"] = {"C_pooled": c_pool, "SE_pooled": se_pool,
                 "predicted": PT["adj"]["C_pred"],
                 "interval": [lo, hi], "PT_LANDS": pt_lands,
                 "directional_line": directional,
                 "oracle_pred": PT["oracle"]["C_pred"],
                 "per_stream": {k: {"C": v[0], "SE": v[1]}
                                for k, v in per_stream.items()}}
    with open(os.path.join(RES, "l1pp_pt.json"), "w") as f:
        json.dump(out["pt"], f, indent=1)
    print(f"P-T: C_pooled={c_pool:.4f}±{se_pool:.4f} vs pred "
          f"{PT['adj']['C_pred']:.4f} [{lo:.4f},{hi:.4f}] -> "
          f"{'LANDS' if pt_lands else 'FAILS'}; directional(+3sig)="
          f"{directional}")
    for k, v in per_stream.items():
        print(f"  C {k}: {v[0]:.4f}±{v[1]:.4f}")

    # ---- peaks vs A3 --------------------------------------------------------
    peaks = {"pooled": {nu: bootstrap_excess(adj, real, float(nu),
                                             seed=2100 + i)
                        for i, nu in enumerate(("1.0", "2.5", "3.0"))}}
    for i, k in enumerate(("adj1", "adj2", "adj3", "oracle_oct1meas")):
        gen = [np.asarray(f, np.float64) for f in GEN[k]]
        peaks[k] = {nu: bootstrap_excess(gen, real, float(nu),
                                         seed=2200 + 10 * i + j)
                    for j, nu in enumerate(("2.5", "3.0"))}
    out["peaks"] = peaks
    delta, percat, atbar = {}, {}, {}
    for nu in ("2.5", "3.0"):
        e, se = peaks["pooled"][nu]["excess"], peaks["pooled"][nu]["se"]
        r, sref = A3[nu]
        hyp = float(np.hypot(se, sref))
        delta[nu] = (r - e) / hyp
        atbar[nu] = bool(abs(e - (r - 2 * hyp)) <= 0.5 * se)
        percat[nu] = "IMPROVED" if delta[nu] >= 2 else "NULL"
    out["delta"] = delta

    # ---- must-not-regress ---------------------------------------------------
    e2e = couplings(adj, [1, 2, 3, 4], n_boot=200, seed=0)
    real_ref = couplings(real, [1, 2, 3, 4], n_boot=50, seed=0)
    catastrophe = []
    reg_rows = {}
    for j in (1, 2, 3, 4):
        reg_rows[str(j)] = {}
        for metric in ("var_slope", "kurtosis"):
            t, tse = real_ref[j][metric], real_ref[j][metric + "_se"]
            v, se = e2e[j][metric], e2e[j][metric + "_se"]
            rel = abs(v - t) / abs(t)
            bar = max(BARS[metric], 3 * float(np.hypot(se, tse) / abs(t)))
            reg_rows[str(j)][metric] = {"rel_err": rel, "bar": bar,
                                        "pass": bool(rel <= bar)}
            if metric == "var_slope" and j in (2, 3, 4) and rel > 0.50:
                catastrophe.append(f"oct{j} var_slope rel {rel:.1%}")

    def T_of(gen, ref, fn):
        a = [fn(np.asarray(f, np.float64)) for f in gen]
        b = [fn(f) for f in ref]
        T, _ = tstat(*stack_profiles(a), *stack_profiles(b))
        return T

    NN_EDGES = np.array(json.load(open(os.path.join(
        RES, "placement_phase_a.json")))["convention"]["nn_edges"])
    parity_T = T_of(adj, real, lambda f: peak_parity_profile(f, k=K_REAL,
                                                             level=1))
    nn_T = T_of(adj, real, lambda f: nn_profile(f, k=K_REAL, edges=NN_EDGES))
    coef_T = T_of(adj, real, lambda f: coef_stats_profile(f, octave=1))
    starlet = {}
    p = os.path.join(RES, "starlet_l1_l1pp_trained.json")
    if os.path.exists(p):
        m = json.load(open(p))
        starlet = {a: m["checks"][a]["all_scored_pass"]
                   for a in m.get("checks", {})}
    gates_ok = all(g["corr_min"] >= 0.99
                   for g in out["identity_gates"].values())
    starlet_ok = all(starlet.values()) if starlet else None
    regression = bool(catastrophe) or parity_T >= 3 or \
        (starlet_ok is False) or not gates_ok
    out["must_not_regress"] = {
        "marginals": reg_rows, "catastrophe": catastrophe,
        "parity_T": parity_T, "coef_T_oct1": coef_T,
        "nn_T_watched": nn_T, "starlet": starlet,
        "identity_ok": gates_ok, "regression": regression}

    # ---- branch (mechanical; NULL keyed on P-T) -----------------------------
    flipped = [nu for nu in ("2.5", "3.0")
               if peaks["pooled"][nu]["ci95"][1] < 0]
    cured = all(peaks["pooled"][nu]["ci95"][0] <= 0 <=
                peaks["pooled"][nu]["ci95"][1] for nu in ("2.5", "3.0"))
    mixed = percat["2.5"] != percat["3.0"]
    if flipped:
        branch = {"branch": "N-FLIPPED", "at": flipped}
    elif regression:
        branch = {"branch": "N-REGRESSED"}
    elif cured:
        branch = {"branch": "N-CURED"}
    elif mixed or any(atbar.values()):
        branch = {"branch": "MIXED/AT-BAR", "governing": "NULL",
                  "at_bar": atbar, "per_nu": percat,
                  "note": "worse governs; one fresh-PRNG disambiguation "
                          "pre-authorized"}
    elif all(percat[nu] == "IMPROVED" for nu in ("2.5", "3.0")):
        branch = {"branch": "N-IMPROVED"}
    else:
        branch = {"branch": "N-NULL-PT-CONFIRMED" if pt_lands
                  else "N-NULL-PT-FAILED"}
    branch["pt_lands"] = pt_lands
    out["adjudication"] = branch

    with open(os.path.join(RES, "l1pp_verdict.json"), "w") as f:
        json.dump(out, f, indent=1)
    for k in ("pooled", "adj1", "adj2", "adj3", "oracle_oct1meas"):
        row = "  ".join(
            f"nu={nu}: {peaks[k][nu]['excess']:+.2%}±{peaks[k][nu]['se']:.2%}"
            for nu in sorted(peaks[k]))
        print(f"peaks {k:>16}: {row}")
    print(f"delta: nu2.5 {delta['2.5']:.2f} nu3.0 {delta['3.0']:.2f}")
    print(f"regress: catastrophe={catastrophe} parity_T={parity_T:.2f} "
          f"nn_T={nn_T:.2f} coef_T={coef_T:.2f} starlet={starlet} "
          f"identity_ok={gates_ok}")
    print("ADJUDICATION:", json.dumps(branch))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", choices=["canary", "main"], required=True)
    canary() if ap.parse_args().leg == "canary" else main_leg()
