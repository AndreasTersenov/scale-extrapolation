"""Phase-3 Stage-0 SCORE phase (env.sh stack; prereg 62db5f0 + amendment A1).

The full battery on the F2-sampled Stage-D substrate (stage0_p3_gen.npz):
  1. Peaks (PRIMARY): bootstrap excesses at nu {1, 2.5, 3} (5000-boot,
     instruments imported from the frozen audit_peak_ci.py) + per-parent
     panel (blocks [(0,10),(10,21),(21,32)]). Adjudicating: arm A, nu 2.5/3.0
     edge excesses vs the committed references (READ AT RUNTIME from
     audit_peak_ci.json, stage_d leg — R12 by construction).
  2. Spacing (descriptive, WATCHED): nn_profile T, K_real=174, frozen edges
     from placement_phase_a.json; parity_T and T_coef_oct1 as context.
  3. Marginal suite (descriptive context): e2e couplings octaves 1-4 vs the
     real test reference (score_stageD convention: real n_boot=50, gen 200;
     bars 10%/15% with 3*SE floors). Catastrophe gate: arm-A var_slope
     rel err > 50% at any SCORED octave (2-4, the prereg's set; octave 1 is
     beyond-edge, reported descriptively only).
  4. Trained-scales leg: peaks CIs for the COMMITTED f2_test_gen.npz
     F2_gowA_e2e (and the pre-F2 committed gen_A as context) vs the
     gowerstreet test tiles. No new generation.

Branch adjudication (mechanical, the prereg table + A1; pure function
`adjudicate_stage0` — unit-tested in tests/test_stage0_adjudicate.py BEFORE
this script first runs, per the pcure verdict-logic lesson):
  gates    : G1/G2 identity-gate failure (from stage0_p3_sample.json) or
             marginal catastrophe.
  S0-FLIPPED (A1): either nu's 95% CI entirely below 0.
  S0-CLOSED : both 95% CIs include 0 AND per-parent panel shows no
             sign-consistent excess (all 3 blocks positive at an
             adjudicating nu = sign-consistent).
  S0-SHRUNK : not CLOSED; both e < r/2 AND both Delta = (r-e)/hypot(s,s_ref)
             >= 2.
  S0-UNCHANGED: otherwise (both-threshold).
  MIXED/AT-BAR: per-nu categories disagree, OR any |e - r/2| <= 0.5*s
             (#11 band). Worse category governs sequencing; ONE fresh-PRNG
             regeneration disambiguation pre-authorized.

Writes results_p2/stage0_p3_verdict.json.
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

NUS_ADJ = ("2.5", "3.0")
BLOCKS = [(0, 10), (10, 21), (21, 32)]
BARS = {"var_slope": 0.10, "kurtosis": 0.15}
K_REAL = 174
CAT_ORDER = {"CLOSED": 0, "SHRUNK": 1, "UNCHANGED": 2}


def adjudicate_stage0(ex, refs, per_parent):
    """Pure branch logic. ex[nu] = {excess, se, ci95}; refs[nu] = (r, s_ref);
    per_parent[nu] = [3 block excesses] (arm A). Returns dict with branch,
    governing category, and every intermediate the readout must show."""
    flipped = {nu: ex[nu]["ci95"][1] < 0 for nu in NUS_ADJ}
    if any(flipped.values()):
        return {"branch": "S0-FLIPPED", "governing": "FLIPPED",
                "flipped_at": [nu for nu in NUS_ADJ if flipped[nu]],
                "note": "A1: single fresh-PRNG disambiguation, then if the "
                        "deficit persists STOP — its own finding"}
    percat, detail = {}, {}
    for nu in NUS_ADJ:
        e, s = ex[nu]["excess"], ex[nu]["se"]
        r, sref = refs[nu]
        lo, hi = ex[nu]["ci95"]
        delta = (r - e) / float(np.hypot(s, sref))
        closed = lo <= 0 <= hi
        shrunk = (not closed) and (e < r / 2) and (delta >= 2)
        percat[nu] = ("CLOSED" if closed else
                      "SHRUNK" if shrunk else "UNCHANGED")
        detail[nu] = {"excess": e, "se": s, "ci95": [lo, hi], "ref": r,
                      "ref_se": sref, "delta_reduction": delta,
                      "at_bar": bool(abs(e - r / 2) <= 0.5 * s),
                      "per_nu_category": percat[nu]}
    sign_consistent = {nu: bool(all(v > 0 for v in per_parent[nu]))
                       for nu in NUS_ADJ}
    at_bar = any(detail[nu]["at_bar"] for nu in NUS_ADJ)
    mixed = percat[NUS_ADJ[0]] != percat[NUS_ADJ[1]]
    worse = max(percat.values(), key=lambda c: CAT_ORDER[c])
    if mixed or at_bar:
        return {"branch": "MIXED/AT-BAR", "governing": worse,
                "per_nu": detail, "sign_consistent_panel": sign_consistent,
                "note": "worse category governs sequencing; ONE fresh-PRNG "
                        "regeneration disambiguation pre-authorized (#11)"}
    if worse == "CLOSED":
        if any(sign_consistent.values()):
            # CIs include 0 but the panel is sign-consistent: CLOSED's second
            # condition fails; the SHRUNK rule is tested next (prereg cascade)
            both_shrunk = all(
                detail[nu]["excess"] < refs[nu][0] / 2
                and detail[nu]["delta_reduction"] >= 2 for nu in NUS_ADJ)
            cat = "S0-SHRUNK" if both_shrunk else "S0-UNCHANGED"
            return {"branch": cat, "governing": cat.replace("S0-", ""),
                    "per_nu": detail, "sign_consistent_panel": sign_consistent,
                    "note": "panel blocked CLOSED (sign-consistent excess)"}
        return {"branch": "S0-CLOSED", "governing": "CLOSED",
                "per_nu": detail, "sign_consistent_panel": sign_consistent}
    return {"branch": f"S0-{worse}", "governing": worse,
            "per_nu": detail, "sign_consistent_panel": sign_consistent}


def main():
    from audit_peak_ci import bootstrap_excess, per_parent_excess
    from measure_generated import couplings
    from parity_localization import coef_stats_profile, peak_parity_profile
    from placement_instruments import nn_profile, stack_profiles, tstat

    AUD = json.load(open(os.path.join(RES, "audit_peak_ci.json")))
    refs = {nu: (AUD["legs"]["stage_d"]["A"][nu]["excess"],
                 AUD["legs"]["stage_d"]["A"][nu]["se"]) for nu in NUS_ADJ}
    SAMPLE = json.load(open(os.path.join(RES, "stage0_p3_sample.json")))
    GEN = np.load(os.path.join(RES, "stage0_p3_gen.npz"), allow_pickle=True)
    real = [np.asarray(f, np.float64) for f in GEN["real"]]

    out = {"convention": "instruments frozen: audit_peak_ci.py peaks, "
                         "placement_instruments nn (K=174, phase-A edges), "
                         "score_stageD marginal bars; refs read at runtime "
                         "from audit_peak_ci.json stage_d/A",
           "sample_gates": SAMPLE["gates"], "edge": {}, "trained": {},
           "descriptive": {}}

    # ---- 1. edge peaks + per-parent panel -----------------------------------
    for i, arm in enumerate(("A", "B")):
        gen = [np.asarray(f, np.float64) for f in GEN[f"gen_{arm}"]]
        out["edge"][arm] = {
            "peaks": {nu: bootstrap_excess(gen, real, float(nu), seed=300 + i)
                      for nu in ("1.0", "2.5", "3.0")},
            "per_parent": per_parent_excess(gen, real, BLOCKS)}

    # ---- 2/3. nn + parity + coef T; marginal suite --------------------------
    NN_EDGES = np.array(json.load(open(os.path.join(
        RES, "placement_phase_a.json")))["convention"]["nn_edges"])

    def T_of(gen, fn):
        a = [fn(np.asarray(f, np.float64)) for f in gen]
        b = [fn(f) for f in real]
        T, _ = tstat(*stack_profiles(a), *stack_profiles(b))
        return T

    real_ref = couplings(real, [1, 2, 3, 4], n_boot=50, seed=0)
    catastrophe = []
    for arm in ("A", "B"):
        gen = [np.asarray(f, np.float64) for f in GEN[f"gen_{arm}"]]
        e2e = couplings(gen, [1, 2, 3, 4], n_boot=200, seed=0)
        rows = {}
        for j in (1, 2, 3, 4):
            rows[str(j)] = {}
            for metric in ("var_slope", "kurtosis"):
                t, tse = real_ref[j][metric], real_ref[j][metric + "_se"]
                v, se = e2e[j][metric], e2e[j][metric + "_se"]
                rel = abs(v - t) / abs(t)
                bar = max(BARS[metric], 3 * float(np.hypot(se, tse) / abs(t)))
                rows[str(j)][metric] = {"value": v, "real": t, "rel_err": rel,
                                        "bar": bar, "pass": bool(rel <= bar)}
                if (arm == "A" and metric == "var_slope" and j in (2, 3, 4)
                        and rel > 0.50):
                    catastrophe.append(f"oct{j} var_slope rel {rel:.1%}")
        out["descriptive"][arm] = {
            "marginals_e2e": rows,
            "nn_T": T_of(gen, lambda f: nn_profile(f, k=K_REAL,
                                                   edges=NN_EDGES)),
            "parity_T": T_of(gen, lambda f: peak_parity_profile(f, k=K_REAL,
                                                                level=1)),
            "T_coef_oct1": T_of(gen, lambda f: coef_stats_profile(f, octave=1))}

    # ---- 4. trained-scales leg (committed maps only) ------------------------
    gow = np.load(os.path.join(RES, "arms_c1t_gowerstreet.npz"),
                  allow_pickle=True)
    F2G = np.load(os.path.join(RES, "f2_test_gen.npz"), allow_pickle=True)
    greal = [np.asarray(f, np.float64) for f in gow["real"]]
    for i, (label, fields) in enumerate(
            (("F2_gowA", F2G["F2_gowA_e2e"]), ("before_gowA", gow["gen_A"]))):
        gen = [np.asarray(f, np.float64) for f in fields]
        out["trained"][label] = {
            nu: bootstrap_excess(gen, greal, float(nu), seed=400 + i)
            for nu in ("1.0", "2.5", "3.0")}

    # ---- adjudication -------------------------------------------------------
    if catastrophe:
        out["adjudication"] = {"branch": "GATE-CATASTROPHE",
                               "failing": catastrophe}
    else:
        ex = {nu: out["edge"]["A"]["peaks"][nu] for nu in NUS_ADJ}
        out["adjudication"] = adjudicate_stage0(
            ex, refs, out["edge"]["A"]["per_parent"])

    # starlet legs read if already scored (wl-challenge-env script)
    for tag in ("edge", "trained"):
        p = os.path.join(RES, f"starlet_l1_stage0_{tag}.json")
        if os.path.exists(p):
            m = json.load(open(p))
            out["descriptive"][f"starlet_{tag}"] = {
                arm: m["checks"][arm]["all_scored_pass"]
                for arm in m.get("checks", {})}

    with open(os.path.join(RES, "stage0_p3_verdict.json"), "w") as f:
        json.dump(out, f, indent=1)

    for arm in ("A", "B"):
        row = "  ".join(
            f"nu={nu}: {out['edge'][arm]['peaks'][nu]['excess']:+.1%}"
            f"±{out['edge'][arm]['peaks'][nu]['se']:.1%}"
            f" ci95[{out['edge'][arm]['peaks'][nu]['ci95'][0]:+.1%},"
            f"{out['edge'][arm]['peaks'][nu]['ci95'][1]:+.1%}]"
            for nu in ("1.0", "2.5", "3.0"))
        print(f"edge {arm}: {row}")
        for nu in NUS_ADJ:
            pp = out["edge"][arm]["per_parent"][nu]
            print(f"  per-parent nu={nu}: [" +
                  ", ".join(f"{v:+.0%}" for v in pp) + "]")
        d = out["descriptive"][arm]
        print(f"  nn_T={d['nn_T']:.2f} parity_T={d['parity_T']:.2f} "
              f"T_coef1={d['T_coef_oct1']:.2f}")
    for label in out["trained"]:
        row = "  ".join(
            f"nu={nu}: {out['trained'][label][nu]['excess']:+.1%}"
            f"±{out['trained'][label][nu]['se']:.1%}"
            for nu in ("2.5", "3.0"))
        print(f"trained {label}: {row}")
    print("ADJUDICATION:", json.dumps(out["adjudication"], indent=1,
                                      default=str))
    print("wrote stage0_p3_verdict.json")


if __name__ == "__main__":
    main()
