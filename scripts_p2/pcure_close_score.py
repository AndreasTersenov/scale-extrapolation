"""R35 orders 7a/7b SCORE phase (env.sh): the program-close verdicts.

(a) 32x joint-window confirmation at n32's joint pick: marginal bars (hc+e2e,
    octaves 2-4, c1t convention) + test-side T_coef oct-1 < 3, with the
    ledger-#11 ambiguity band (0.15 around the 3.0 defect bar; 10% of bar on
    marginal entries) — inside-band landings reported AT-THE-BAR.
(b) seed mini-ensemble: e (raw T_coef oct-1 at each seed's own caged pick),
    n=5 with the committed seed-0 (15.3) and seed-1 (6.861); descriptive.

Writes results_p2/pcure_confirm_verdict.json + results_p2/pcure_seed_ensemble.json.
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
from parity_localization import coef_stats_profile
from placement_instruments import stack_profiles, tstat

TRUTH = json.load(open(os.path.join(RES, "sandbox_truth_normconv.json")))["truth"]
BARS = {"var_slope": 0.10, "kurtosis": 0.15}
BAND_DEFECT, BAND_MARGIN = 0.15, 0.10
tiles = np.load(os.path.join(REPO, "data_cache", "tiles_sandbox.npz"))["sandbox"]
test = tiles[-32:].astype(np.float64)


def T_of(gen, octave=1):
    a = [coef_stats_profile(np.asarray(f, np.float64), octave) for f in gen]
    b = [coef_stats_profile(f, octave) for f in test]
    T, _ = tstat(*stack_profiles(a), *stack_profiles(b))
    return T


def check(metric, val, se, j):
    t, tse = TRUTH[str(j)][metric], TRUTH[str(j)][metric + "_se"]
    rel = abs(val - t) / abs(t)
    bar = max(BARS[metric], 3 * float(np.hypot(se, tse) / abs(t)))
    at_bar = abs(rel - bar) <= BAND_MARGIN * bar
    return {"value": val, "rel_err": rel, "bar": bar,
            "pass": bool(rel <= bar), "at_bar_band": bool(at_bar)}


# ---- (a) confirmation -------------------------------------------------------
HC = json.load(open(os.path.join(RES, "pcure_confirm_hc.json")))
gen = np.load(os.path.join(RES, "pcure_confirm_gen.npz"))["gen_A"]
e2e = couplings([np.asarray(f, np.float64) for f in gen], [1, 2, 3, 4],
                n_boot=200, seed=0)
rows, all_pass, any_at_bar = {}, True, False
for level, src in (("head-conditional", HC["hc"]), ("end-to-end", e2e)):
    for j in (2, 3, 4):
        s = src[str(j)] if str(j) in src else src[j]
        for metric in ("var_slope", "kurtosis"):
            r = check(metric, s[metric], s[metric + "_se"], j)
            rows.setdefault(level, {}).setdefault(str(j), {})[metric] = r
            all_pass &= r["pass"]
            any_at_bar |= r["at_bar_band"]
t1 = T_of(gen, 1)
defect_at_bar = abs(t1 - 3.0) <= BAND_DEFECT
# Verdict per the PRE-STATED rule (the first emission of this script used a
# looser at-bar trigger — any pass-side at-bar marginal — which the
# pre-statement does not support; corrected before adjudication, disclosed
# in the readout): the band matters only where it can change the outcome.
fail_side_at_bar = any(
    r["at_bar_band"] and not r["pass"]
    for lv in rows for j in rows[lv] for r in rows[lv][j].values())
if defect_at_bar or fail_side_at_bar:
    verdict = "AT-THE-BAR (disambiguate on repl64, pre-authorized)"
elif all_pass and t1 < 3.0:
    verdict = "CONFIRMED"
else:
    verdict = "NOT-CONFIRMED"
fails = [f"{lv} oct{j} {m} rel {r['rel_err']:.1%} bar {r['bar']:.1%}"
         for lv in rows for j in rows[lv] for m, r in rows[lv][j].items()
         if not r["pass"]]
out_a = {"joint_pick": HC["joint_pick"], "T_coef_oct1_test": t1,
         "defect_at_bar_band": bool(defect_at_bar),
         "marginals": rows, "marginals_all_pass": bool(all_pass),
         "any_marginal_at_bar_band": bool(any_at_bar),
         "failing": fails, "verdict": verdict}
with open(os.path.join(RES, "pcure_confirm_verdict.json"), "w") as f:
    json.dump(out_a, f, indent=1)
print(f"(a) n32@{HC['joint_pick']}: T_coef={t1:.2f} marginals_all_pass="
      f"{all_pass} at_bar={any_at_bar or defect_at_bar} -> {verdict}")
if fails:
    print("    failing:", "; ".join(fails))

# ---- (b) seed ensemble ------------------------------------------------------
ens = {"0_committed": {"e": 15.3, "pick": 7500},
       "1": {"e": 6.861, "pick": 5000}}
for seed, arm in ((2, "n1s2"), (3, "n1s3"), (4, "n1s4")):
    d = np.load(os.path.join(RES, f"arms_pcure_{arm}.npz"), allow_pickle=True)
    sel = json.load(open(os.path.join(RES, f"pcure_selection_{arm}.json")))
    ens[str(seed)] = {"e": T_of(d["gen_A"], 1),
                      "pick": sel["A"]["selected_step"]}
vals = np.array([v["e"] for v in ens.values()])
out_b = {"ensemble": ens, "mean": float(vals.mean()),
         "sd": float(vals.std(ddof=1)),
         "range": [float(vals.min()), float(vals.max())]}
with open(os.path.join(RES, "pcure_seed_ensemble.json"), "w") as f:
    json.dump(out_b, f, indent=1)
print("(b) seed ensemble:",
      {k: (round(v['e'], 2), v['pick']) for k, v in ens.items()},
      f"mean={out_b['mean']:.1f} sd={out_b['sd']:.1f}")
