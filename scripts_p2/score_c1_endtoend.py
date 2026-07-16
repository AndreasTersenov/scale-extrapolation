"""C1 end-to-end scoring + mechanical bar adjudication (CPU, env.sh stack).

Scores the arms npz (gen_A/gen_B stacks) with the FROZEN production scorer
(scripts/measure_generated.couplings) and adjudicates the pre-registered bars
(log/2026-07-16-prereg-c1-sandbox.md) against the exact truth
(results_p2/sandbox_truth.json), together with the head-conditional results
(results_p2/c1_conditional_sandbox.json). Prints the verdict table and writes
results_p2/c1_verdict_sandbox.json. All rules are the prereg's, applied verbatim.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from measure_generated import couplings  # frozen production path

# normalized-convention truth (pre-readout amendment
# log/2026-07-16-c1-amendment-normconv.md): generated fields live in the per-tile
# normalized convention, so the exact-truth reference must share it
TRUTH = json.load(open(os.path.join(
    REPO, "results_p2", "sandbox_truth_normconv.json")))["truth"]
COND = json.load(open(os.path.join(REPO, "results_p2",
                                   "c1_conditional_sandbox.json")))
ARMS = np.load(os.path.join(REPO, "results_p2", "arms_c1_sandbox.npz"),
               allow_pickle=True)

BARS = {"var_slope": 0.10, "kurtosis": 0.15}
TRAINED = [2, 3, 4]


def check(metric, val, se, j):
    t = TRUTH[str(j)][metric]
    tse = TRUTH[str(j)][metric + "_se"]
    rel = abs(val - t) / abs(t)
    se_rel = float(np.hypot(se, tse) / abs(t))
    bar = max(BARS[metric], 3 * se_rel)
    return {"value": val, "se": se, "truth": t, "rel_err": rel, "bar": bar,
            "pass": bool(rel <= bar)}


out = {"levels": {}, "curve": {}, "amplitude": {}}
verdicts = []

# ---- end-to-end (frozen scorer) --------------------------------------------------
e2e = {}
for arm in ("A", "B"):
    fields = [np.asarray(f, dtype=np.float64) for f in ARMS[f"gen_{arm}"]]
    e2e[arm] = couplings(fields, [1, 2, 3, 4], n_boot=200, seed=0)
    # amplitude sanity for the DEGENERATE branch: detail_std vs the arms npz's own
    # normalized real stack (identical convention by construction; amendment §2)
    if "real_ref" not in out:
        real_fields = [np.asarray(f, dtype=np.float64) for f in ARMS["real"]]
        out["real_ref"] = couplings(real_fields, [1, 2, 3, 4], n_boot=50, seed=0)
    amp = {}
    for j in TRAINED:
        ds = e2e[arm][j]["detail_std"]
        tds = out["real_ref"][j]["detail_std"]
        amp[j] = {"gen": ds, "real": tds, "rel": abs(ds - tds) / tds}
    out["amplitude"][arm] = amp

print("=== C1 SANDBOX VERDICT TABLE (prereg rules verbatim) ===\n")
rows = []
for arm in ("A", "B"):
    for level, src in (("head-conditional", COND[arm]["final"]),
                       ("end-to-end", e2e[arm])):
        for j in TRAINED:
            s = src[str(j)] if isinstance(src, dict) and str(j) in src else src[j]
            for metric in ("var_slope", "kurtosis"):
                r = check(metric, s[metric], s[metric + "_se"], j)
                rows.append((arm, level, j, metric, r))
                out["levels"].setdefault(arm, {}).setdefault(level, {}).setdefault(
                    str(j), {})[metric] = r

hdr = f"{'arm':>3} {'level':>16} {'oct':>3} {'metric':>9} | {'gen':>8} {'truth':>8} {'rel':>6} {'bar':>6} | P/F"
print(hdr)
for arm, level, j, metric, r in rows:
    print(f"{arm:>3} {level:>16} {j:>3} {metric:>9} | {r['value']:8.3f} "
          f"{r['truth']:8.3f} {r['rel_err']:6.1%} {r['bar']:6.1%} | "
          f"{'PASS' if r['pass'] else 'FAIL'}")

# ---- checkpoint-curve collapse signature (running-peak rule) ----------------------
print("\ncheckpoint curve (oct-2 head-conditional var_slope):")
collapse = {}
for arm in ("A", "B"):
    curve = {int(k): v["var_slope"] for k, v in COND[arm]["curve_oct2"].items()}
    curve[20000] = COND[arm]["final"]["2"]["var_slope"]
    steps = sorted(curve)
    vals = [curve[s] for s in steps]
    peak = -np.inf
    fired = False
    for v in vals:
        peak = max(peak, v)
        if peak - v >= 0.10:
            fired = True
    collapse[arm] = {"curve": {str(s): curve[s] for s in steps},
                     "collapse_signature": fired}
    print(f"  arm {arm}: " + " ".join(f"{s//1000}k:{curve[s]:.3f}" for s in steps)
          + f"  -> collapse={'FIRES' if fired else 'no'}")
out["curve"] = collapse

# ---- branch adjudication (prereg precedence) --------------------------------------
def level_pass(arm, level, metric):
    return all(out["levels"][arm][level][str(j)][metric]["pass"] for j in TRAINED)

branch = {}
for arm in ("A", "B"):
    deg = any(out["amplitude"][arm][j]["rel"] > 0.25 for j in TRAINED)
    hc_disp = level_pass(arm, "head-conditional", "var_slope")
    e2_disp = level_pass(arm, "end-to-end", "var_slope")
    kurt_all = (level_pass(arm, "head-conditional", "kurtosis")
                and level_pass(arm, "end-to-end", "kurtosis"))
    coll = collapse[arm]["collapse_signature"] or not hc_disp
    if deg:
        b = "B-C1-DEG"
    elif coll:
        b = "B-C1-COLL"
    elif hc_disp and not e2_disp:
        b = "B-C1-REC"
    elif hc_disp and e2_disp and not kurt_all:
        b = "B-C1-TAILS"
    elif hc_disp and e2_disp and kurt_all:
        b = "B-C1-CAL"
    else:
        b = "other"
    branch[arm] = b
    print(f"\narm {arm} branch: {b}")
out["branch"] = branch
out["leg2_trigger_dispersion_pass"] = bool(
    all(level_pass(a, "head-conditional", "var_slope")
        and level_pass(a, "end-to-end", "var_slope") for a in ("A", "B")))
print(f"\ngowerstreet-leg trigger (dispersion bar, both arms, both levels): "
      f"{'PASS' if out['leg2_trigger_dispersion_pass'] else 'FAIL'}")

with open(os.path.join(REPO, "results_p2", "c1_verdict_sandbox.json"), "w") as f:
    json.dump(out, f, indent=1)
print("\nwrote results_p2/c1_verdict_sandbox.json")
