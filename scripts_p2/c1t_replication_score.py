"""R18 replication SCORE phase (env.sh stack — pywt/frozen scorer).

Reads the sample phase's outputs (c1t_repl64_gen.npz + c1t_repl64_hc.json),
scores end-to-end with the frozen production scorer, adjudicates the identical
bars on all 24 checks, writes results_p2/c1t_repl64.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from measure_generated import couplings

TRUTH = json.load(open(os.path.join(REPO, "results_p2",
                                    "sandbox_truth_normconv.json")))["truth"]
HC = json.load(open(os.path.join(REPO, "results_p2", "c1t_repl64_hc.json")))
GEN = np.load(os.path.join(REPO, "results_p2", "c1t_repl64_gen.npz"))
BARS = {"var_slope": 0.10, "kurtosis": 0.15}
TRAINED = [2, 3, 4]


def check(metric, val, se, j):
    t, tse = TRUTH[str(j)][metric], TRUTH[str(j)][metric + "_se"]
    rel = abs(val - t) / abs(t)
    bar = max(BARS[metric], 3 * float(np.hypot(se, tse) / abs(t)))
    return {"value": val, "se": se, "truth": t, "rel_err": rel, "bar": bar,
            "pass": bool(rel <= bar)}


out = {"meta": HC["meta"], "levels": {}}
rows = []
for arm in ("A", "B"):
    fields = [np.asarray(f, dtype=np.float64) for f in GEN[f"gen_{arm}"]]
    e2e = couplings(fields, [1, 2, 3, 4], n_boot=200, seed=0)
    for level, src in (("head-conditional", HC["hc_full"][arm]),
                       ("end-to-end", e2e)):
        for j in TRAINED:
            s = src[str(j)] if isinstance(src, dict) and str(j) in src else src[j]
            for metric in ("var_slope", "kurtosis"):
                r = check(metric, s[metric], s[metric + "_se"], j)
                rows.append((arm, level, j, metric, r))
                out["levels"].setdefault(arm, {}).setdefault(level, {}).setdefault(
                    str(j), {})[metric] = r
    out.setdefault("e2e_oct1", {})[arm] = {k_: e2e[1][k_] for k_ in
                                           ("var_slope", "kurtosis")}

print("=== R18 REPLICATION VERDICT (64 fresh fields, frozen ckpts) ===")
n_pass = 0
for arm, level, j, metric, r in rows:
    n_pass += r["pass"]
    print(f"{arm:>3} {level:>16} {j:>3} {metric:>9} | {r['value']:8.3f} "
          f"{r['truth']:8.3f} {r['rel_err']:6.1%} {r['bar']:6.1%} | "
          f"{'PASS' if r['pass'] else 'FAIL'}")
out["n_pass"] = n_pass
out["all_pass"] = bool(n_pass == len(rows))
print(f"\n{n_pass}/{len(rows)} bars pass -> "
      f"{'REPLICATED (C1T-CAL confirmed)' if out['all_pass'] else 'PARTIAL'}")
with open(os.path.join(REPO, "results_p2", "c1t_repl64.json"), "w") as f:
    json.dump(out, f, indent=1)
print("wrote results_p2/c1t_repl64.json")
