"""C1 gowerstreet leg — DESCRIPTIVE readout (no adjudication tonight, per prereg:
the morning reconvene adjudicates against the G-1c project bars).

Tables: per octave, both arms — end-to-end (frozen scorer, gen vs the npz's real
stack, like-for-like normalized convention) and head-conditional (generated detail
given real held-out coarse vs the REAL (w,c) response of the same fields).
Phase-1 comparators quoted from the frozen record for the eye.
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
from sandbox.haar import octave_wc_pooled
from sandbox.truth_stats import estimand_scalars

ARMS = np.load(os.path.join(REPO, "results_p2", "arms_c1_gowerstreet.npz"),
               allow_pickle=True)
COND = json.load(open(os.path.join(REPO, "results_p2",
                                   "c1_conditional_gowerstreet.json")))

out = {}

# real-field conditional baseline (same normalized heldout fields, estimand code)
real_fields = [np.asarray(f, dtype=np.float64) for f in ARMS["real"]]
real_cond = {}
for j in (1, 2, 3, 4):
    per_field = [octave_wc_pooled(f, j) for f in real_fields]
    w = np.concatenate([p[0] for p in per_field])
    c = np.concatenate([p[1] for p in per_field])
    s = estimand_scalars(w, c)
    rng = np.random.default_rng(0)
    boot = np.empty((200, 2))
    for t in range(200):
        idx = rng.integers(0, len(per_field), len(per_field))
        wb = np.concatenate([per_field[i][0] for i in idx])
        cb = np.concatenate([per_field[i][1] for i in idx])
        sb = estimand_scalars(wb, cb)
        boot[t] = (sb["var_slope"], sb["kurtosis"])
    real_cond[j] = {"var_slope": s["var_slope"],
                    "var_slope_se": float(np.nanstd(boot[:, 0], ddof=1)),
                    "kurtosis": s["kurtosis"],
                    "kurtosis_se": float(np.nanstd(boot[:, 1], ddof=1))}
out["real_conditional"] = real_cond

e2e_real = couplings(real_fields, [1, 2, 3, 4], n_boot=200, seed=0)
out["real_endtoend"] = e2e_real
print("=== C1 GOWERSTREET (DESCRIPTIVE) ===")
print("\nreal reference (64 heldout, normalized conv):")
for j in (1, 2, 3, 4):
    print(f"  oct{j}: cond vs={real_cond[j]['var_slope']:.3f}±"
          f"{real_cond[j]['var_slope_se']:.3f} kurt={real_cond[j]['kurtosis']:.2f}"
          f" | stack vs={e2e_real[j]['var_slope']:.3f} kurt={e2e_real[j]['kurtosis']:.2f}")

for arm in ("A", "B"):
    gen_fields = [np.asarray(f, dtype=np.float64) for f in ARMS[f"gen_{arm}"]]
    e2e = couplings(gen_fields, [1, 2, 3, 4], n_boot=200, seed=0)
    out[f"endtoend_{arm}"] = e2e
    print(f"\narm {arm}:")
    print(f"{'oct':>4} {'level':>16} | {'gen vs':>8} {'real vs':>8} {'ratio':>6} | {'gen k':>6} {'real k':>6}")
    for j in (1, 2, 3, 4):
        hc = COND[arm]["final"].get(str(j))
        rc = real_cond[j]
        if hc:
            print(f"{j:>4} {'head-cond':>16} | {hc['var_slope']:8.3f} "
                  f"{rc['var_slope']:8.3f} {hc['var_slope']/rc['var_slope']:6.2f} | "
                  f"{hc['kurtosis']:6.2f} {rc['kurtosis']:6.2f}")
        er = e2e_real[j]
        print(f"{j:>4} {'end-to-end':>16} | {e2e[j]['var_slope']:8.3f} "
              f"{er['var_slope']:8.3f} {e2e[j]['var_slope']/er['var_slope']:6.2f} | "
              f"{e2e[j]['kurtosis']:6.2f} {er['kurtosis']:6.2f}")

# checkpoint curve
print("\ncheckpoint curve (oct-2 head-conditional var_slope):")
for arm in ("A", "B"):
    curve = {int(k): v["var_slope"] for k, v in COND[arm]["curve_oct2"].items()}
    curve[20000] = COND[arm]["final"]["2"]["var_slope"]
    steps = sorted(curve)
    peak = -np.inf
    fired = False
    for v in (curve[s] for s in steps):
        peak = max(peak, v)
        if peak - v >= 0.10:
            fired = True
    out[f"curve_{arm}"] = {str(s): curve[s] for s in steps}
    out[f"collapse_{arm}"] = fired
    print(f"  arm {arm}: " + " ".join(f"{s//1000}k:{curve[s]:.3f}" for s in steps)
          + f"  -> collapse signature: {'FIRES' if fired else 'no'}")

print("\nphase-1 comparators (frozen record, NLL-head 4a): end-to-end oct-2 0.746 "
      "vs real 1.020 (-27%); sigma-channel own-ceiling 0.956-0.996; kurtosis oct-2 "
      "~3.0 vs 6.7")

with open(os.path.join(REPO, "results_p2", "c1_descriptive_gowerstreet.json"),
          "w") as f:
    json.dump(out, f, indent=1, default=str)
print("\nwrote results_p2/c1_descriptive_gowerstreet.json")
