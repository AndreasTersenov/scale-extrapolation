"""C1-t gowerstreet leg: DESCRIPTIVE readout (reconvene adjudicates).

Real field — no exact truth. Compares the selected-checkpoint generation against
the REAL TEST fields (same convention, frozen production scorer), and answers the
standing prediction P(e2e oct-2 kurtosis deficit halves vs C1) by computing each
arm's relative deficit against its own real reference: C1-t from
arms_c1t_gowerstreet.npz (test-32), C1 from arms_c1_gowerstreet.npz (its 64
held-out; reference sets differ by construction — stated, not hidden).
Writes results_p2/c1t_gow_descriptive.json.
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
from sandbox.truth_stats import tail_q999

SEL = json.load(open(os.path.join(REPO, "results_p2",
                                  "c1t_selection_gowerstreet.json")))
T = np.load(os.path.join(REPO, "results_p2", "arms_c1t_gowerstreet.npz"),
            allow_pickle=True)
C1 = np.load(os.path.join(REPO, "results_p2", "arms_c1_gowerstreet.npz"),
             allow_pickle=True)

out = {"selected_step": {a: SEL[a]["selected_step"] for a in ("A", "B")}}
print(f"selected checkpoints: A @{out['selected_step']['A']} "
      f"B @{out['selected_step']['B']}\n")


def q999_of(fields, j):
    return tail_q999(np.concatenate(
        [octave_wc_pooled(f, j)[0].astype(np.float32) for f in fields]))


def stack(npz, key):
    return [np.asarray(f, dtype=np.float64) for f in npz[key]]


real_t = couplings(stack(T, "real"), [1, 2, 3, 4], n_boot=50, seed=0)
real_c1 = couplings(stack(C1, "real"), [1, 2, 3, 4], n_boot=50, seed=0)
out["real_test_ref"] = {str(j): real_t[j] for j in [1, 2, 3, 4]}

print(f"{'':>16} {'oct':>3} | {'C1-t gen':>9} {'real':>7} {'deficit':>8} | "
      f"{'C1 gen':>7} {'C1 deficit':>10}")
for arm in ("A", "B"):
    g = couplings(stack(T, f"gen_{arm}"), [1, 2, 3, 4], n_boot=200, seed=0)
    g1 = couplings(stack(C1, f"gen_{arm}"), [1, 2, 3, 4], n_boot=50, seed=0)
    a = {}
    for j in [1, 2, 3, 4]:
        row = {}
        for m in ("var_slope", "kurtosis"):
            d_t = (g[j][m] - real_t[j][m]) / abs(real_t[j][m])
            d_1 = (g1[j][m] - real_c1[j][m]) / abs(real_c1[j][m])
            row[m] = {"gen": g[j][m], "gen_se": g[j][m + "_se"],
                      "real": real_t[j][m], "rel_deficit": d_t,
                      "c1_gen": g1[j][m], "c1_rel_deficit": d_1,
                      "halved": bool(abs(d_t) <= 0.5 * abs(d_1))}
        row["q999_ratio"] = (q999_of(stack(T, f"gen_{arm}"), j)
                             / q999_of(stack(T, "real"), j))
        a[str(j)] = row
        print(f"arm {arm} kurtosis {j:>3} | {row['kurtosis']['gen']:9.3f} "
              f"{row['kurtosis']['real']:7.3f} "
              f"{row['kurtosis']['rel_deficit']:+8.1%} | "
              f"{row['kurtosis']['c1_gen']:7.3f} "
              f"{row['kurtosis']['c1_rel_deficit']:+10.1%}"
              f"  halved={row['kurtosis']['halved']}")
    for j in [2]:
        r = a[str(j)]["var_slope"]
        print(f"arm {arm} var_slope {j:>2} | {r['gen']:9.3f} {r['real']:7.3f} "
              f"{r['rel_deficit']:+8.1%} | {r['c1_gen']:7.3f} "
              f"{r['c1_rel_deficit']:+10.1%}")
    out[arm] = a
    hc2 = SEL[arm]["final"]["2"]
    print(f"arm {arm} head-cond oct2 @sel: vs={hc2['var_slope']:.3f} "
          f"kurt={hc2['kurtosis']:.2f} q999={hc2['q999']:.2f}\n")
    out[arm + "_hc"] = SEL[arm]["final"]

with open(os.path.join(REPO, "results_p2", "c1t_gow_descriptive.json"), "w") as f:
    json.dump(out, f, indent=1, default=str)
print("wrote results_p2/c1t_gow_descriptive.json")
