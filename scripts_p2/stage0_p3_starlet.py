"""Phase-3 Stage-0 starlet-l1 legs (~/wl-challenge-env, torch, CPU).

Frozen scorer (score_starlet_l1.py conventions verbatim — measure_sets +
adjudicate, sigma = real.std(), 10% floors, SCORED octaves 2-4):
  edge leg    : stage0_p3_gen.npz (real / gen_A=F2 A@9000 / gen_B=F2 B@7500)
                — run_leg verbatim, leg_idx=6 (fresh bootstrap seed stream,
                disjoint from committed legs 1-4).
  trained leg : committed f2_test_gen.npz F2_gowA_e2e as gen_A vs the
                arms_c1t_gowerstreet.npz real stack, leg_idx=7 (adjudicate
                skips the absent gen_B). No new generation.

Writes results_p2/starlet_l1_stage0_{edge,trained}.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts_p2.score_starlet_l1 import (  # noqa: E402
    SCORED,
    adjudicate,
    measure_sets,
    run_leg,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")

run_leg("stage0_edge", 6, os.path.join(RES, "stage0_p3_gen.npz"),
        os.path.join(RES, "starlet_l1_stage0_edge.json"), with_secondary=True)

gow = np.load(os.path.join(RES, "arms_c1t_gowerstreet.npz"), allow_pickle=True)
F2G = np.load(os.path.join(RES, "f2_test_gen.npz"), allow_pickle=True)
sets = {"real": np.asarray(gow["real"], dtype=np.float64),
        "gen_A": np.asarray(F2G["F2_gowA_e2e"], dtype=np.float64)}
sigma = float(sets["real"].std())
meas = measure_sets(sets, sigma, leg_idx=7)
meas["leg"] = "stage0_trained"
meas["source_npz"] = "results_p2/f2_test_gen.npz (F2_gowA_e2e, committed)"
meas["n_maps"] = {k: int(v.shape[0]) for k, v in sets.items()}
meas["checks"] = adjudicate(meas)
for arm in meas["checks"]:
    meas["checks"][arm]["all_scored_pass"] = bool(all(
        meas["checks"][arm][lab]["pass"] for lab in SCORED))
out_json = os.path.join(RES, "starlet_l1_stage0_trained.json")
with open(out_json, "w") as fh:
    json.dump(meas, fh, indent=1)
print(f"[stage0_trained] sigma={sigma:.6g}  gen_A:"
      f"{'PASS' if meas['checks']['gen_A']['all_scored_pass'] else 'FAIL'}")
