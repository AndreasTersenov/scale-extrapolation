"""Stage-3 (b) starlet edge leg (~/wl-challenge-env; frozen scorer).
Usage: stage3_b_starlet.py <tag>. leg_idx 11 (dryrun) / 12 (blind).
Writes results_p2/starlet_l1_stage3_<tag>.json."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts_p2.score_starlet_l1 import SCORED, adjudicate, measure_sets  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")
tag = sys.argv[1]
GEN = np.load(os.path.join(RES, f"stage3_{tag}_final.npz"))
sets = {"real": np.asarray(np.load(os.path.join(RES, "arms_stageD.npz"),
                                   allow_pickle=True)["real"], np.float64),
        "gen_A": np.concatenate([np.asarray(GEN[k], np.float64)
                                 for k in ("final1", "final2", "final3")])}
sigma = float(sets["real"].std())
meas = measure_sets(sets, sigma, leg_idx=11 if tag == "dryrun" else 12)
meas["leg"] = f"stage3_{tag}"
meas["n_maps"] = {k: int(v.shape[0]) for k, v in sets.items()}
meas["checks"] = adjudicate(meas)
for arm in meas["checks"]:
    meas["checks"][arm]["all_scored_pass"] = bool(all(
        meas["checks"][arm][lab]["pass"] for lab in SCORED))
with open(os.path.join(RES, f"starlet_l1_stage3_{tag}.json"), "w") as fh:
    json.dump(meas, fh, indent=1)
print(f"[stage3_{tag}] starlet gen_A:"
      f"{'PASS' if meas['checks']['gen_A']['all_scored_pass'] else 'FAIL'}")
