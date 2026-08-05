"""O-arm starlet trained leg (~/wl-challenge-env; frozen scorer, leg 15)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts_p2.score_starlet_l1 import SCORED, adjudicate, measure_sets  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")
G = np.load(os.path.join(RES, "stage3_oracle_final.npz"))
sets = {"real": np.asarray(np.load(os.path.join(
            RES, "arms_c1t_gowerstreet.npz"), allow_pickle=True)["real"],
            np.float64),
        "gen_A": np.concatenate([np.asarray(G[k], np.float64)
                                 for k in ("final1", "final2", "final3")])}
meas = measure_sets(sets, float(sets["real"].std()), leg_idx=15)
meas["leg"] = "oct1fix_oracle"
meas["n_maps"] = {k: int(v.shape[0]) for k, v in sets.items()}
meas["checks"] = adjudicate(meas)
for arm in meas["checks"]:
    meas["checks"][arm]["all_scored_pass"] = bool(all(
        meas["checks"][arm][lab]["pass"] for lab in SCORED))
with open(os.path.join(RES, "starlet_l1_oct1fix.json"), "w") as fh:
    json.dump(meas, fh, indent=1)
print(f"[oct1fix] starlet gen_A:"
      f"{'PASS' if meas['checks']['gen_A']['all_scored_pass'] else 'FAIL'}")
