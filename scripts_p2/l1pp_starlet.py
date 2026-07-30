"""L1'' starlet-l1 trained leg (~/wl-challenge-env, torch, CPU; frozen
scorer, leg_idx=10): pooled adjudicating streams vs gowerstreet real.
Writes results_p2/starlet_l1_l1pp_trained.json."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts_p2.score_starlet_l1 import SCORED, adjudicate, measure_sets  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")
GEN = np.load(os.path.join(RES, "l1pp_main_gen.npz"))
sets = {"real": np.asarray(np.load(os.path.join(
            RES, "arms_c1t_gowerstreet.npz"), allow_pickle=True)["real"],
            dtype=np.float64),
        "gen_A": np.concatenate([np.asarray(GEN[k], np.float64)
                                 for k in ("adj1", "adj2", "adj3")])}
sigma = float(sets["real"].std())
meas = measure_sets(sets, sigma, leg_idx=10)
meas["leg"] = "l1pp_trained"
meas["n_maps"] = {k: int(v.shape[0]) for k, v in sets.items()}
meas["checks"] = adjudicate(meas)
for arm in meas["checks"]:
    meas["checks"][arm]["all_scored_pass"] = bool(all(
        meas["checks"][arm][lab]["pass"] for lab in SCORED))
with open(os.path.join(RES, "starlet_l1_l1pp_trained.json"), "w") as fh:
    json.dump(meas, fh, indent=1)
print(f"[l1pp_trained] gen_A:"
      f"{'PASS' if meas['checks']['gen_A']['all_scored_pass'] else 'FAIL'}")
