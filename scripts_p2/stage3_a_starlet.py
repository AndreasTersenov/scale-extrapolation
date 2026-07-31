"""Stage-3 (a) starlet trained legs for seed1/seed2 (~/wl-challenge-env;
frozen scorer; leg_idx 13/14). Writes starlet_l1_stage3a_seed{1,2}.json."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts_p2.score_starlet_l1 import SCORED, adjudicate, measure_sets  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")
real = np.asarray(np.load(os.path.join(RES, "arms_c1t_gowerstreet.npz"),
                          allow_pickle=True)["real"], np.float64)
for i, seed in enumerate(("seed1", "seed2")):
    G = np.load(os.path.join(RES, f"stage3_{seed}_final.npz"))
    sets = {"real": real,
            "gen_A": np.concatenate([np.asarray(G[k], np.float64)
                                     for k in ("final1", "final2",
                                               "final3")])}
    meas = measure_sets(sets, float(real.std()), leg_idx=13 + i)
    meas["leg"] = f"stage3a_{seed}"
    meas["n_maps"] = {k: int(v.shape[0]) for k, v in sets.items()}
    meas["checks"] = adjudicate(meas)
    for arm in meas["checks"]:
        meas["checks"][arm]["all_scored_pass"] = bool(all(
            meas["checks"][arm][lab]["pass"] for lab in SCORED))
    with open(os.path.join(RES, f"starlet_l1_stage3a_{seed}.json"),
              "w") as fh:
        json.dump(meas, fh, indent=1)
    print(f"[{seed}] starlet gen_A:"
          f"{'PASS' if meas['checks']['gen_A']['all_scored_pass'] else 'FAIL'}")
