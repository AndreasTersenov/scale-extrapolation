"""L1' starlet-l1 legs (~/wl-challenge-env, torch, CPU; frozen scorer).

trained leg (leg_idx=8): pooled adjudicating streams (96 maps) as gen_A vs
the gowerstreet real stack. edge leg (leg_idx=9): the edge continuity
stream vs the stage-D real stack. Writes
results_p2/starlet_l1_l1p_{trained,edge}.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts_p2.score_starlet_l1 import SCORED, adjudicate, measure_sets  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")

GEN = np.load(os.path.join(RES, "l1p_main_gen.npz"))
LEGS = {
    "trained": (8, np.concatenate([GEN[k] for k in ("adj1", "adj2", "adj3")]),
                np.load(os.path.join(RES, "arms_c1t_gowerstreet.npz"),
                        allow_pickle=True)["real"]),
    "edge": (9, GEN["edge"],
             np.load(os.path.join(RES, "arms_stageD.npz"),
                     allow_pickle=True)["real"]),
}
for tag, (leg_idx, gen, real) in LEGS.items():
    sets = {"real": np.asarray(real, dtype=np.float64),
            "gen_A": np.asarray(gen, dtype=np.float64)}
    sigma = float(sets["real"].std())
    meas = measure_sets(sets, sigma, leg_idx=leg_idx)
    meas["leg"] = f"l1p_{tag}"
    meas["n_maps"] = {k: int(v.shape[0]) for k, v in sets.items()}
    meas["checks"] = adjudicate(meas)
    for arm in meas["checks"]:
        meas["checks"][arm]["all_scored_pass"] = bool(all(
            meas["checks"][arm][lab]["pass"] for lab in SCORED))
    out = os.path.join(RES, f"starlet_l1_l1p_{tag}.json")
    with open(out, "w") as fh:
        json.dump(meas, fh, indent=1)
    print(f"[l1p_{tag}] gen_A:"
          f"{'PASS' if meas['checks']['gen_A']['all_scored_pass'] else 'FAIL'}")
