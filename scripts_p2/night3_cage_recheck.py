"""A-N3-2 corrected-cage recomputation (env.sh; mechanical, pre-stated in
prereg 2026-08-05-night3). Re-scores the FROZEN selection formula
(max(rel_vs/0.10, rel_kurt/0.15), argmin, ties -> earlier) from the
committed curve_val raw per-ckpt values against the CORRECT gowerstreet
octave-2 reference, for the three legs that trained with the sandbox
default (oracle, stage3 seed1, seed2). Writes
results_p2/night3_cage_recheck.json."""
from __future__ import annotations

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")

ref = json.load(open(os.path.join(RES, "gowerstreet_val_ref.json")))["truth"]["2"]
LEGS = {"oracle": "oct1fix_oracle_selection.json",
        "seed1": "stage3_seed1_selection.json",
        "seed2": "stage3_seed2_selection.json"}

out = {"reference": {"var_slope": ref["var_slope"],
                     "kurtosis": ref["kurtosis"]}}
for leg, fn in LEGS.items():
    sel = json.load(open(os.path.join(RES, fn)))["A"]
    curve = sel["curve_val"]
    rescored = {}
    for step, row in curve.items():
        rv = abs(row["var_slope"] - ref["var_slope"]) / abs(ref["var_slope"])
        rk = abs(row["kurtosis"] - ref["kurtosis"]) / abs(ref["kurtosis"])
        rescored[step] = max(rv / 0.10, rk / 0.15)
    steps = sorted(int(s) for s in rescored)
    pick = min(steps, key=lambda s: (rescored[str(s)], s))
    out[leg] = {"committed_pick": sel["selected_step"],
                "corrected_pick": pick,
                "corrected_score": rescored[str(pick)],
                "committed_pick_corrected_score":
                    rescored[str(sel["selected_step"])],
                "differs": bool(pick != sel["selected_step"]),
                "curve_corrected": {str(s): rescored[str(s)]
                                    for s in steps}}
    print(f"{leg}: committed @{sel['selected_step']} -> corrected @{pick} "
          f"(score {rescored[str(pick)]:.3f}; committed pick rescores to "
          f"{rescored[str(sel['selected_step'])]:.3f}) "
          f"{'DIFFERS' if out[leg]['differs'] else 'same'}")
with open(os.path.join(RES, "night3_cage_recheck.json"), "w") as f:
    json.dump(out, f, indent=1)
