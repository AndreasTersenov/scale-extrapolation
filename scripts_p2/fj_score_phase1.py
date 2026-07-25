"""Arms F/J, SCORE phase 1 (env.sh stack; R32 GO).

From fj_val_gen.npz + fj_val_hc.json (all VALIDATION-side):
 1. F offsets: e2e field-space offsets per octave/channel (pywt units) for
    sandbox A@7500, B@2500 (descriptive) and gowerstreet A@16000, with the
    prereg's K-fold stability gate; hc offsets copied from the JAX phase
    (model units, wfm order) with fold spread disclosed.
 2. J curve: T_coef_val (octave-1 coefficient D4-statistic vs the 32 VAL
    tiles — cage discipline: the val-side reference for a val-side pick) for
    every arm-A checkpoint; joint pick = frozen criterion over the COMMITTED
    marginal curve (c1t_selection_sandbox.json curve_val) + T_coef_val.

Writes results_p2/fj_offsets.json + results_p2/fj_joint_pick.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")
sys.path.insert(0, os.path.join(REPO, "scripts_p2"))

from fj_lib import estimate_offsets, joint_pick, offsets_pywt_to_wfm, stability_gate
from parity_localization import coef_stats_profile
from placement_instruments import stack_profiles, tstat


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def t_coef(gen_fields, ref_fields, octave=1):
    a = [coef_stats_profile(np.asarray(f, np.float64), octave) for f in gen_fields]
    b = [coef_stats_profile(np.asarray(f, np.float64), octave) for f in ref_fields]
    T, z = tstat(*stack_profiles(a), *stack_profiles(b))
    return T


GEN = np.load(os.path.join(RES, "fj_val_gen.npz"), allow_pickle=True)
HC = json.load(open(os.path.join(RES, "fj_val_hc.json")))
tiles = np.load(os.path.join(REPO, "data_cache", "tiles_sandbox.npz"))["sandbox"]
val = tiles[-64:-32].astype(np.float64)
gtiles = np.load(os.path.join(REPO, "data_cache", "tiles_pnull.npz"))["gowerstreet"]
gval = gtiles[266:298].astype(np.float64)

# ---- 1. F offsets ----------------------------------------------------------
offsets = {"equivalence_gate": HC["equivalence"], "e2e": {}, "hc": {}}
for key, fields in (("sandbox_A@7500", GEN["armA_s7500"]),
                    ("sandbox_B@2500", GEN["armB_s2500"]),
                    ("gowerstreet_A@16000", GEN["gowA_s16000"])):
    est = estimate_offsets(list(fields), octaves=(1, 2, 3, 4))
    gate = stability_gate(list(fields), octaves=(1, 2, 3, 4), n_folds=4)
    wfm = offsets_pywt_to_wfm(est["mean"])
    offsets["e2e"][key] = {
        "mean_pywt": {str(j): est["mean"][j].tolist() for j in est["mean"]},
        "se_pywt": {str(j): est["se"][j].tolist() for j in est["se"]},
        "mean_wfm": {str(j): wfm[j].tolist() for j in wfm},
        "stability": gate}
    log(f"{key}: oct1 offsets (pywt HVD) "
        f"{np.round(est['mean'][1], 4).tolist()} "
        f"stability worst_ratio={gate['worst_ratio']:.2f} "
        f"fires={gate['fires']}")
offsets["hc"] = HC["hc_channel_means"]

with open(os.path.join(RES, "fj_offsets.json"), "w") as f:
    json.dump(offsets, f, indent=1)

# ---- 2. J curve + pick -----------------------------------------------------
SEL = json.load(open(os.path.join(RES, "c1t_selection_sandbox.json")))
curve = {}
for key in GEN.files:
    if not key.startswith("armA_s"):
        continue
    step = key.split("_s")[1]
    marg = SEL["A"]["curve_val"][step]
    curve[step] = {"rel_vs": marg["rel_vs"], "rel_kurt": marg["rel_kurt"],
                   "T_coef": t_coef(GEN[key], val)}
pick = joint_pick(curve)
marginal_pick = SEL["A"]["selected_step"]
log(f"J: joint pick arm A = {pick} (marginal cage picked {marginal_pick}); "
    f"T_coef_val@joint={curve[str(pick)]['T_coef']:.1f} "
    f"@marginal={curve[str(marginal_pick)]['T_coef']:.1f}")

curve_B = {}
for key in GEN.files:
    if not key.startswith("armB_s"):
        continue
    step = key.split("_s")[1]
    marg = SEL["B"]["curve_val"][step]
    curve_B[step] = {"rel_vs": marg["rel_vs"], "rel_kurt": marg["rel_kurt"],
                     "T_coef": t_coef(GEN[key], val)}
pick_B = joint_pick(curve_B)

with open(os.path.join(RES, "fj_joint_pick.json"), "w") as f:
    json.dump({"A": {"joint_pick": pick, "marginal_pick": marginal_pick,
                     "curve": curve},
               "B_descriptive": {"joint_pick": pick_B,
                                 "marginal_pick": SEL["B"]["selected_step"],
                                 "curve": curve_B}}, f, indent=1)
log("wrote fj_offsets.json + fj_joint_pick.json")
