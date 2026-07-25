"""Arm D SCORE phase (env.sh stack): the frozen branch adjudication
(log/2026-07-25-prereg-parity-cure.md D block, R32 weights; R34 sequencing;
execution reading documented in the F2 readout append).

Per data arm (model arm A), vs the shared 32 test tiles:
  e(arm) = T_coef oct-1 (RAW mode, the runner's npz gen_A at the caged pick)
  — comparable to the committed 1x baseline 15.3. Watch list: oct-2 T, meanD,
  corrHV, parity. Joint-window indicator: the J criterion re-run on the arm's
  T_coef_val curve (2500-stride) + its committed marginal curve; the window
  "exists" reading is the J convention (witnessed by the pick).
  F2-mode production texture at the pick: descriptive.

Branches (frozen): D-WIDEN: e(N8) < 3 AND e(N32) < 3 (clean at the caged
pick at 8x, stays clean at 32x). D-SHRINK: e(N8) <= 15.3/2 but >= 3.
D-FLAT: e(N8) > 15.3/2 AND e(N32) > 15.3/2. Currency (conditional):
CUR-PARENTS e(cens) > 1.25*e(n8); CUR-ENSEMBLE e(cens) < 0.8*e(n8);
CUR-EQUIV between. N1-seed1: descriptive rider (seed jitter + marginal bars).

Writes results_p2/pcure_verdict.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts_p2"))

from fj_lib import joint_pick
from measure_generated import couplings
from parity_localization import COEF_STAT_NAMES, coef_stats_profile, peak_parity_profile
from placement_instruments import stack_profiles, tstat

TRUTH = json.load(open(os.path.join(RES, "sandbox_truth_normconv.json")))["truth"]
BASELINE_1X = 15.3
K = 160
ARMS = ("n1s1", "n8", "n32", "cens")

tiles = np.load(os.path.join(REPO, "data_cache", "tiles_sandbox.npz"))["sandbox"]
test = tiles[-32:].astype(np.float64)
val = tiles[-64:-32].astype(np.float64)
CURVES = np.load(os.path.join(RES, "pcure_curves_gen.npz"), allow_pickle=True)


def T_of(gen, ref, octave=1, with_z=False):
    a = [coef_stats_profile(np.asarray(f, np.float64), octave) for f in gen]
    b = [coef_stats_profile(np.asarray(f, np.float64), octave) for f in ref]
    T, z = tstat(*stack_profiles(a), *stack_profiles(b))
    return (T, z) if with_z else T


def parity_of(gen, ref):
    a = [peak_parity_profile(np.asarray(f, np.float64), k=K, level=1)
         for f in gen]
    b = [peak_parity_profile(np.asarray(f, np.float64), k=K, level=1)
         for f in ref]
    T, _ = tstat(*stack_profiles(a), *stack_profiles(b))
    return T


out = {"baseline_1x_committed": BASELINE_1X, "arms": {}}
e = {}
for arm in ARMS:
    d = np.load(os.path.join(RES, f"arms_pcure_{arm}.npz"), allow_pickle=True)
    sel = json.load(open(os.path.join(RES, f"pcure_selection_{arm}.json")))
    pick = sel["A"]["selected_step"]
    gen = d["gen_A"]
    T1, z1 = T_of(gen, test, 1, with_z=True)
    T2 = T_of(gen, test, 2)
    e[arm] = T1
    # joint-window indicator: J criterion on the arm's val curve
    curve = {}
    for si in range(2500, 20001, 2500):
        key = f"{arm}_val_s{si}"
        if key not in CURVES.files:
            continue
        marg = sel["A"]["curve_val"][str(si)]
        curve[str(si)] = {"rel_vs": marg["rel_vs"],
                          "rel_kurt": marg["rel_kurt"],
                          "T_coef": T_of(CURVES[key], val, 1)}
    jp = joint_pick(curve) if curve else None
    f2key = f"{arm}_test_f2_s{pick}"
    f2_T1 = T_of(CURVES[f2key], test, 1) if f2key in CURVES.files else None
    out["arms"][arm] = {
        "caged_pick": pick, "T_coef_oct1_raw": T1, "T_coef_oct2_raw": T2,
        "meanD_oct2_z": float(T_of(gen, test, 2, with_z=True)[1][2]),
        "corrHV_oct1_z": float(z1[6]), "parity_T_raw": parity_of(gen, test),
        "curve": curve, "joint_pick": jp,
        "T_coef_val_at_joint": (curve[str(jp)]["T_coef"] if jp else None),
        "f2_mode_T_coef_oct1": f2_T1}
    a = out["arms"][arm]
    print(f"{arm}: pick={pick} T_coef1_raw={T1:.1f} T2={T2:.1f} "
          f"parity={a['parity_T_raw']:.1f} corrHV={a['corrHV_oct1_z']:+.1f} "
          f"joint_pick={jp} f2_T1={f2_T1 and round(f2_T1,1)}")

# ---- branch (frozen) --------------------------------------------------------
if e["n8"] < 3.0 and e["n32"] < 3.0:
    branch = "D-WIDEN"
elif e["n8"] <= BASELINE_1X / 2 and e["n8"] >= 3.0:
    branch = "D-SHRINK"
elif e["n8"] > BASELINE_1X / 2 and e["n32"] > BASELINE_1X / 2:
    branch = "D-FLAT"
else:
    branch = "UNLISTED (mixed n8/n32 pattern)"
out["branch"] = branch
r = e["cens"] / e["n8"] if e["n8"] > 0 else np.inf
currency = ("CUR-PARENTS" if r > 1.25 else
            "CUR-ENSEMBLE" if r < 0.8 else "CUR-EQUIV")
out["currency"] = {"ratio_cens_over_n8": float(r), "branch": currency,
                   "conditional_on": branch in ("D-WIDEN", "D-SHRINK")}
print(f"\nbranch: {branch} (e: n1s1={e['n1s1']:.1f} n8={e['n8']:.1f} "
      f"n32={e['n32']:.1f} cens={e['cens']:.1f}; committed 1x {BASELINE_1X})")
print(f"currency: {currency} (ratio {r:.2f}; "
      f"{'ADJUDICATING' if out['currency']['conditional_on'] else 'descriptive'})")

# seed rider: N1-seed1 marginal verdict summary from the runner's selection json
sel1 = json.load(open(os.path.join(RES, "pcure_selection_n1s1.json")))
out["seed_rider"] = {"selected_step": sel1["A"]["selected_step"],
                     "final_test": sel1["A"]["final"]}

with open(os.path.join(RES, "pcure_verdict.json"), "w") as f:
    json.dump(out, f, indent=1)
print("wrote results_p2/pcure_verdict.json")
