"""STAGE D scoring + mechanical adjudication (CPU, env.sh stack).

The bet (PLAN-phase2 §4): does the substrate extrapolate INTO the held-out finest
octave (2) under the deployment protocol? Adjudicated at octave 2, BOTH levels
(head-conditional = conditional calibration given real coarse; end-to-end =
recursion from octave 4), vs the REAL TEST fields' own statistics:
var_slope <= max(10%, 3*SE_rel) AND kurtosis <= max(15%, 3*SE_rel). Arm B (the
curve-extrapolated dial) carries P-D; arm A is the scale-blind control.
AMBIGUOUS = FAIL (standing asymmetry). Peak audit (nu 1/2.5/3, 32-field means)
descriptive; octaves 3/4/1 descriptive. starlet-l1/scattering NOT in the suite
(hardening not landed — stated, per the prereg). Writes
results_p2/stageD_verdict.json.
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
                                  "c1t_selection_stageD.json")))
ARMS = np.load(os.path.join(REPO, "results_p2", "arms_stageD.npz"),
               allow_pickle=True)
BARS = {"var_slope": 0.10, "kurtosis": 0.15}
EDGE = 2


def peaks_count(field, nu):
    f = (field - field.mean()) / field.std()
    c = f[1:-1, 1:-1]
    is_max = np.ones(c.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            is_max &= c > f[1 + dy:f.shape[0] - 1 + dy, 1 + dx:f.shape[1] - 1 + dx]
    return int((is_max & (c > nu)).sum())


real = [np.asarray(f, dtype=np.float64) for f in ARMS["real"]]
real_ref = couplings(real, [1, 2, 3, 4], n_boot=50, seed=0)
out = {"selected_step": {a: SEL[a]["selected_step"] for a in ("A", "B")},
       "real_ref": {str(j): real_ref[j] for j in [1, 2, 3, 4]},
       "levels": {}, "peaks": {}, "suite_note":
       "starlet-l1/scattering absent (hardening not landed at submission)"}
print(f"selected: A @{out['selected_step']['A']} B @{out['selected_step']['B']}\n")


def check(metric, val, se, ref):
    t, tse = ref[metric], ref[metric + "_se"]
    rel = abs(val - t) / abs(t)
    bar = max(BARS[metric], 3 * float(np.hypot(se, tse) / abs(t)))
    return {"value": val, "se": se, "real": t, "rel_err": rel, "bar": bar,
            "pass": bool(rel <= bar)}


rows = []
e2e = {}
for arm in ("A", "B"):
    fields = [np.asarray(f, dtype=np.float64) for f in ARMS[f"gen_{arm}"]]
    e2e[arm] = couplings(fields, [1, 2, 3, 4], n_boot=200, seed=0)
    for level, src in (("head-conditional", SEL[arm]["final"][str(EDGE)]),
                       ("end-to-end", e2e[arm][EDGE])):
        for metric in ("var_slope", "kurtosis"):
            r = check(metric, src[metric], src[metric + "_se"], real_ref[EDGE])
            rows.append((arm, level, metric, r))
            out["levels"].setdefault(arm, {}).setdefault(level, {})[metric] = r
    pk = {}
    for nu in (1.0, 2.5, 3.0):
        g = np.array([peaks_count(f, nu) for f in fields], float)
        rl = np.array([peaks_count(f, nu) for f in real], float)
        pk[str(nu)] = {"gen_mean": float(g.mean()),
                       "real_mean": float(rl.mean()),
                       "excess": float((g.mean() - rl.mean()) / rl.mean())}
    out["peaks"][arm] = pk

print("=== STAGE D VERDICT — the held-out EDGE (octave 2), vs real TEST ===\n")
print(f"{'arm':>3} {'level':>16} {'metric':>9} | {'gen':>8} {'real':>8} "
      f"{'rel':>6} {'bar':>6} | P/F")
for arm, level, metric, r in rows:
    print(f"{arm:>3} {level:>16} {metric:>9} | {r['value']:8.3f} "
          f"{r['real']:8.3f} {r['rel_err']:6.1%} {r['bar']:6.1%} | "
          f"{'PASS' if r['pass'] else 'FAIL'}")

def arm_pass(arm):
    return all(out["levels"][arm][lv][m]["pass"]
               for lv in ("head-conditional", "end-to-end")
               for m in ("var_slope", "kurtosis"))

pa, pb = arm_pass("A"), arm_pass("B")
branch = ("D-PASS-BOTH" if pa and pb else "D-PASS-B-ONLY" if pb else
          "D-PASS-A-ONLY" if pa else "D-FAIL-BOTH")
out["branch"] = branch
kA = abs(out["levels"]["A"]["end-to-end"]["kurtosis"]["rel_err"])
kB = abs(out["levels"]["B"]["end-to-end"]["kurtosis"]["rel_err"])
out["dial_beats_scaleblind"] = bool(kB < kA)
print(f"\nbranch: {branch}   P-D (arm B, the bet): "
      f"{'PASS' if pb else 'FAIL'}")
print(f"dial-beats-scale-blind (|e2e kurt deficit| B {kB:.1%} vs A {kA:.1%}): "
      f"{out['dial_beats_scaleblind']}")

print("\nPeak audit at the edge fields (descriptive; excess vs real):")
for arm in ("A", "B"):
    print(f"  arm {arm}: " + "  ".join(
        f"nu={nu}: {out['peaks'][arm][nu]['excess']:+.0%}"
        for nu in ("1.0", "2.5", "3.0")))

print("\nDescriptive other octaves (e2e kurtosis vs real):")
for arm in ("A", "B"):
    for j in (3, 4, 1):
        d = (e2e[arm][j]["kurtosis"] - real_ref[j]["kurtosis"]) \
            / abs(real_ref[j]["kurtosis"])
        out.setdefault("other_octaves", {}).setdefault(arm, {})[str(j)] = d
        print(f"  arm {arm} oct{j}: {d:+.1%}", end="")
    print()

with open(os.path.join(REPO, "results_p2", "stageD_verdict.json"), "w") as f:
    json.dump(out, f, indent=1)
print("\nwrote results_p2/stageD_verdict.json")
