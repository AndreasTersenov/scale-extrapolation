"""Score the NLL-noise forensic stacks with the frozen scorer and print the
pre-registered branch adjudication (descriptive; prereg
log/2026-07-17-prereg-forensic-nllnoise.md)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from measure_generated import couplings

D = np.load(os.path.join(REPO, "results_p2", "forensic_nllnoise.npz"))
REFS = {"real_oct2": 1.020, "withnoise_A": 0.746, "withnoise_B": 0.734,
        "c1_A": 0.921, "c1_B": 0.927}

out = {"refs": REFS}
real_fields = [np.asarray(f, dtype=np.float64) for f in D["real"]]
real = couplings(real_fields, [1, 2, 3, 4], n_boot=200, seed=0)
out["real"] = real

print("=== NLL-HEAD NOISE FORENSIC (mean-path generation, frozen 4a ckpts) ===\n")
print(f"{'arm':>4} {'oct':>4} | {'vs':>7} {'(SE)':>7} | {'kurt':>6} | {'dstd':>6} {'dstd/real':>9}")
vals = {}
for arm in ("A", "B"):
    gen_fields = [np.asarray(f, dtype=np.float64) for f in D[f"gen_{arm}"]]
    c = couplings(gen_fields, [1, 2, 3, 4], n_boot=200, seed=0)
    out[f"gen_{arm}"] = c
    for j in (1, 2, 3, 4):
        r = c[j]
        ratio = r["detail_std"] / real[j]["detail_std"]
        print(f"{arm:>4} {j:>4} | {r['var_slope']:7.3f} {r['var_slope_se']:7.3f} | "
              f"{r['kurtosis']:6.2f} | {r['detail_std']:6.3f} {ratio:9.2f}")
    vals[arm] = c[2]["var_slope"]
    print()

print(f"reference points: real oct-2 {REFS['real_oct2']}; 4a WITH noise "
      f"{REFS['withnoise_A']}/{REFS['withnoise_B']}; C1 {REFS['c1_A']}/{REFS['c1_B']}\n")


def branch(v):
    if v > 1.10:
        return "F-OVERSHOOT"
    if v >= 0.87:
        return "F-CLOSE"
    if v > 0.78:
        return "F-MID"
    return "F-COLLAPSE"


ba, bb = branch(vals["A"]), branch(vals["B"])
overall = ba if ba == bb else "OTHER (arms split: " + ba + "/" + bb + ")"
print(f"oct-2 mean-path var_slope: arm A {vals['A']:.3f} -> {ba}; "
      f"arm B {vals['B']:.3f} -> {bb}")
print(f"BRANCH: {overall}")
out["branch"] = {"A": ba, "B": bb, "overall": overall}

with open(os.path.join(REPO, "results_p2", "forensic_nllnoise.json"), "w") as f:
    json.dump(out, f, indent=1)
print("\nwrote results_p2/forensic_nllnoise.json")
