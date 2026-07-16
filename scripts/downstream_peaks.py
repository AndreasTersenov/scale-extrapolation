#!/usr/bin/env python
"""Step 3: the downstream-bias demo — peak counts on the frozen generator's fields.

Pre-registered log/2026-07-16-prereg-step3-downstream.md (criteria fixed before this
script ran). Peak counts at nu in {1,2,3} for real vs arm A/B (frozen 4a run, 64
held-out starts); z via bootstrap-over-maps. The demo's conjunction: P(k)-level checks
PASS (P4 within 10% everywhere) while the peak observable is biased.
"""
import json
import os
try:
    os.sched_setaffinity(0, set(range(4)))
except Exception:
    pass
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pilotstats import peak_counts, z_stack

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K, BLUE, ORANGE, VERM = "#000000", "#0072B2", "#E69F00", "#D55E00"

d = np.load(os.path.join(REPO, "results", "arms_aug.npz"))
real, gA, gB = d["real"], d["gen_A"], d["gen_B"]
NUS = [1.0, 1.5, 2.0, 2.5, 3.0]          # {1,2,3} adjudicated; 1.5/2.5 for the curve
out = {"nus": NUS, "counts": {}, "z": {}}
for name, f in (("real", real), ("A", gA), ("B", gB)):
    per = {nu: peak_counts(f, nu) for nu in NUS}
    out["counts"][name] = {nu: [float(per[nu].mean()),
                                float(per[nu].std(ddof=1) / np.sqrt(len(per[nu])))]
                           for nu in NUS}
    if name != "real":
        out["z"][name] = {nu: z_stack(per[nu], peak_counts(real, nu)) for nu in NUS}
print("mean peak counts per 128^2 map (real / A / B) and z:")
for nu in NUS:
    r = out["counts"]["real"][nu][0]
    a, b = out["counts"]["A"][nu][0], out["counts"]["B"][nu][0]
    print(f"  nu={nu}: real {r:7.1f}  A {a:7.1f} (z={out['z']['A'][nu]:+.1f})  "
          f"B {b:7.1f} (z={out['z']['B'][nu]:+.1f})")
json.dump(out, open(os.path.join(REPO, "results", "downstream_peaks.json"), "w"),
          indent=1)

fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.9))
for name, col, mk in (("real", K, "o"), ("A", BLUE, "s"), ("B", ORANGE, "D")):
    m = [out["counts"][name][nu][0] for nu in NUS]
    e = [out["counts"][name][nu][1] for nu in NUS]
    lab = {"real": "REAL held-out maps", "A": "arm A (frozen generator)",
           "B": "arm B (frozen generator)"}[name]
    ax[0].errorbar(NUS, m, yerr=e, color=col, marker=mk, lw=2, capsize=3, label=lab)
ax[0].set_yscale("log")
ax[0].set_xlabel("peak height threshold ν  [σ of the map]")
ax[0].set_ylabel("peaks per 128×128 map")
ax[0].set_title("The observable a WL analysis would use\n(peak counts vs threshold)")
ax[0].legend(fontsize=9)
ax[0].grid(alpha=0.25)
x = np.arange(len(NUS))
for name, col, off in (("A", BLUE, -0.17), ("B", ORANGE, 0.17)):
    ax[1].bar(x + off, [out["z"][name][nu] for nu in NUS], 0.32, color=col,
              label=f"arm {name}")
ax[1].axhline(-3, color=VERM, ls="--", lw=1.5)
ax[1].axhline(3, color=VERM, ls="--", lw=1.5)
ax[1].axhline(0, color=K, lw=1)
ax[1].set_xticks(x)
ax[1].set_xticklabels([f"ν={nu:g}" for nu in NUS])
ax[1].set_ylabel("bias z  (generated − real)")
ax[1].set_title("The bias P(k)-level checks miss\n(P4 amplitude passed at ≤7% everywhere)")
ax[1].legend(fontsize=9)
ax[1].grid(alpha=0.25)
fig.suptitle("Step 3 — downstream-bias demo: power-level checks pass, the peak observable is biased",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(REPO, "results", "downstream_peaks.png"), dpi=130,
            bbox_inches="tight")
print("wrote results/downstream_peaks.{json,png}")
