#!/usr/bin/env python
"""4b' lever-bar readout figure: heads vs recursion, before/after conditioning corruption.

Dashed = what each head achieves GIVEN REAL COARSE (its ceiling). Solid = what the full
coarse-to-fine recursion delivers end-to-end. The lever was supposed to pull the solid
line up to the dashed one; instead it lifted the dashed line (a free head improvement)
and left the solid one at the floor.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K, BLUE, GREEN, ORANGE = "#000000", "#0072B2", "#009E73", "#E69F00"
octs = [2, 3, 4]

real = {2: (1.020, 0.038), 3: (0.801, 0.037), 4: (0.532, 0.033)}
ceil4a = json.load(open(os.path.join(REPO, "results", "scores", "ceiling_4a.json")))
own = {("0.1", "A"): [0.991, 0.732, 0.489], ("0.1", "B"): [1.015, 0.723, 0.497],
       ("0.3", "A"): [0.996, 0.748, 0.489], ("0.3", "B"): [1.007, 0.740, 0.471]}
e2e = {}
for tag, path in (("4a", "arms_aug_score.json"), ("0.1", "arms_4bp_s0.1_score.json"),
                  ("0.3", "arms_4bp_s0.3_score.json")):
    s = json.load(open(os.path.join(REPO, "results", path)))
    for arm in "AB":
        e2e[(tag, arm)] = [(s[f"arm{arm}"][str(j)]["var_slope"],
                            s[f"arm{arm}"][str(j)]["var_slope_se"]) for j in octs]

fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.2), sharey=True)
for a, arm in zip(ax, "AB"):
    a.errorbar(octs, [real[j][0] for j in octs], yerr=[real[j][1] for j in octs],
               color=K, marker="o", lw=2.4, capsize=3, label="REAL fields")
    c = [ceil4a[f"arm{arm}_oct{j}"]["ceiling"] for j in octs]
    ce = [ceil4a[f"arm{arm}_oct{j}"]["se"] for j in octs]
    a.errorbar(octs, c, yerr=ce, color=BLUE, ls="--", marker="^", lw=1.8, capsize=3,
               label="head ceiling, 4a model (given REAL coarse)")
    a.plot(octs, own[("0.3", arm)], "--v", color=GREEN, lw=1.8,
           label="head ceiling, corrupted model s=0.3")
    y, e = zip(*e2e[("4a", arm)])
    a.errorbar(octs, y, yerr=e, color=BLUE, marker="s", lw=2.2, capsize=3,
               label="END-TO-END, 4a (no corruption)")
    y, e = zip(*e2e[("0.3", arm)])
    a.errorbar(octs, y, yerr=e, color=GREEN, marker="D", lw=2.2, capsize=3,
               label="END-TO-END, corrupted s=0.3")
    y, e = zip(*e2e[("0.1", arm)])
    a.errorbar(octs, y, yerr=e, color=ORANGE, marker="x", lw=1.2, ls=":", capsize=2,
               label="END-TO-END, corrupted s=0.1")
    a.set_xticks(octs)
    a.set_xlabel("octave (2 = deepest recursion level tested)")
    a.set_title(f"arm {arm}")
    a.invert_xaxis()
    a.grid(alpha=0.25)
ax[0].set_ylabel("var_slope (conditional-variance modulation)")
ax[0].legend(fontsize=8.2, loc="upper left")
fig.suptitle("4b' readout — LEVER BAR FAILED: corruption lifted the HEADS (dashed, now ~at real)\n"
             "but not the RECURSION (solid, unchanged at octave 2): the gap is compounding, and s=0 sampling never engages the robustness",
             fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.90])
out = os.path.join(REPO, "results", "figures", "readouts", "readout_4bp.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
