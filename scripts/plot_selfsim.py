#!/usr/bin/env python
"""Step-4 readout figure: same frozen architecture, drifting vs scale-invariant field.

Left: gowerstreet (drifting couplings) — the end-to-end curve sags away from real.
Right: the synthesized scale-invariant in-class control — end-to-end sits on real
everywhere including the extrapolated octave (residual 2.5% vs 39-65%).
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K, BLUE, ORANGE, VERM = "#000000", "#0072B2", "#E69F00", "#D55E00"
octs = [1, 2, 3, 4]

fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.9), sharey=True)
for a, path, title in (
        (ax[0], "arms_aug_score.json",
         "GOWERSTREET (couplings drift ×2 across octaves)\nthe compounding cap: 39–65% "
         "error at the extrapolated octave"),
        (ax[1], "arms_selfsim_score.json",
         "SCALE-INVARIANT control (same architecture, no drift)\nresidual 2.5% — the cap "
         "is the field's drift, not the architecture")):
    s = json.load(open(os.path.join(REPO, "results", path)))
    for which, col, mk, lab in (("real", K, "o", "real fields"),
                                ("armA", BLUE, "s", "end-to-end, arm A"),
                                ("armB", ORANGE, "D", "end-to-end, arm B")):
        y = [s[which][str(j)]["var_slope"] for j in octs]
        e = [s[which][str(j)]["var_slope_se"] for j in octs]
        a.errorbar(octs, y, yerr=e, color=col, marker=mk, lw=2, capsize=3, label=lab)
    a.axvspan(0.6, 1.5, color="#D55E0018")
    a.set_xticks(octs)
    a.set_xticklabels(["1\n(extrapolated)", "2", "3", "4"])
    a.invert_xaxis()
    a.set_xlabel("octave")
    a.set_title(title, fontsize=10)
    a.grid(alpha=0.25)
ax[0].set_ylabel("var_slope (conditional-variance modulation)")
ax[0].legend(fontsize=9)
fig.suptitle("Step 4 — the self-similar control bounds the negative claim: where its assumption holds, the method works",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])
out = os.path.join(REPO, "results", "selfsim_control.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
