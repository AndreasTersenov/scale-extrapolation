#!/usr/bin/env python
"""4b'-ii stop figure: white-noise conditioning corruption cannot reproduce the
measured end-to-end attenuation within the trained range.

Curves: the s=0.3 head's implied var_slope when fed REAL coarse corrupted at level s
(s_told=0). Horizontal lines: the MEASURED end-to-end values the matching had to hit.
The curves never get there for s <= 0.35 (except arm A oct 3, at 0.335 > s_max=0.3):
s_matched is outside the trained conditioning range at every arm/octave -> the ruling's
pre-named stop fires before any generation.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K, BLUE, VERM, GREEN = "#000000", "#0072B2", "#D55E00", "#009E73"
d = json.load(open(os.path.join(REPO, "results", "smatched_4bpii.json")))
grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.9))
for a, arm in zip(ax, "AB"):
    for j, col in ((2, BLUE), (3, GREEN)):
        m = d["attenuation_match"][f"arm{arm}_oct{j}"]
        a.plot(grid, m["curve"], "-o", ms=4, color=col,
               label=f"head response on real coarse + s·noise (octave {j})")
        a.axhline(m["target"], color=col, ls="--", lw=1.6)
        a.text(0.002, m["target"] + 0.006, f"measured end-to-end, octave {j}",
               color=col, fontsize=8.5)
    a.axvspan(0.3, 0.35, color=VERM, alpha=0.10)
    a.axvline(0.3, color=VERM, ls=":", lw=1.5)
    a.text(0.302, a.get_ylim()[0] + 0.02, " trained range ends (s_max=0.3)",
           color=VERM, fontsize=8.5, rotation=90, va="bottom")
    a.set_xlabel("injected white-noise corruption level s (relative to coarse std)")
    a.set_title(f"arm {arm}")
    a.grid(alpha=0.25)
ax[0].set_ylabel("implied var_slope of the head's response")
ax[0].legend(fontsize=8.5, loc="lower left")
fig.suptitle("4b'-ii stops at its own gate: to mimic the real conditioning drift, white noise would need s far beyond\n"
             "the trained range — the drift is NOT additive noise (solid curves never reach the dashed targets before the red line)",
             fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.90])
out = os.path.join(REPO, "results", "smatched_4bpii.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
