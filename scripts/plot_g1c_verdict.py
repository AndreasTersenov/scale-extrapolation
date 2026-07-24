#!/usr/bin/env python
"""G-1c verdict figure: the frozen bars applied to the NLL-head arms run, visually.

Reads results/scores/arms_nll_score.json (the frozen scaledrift instrument's output) and shows
var_slope and kurtosis vs octave for real / arm A / arm B with bootstrap error bars.
The bars: var_slope within 1 sigma AND kurtosis within 2 sigma at octaves 2,3,4.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K, BLUE, VERM = "#000000", "#0072B2", "#D55E00"

s = json.load(open(os.path.join(REPO, "results", "scores", "arms_nll_score.json")))
octs = [1, 2, 3, 4]


def get(which, metric, se):
    d = s[which]
    return (np.array([d[str(j)][metric] for j in octs]),
            np.array([d[str(j)][se] for j in octs]))


fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.0))
for a, metric, se, nb, bar in ((ax[0], "var_slope", "var_slope_se", 1,
                                "bar: within 1σ of real"),
                               (ax[1], "kurtosis", "kurtosis_se", 2,
                                "bar: within 2σ of real")):
    for which, col, lab, mk in (("real", K, "REAL held-out fields", "o"),
                                ("armA", BLUE, "arm A + NLL head", "s"),
                                ("armB", VERM, "arm B + NLL head", "D")):
        y, e = get(which, metric, se)
        a.errorbar(octs, y, yerr=(nb * e if which == "real" else e), color=col,
                   marker=mk, ms=6, lw=2, capsize=3, label=lab)
    a.axvspan(1.5, 4.4, color="#88888822")
    a.text(2.9, a.get_ylim()[1] * 0.02 + a.get_ylim()[0],
           "TRAINED octaves (where the bar applies)", fontsize=9, ha="center",
           color="#555555")
    a.set_xticks(octs)
    a.set_xticklabels(["1\n(extrapolated)", "2", "3", "4"])
    a.set_xlabel("octave (finer ← → coarser)")
    a.invert_xaxis()
    a.grid(alpha=0.25)
    a.legend(fontsize=9)
ax[0].set_ylabel("var_slope  (how detail spread grows with coarse brightness)")
ax[0].set_title("Dispersion bar — FAILED\n(oct 2: 7–8σ low, oct 3: 5–7σ low, both arms; "
                "real band = ±1σ)", color=VERM, fontsize=11)
ax[1].set_ylabel("excess kurtosis of detail  (tail heaviness)")
ax[1].set_title("Kurtosis check — FAILED\n(generated details far too tame; "
                "real band = ±2σ)", color=VERM, fontsize=11)
fig.suptitle("G-1c verdict on the Gaussian-NLL head (10k-step checkpoint, frozen bars verbatim): STOP → reconvene",
             fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(REPO, "results", "figures", "readouts", "g1c_verdict.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
