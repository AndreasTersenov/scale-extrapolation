#!/usr/bin/env python
"""Attempt-5 (final) readout: the compounding limit is informational — one figure.

Left: octave-2 (arm A) responses of each model on REAL vs DRIFTED (4a-generated)
coarse, with its end-to-end result alongside: end-to-end == the drifted-input response
(4a: 0.742 vs 0.746), and self-conditioning moves that response DOWN to the honest
information limit (~0.5), not up. Right: the whole generator program's arc at octave 2
against the frozen bars — where it froze and why.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K, BLUE, GREEN, ORANGE, VERM = "#000000", "#0072B2", "#009E73", "#E69F00", "#D55E00"

REAL2 = (1.020, 0.038)

fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.2))

# ---- left: the mechanism (octave 2, arm A) ----
models = ["4a model\n(trained on real coarse)", "attempt 5, p=0.5\n(self-conditioned)"]
on_real = [0.956, 0.992]
on_drift = [0.742, 0.530]
e2e = [0.746, 0.482]
x = np.arange(2)
w = 0.26
ax[0].bar(x - w, on_real, w, color=BLUE, label="response given REAL coarse (ceiling)")
ax[0].bar(x, on_drift, w, color=ORANGE,
          label="response given DRIFTED (generated) coarse")
ax[0].bar(x + w, e2e, w, color=GREEN, label="END-TO-END result")
ax[0].axhline(REAL2[0], color=K, ls="--", lw=1.8)
ax[0].text(-0.42, REAL2[0] + 0.012, "real fields", fontsize=9)
ax[0].axhline(0.53, color=VERM, ls=":", lw=1.5)
ax[0].text(1.02, 0.49, "honest information limit of\ndrifted conditioning (~0.5)",
           color=VERM, fontsize=8.5)
for xi, (od, ee) in enumerate(zip(on_drift, e2e)):
    ax[0].annotate("", xy=(xi + w, ee + 0.02), xytext=(xi, od + 0.02),
                   arrowprops=dict(arrowstyle="-", ls=":", color="#555555"))
ax[0].set_xticks(x)
ax[0].set_xticklabels(models, fontsize=9)
ax[0].set_ylabel("var_slope at octave 2 (arm A)")
ax[0].set_ylim(0, 1.12)
ax[0].set_title("End-to-end EQUALS the drifted-input response —\nand honesty about drift is LOWER than over-trust", fontsize=10.5)
ax[0].legend(fontsize=8.2, loc="lower left")

# ---- right: the program arc at octave 2 (arm A end-to-end) ----
labels = ["G-1c\n(NLL head)", "4a\n(+8x data)", "4b'\n(+corruption)", "a5 p=0.5\n(self-cond)", "a5 p=1.0"]
vals = [0.711, 0.746, 0.773, 0.482, 0.467]
errs = [0.009, 0.014, 0.018, 0.011, 0.010]
xx = np.arange(5)
ax[1].errorbar(xx, vals, yerr=errs, fmt="o-", color=GREEN, lw=2, capsize=3,
               label="end-to-end, octave 2 (arm A)")
ax[1].axhline(REAL2[0], color=K, ls="--", lw=1.8, label="real fields (project bar)")
ax[1].fill_between([-0.4, 4.4], REAL2[0] - REAL2[1], REAL2[0] + REAL2[1],
                   color="#88888833")
ax[1].axhline(0.996, color=BLUE, ls=":", lw=1.8,
              label="head ceiling given real coarse (calibrated)")
ax[1].set_xticks(xx)
ax[1].set_xticklabels(labels, fontsize=8.5)
ax[1].set_ylabel("var_slope at octave 2 (arm A)")
ax[1].set_title("The generator program's arc — frozen here:\nthe residual gap is the measured informational compounding limit", fontsize=10.5)
ax[1].legend(fontsize=8.5, loc="lower left")
ax[1].grid(alpha=0.25)

fig.suptitle("Attempt 5 (FINAL): lever bar FAILED, branch B5 — self-conditioning installs the honest attenuated conditional and makes generation worse.\n"
             "GENERATOR FROZEN at calibrated heads + measured compounding limit (pre-committed reshape).", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.88])
out = os.path.join(REPO, "results", "figures", "readouts", "readout_a5.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
