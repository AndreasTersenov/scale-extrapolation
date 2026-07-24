#!/usr/bin/env python
"""Phase-1 result story in four panels (values are the measured results from the logs)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Okabe-Ito colourblind-safe palette
K, BLUE, VERM, GREEN, ORANGE = "#000000", "#0072B2", "#D55E00", "#009E73", "#E69F00"
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# ---- Panel A: P5 break — arm A var_slope at the extrapolated octave, 3 configs ----
a = ax[0, 0]
real1 = 1.117
configs = ["additive\n10k", "FiLM\n10k", "FiLM big\n25k"]
armA1 = [0.822, 0.824, 0.539]
armA1_se = [0.006, 0.006, 0.003]
z = [5.8, 5.7, 11.3]
x = np.arange(3)
a.axhline(real1, color=K, ls="--", lw=1.5, label="real (held-out)")
a.axhspan(real1 - 0.051, real1 + 0.051, color=K, alpha=0.08)
a.errorbar(x, armA1, yerr=armA1_se, fmt="o", ms=9, color=BLUE, capsize=4, label="arm A (no scale input)")
for xi, yi, zi in zip(x, armA1, z):
    a.annotate(f"{zi:.0f}σ low", (xi, yi), textcoords="offset points", xytext=(0, -16),
               ha="center", color=BLUE, fontsize=9)
a.set_xticks(x); a.set_xticklabels(configs); a.set_ylim(0.3, 1.25)
a.set_ylabel("var_slope at octave 1 (extrapolated)")
a.set_title("P5 — the break is real and robust\n(weight-tied arm A is wrong at the untrained octave)")
a.legend(loc="lower left", fontsize=8)

# ---- Panel B: dispersion-collapse curve (var_slope vs training steps) ----
b = ax[0, 1]
steps = np.array([2000, 4000, 6000, 8000, 12000, 25000])
vs2 = np.array([0.847, 0.751, 0.751, 0.748, 0.757, 0.513])
loss = [1.30, 1.11, 0.32, 0.34, 0.15, 0.15]
real2 = 1.020
b.axhline(real2, color=K, ls="--", lw=1.5, label="real (octave 2)")
b.axhspan(real2 - 0.048, real2 + 0.048, color=K, alpha=0.08)
b.plot(steps, vs2, "-o", color=GREEN, lw=2, ms=8, label="generated (arm A)")
for s, v, l in zip(steps, vs2, loss):
    b.annotate(f"loss {l:.2f}", (s, v), textcoords="offset points", xytext=(0, 9),
               ha="center", fontsize=7.5, color="0.4")
b.annotate("peak dispersion\n(~2k steps)", (2000, 0.847), textcoords="offset points",
           xytext=(35, -6), fontsize=8, color=GREEN,
           arrowprops=dict(arrowstyle="->", color=GREEN))
b.set_xscale("log"); b.set_xlabel("training steps (log)")
b.set_ylabel("var_slope at octave 2 (trained)")
b.set_ylim(0.3, 1.1)
b.set_title("Generator finding — conditional variance COLLAPSES with training\n"
            "(var_slope peaks early then falls; loss keeps dropping)")
b.legend(loc="upper right", fontsize=8)

# ---- Panel C: no intervention natively reaches real dispersion (octave 2) ----
c = ax[1, 0]
labels = ["baseline\n10k ODE", "churn 8\n(SDE, a)", "2k peak\n(b)", "λ=0.3\n(c objective)",
          "2k + churn4\n(a+b)"]
vals = [0.77, 0.86, 0.847, 0.82, 0.97]
cols = [BLUE, ORANGE, GREEN, VERM, "#666666"]
xc = np.arange(len(labels))
c.axhline(real2, color=K, ls="--", lw=1.5, label="real (octave 2)")
c.axhspan(real2 - 0.048, real2 + 0.048, color=K, alpha=0.10, label="±1σ target band")
c.bar(xc, vals, color=cols, width=0.6)
for xi, yi in zip(xc, vals):
    c.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 3),
               ha="center", fontsize=8)
c.set_xticks(xc); c.set_xticklabels(labels, fontsize=8); c.set_ylim(0, 1.15)
c.set_ylabel("var_slope at octave 2 (trained)")
c.set_title("Neither sampler (a), checkpoint (b), nor objective (c) reaches\n"
            "real dispersion natively — all fall short of the band")
c.legend(loc="lower right", fontsize=8)

# ---- Panel D: the 90% repair signal (octave 1 vs churn, 2k checkpoint) ----
d = ax[1, 1]
churn = [0, 4, 8]
armA = [0.86, 0.87, 0.97]
armB = [0.88, 1.14, 1.33]
d.axhline(real1, color=K, ls="--", lw=1.5, label="real (octave 1)")
d.axhspan(real1 - 0.051, real1 + 0.051, color=K, alpha=0.08)
d.plot(churn, armA, "-o", color=BLUE, lw=2, ms=8, label="arm A (no scale coord)")
d.plot(churn, armB, "-s", color=VERM, lw=2, ms=8, label="arm B (running-coupling coord)")
d.annotate("arm B reaches real\n→ 90% repair", (4, 1.14), textcoords="offset points",
           xytext=(10, 6), fontsize=8, color=VERM)
d.set_xlabel("SDE churn ε₀ (dispersion restored →)")
d.set_ylabel("var_slope at octave 1 (extrapolated)")
d.set_title("The payoff signal — once dispersion is restored, the repair WORKS\n"
            "(arm B hits real while arm A stays broken)")
d.legend(loc="upper left", fontsize=8)

fig.suptitle("Phase-1 break & repair: P5 confirmed; P6 blocked by generator variance under-dispersion (not the conditioning)",
             fontsize=12.5, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
out = os.path.join(REPO, "results", "figures", "explainers", "phase1_story.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
