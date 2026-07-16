"""Gate-A milestone figure: what the sandbox is (maps) + instrument-vs-truth (bars).

Top row: one parent lognormal field and three EXACT conditional redraws sharing its
level-4 Gaussian coarse (same large-scale layout, legitimately different small-scale
dice) — shared color scale.
Bottom row: per-octave var_slope and kurtosis, truth (16384 exact fields, batch-means
SE) vs frozen instrument (256 parents, production bootstrap), with the Gate-A bars.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

SCRATCH = os.path.expanduser("~/links/scratch/scale-extrap-p2")

ens = np.load(os.path.join(SCRATCH, "sandbox_ens_f32.npy"), mmap_mode="r")
parents = np.load(os.path.join(SCRATCH, "sandbox_parents_f32.npy"))
truth = json.load(open(os.path.join(REPO, "results_p2", "sandbox_truth.json")))["truth"]
inst = json.load(open(os.path.join(REPO, "results_p2", "gateA_instrument.json")))["primary_n256"]

fig = plt.figure(figsize=(13, 8))
gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.25)

# --- top: parent + 3 conditional redraws (parent index 0 — first, no cherry-picking)
pi = 0
maps = [parents[pi]] + [np.asarray(ens[pi, r]) for r in range(3)]
titles = ["parent field (index 0)",
          "conditional redraw 1", "conditional redraw 2", "conditional redraw 3"]
vmin, vmax = np.percentile(np.stack(maps), [1, 99])
for k, (m, t) in enumerate(zip(maps, titles)):
    ax = fig.add_subplot(gs[0, k])
    ax.imshow(m, cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set_title(t, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
fig.text(0.5, 0.965,
         "Stage A sandbox: exact conditional samples — same 8$\\times$8 Gaussian coarse, "
         "different fine detail (shared color scale, 1–99 pct)",
         ha="center", fontsize=11)

# --- bottom: truth vs instrument with Gate-A bars
octs = [1, 2, 3, 4]
for col, (metric, floor, label) in enumerate(
        [("var_slope", 0.05, "var_slope"), ("kurtosis", 0.10, "excess kurtosis")]):
    ax = fig.add_subplot(gs[1, 2 * col:2 * col + 2])
    tv = [truth[str(j)][metric] for j in octs]
    tse = [truth[str(j)][metric + "_se"] for j in octs]
    iv = [inst[str(j)][metric] for j in octs]
    ise = [inst[str(j)][metric + "_se"] for j in octs]
    bars = [max(floor, 3 * np.hypot(a, b) / abs(t))
            for a, b, t in zip(ise, tse, tv)]
    band_lo = [t * (1 - b) for t, b in zip(tv, bars)]
    band_hi = [t * (1 + b) for t, b in zip(tv, bars)]
    ax.fill_between(octs, band_lo, band_hi, alpha=0.18, color="tab:green",
                    label="Gate-A tolerance")
    ax.errorbar(octs, tv, yerr=tse, fmt="o-", color="k", lw=2, capsize=3,
                label="TRUTH (16384 exact fields)")
    ax.errorbar(octs, iv, yerr=ise, fmt="s--", color="tab:red", capsize=3,
                label="frozen instrument (256 fields)")
    ax.set_xticks(octs)
    ax.set_xlabel("octave (1 = finest)")
    ax.set_title(f"{label}: instrument recovers exact truth", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

out = os.path.join(REPO, "results_p2", "gateA.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
