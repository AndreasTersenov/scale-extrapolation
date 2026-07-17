"""Overnight-run results ON THE MAPS (Andreas's request, 2026-07-17).

Honesty rules as in SPEC-gallery: held-out index 0 everywhere (first, no
cherry-picking), arm A displayed (arm B similar, quoted in captions where it
differs), shared color scales stated per figure, generated panels are conditional
SAMPLES from the same octave-4 coarse, measured statistics quoted per panel from
the committed jsons (no recomputation except the on-map peak circles, which are
computed on the displayed field itself).

FIG 1  maps_channels.png — one field, four generators (the channel story).
FIG 2  maps_peaks.png    — two opposite peak-failure signatures, circled.
FIG 3  maps_locality.png — B1's r*≈1 drawn on the coarse field.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from sandbox.haar import haar_level

AUG = np.load(os.path.join(REPO, "results", "arms_aug.npz"))          # 4a with-noise
FOR = np.load(os.path.join(REPO, "results_p2", "forensic_nllnoise.npz"))
C1 = np.load(os.path.join(REPO, "results_p2", "arms_c1_gowerstreet.npz"))

real = np.asarray(C1["real"][0])
assert np.allclose(real, np.asarray(AUG["real"][0]), atol=1e-5)
assert np.allclose(real, np.asarray(FOR["real"][0]), atol=1e-5)

# ---------------------------------------------------------------- FIG 1: channels
panels = [
    (real, "REAL held-out field (index 0)",
     "oct-2 var_slope 1.020 · kurtosis 6.7"),
    (np.asarray(AUG["gen_A"][0]), "4a generator: $\\mu$ + white-noise $\\sigma$-head\n(the frozen phase-1 sampler)",
     "0.746 (−27%) · kurt 3.0 — modulation DILUTED"),
    (np.asarray(FOR["gen_A"][0]), "same ckpt, noise removed: $\\mu$-cascade only\n(yesterday's forensic)",
     "1.536 (+51%) · kurt 32.5 — skeleton at 1/3 amplitude"),
    (np.asarray(C1["gen_A"][0]), "C1: plain CFM + augmentation, no head\n(the overnight arm)",
     "0.921 (−10%) · kurt 4.4 — tails still tame"),
]
vmin, vmax = np.percentile(real, [1, 99])
fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
for ax, (m, title, stats) in zip(axes, panels):
    ax.imshow(m, cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(stats, fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("One held-out field, one octave-4 coarse start, three samplers — color scale fixed to the REAL map (1–99 pct); "
             "quoted: measured end-to-end oct-2 var_slope (deficit) and kurtosis, arm A",
             fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(os.path.join(REPO, "results_p2", "maps_channels.png"), dpi=150,
            bbox_inches="tight")
print("wrote maps_channels.png")

# ------------------------------------------------------------------ FIG 2: peaks
def peaks(field, nu):
    c = field[1:-1, 1:-1]
    is_max = np.ones(c.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            is_max &= c > field[1 + dy:field.shape[0] - 1 + dy,
                                1 + dx:field.shape[1] - 1 + dx]
    ys, xs = np.nonzero(is_max & (c > nu))
    return ys + 1, xs + 1

peak_panels = [
    (real, "REAL",
     "mean counts/field: 590 (ν=1) · 180 (ν=2.5) · 123 (ν=3)"),
    (np.asarray(AUG["gen_A"][0]), "4a generator (NLL head)",
     "777 (+32%, z=+14) · 176 (−2%) · 105 (−14%, z=−6.6)\nsign-flip: spurious low peaks, MISSING extremes"),
    (np.asarray(C1["gen_A"][0]), "C1 generator (plain CFM)",
     "716 (+21%, z=+9.1) · 211 (+17%, z=+8.7) · 142 (+16%, z=+7.4)\nuniform UP-tilt: excess peaks at EVERY threshold"),
]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0))
for ax, (m, title, stats) in zip(axes, peak_panels):
    ax.imshow(m, cmap="gray", vmin=vmin, vmax=vmax)
    ys, xs = peaks(m, 2.5)
    for y, x in zip(ys, xs):
        ax.add_patch(Circle((x, y), 2.6, fill=False, color="red", lw=0.9))
    ax.set_title(f"{title} — {len(ys)} peaks ≥2.5σ on THIS map", fontsize=9.5)
    ax.set_xlabel(stats, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Where the peaks go — red circles: local maxima ≥2.5σ on the displayed (index-0) field; captions: measured 64-field "
             "means from the committed jsons. Both generators pass power-level checks; the peak function disagrees, in opposite ways.",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(os.path.join(REPO, "results_p2", "maps_peaks.png"), dpi=150,
            bbox_inches="tight")
print("wrote maps_peaks.png")

# --------------------------------------------------------------- FIG 3: locality
cA = real.copy()
for _ in range(2):
    cA, _bands = haar_level(cA)          # octave-2 coarse (32x32)
py, px = 16, 18                          # a representative interior point (fixed)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6),
                         gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
ax.imshow(cA, cmap="inferno")
ax.add_patch(Circle((px, py), 1.5, fill=False, color="cyan", lw=2.2))
ax.add_patch(Circle((px, py), 8, fill=False, color="white", lw=1.2,
                    linestyle="--"))
ax.plot([px], [py], "c+", ms=10, mew=2)
ax.set_title("octave-2 coarse of the same field (32²)\ncyan r=1: ALL usable context — white r=8: adds nothing",
             fontsize=9.5)
ax.set_xticks([]); ax.set_yticks([])
import json
curves = json.load(open(os.path.join(REPO, "results_p2", "stageB1_curves.json")))
row = curves["gowerstreet"]["2"]
grid = sorted(int(r) for r in row["w"].keys())
v0 = row["w"]["0"]["ridge"]
ax = axes[1]
ax.errorbar(grid, [row["w"][str(r)]["ridge"] / v0 for r in grid],
            yerr=[row["w"][str(r)]["ridge_se"] / v0 for r in grid],
            fmt="o-", color="tab:orange")
ax.axvline(1, color="cyan", lw=2)
ax.axvline(8, color="gray", lw=1, ls="--")
ax.set_xlabel("context radius r (coarse pixels)")
ax.set_ylabel("remaining detail variance V(r)/V(center)")
ax.set_title("measured: one 9% drop at r=1, then flat (octave-2 grid to r=8; octave 1 flat to r=12)", fontsize=9.5)
ax.grid(alpha=0.3)
fig.suptitle("B1 locality on the map: predicting the next level of detail at the marked point uses ~one coarse pixel of context",
             fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(REPO, "results_p2", "maps_locality.png"), dpi=150,
            bbox_inches="tight")
print("wrote maps_locality.png")
