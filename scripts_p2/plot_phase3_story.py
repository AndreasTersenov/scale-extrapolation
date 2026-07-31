"""DESCRIPTIVE story figures for the phase-3 close (committed maps only;
prereg-exempt re-rendering — no new adjudication).

V1 phase3_story_blind.png    — the one-shot on maps: real vs the blind
                               fresh-seed final config at the held-out
                               octave, native vs declared resolution,
                               peaks marked, counts annotated.
V2 phase3_story_seeds.png    — the robustness finding on maps: the three
                               training seeds at the declared resolution
                               (why two fresh seeds fail the 0.5-px rule).
V3 phase3_story_minkowski.png— what the untouched judge saw: per-entry z
                               decomposition of T_MF = 6.35 (delivers
                               FIGURES.md F20) + excursion-set masks at
                               the two most deviant entries.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts_p2"))

from audit_peak_ci import peaks_count
from minkowski_judge import NUS, judge_T

NU = 2.5


def peak_xy(field, nu=NU):
    f = (field - field.mean()) / field.std()
    c = f[1:-1, 1:-1]
    is_max = np.ones(c.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            is_max &= c > f[1 + dy:f.shape[0] - 1 + dy,
                            1 + dx:f.shape[1] - 1 + dx]
    ys, xs = np.nonzero(is_max & (c > nu))
    return xs + 1, ys + 1


real = [np.asarray(f, np.float64) for f in
        np.load(os.path.join(RES, "arms_stageD.npz"),
                allow_pickle=True)["real"]]
B = np.load(os.path.join(RES, "stage3_blind_final.npz"))
blind = [np.asarray(f, np.float64) for k in ("final1", "final2", "final3")
         for f in B[k]]
counts = [peaks_count(f, NU) for f in real]
tile = int(np.argmin(np.abs(np.array(counts) - np.median(counts))))

# ---- V1: the one-shot on maps ------------------------------------------------
stacks = {"REAL (held-out octave)": real[tile],
          "blind fresh seed — final config": np.asarray(B["final1"][tile],
                                                        np.float64)}
ref = real[tile]
vmin, vmax = np.percentile((ref - ref.mean()) / ref.std(), [1, 99.5])
fig, axes = plt.subplots(2, 2, figsize=(7.6, 8.2),
                         gridspec_kw={"hspace": 0.16})
for col, (label, f) in enumerate(stacks.items()):
    for row, sig in enumerate((0.0, 0.5)):
        g = f if sig == 0 else gaussian_filter(f, sig)
        gs = (g - g.mean()) / g.std()
        ax = axes[row, col]
        ax.imshow(gs, cmap="magma", vmin=vmin, vmax=vmax, origin="lower",
                  interpolation="nearest")
        xs, ys = peak_xy(g)
        ax.plot(xs, ys, "o", ms=5, mfc="none", mec="cyan", mew=1.1)
        ax.set_title(f"{label}\n{peaks_count(g, NU)} peaks (ν≥{NU})"
                     if row == 0 else f"{peaks_count(g, NU)} peaks",
                     fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
axes[0, 0].set_ylabel("native resolution", fontsize=11)
axes[1, 0].set_ylabel("declared resolution (σ=0.5 px)", fontsize=11)
fig.suptitle(f"The one-shot blind test on maps (tile {tile}; same coarse "
             "conditioning)\nstack verdict: peaks PASS at declared "
             "resolution (−1.1%/−4.0%)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(RES, "phase3_story_blind.png"), dpi=150)
plt.close(fig)
print("wrote phase3_story_blind.png")

# ---- V2: the seed story ------------------------------------------------------
greal = [np.asarray(f, np.float64) for f in
         np.load(os.path.join(RES, "arms_c1t_gowerstreet.npz"),
                 allow_pickle=True)["real"]]
gcounts = [peaks_count(f, NU) for f in greal]
gtile = int(np.argmin(np.abs(np.array(gcounts) - np.median(gcounts))))
seed_stacks = {
    "REAL": greal[gtile],
    "seed 0 (shipped)\npick @16000": np.asarray(np.load(os.path.join(
        RES, "l1pp_main_gen.npz"))["adj1"][gtile], np.float64),
    "seed 1 (fresh)\npick @3500": np.asarray(np.load(os.path.join(
        RES, "stage3_seed1_final.npz"))["final1"][gtile], np.float64),
    "seed 2 (fresh)\npick @5500": np.asarray(np.load(os.path.join(
        RES, "stage3_seed2_final.npz"))["final1"][gtile], np.float64)}
STACK_EX = {"seed 0 (shipped)\npick @16000": "−1.1%/−0.3%  PASS",
            "seed 1 (fresh)\npick @3500": "+3.6%/+6.5%  FAIL",
            "seed 2 (fresh)\npick @5500": "+4.5%/+6.6%  FAIL"}
fig, axes = plt.subplots(2, 4, figsize=(13.2, 7.6),
                         gridspec_kw={"hspace": 0.2})
rref = greal[gtile]
vmin, vmax = np.percentile((rref - rref.mean()) / rref.std(), [1, 99.5])
for col, (label, f) in enumerate(seed_stacks.items()):
    for row, sig in enumerate((0.0, 0.5)):
        g = f if sig == 0 else gaussian_filter(f, sig)
        gs = (g - g.mean()) / g.std()
        ax = axes[row, col]
        ax.imshow(gs, cmap="magma", vmin=vmin, vmax=vmax, origin="lower",
                  interpolation="nearest")
        xs, ys = peak_xy(g)
        ax.plot(xs, ys, "o", ms=4.5, mfc="none", mec="cyan", mew=1.0)
        n = peaks_count(g, NU)
        top = f"{label}\n{n} peaks" if row == 0 else f"{n} peaks"
        if row == 1 and label in STACK_EX:
            top += f"\nstack: {STACK_EX[label]}"
        ax.set_title(top, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
axes[0, 0].set_ylabel("native", fontsize=11)
axes[1, 0].set_ylabel("σ = 0.5 px", fontsize=11)
fig.suptitle("The robustness finding on maps (trained scales, tile "
             f"{gtile}): fresh seeds keep extra peaks AT the declared "
             "resolution — the 0.5-px claim attaches to the selected "
             "configuration", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(os.path.join(RES, "phase3_story_seeds.png"), dpi=150)
plt.close(fig)
print("wrote phase3_story_seeds.png")

# ---- V3: what the Minkowski judge saw ---------------------------------------
gs_stack = [gaussian_filter(f, 0.5) for f in blind]
rs_stack = [gaussian_filter(f, 0.5) for f in real]
T, z = judge_T(gs_stack, rs_stack, seed=20260845)
labels = ([f"V0 area\nν={n:g}" for n in NUS]
          + [f"V1 boundary\nν={n:g}" for n in NUS]
          + [f"V2 Euler\nν={n:g}" for n in NUS])
order = np.argsort(-np.abs(z))
top2 = order[:2]
fig = plt.figure(figsize=(13.5, 4.6))
gsx = fig.add_gridspec(1, 3, width_ratios=[2.1, 1, 1])
ax = fig.add_subplot(gsx[0, 0])
colors = (["tab:blue"] * 6 + ["tab:orange"] * 6 + ["tab:green"] * 6)
ax.bar(range(18), z, color=colors)
ax.axhline(3.5, color="r", ls="--", lw=0.8)
ax.axhline(-3.5, color="r", ls="--", lw=0.8)
ax.set_xticks(range(18))
ax.set_xticklabels(labels, fontsize=6.5, rotation=60)
ax.set_ylabel("z (gen − real)")
ax.set_title(f"Where T_MF = {T:.2f} lives: the 18 judge entries "
             "(declared resolution)", fontsize=10)
mask_nu = []
for idx in top2:
    nu = NUS[idx % 6]
    if nu not in mask_nu:
        mask_nu.append(nu)
while len(mask_nu) < 2:
    mask_nu.append(NUS[3])
for i, nu in enumerate(mask_nu[:2]):
    ax = fig.add_subplot(gsx[0, 1 + i])
    r = gaussian_filter(real[tile], 0.5)
    g = gaussian_filter(np.asarray(B["final1"][tile], np.float64), 0.5)
    r = (r - r.mean()) / r.std()
    g = (g - g.mean()) / g.std()
    comp = np.zeros(r.shape + (3,))
    comp[..., 0] = (g > nu)  # red: generated excursion
    comp[..., 2] = (r > nu)  # blue: real excursion (overlap -> magenta)
    ax.imshow(comp, origin="lower", interpolation="nearest")
    ax.set_title(f"excursion sets at ν={nu:g}\nred=gen, blue=real, "
                 "magenta=both", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(RES, "phase3_story_minkowski.png"), dpi=150)
print("wrote phase3_story_minkowski.png; top entries:",
      [(labels[i].replace(chr(10), " "), round(float(z[i]), 2))
       for i in order[:4]])
