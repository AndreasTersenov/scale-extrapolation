#!/usr/bin/env python
"""A one-page 'where are we' roadmap of the whole investigation and its current state."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GREEN, RED, AMBER, BLUE, GREY = "#2c7c3f", "#c0392b", "#d59a00", "#0072B2", "#555555"

# (y, title, body, status-color, tag)
steps = [
    (10.4, "STAGE 0 — measure real fields", "The non-Gaussian statistics DRIFT with scale, and the drift is low-dimensional (2 numbers).", GREEN, "DONE"),
    (9.0, "Toy ladder (i)(ii)(iii)", "Single-octave overfit ✓, coarse-to-fine recursion ✓, GRF end-to-end NULL (P-null) ✓.", GREEN, "GREEN"),
    (7.6, "P4 — power spectrum extrapolates", "Both arms get the amplitude right at the untrained octave (~5%).", GREEN, "PASS"),
    (6.2, "P5 — THE BREAK (load-bearing)", "Weight-tied arm A (no scale input) gets the conditional non-Gaussianity WRONG\nat the untrained octave (5.8-11.3 sigma). Confirmed, robust.", GREEN, "CONFIRMED"),
    (4.6, "P6 — the repair", "Arm B (given the 2-D scale coordinate) should fix it. BLOCKED: see diagnosis.", RED, "BLOCKED"),
    (3.2, "DIAGNOSIS (reconvene: K-T2 did NOT fire)", "The conditioning is fine (FiLM moves the right way). The blocker is the GENERATOR:\nL2 flow matching under-disperses conditional variance, WORSE with training.", AMBER, "cause found"),
    (1.6, "Variance-faithful program", "(a) SDE churn, (b) checkpoint sweep, (c)+(c') dispersion penalties — ALL fall short of\nthe fidelity bar. Twice-confirmed: a training penalty can't fix a deterministic-sampler problem.", RED, "a,b,c,c' fail"),
    (0.1, "4a CONFIRMED -> 4b' lever FAILED -> 4b'-ii stopped at its own gate", "4a: collapse ELIMINATED by 8x data (memorization law causally confirmed; augmentation\nfrozen in). 4b': corruption fixed the HEADS (~at real) but not the recursion. 4b'-ii\n(inference-time matched corruption): the pre-generation drift measurement fired the stop --\nthe drift is NOT additive noise (white noise can't reproduce the damage within the trained\nrange). Next lever must be drift-shaped (self-conditioning) -- reconvene decides.\nFigures: smatched_4bpii.png, readout_4bp.png, signature_4a.png.", "#D55E00", "STOP->reconvene"),
]
fig, ax = plt.subplots(figsize=(12, 11))
ax.set_xlim(0, 10); ax.set_ylim(-0.6, 11.4); ax.axis("off")
for i, (y, title, body, col, tag) in enumerate(steps):
    box = FancyBboxPatch((0.4, y), 8.2, 1.15, boxstyle="round,pad=0.04,rounding_size=0.12",
                         linewidth=2, edgecolor=col, facecolor=col + "14")
    ax.add_patch(box)
    ax.text(0.65, y + 0.83, title, fontsize=12.5, fontweight="bold", color=col, va="center")
    ax.text(0.65, y + 0.33, body, fontsize=9.6, color="#222222", va="center", wrap=True)
    ax.text(9.35, y + 0.58, tag, fontsize=9.5, fontweight="bold", color=col, ha="center",
            va="center", rotation=0,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=1.5))
    if i < len(steps) - 1:
        ny = steps[i + 1][0]
        ax.add_patch(FancyArrowPatch((4.5, y), (4.5, ny + 1.15), arrowstyle="-|>",
                                     mutation_scale=16, color=GREY, lw=1.5))
# side note: the prior
ax.text(5.0, -0.45, "Prior for when the generator is fixed:  once dispersion is restored (by churn), arm B already hit ~90% repair "
        "→ P6 is expected to PASS.", fontsize=9.6, style="italic", color=BLUE, ha="center")
ax.set_title("Where we are: the break (P5) is proven; the repair (P6) is blocked by ONE generator-side problem",
             fontsize=13.5, fontweight="bold", pad=14)
fig.tight_layout()
out = os.path.join(REPO, "results", "roadmap.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
