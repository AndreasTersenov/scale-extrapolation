"""Topology explainer (descriptive re-render, committed maps; the scored
declared-resolution domain σ=0.5 px throughout).

Panel A: the full Euler-characteristic curve V2(ν) (dense grid, mean ±
         SE bands) — real vs the blind-run final config; judge thresholds
         marked. The field-standard Minkowski representation.
Panel B/C: the ν=0 excursion set of one tile, real vs generated, with
         CONNECTED COMPONENTS individually colored and counts annotated
         (components, holes, χ) — the topology difference as countable
         islands.
Panel D: per-map χ(ν=0) distributions (32 real vs 96 generated maps).

Writes results_p2/phase3_story_topology.png.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, label

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts_p2"))

from audit_peak_ci import peaks_count
from minkowski_judge import minkowski_vector

NU_GRID = tuple(np.arange(-3.0, 3.51, 0.25))
N = len(NU_GRID)

real = [gaussian_filter(np.asarray(f, np.float64), 0.5) for f in
        np.load(os.path.join(RES, "arms_stageD.npz"),
                allow_pickle=True)["real"]]
B = np.load(os.path.join(RES, "stage3_blind_final.npz"))
gen = [gaussian_filter(np.asarray(f, np.float64), 0.5)
       for k in ("final1", "final2", "final3") for f in B[k]]


def v2_curves(stack):
    vecs = np.array([minkowski_vector(f, NU_GRID) for f in stack])
    v2 = vecs[:, 2 * N:3 * N]
    return v2.mean(0), v2.std(0, ddof=1) / np.sqrt(len(stack)), v2


def components_and_holes(mask):
    """4-connectivity components of the mask; holes = complement components
    not touching the border (matching the judge's chi convention family)."""
    s4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    lab, ncomp = label(mask, structure=s4)
    labc, ncompl = label(~mask, structure=s4)
    border = set(np.concatenate([labc[0], labc[-1], labc[:, 0],
                                 labc[:, -1]]))
    border.discard(0)
    holes = ncompl - len(border)
    return lab, ncomp, holes


m_r, se_r, v2_r = v2_curves(real)
m_g, se_g, v2_g = v2_curves(gen)
i0 = NU_GRID.index(0.0) if 0.0 in NU_GRID else int(np.argmin(np.abs(NU_GRID)))
chi0_r = v2_r[:, i0] * real[0].size
chi0_g = v2_g[:, i0] * gen[0].size

counts = [peaks_count(f, 2.5) for f in real]
tile = int(np.argmin(np.abs(np.array(counts) - np.median(counts))))
rt = (real[tile] - real[tile].mean()) / real[tile].std()
gt = (gen[tile] - gen[tile].mean()) / gen[tile].std()

fig = plt.figure(figsize=(14.5, 8.6))
gsx = fig.add_gridspec(2, 3, width_ratios=[1.5, 1, 1], height_ratios=[1, 1])

# A: V2 curves
ax = fig.add_subplot(gsx[:, 0])
nu = np.array(NU_GRID)
sc = real[0].size
ax.fill_between(nu, (m_r - 3 * se_r) * sc, (m_r + 3 * se_r) * sc,
                color="k", alpha=0.25, label="REAL (±3 SE)")
ax.plot(nu, m_r * sc, "k-", lw=2)
ax.fill_between(nu, (m_g - 3 * se_g) * sc, (m_g + 3 * se_g) * sc,
                color="tab:red", alpha=0.25,
                label="generated, blind run (±3 SE)")
ax.plot(nu, m_g * sc, color="tab:red", lw=2)
for t in (-2, -1, 0, 1, 2, 3):
    ax.axvline(t, color="gray", lw=0.5, ls=":")
ax.axhline(0, color="gray", lw=0.7)
ax.set_xlabel("threshold ν")
ax.set_ylabel("Euler characteristic χ(ν) per map")
ax.set_title("The Euler-characteristic curve (declared resolution):\n"
             "generated maps have too-high χ around the median level set",
             fontsize=11)
ax.legend(fontsize=9)

# B/C: labeled components at nu=0
for i, (name, f) in enumerate((("REAL", rt), ("generated", gt))):
    ax = fig.add_subplot(gsx[0, 1 + i])
    lab, ncomp, holes = components_and_holes(f > 0)
    rng = np.random.default_rng(3)
    lut = np.concatenate([[0], rng.permutation(np.arange(1, ncomp + 1))])
    show = lut[lab]
    ax.imshow(np.ma.masked_equal(show, 0), cmap="tab20", origin="lower",
              interpolation="nearest")
    ax.set_facecolor("black")
    ax.set_title(f"{name}: ν=0 excursion set, tile {tile}\n"
                 f"{ncomp} components, {holes} holes "
                 f"(χ≈{ncomp - holes:+d})", fontsize=9.5)
    ax.set_xticks([])
    ax.set_yticks([])

# D: chi(0) distributions
ax = fig.add_subplot(gsx[1, 1:])
bins = np.linspace(min(chi0_r.min(), chi0_g.min()) - 5,
                   max(chi0_r.max(), chi0_g.max()) + 5, 28)
ax.hist(chi0_r, bins=bins, density=True, alpha=0.55, color="k",
        label=f"REAL (32 maps): mean {chi0_r.mean():+.1f}")
ax.hist(chi0_g, bins=bins, density=True, alpha=0.55, color="tab:red",
        label=f"generated (96 maps): mean {chi0_g.mean():+.1f}")
ax.set_xlabel("per-map Euler characteristic at ν=0")
ax.set_ylabel("density")
ax.set_title("Per-map χ(0): the whole generated population is shifted — "
             "more fragmented half-level sets", fontsize=10)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(RES, "phase3_story_topology.png"), dpi=150)
print(f"wrote phase3_story_topology.png; chi0 real {chi0_r.mean():+.1f} "
      f"gen {chi0_g.mean():+.1f}; tile {tile}")
