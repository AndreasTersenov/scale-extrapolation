#!/usr/bin/env python
"""SPEC-gallery.md renderer — four map-level figures from FROZEN artifacts only.

Honesty rules (binding, from the SPEC): field index 0 (first held-out) everywhere;
shared color scales within comparison rows (stated per figure); every generated panel
captioned as a conditional SAMPLE; measured statistics quoted per panel from the
committed score jsons (no recomputation); seeds/indices logged in
log/2026-07-16-gallery.md. FIG-G1..G3 use the committed run outputs (arms_aug.npz /
arms_selfsim.npz, generation key PRNGKey(1) from their runs); FIG-G4 draws nine new
conditional samples from the frozen 4a arm-A checkpoint with keys PRNGKey(1..9).
"""
import json
import os
try:
    os.sched_setaffinity(0, set(range(4)))
except Exception:
    pass
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wfm import haar
from wfm.generate import generate_recursive
from wfm.model import ConditionalUNet

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDX = 0                                     # SPEC rule 1: first held-out index
CMAP = "inferno"

aug = np.load(os.path.join(REPO, "results", "arms_aug.npz"))
ssim = np.load(os.path.join(REPO, "results", "arms_selfsim.npz"))
sc_g = json.load(open(os.path.join(REPO, "results", "arms_aug_score.json")))
sc_s = json.load(open(os.path.join(REPO, "results", "arms_selfsim_score.json")))
peaks = json.load(open(os.path.join(REPO, "results", "downstream_peaks.json")))
coords = json.load(open(os.path.join(REPO, "data_cache",
                                     "running_couplings.json")))["gowerstreet"]


def clim(*imgs, lo=1, hi=99.5):
    a = np.concatenate([np.asarray(i).ravel() for i in imgs])
    return np.percentile(a, lo), np.percentile(a, hi)


def show(ax, img, vmin, vmax, cmap=CMAP):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])


def coarse_at(field, j):
    if j == 0:
        return np.asarray(field)
    _, c = haar.octave_pair(jnp.asarray(field)[None, :, :, None], j)
    return np.asarray(c)[0, :, :, 0]


def band_at(field, j):
    det, c = haar.octave_pair(jnp.asarray(field)[None, :, :, None], j)
    b = haar.idwt2(jnp.zeros_like(c), (det[..., 0:1], det[..., 1:2], det[..., 2:3]))
    for _ in range(j - 1):
        b = haar.idwt2(b, (jnp.zeros_like(b),) * 3)
    return np.asarray(b)[0, :, :, 0]


# ================= FIG-G1: the ladder walkthrough (arm B, coupling dial visible) ====
real0, genB0 = aug["real"][IDX], aug["gen_B"][IDX]
rows = [4, 3, 2, 1, 0]
fig, ax = plt.subplots(5, 2, figsize=(8.6, 21.0))
for r, j in enumerate(rows):
    ra, ga = coarse_at(real0, j), coarse_at(genB0, j)
    v = clim(ra, ga)                                   # shared scale per row
    show(ax[r, 0], ra, *v)
    show(ax[r, 1], ga, *v)
    n = 128 // (2 ** j)
    if j == 4:
        ax[r, 0].set_title(f"REAL, octave 4 coarse ({n}x{n})", fontsize=10)
        ax[r, 1].set_title("the SHARED start (identical by construction)", fontsize=10)
    else:
        js = j + 1                # the map at octave j is made by sampling octave-js detail
        vs_g = sc_g["armB"][str(js)]["var_slope"]
        vs_r = sc_g["real"][str(js)]["var_slope"]
        lab0 = f"REAL at octave {j} ({n}x{n})" if j > 0 else "REAL, full resolution"
        ax[r, 0].set_title(lab0, fontsize=10)
        cap = (f"SAMPLED octave-{js} detail added, dial=[{coords[str(js)][0]:.2f}, "
               f"{coords[str(js)][1]:.1f}]\nmeasured var_slope at octave {js} "
               f"(64 maps): {vs_g:.2f} vs real {vs_r:.2f}")
        if j == 0:
            cap += ("\na conditional SAMPLE: same start, different detail dice "
                    "(pixel match NOT expected)")
        ax[r, 1].set_title(cap, fontsize=8.5)
    ax[r, 0].set_ylabel(f"octave {j}" if j > 0 else "full", fontsize=10,
                        fontweight="bold")
fig.suptitle("FIG-G1 — how it works: the generator climbs the ladder from the shared 8x8 start,\n"
             "sampling each octave's detail conditioned on the map above + the coupling dial "
             "(arm B, frozen 4a checkpoint; held-out field #0)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.subplots_adjust(hspace=0.18)
fig.savefig(os.path.join(REPO, "results", "gallery_ladder.png"), dpi=130,
            bbox_inches="tight")
plt.close(fig)
print("wrote results/gallery_ladder.png")

# ================= FIG-G2: the boundary, in maps (arm A both rows) =================
fig, ax = plt.subplots(2, 2, figsize=(10.6, 11.2))
rows2 = [
    ("gowerstreet (couplings DRIFT x2 across octaves)", aug["real"][IDX],
     aug["gen_A"][IDX], sc_g, "1.117±0.051", "0.680±0.011",
     "9.8±1.5", "2.6±0.2"),
    ("self-similar control (NO drift; same architecture)", ssim["real"][IDX],
     ssim["gen_A"][IDX], sc_s, "0.558±0.002", "0.544±0.002",
     "1.02±0.02", "0.98±0.02"),
]
for r, (name, rf, gf, sc, vsr, vsg, kur, kug) in enumerate(rows2):
    rb, gb = band_at(rf, 1), band_at(gf, 1)
    s = np.percentile(np.abs(np.concatenate([rb.ravel(), gb.ravel()])), 99)
    show(ax[r, 0], rb, -s, s, cmap="RdBu_r")           # shared scale per row
    show(ax[r, 1], gb, -s, s, cmap="RdBu_r")
    ax[r, 0].set_title(f"REAL extrapolated-octave detail\nvar_slope {vsr}, "
                       f"kurtosis {kur}", fontsize=9.5)
    ax[r, 1].set_title(f"GENERATED (conditional sample, same start)\nvar_slope {vsg}, "
                       f"kurtosis {kug}", fontsize=9.5)
    ax[r, 0].set_ylabel(name, fontsize=9.5, fontweight="bold")
fig.suptitle("FIG-G2 — the boundary, in maps: where the between-scale law is scale-invariant the method\n"
             "works (bottom row: statistically indistinguishable); where it drifts, the drift is the error\n"
             "(top row: generated texture too uniform, extremes missing). Octave-1 band, held-out field #0,\n"
             "arm A frozen checkpoints; one shared color scale per row; panels are SAMPLES (no pixel match)",
             fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.925])
fig.subplots_adjust(hspace=0.16)
fig.savefig(os.path.join(REPO, "results", "gallery_boundary.png"), dpi=130,
            bbox_inches="tight")
plt.close(fig)
print("wrote results/gallery_boundary.png")

# ================= FIG-G3: where the peaks go (arm A map + measured inset) =========
def peak_coords(f, lo, hi=None):
    c = f[1:-1, 1:-1]
    is_max = np.ones(c.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            is_max &= c > f[1 + dy:f.shape[0] - 1 + dy, 1 + dx:f.shape[1] - 1 + dx]
    sel = is_max & (c > lo) & (c <= hi if hi is not None else np.ones_like(is_max))
    y, x = np.where(sel)
    return x + 1, y + 1


realA, genA = aug["real"][IDX], aug["gen_A"][IDX]
fig = plt.figure(figsize=(13.8, 5.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.55], wspace=0.12)
v = clim(realA, genA)                                   # shared scale across both maps
for col, (f, name) in enumerate(((realA, "REAL held-out map #0"),
                                 (genA, "GENERATED (conditional sample, arm A)"))):
    a = fig.add_subplot(gs[0, col])
    show(a, f, *v)
    x, y = peak_coords(f, 2.5)
    a.scatter(x, y, s=70, facecolors="none", edgecolors="#00E5FF", lw=1.3,
              label=f"peaks ≥ 2.5σ  (n={len(x)})")
    if col == 1:                       # 1.0-1.5σ band on the generated map only (SPEC)
        x2, y2 = peak_coords(f, 1.0, 1.5)
        a.scatter(x2, y2, s=4, color="#7CFC00", alpha=0.6,
                  label=f"peaks 1.0–1.5σ  (n={len(x2)})")
    a.legend(fontsize=8, loc="lower left", framealpha=0.85)
    a.set_title(name, fontsize=10.5)
ai = fig.add_subplot(gs[0, 2])
xb = np.arange(2)
for k, (who, col) in enumerate((("real", "#000000"), ("A", "#0072B2"),
                                ("B", "#E69F00"))):
    m = [peaks["counts"][who]["1.0"][0], peaks["counts"][who]["3.0"][0]]
    ai.bar(xb + (k - 1) * 0.26, m, 0.26, color=col,
           label={"real": "real", "A": "arm A", "B": "arm B"}[who])
ai.set_xticks(xb)
ai.set_xticklabels([f"ν=1\nz=+{peaks['z']['A']['1.0']:.0f}/+{peaks['z']['B']['1.0']:.0f}",
                    f"ν=3\nz={peaks['z']['A']['3.0']:.0f}/{peaks['z']['B']['3.0']:.0f}"])
ai.set_ylabel("peaks per map (measured, 64 maps)", labelpad=2)
ai.yaxis.set_label_position("right")
ai.yaxis.tick_right()
ai.set_title("the measured bias\n(downstream_peaks.json)", fontsize=9.5, pad=14)
ai.legend(fontsize=8)
ai.grid(alpha=0.25)
fig.suptitle("FIG-G3 — where the peaks go: power-level checks pass at ≤7% on this generator, yet the peak function is\n"
             "biased +14σ (spurious low peaks, green dots) to −12σ (missing extremes, cyan circles). Shared color scale;\n"
             "the generated panel is a conditional sample — compare peak POPULATIONS, not positions",
             fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(os.path.join(REPO, "results", "gallery_peaks.png"), dpi=130,
            bbox_inches="tight")
plt.close(fig)
print("wrote results/gallery_peaks.png")

# ================= FIG-G4: the ensemble view (nine new samples, seeds 1..9) ========
with open(os.path.join(REPO, "data_cache", "ckpt_aug", "armA_gowerstreet.pkl"),
          "rb") as fh:
    ck = pickle.load(fh)
model = ConditionalUNet(out_channels=3, channels=tuple(ck["channels"]),
                        bottleneck=ck["channels"][-1] * 2, cond_dim=ck["cond_dim"],
                        cond_mode=ck["cond_mode"], variance_head=True)
_, c4 = haar.octave_pair(jnp.asarray(realA)[None, :, :, None], 4)
std = {int(k): v for k, v in ck["std_by_j"].items()}
samples = [np.asarray(generate_recursive(model.apply, ck["params"], c4, 4,
                                         jax.random.PRNGKey(s), std,
                                         nll=True))[0, :, :, 0]
           for s in range(1, 10)]
fig = plt.figure(figsize=(14.6, 10.6))
gs = fig.add_gridspec(3, 4, wspace=0.06, hspace=0.10)
v = clim(realA, *samples)                               # one shared scale, all panels
a = fig.add_subplot(gs[:, 0])
show(a, realA, *v)
a.set_title("the ONE real field\n(held-out #0)", fontsize=11)
for spine in a.spines.values():
    spine.set_edgecolor("#D55E00"); spine.set_linewidth(3)
for k, smp in enumerate(samples):
    a = fig.add_subplot(gs[k // 3, 1 + k % 3])
    show(a, smp, *v)
    a.set_title(f"sample {k + 1}  (seed {k + 1})", fontsize=9)
fig.suptitle("FIG-G4 — what \"sampling the conditional\" means: nine generated maps from the SAME 8x8 coarse start\n"
             "(arm A frozen 4a checkpoint, seeds 1–9, shared color scale). Large-scale structure is inherited from the\n"
             "start; every fine texture is a fresh draw — the real field is ONE more draw from the (ideal) same law",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(REPO, "results", "gallery_ensemble.png"), dpi=130,
            bbox_inches="tight")
plt.close(fig)
print("wrote results/gallery_ensemble.png")
