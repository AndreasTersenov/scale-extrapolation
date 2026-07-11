#!/usr/bin/env python
"""Actual-maps explainer figures (no statistics, just the fields themselves).

1. maps_fields.png   -- what the three datasets look like (gowerstreet / GRF / hf_pm).
2. maps_ladder.png   -- one gowerstreet tile taken apart into the octave ladder the
                        generator climbs (coarse thumbnails + the detail band added at
                        each rung).
3. maps_break.png    -- the break in pictures: held-out REAL maps next to what arm A /
                        arm B (current FiLM checkpoint, pre-NLL) generate from the SAME
                        8x8 coarse start, plus fine-texture zooms.
"""
import os
try:
    os.sched_setaffinity(0, set(range(4)))
except Exception:
    pass
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wfm import haar

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMAP = "inferno"


def norm(t):
    return (t - t.mean()) / t.std()


def show(ax, img, title, vmin=None, vmax=None, fs=10, cmap=CMAP):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=fs)
    ax.set_xticks([]); ax.set_yticks([])


def clim(img, lo=1, hi=99.5):
    return np.percentile(img, lo), np.percentile(img, hi)


# ---------------------------------------------------------------- 1. the data zoo
tiles = np.load(os.path.join(REPO, "data_cache", "tiles_pnull.npz"))
fig, ax = plt.subplots(3, 3, figsize=(10.2, 10.6))
rows = [("gowerstreet", "GOWERSTREET — the training field (simulated weak-lensing mass maps)"),
        ("grf", "GRF — the Gaussian control (self-similar by construction: the null test)"),
        ("hf_pm_1024", "HF_PM — a different simulation (the zero-retrain transfer target)")]
for r, (key, label) in enumerate(rows):
    stack = tiles[key][:3].astype(float)
    v = clim(np.array([norm(s) for s in stack]))
    for c in range(3):
        show(ax[r, c], norm(stack[c]), "", vmin=v[0], vmax=v[1])
    ax[r, 0].set_ylabel(label.split(" — ")[0], fontsize=11, fontweight="bold")
    ax[r, 1].set_title(label, fontsize=10.5)
fig.suptitle("The actual fields (three 128x128 tiles of each; brighter = more mass)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(REPO, "results", "maps_fields.png"), dpi=130,
            bbox_inches="tight")
print("wrote results/maps_fields.png")

# ------------------------------------------------------- 2. the octave ladder
f = norm(tiles["gowerstreet"][5].astype(float))[None, :, :, None]
fj = jnp.asarray(f)
fig = plt.figure(figsize=(13.5, 6.0))
gs = fig.add_gridspec(2, 5, hspace=0.28, wspace=0.08)
axf = fig.add_subplot(gs[0, 0])
v = clim(f[0, :, :, 0])
show(axf, f[0, :, :, 0], "the full 128x128 map", vmin=v[0], vmax=v[1])
for i, j in enumerate((1, 2, 3, 4)):
    _, coarse = haar.octave_pair(fj, j)
    img = np.asarray(coarse)[0, :, :, 0]
    axc = fig.add_subplot(gs[0, i + 1])
    show(axc, img, f"coarse at octave {j}  ({img.shape[0]}x{img.shape[0]})",
         vmin=clim(img)[0], vmax=clim(img)[1])
axz = fig.add_subplot(gs[1, 0])
axz.axis("off")
axz.text(0.5, 0.5, "the DETAIL each\nrung adds back\n(band per octave)\n"
                   "→ this is what the\ngenerator must invent",
         ha="center", va="center", fontsize=11)
for i, j in enumerate((1, 2, 3, 4)):
    det, coarse = haar.octave_pair(fj, j)
    band = haar.idwt2(jnp.zeros_like(coarse),
                      (det[..., 0:1], det[..., 1:2], det[..., 2:3]))
    for _ in range(j - 1):
        z3 = (jnp.zeros_like(band),) * 3
        band = haar.idwt2(band, z3)
    img = np.asarray(band)[0, :, :, 0]
    axb = fig.add_subplot(gs[1, i + 1])
    s = np.percentile(np.abs(img), 99)
    show(axb, img, f"detail band, octave {j}", vmin=-s, vmax=s, cmap="RdBu_r")
fig.suptitle("One real tile taken apart: the generator starts from the 8x8 thumbnail (right) "
             "and must invent every detail band on the way back", fontsize=12.5)
fig.savefig(os.path.join(REPO, "results", "maps_ladder.png"), dpi=130,
            bbox_inches="tight")
print("wrote results/maps_ladder.png")

# ------------------------------------------------- 3. real vs generated (the break)
d = np.load(os.path.join(REPO, "results", "arms_film.npz"), allow_pickle=True)
real, gA, gB = d["real"], d["gen_A"], d["gen_B"]
idx = [0, 3]
Z = 40                                   # zoom crop size
fig, ax = plt.subplots(3, 4, figsize=(12.6, 10.0))
for r, i in enumerate(idx):
    _, c4 = haar.octave_pair(jnp.asarray(real[i])[None, :, :, None], 4)
    thumb = np.asarray(c4)[0, :, :, 0]
    v = clim(real[i])
    show(ax[r, 0], thumb, "the 8x8 coarse start\n(all the generator is given)" if r == 0
         else "", vmin=clim(thumb)[0], vmax=clim(thumb)[1])
    show(ax[r, 1], real[i], "REAL held-out map" if r == 0 else "", vmin=v[0], vmax=v[1])
    show(ax[r, 2], gA[i], "arm A generates\n(no scale input)" if r == 0 else "",
         vmin=v[0], vmax=v[1])
    show(ax[r, 3], gB[i], "arm B generates\n(+ running-coupling coordinate)" if r == 0
         else "", vmin=v[0], vmax=v[1])
    ax[r, 0].set_ylabel(f"held-out field #{i}", fontsize=10, fontweight="bold")
i = idx[0]
v = clim(real[i][:Z, :Z])
ax[2, 0].axis("off")
ax[2, 0].text(0.5, 0.5, "ZOOM on the fine\ntexture (40x40 corner\nof field #0) →",
              ha="center", va="center", fontsize=11, transform=ax[2, 0].transAxes)
show(ax[2, 1], real[i][:Z, :Z], "real texture", vmin=v[0], vmax=v[1])
show(ax[2, 2], gA[i][:Z, :Z], "arm A texture", vmin=v[0], vmax=v[1])
show(ax[2, 3], gB[i][:Z, :Z], "arm B texture", vmin=v[0], vmax=v[1])
fig.suptitle("The experiment in pictures (CURRENT pre-NLL generator, FiLM checkpoint): same coarse start, "
             "same large blobs --\nthe contest is entirely in the fine texture, which is where the "
             "under-dispersion lives", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(REPO, "results", "maps_break.png"), dpi=130,
            bbox_inches="tight")
print("wrote results/maps_break.png")
