#!/usr/bin/env python
"""Map-level explainers for the attempt-4a result (eyes-first, no statistics).

1. maps_4a_texture.png   -- same 8x8 start: REAL map vs the G-1c generator (collapsed
                            sigma) vs the 4a augmented generator (live sigma), + zooms.
2. maps_4a_diversity.png -- the dice-roll test: two independent draws from the SAME
                            coarse start, per generator, plus their |difference| and
                            the sigma-map. A collapsed generator produces near-clones
                            (dark difference); a variance-faithful one rolls the dice.
"""
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
from wfm.dataset import normalize_tiles
from wfm.generate import generate_recursive
from wfm.model import ConditionalUNet

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMAP = "inferno"


def show(ax, img, title, vmin=None, vmax=None, fs=10, cmap=CMAP):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=fs)
    ax.set_xticks([]); ax.set_yticks([])


def clim(img, lo=1, hi=99.5):
    return np.percentile(img, lo), np.percentile(img, hi)


def load_arm_a(ckpt_dir):
    with open(os.path.join(REPO, "data_cache", ckpt_dir, "armA_gowerstreet.pkl"),
              "rb") as fh:
        c = pickle.load(fh)
    model = ConditionalUNet(out_channels=3, channels=tuple(c["channels"]),
                            bottleneck=c["channels"][-1] * 2, cond_dim=0,
                            cond_mode=c["cond_mode"], variance_head=True)
    return model, c


# ------------------------------------------------ figure 1: texture, from saved runs
base = np.load(os.path.join(REPO, "results", "arms_nll.npz"))
aug = np.load(os.path.join(REPO, "results", "arms_aug.npz"))
assert np.allclose(base["real"], aug["real"])            # same held-out fields
real, gN, gA = base["real"], base["gen_A"], aug["gen_A"]
i, Z = 0, 40
_, c4 = haar.octave_pair(jnp.asarray(real[i])[None, :, :, None], 4)
thumb = np.asarray(c4)[0, :, :, 0]
fig, ax = plt.subplots(2, 4, figsize=(12.8, 6.9))
v = clim(real[i])
show(ax[0, 0], thumb, "the 8x8 coarse start\n(all any generator is given)",
     vmin=clim(thumb)[0], vmax=clim(thumb)[1])
show(ax[0, 1], real[i], "REAL held-out map", vmin=v[0], vmax=v[1])
show(ax[0, 2], gN[i], "G-1c generator\n(σ collapsed: memorized texture)",
     vmin=v[0], vmax=v[1])
show(ax[0, 3], gA[i], "4a generator (8x augmented:\nσ-channel alive, ~90%)",
     vmin=v[0], vmax=v[1])
from matplotlib.patches import Rectangle
for col in (1, 2, 3):        # mark the SAME zoom window on each full map
    ax[0, col].add_patch(Rectangle((-0.5, -0.5), Z, Z, fill=False, edgecolor="white",
                                   ls="--", lw=1.4))
vz = clim(real[i][:Z, :Z])
ax[1, 0].axis("off")
ax[1, 0].text(0.5, 0.5, "ZOOM: the SAME 40x40\nwindow of each map\n(dashed box above) →",
              ha="center", va="center", fontsize=11, transform=ax[1, 0].transAxes)
for col, (img, lab) in enumerate(((real[i], "real texture"), (gN[i], "G-1c texture"),
                                  (gA[i], "4a texture")), start=1):
    show(ax[1, col], img[:Z, :Z], "", vmin=vz[0], vmax=vz[1])
    ax[1, col].set_xlabel(lab, fontsize=10.5)
fig.suptitle("Same start, three maps: the real field vs the collapsed generator vs the augmented one (arm A)",
             fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.subplots_adjust(hspace=0.16)
fig.savefig(os.path.join(REPO, "results", "maps_4a_texture.png"), dpi=130,
            bbox_inches="tight")
print("wrote results/maps_4a_texture.png")

# ------------------------------------- figure 2: the dice-roll (diversity) test
tiles = np.load(os.path.join(REPO, "data_cache", "tiles_pnull.npz"))["gowerstreet"]
heldout = normalize_tiles(tiles.astype(np.float32)[-64:])
_, coarse4 = haar.octave_pair(heldout[i:i + 1], 4)
det2, coarse2 = haar.octave_pair(heldout[i:i + 1], 2)

rows = []
for name, ckpt_dir in (("G-1c generator (σ collapsed)", "ckpt_nll"),
                       ("4a generator (σ alive)", "ckpt_aug")):
    model, c = load_arm_a(ckpt_dir)
    std = {int(k): v for k, v in c["std_by_j"].items()}
    draws = [np.asarray(generate_recursive(model.apply, c["params"], coarse4, 4,
                                           jax.random.PRNGKey(k), std, cond_fn=None,
                                           nll=True))[0, :, :, 0] for k in (1, 2)]
    out0 = model.apply({"params": c["params"]},
                       jnp.zeros(coarse2.shape[:3] + (3,)), jnp.zeros((1,)),
                       coarse2, None)
    sig = np.exp(np.clip(np.asarray(out0[..., 3:]), -5, 3))[0].mean(-1)
    rows.append((name, draws, sig))

fig, ax = plt.subplots(2, 4, figsize=(12.8, 6.9))
v = clim(np.concatenate([r[1][0] for r in rows]))
dmax = max(np.abs(rows[0][1][0] - rows[0][1][1]).max(),
           np.abs(rows[1][1][0] - rows[1][1][1]).max()) * 0.6
smax = max(rows[0][2].max(), rows[1][2].max())
for r, (name, draws, sig) in enumerate(rows):
    show(ax[r, 0], draws[0], "draw 1" if r == 0 else "", vmin=v[0], vmax=v[1])
    show(ax[r, 1], draws[1], "draw 2 (same start, new dice)" if r == 0 else "",
         vmin=v[0], vmax=v[1])
    show(ax[r, 2], np.abs(draws[0] - draws[1]),
         "|draw 1 − draw 2|\n(dark = clones)" if r == 0 else "", vmin=0, vmax=dmax)
    show(ax[r, 3], sig, "the σ map it rolls with\n(octave 2)" if r == 0 else "",
         vmin=0, vmax=smax)
    ax[r, 0].set_ylabel(name, fontsize=10.5, fontweight="bold")
fig.suptitle("The dice-roll test: two draws from the SAME 8x8 start — a collapsed generator produces near-clones;\n"
             "the augmented one actually rolls the dice (shared color scales per column)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.subplots_adjust(hspace=0.1)
fig.savefig(os.path.join(REPO, "results", "maps_4a_diversity.png"), dpi=130,
            bbox_inches="tight")
print("wrote results/maps_4a_diversity.png")
