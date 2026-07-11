#!/usr/bin/env python
"""G-1c failure diagnosis for the Gaussian-NLL head (phase 1c, step 1).

No sampling: works directly from the checkpoints. For each octave and arm, one forward
at the anchor (x_t=0, t=0 | coarse, cond) gives the model's conditional mean mu and
noise sigma=e^g per pixel. The generated conditional variance per coarse quantile bin is
EXACTLY Var(mu|bin) + mean(sigma^2|bin), so we can decompose the var_slope deficit into
"the mean under-modulates" vs "the sigma-head under-modulates" — the question the
reconvene needs answered. Also traces the head's implied var_slope across the
2k-granularity training checkpoints (does it collapse with training, like L2-CFM did?).

Outputs: printed table, results/nll_diagnosis.npz, and two figures
(results/nll_diagnosis.png, results/nll_sigma_maps.png).
"""
import os
try:
    os.sched_setaffinity(0, set(range(4)))
except Exception:
    pass
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wfm import haar
from wfm.dataset import normalize_tiles
from wfm.model import ConditionalUNet

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K, BLUE, GREEN, ORANGE, VERM = "#000000", "#0072B2", "#009E73", "#E69F00", "#D55E00"
N_BINS = 10

tiles = np.load(os.path.join(REPO, "data_cache", "tiles_pnull.npz"))["gowerstreet"]
heldout = normalize_tiles(tiles.astype(np.float32)[-64:])
coords = json.load(open(os.path.join(REPO, "data_cache", "running_couplings.json")))
coords = {int(j): np.asarray(v) for j, v in coords["gowerstreet"].items()}


def load_ckpt(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def head_forward(params, channels, cond_dim, cond_mode, coarse, cond_vec):
    model = ConditionalUNet(out_channels=3, channels=tuple(channels),
                            bottleneck=channels[-1] * 2, cond_dim=cond_dim,
                            cond_mode=cond_mode, variance_head=True)
    B = coarse.shape[0]
    zeros = jnp.zeros(coarse.shape[:3] + (3,))
    out = model.apply({"params": params}, zeros, jnp.zeros((B,)), coarse, cond_vec)
    return np.asarray(out[..., :3]), np.exp(np.clip(np.asarray(out[..., 3:]), -5, 3))


def decompose(mu, sig, detail_n, coarse):
    """Per-coarse-quantile-bin variance profiles, scaledrift-style (standardized w,c).

    Returns c_center, var_real, var_gen, var_from_mean, var_from_noise (gen profiles
    normalized by the GENERATED pooled variance, real by the real pooled variance —
    exactly how coupling_scalars standardizes w before binning)."""
    c = np.tile(np.asarray(coarse).reshape(-1), 3)
    c = (c - c.mean()) / c.std()
    w = np.asarray(detail_n).transpose(3, 0, 1, 2).reshape(-1) if detail_n.ndim == 4 \
        else np.asarray(detail_n).reshape(-1)
    w = np.asarray(detail_n).reshape(-1, 3).T.reshape(-1)  # channel-major like octave_wc
    m = np.asarray(mu).reshape(-1, 3).T.reshape(-1)
    s2 = (np.asarray(sig).reshape(-1, 3).T.reshape(-1)) ** 2
    edges = np.quantile(c, np.linspace(0, 1, N_BINS + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    idx = np.clip(np.digitize(c, edges) - 1, 0, N_BINS - 1)
    w = w / w.std()
    pooled_gen = m.var() + s2.mean()
    cc = np.array([c[idx == b].mean() for b in range(N_BINS)])
    v_real = np.array([w[idx == b].var() for b in range(N_BINS)])
    v_mean = np.array([m[idx == b].var() for b in range(N_BINS)]) / pooled_gen
    v_noise = np.array([s2[idx == b].mean() for b in range(N_BINS)]) / pooled_gen
    return cc, v_real, v_mean + v_noise, v_mean, v_noise


def slope(cc, v):
    return float(np.polyfit(cc, v, 1)[0])


ck = {arm: load_ckpt(os.path.join(REPO, "data_cache", "ckpt_nll",
                                  f"arm{arm}_gowerstreet.pkl")) for arm in "AB"}
prof = {}
print(f"{'oct':>3} {'arm':>3} {'real slope':>11} {'gen slope':>10} "
      f"{'mean part':>10} {'noise part':>11}")
for j in (1, 2, 3, 4):
    det, coarse = haar.octave_pair(heldout, j)
    det_n = np.asarray(det) / ck["A"]["std_by_j"][j]
    for arm in "AB":
        c = ck[arm]
        cv = None if c["cond_dim"] == 0 else jnp.broadcast_to(
            jnp.asarray(coords[j] / np.asarray(c["coord_norm"]), jnp.float32),
            (coarse.shape[0], 2))
        mu, sig = head_forward(c["params"], c["channels"], c["cond_dim"],
                               c["cond_mode"], coarse, cv)
        cc, vr, vg, vm, vn = decompose(mu, sig, det_n, coarse)
        prof[(j, arm)] = (cc, vr, vg, vm, vn)
        print(f"{j:>3} {arm:>3} {slope(cc, vr):>11.3f} {slope(cc, vg):>10.3f} "
              f"{slope(cc, vm):>10.3f} {slope(cc, vn):>11.3f}")

# ------- head modulation vs training steps (arm A, octaves 2 and 3) -------
steps_axis, curve = [2000, 4000, 6000, 8000, 10000], {2: [], 3: []}
for s in steps_axis:
    p = (ck["A"]["params"] if s == 10000 else
         load_ckpt(os.path.join(REPO, "data_cache", "ckpt_nll",
                                f"armA_gowerstreet_s{s}.pkl"))["params"])
    for j in (2, 3):
        det, coarse = haar.octave_pair(heldout, j)
        det_n = np.asarray(det) / ck["A"]["std_by_j"][j]
        mu, sig = head_forward(p, ck["A"]["channels"], 0, ck["A"]["cond_mode"],
                               coarse, None)
        cc, vr, vg, vm, vn = decompose(mu, sig, det_n, coarse)
        curve[j].append(slope(cc, vg))

np.savez(os.path.join(REPO, "results", "nll_diagnosis.npz"),
         steps=steps_axis, curve2=curve[2], curve3=curve[3],
         **{f"prof_{j}{arm}": np.array(prof[(j, arm)]) for j, arm in prof})

# ------------------------------- figure 1: profiles + training curve
fig, ax = plt.subplots(2, 2, figsize=(12.2, 9.2))
for a, j in zip(ax.ravel()[:3], (2, 3, 4)):
    cc, vr, vg, vm, vn = prof[(j, "B")]
    a.plot(cc, vr, "-o", color=K, lw=2.4, ms=4, label="REAL detail variance")
    a.plot(cc, vg, "--s", color=GREEN, ms=4, label="generated total (mean+noise)")
    a.plot(cc, vn, ":", color=ORANGE, lw=2.2, label="…from the σ-head noise  e^{2g}")
    a.plot(cc, vm, ":", color=BLUE, lw=2.2, label="…from the predicted mean μ")
    a.set_title(f"octave {j} (trained) — arm B", fontsize=11)
    a.set_xlabel("coarse-field value at the pixel (standardized, binned)")
    a.set_ylabel("variance of detail in the bin / pooled")
    a.grid(alpha=0.25)
    if j == 2:
        a.legend(fontsize=8.5)
a = ax.ravel()[3]
a.plot(steps_axis, curve[2], "-o", color=VERM, label="octave 2")
a.plot(steps_axis, curve[3], "-s", color=BLUE, label="octave 3")
for j, col in ((2, VERM), (3, BLUE)):
    cc, vr, *_ = prof[(j, "A")]
    a.axhline(slope(cc, vr), color=col, ls="--", alpha=0.5)
a.set_title("does the head collapse with training?\n(implied var_slope vs steps, arm A; "
            "dashed = real)", fontsize=11)
a.set_xlabel("training steps")
a.set_ylabel("implied var_slope of generated detail")
a.legend(fontsize=9)
a.grid(alpha=0.25)
fig.suptitle("G-1c diagnosis: WHERE the conditional-variance modulation is missing "
             "(decomposition is exact, no sampling)", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(REPO, "results", "nll_diagnosis.png"), dpi=130,
            bbox_inches="tight")
print("wrote results/nll_diagnosis.png")

# ------------------------------- figure 2: sigma maps (octave 2, one field)
det, coarse = haar.octave_pair(heldout, 2)
det_n = np.asarray(det) / ck["A"]["std_by_j"][2]
mu, sig = head_forward(ck["A"]["params"], ck["A"]["channels"], 0,
                       ck["A"]["cond_mode"], coarse, None)
i = 0
fig, ax = plt.subplots(1, 3, figsize=(12.6, 4.4))
im0 = ax[0].imshow(np.asarray(coarse)[i, :, :, 0], cmap="inferno")
ax[0].set_title("the coarse field the model sees\n(octave 2, held-out field #0)")
amp = np.abs(det_n[i]).mean(-1)
im1 = ax[1].imshow(amp, cmap="inferno", vmax=np.percentile(amp, 99))
ax[1].set_title("REAL detail amplitude |detail|\n(bright = strong fine texture)")
im2 = ax[2].imshow(sig[i].mean(-1), cmap="inferno")
ax[2].set_title("the σ-head's predicted noise level e^g\n(should modulate like the middle panel)")
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.suptitle("Is the learned σ map modulating with the environment?", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(REPO, "results", "nll_sigma_maps.png"), dpi=130,
            bbox_inches="tight")
print("wrote results/nll_sigma_maps.png")
