#!/usr/bin/env python
"""4b'-ii pre-requirement: measure the conditioning drift and derive s_matched.

NO generation is run here (ruling: compute BEFORE any generation run). Everything comes
from existing artifacts: the s=0.3 checkpoints, the s=0.3 end-to-end fields
(results/arms_4bp_s0.3.npz), and the real held-out tiles.

Primary estimator (attenuation matching, in the corruption model's own units):
  s_matched_j = the white-noise level s at which the head, fed CORRUPTED REAL coarse
  (c + s*std(c)*eps) with s_told = 0 -- exactly the mode the failed end-to-end run
  operated in -- reproduces the MEASURED end-to-end var_slope at octave j.
Cross-check estimator: the pixel-aligned relative residual between generated and real
coarse at octave j (same held-out fields, same octave-4 start); NOTE this includes the
legitimate conditional sampling spread, so it is an upper-bound-flavored statistic.

Also recomputes the s=0.3 model's OWN given-real-coarse ceiling with bootstrap SEs
(the 4b'-ii lever-bar reference needs a sigma).
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
import jax
import jax.numpy as jnp

from wfm import haar
from wfm.dataset import normalize_tiles
from wfm.model import ConditionalUNet

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_BINS, N_BOOT = 10, 200

tiles = np.load(os.path.join(REPO, "data_cache", "tiles_pnull.npz"))["gowerstreet"]
heldout = normalize_tiles(tiles.astype(np.float32)[-64:])
coords = {int(j): np.asarray(v) for j, v in
          json.load(open(os.path.join(REPO, "data_cache",
                                      "running_couplings.json")))["gowerstreet"].items()}
e2e = json.load(open(os.path.join(REPO, "results", "arms_4bp_s0.3_score.json")))


def load_ck(arm):
    p = os.path.join(REPO, "data_cache", "ckpt_4bp_s0.3", f"arm{arm}_gowerstreet.pkl")
    with open(p, "rb") as fh:
        ck = pickle.load(fh)
    model = ConditionalUNet(out_channels=3, channels=tuple(ck["channels"]),
                            bottleneck=ck["channels"][-1] * 2, cond_dim=ck["cond_dim"],
                            cond_mode=ck["cond_mode"], variance_head=True)
    return model, ck


def cond_vec(ck, arm, j, B, s_told=0.0):
    parts = []
    if arm == "B":
        parts.append(jnp.broadcast_to(
            jnp.asarray(coords[j] / np.asarray(ck["coord_norm"]), jnp.float32), (B, 2)))
    parts.append(jnp.full((B, 1), s_told, jnp.float32))
    return jnp.concatenate(parts, axis=1)


def implied_slope(model, ck, arm, j, coarse_in, s_told, boot=False):
    B = coarse_in.shape[0]
    o = model.apply({"params": ck["params"]}, jnp.zeros(coarse_in.shape[:3] + (3,)),
                    jnp.zeros((B,)), coarse_in, cond_vec(ck, arm, j, B, s_told))
    mu_f = [np.asarray(o[i, ..., :3]).reshape(-1, 3).T.reshape(-1) for i in range(B)]
    s2_f = [np.exp(2 * np.clip(np.asarray(o[i, ..., 3:]), -5, 3))
            .reshape(-1, 3).T.reshape(-1) for i in range(B)]
    c_f = [np.tile(np.asarray(coarse_in[i]).reshape(-1), 3) for i in range(B)]

    def slope(idx_fields):
        c = np.concatenate([c_f[i] for i in idx_fields])
        mu = np.concatenate([mu_f[i] for i in idx_fields])
        s2 = np.concatenate([s2_f[i] for i in idx_fields])
        c = (c - c.mean()) / c.std()
        edges = np.quantile(c, np.linspace(0, 1, N_BINS + 1))
        edges[0] -= 1e-9; edges[-1] += 1e-9
        idx = np.clip(np.digitize(c, edges) - 1, 0, N_BINS - 1)
        pooled = mu.var() + s2.mean()
        cc = np.array([c[idx == b].mean() for b in range(N_BINS)])
        v = np.array([(mu[idx == b].var() + s2[idx == b].mean())
                      for b in range(N_BINS)])
        return float(np.polyfit(cc, v / pooled, 1)[0])

    point = slope(range(B))
    if not boot:
        return point, None
    rng = np.random.default_rng(1)
    boots = [slope(rng.integers(0, B, B)) for _ in range(N_BOOT)]
    return point, float(np.std(boots, ddof=1))


# ---- 1. s=0.3 model's OWN ceiling with SEs (lever-bar reference) ----
print("== own ceiling of the s=0.3 model (given real coarse, s_told=0), with SEs ==")
ceil = {}
for arm in "AB":
    model, ck = load_ck(arm)
    for j in (2, 3, 4):
        _, coarse = haar.octave_pair(heldout, j)
        p, se = implied_slope(model, ck, arm, j, coarse, 0.0, boot=True)
        ceil[f"arm{arm}_oct{j}"] = {"ceiling": p, "se": se}
        print(f"  arm {arm} oct {j}: {p:.3f} +- {se:.3f}")

# ---- 2. primary estimator: attenuation matching (arm A primary, B reported) ----
print("== attenuation matching: implied slope on corrupted REAL coarse, s_told=0 ==")
key = jax.random.PRNGKey(0)
grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
match = {}
for arm in "AB":
    model, ck = load_ck(arm)
    for j in (2, 3):
        _, coarse = haar.octave_pair(heldout, j)
        target = e2e[f"arm{arm}"][str(j)]["var_slope"]
        curve = []
        for s in grid:
            vals = []
            for rep in range(3):
                key, k = jax.random.split(key)
                cin = coarse + s * coarse.std() * jax.random.normal(k, coarse.shape)
                vals.append(implied_slope(model, ck, arm, j, cin, 0.0)[0])
            curve.append(float(np.mean(vals)))
        # first grid interval where the (monotone-ish) curve crosses the target
        s_m = None
        for a in range(len(grid) - 1):
            lo, hi = curve[a], curve[a + 1]
            if (lo - target) * (hi - target) <= 0:
                s_m = grid[a] + (grid[a + 1] - grid[a]) * (lo - target) / (lo - hi + 1e-12)
                break
        match[f"arm{arm}_oct{j}"] = {"target": target, "curve": curve, "s_matched": s_m}
        print(f"  arm {arm} oct {j}: target {target:.3f}; curve "
              f"{[round(v, 3) for v in curve]}; s_matched = "
              f"{'NOT in [0,0.35]' if s_m is None else round(s_m, 3)}")

# ---- 3. cross-check: pixel-aligned generated-vs-real coarse residual ----
print("== cross-check: aligned relative coarse residual (upper-bound-flavored) ==")
d = np.load(os.path.join(REPO, "results", "arms_4bp_s0.3.npz"))
aligned = {}
for arm in "AB":
    gen = jnp.asarray(d[f"gen_{arm}"])[..., None]
    real = jnp.asarray(d["real"])[..., None]
    for j in (2, 3):
        _, cg = haar.octave_pair(gen, j)
        _, cr = haar.octave_pair(real, j)
        s_a = float(jnp.std(cg - cr) / jnp.std(cr))
        aligned[f"arm{arm}_oct{j}"] = s_a
        print(f"  arm {arm} oct {j}: s_aligned = {s_a:.3f}")

json.dump({"own_ceiling_s03": ceil, "attenuation_match": match,
           "aligned_residual": aligned},
          open(os.path.join(REPO, "results", "smatched_4bpii.json"), "w"), indent=1)
print("wrote results/smatched_4bpii.json")
