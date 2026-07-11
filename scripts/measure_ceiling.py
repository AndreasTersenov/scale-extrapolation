#!/usr/bin/env python
"""Measure the given-real-coarse CEILING for the 4b' lever bar (reconvene ruling 4).

For the existing 20k AUGMENTED checkpoints (no new training), compute the head's
implied var_slope at octaves 2, 3, 4 — conditioning every octave on REAL held-out
coarse — with a bootstrap-over-fields SE. This is what a perfect de-compounding could
achieve; the 4b' lever bar is "end-to-end within 1 sigma (combined) of this ceiling".

Decomposition is exact (Var(mu|bin) + E[e^{2g}|bin]; 10 coarse-quantile bins), same
estimator as scripts/diagnose_nll.py / signature_4a.py.
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

from wfm import haar
from wfm.dataset import normalize_tiles
from wfm.model import ConditionalUNet

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_BINS, N_BOOT = 10, 200

tiles = np.load(os.path.join(REPO, "data_cache", "tiles_pnull.npz"))["gowerstreet"]
heldout = normalize_tiles(tiles.astype(np.float32)[-64:])
coords = json.load(open(os.path.join(REPO, "data_cache", "running_couplings.json")))
coords = {int(j): np.asarray(v) for j, v in coords["gowerstreet"].items()}


def slope_from_fields(c_f, mu_f, s2_f, idx_fields):
    """Pooled implied var_slope over the selected fields (scaledrift-style bins)."""
    c = np.concatenate([c_f[i] for i in idx_fields])
    mu = np.concatenate([mu_f[i] for i in idx_fields])
    s2 = np.concatenate([s2_f[i] for i in idx_fields])
    c = (c - c.mean()) / c.std()
    edges = np.quantile(c, np.linspace(0, 1, N_BINS + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    idx = np.clip(np.digitize(c, edges) - 1, 0, N_BINS - 1)
    pooled = mu.var() + s2.mean()
    cc = np.array([c[idx == b].mean() for b in range(N_BINS)])
    v = np.array([(mu[idx == b].var() + s2[idx == b].mean()) for b in range(N_BINS)])
    return float(np.polyfit(cc, v / pooled, 1)[0])


out = {}
rng = np.random.default_rng(0)
for arm in "AB":
    with open(os.path.join(REPO, "data_cache", "ckpt_aug",
                           f"arm{arm}_gowerstreet.pkl"), "rb") as fh:
        ck = pickle.load(fh)
    model = ConditionalUNet(out_channels=3, channels=tuple(ck["channels"]),
                            bottleneck=ck["channels"][-1] * 2,
                            cond_dim=ck["cond_dim"], cond_mode=ck["cond_mode"],
                            variance_head=True)
    for j in (2, 3, 4):
        det, coarse = haar.octave_pair(heldout, j)
        cv = None if ck["cond_dim"] == 0 else jnp.broadcast_to(
            jnp.asarray(coords[j] / np.asarray(ck["coord_norm"]), jnp.float32),
            (coarse.shape[0], ck["cond_dim"]))
        o = model.apply({"params": ck["params"]},
                        jnp.zeros(coarse.shape[:3] + (3,)),
                        jnp.zeros((coarse.shape[0],)), coarse, cv)
        n = coarse.shape[0]
        c_f = [np.tile(np.asarray(coarse[i]).reshape(-1), 3) for i in range(n)]
        mu_f = [np.asarray(o[i, ..., :3]).reshape(-1, 3).T.reshape(-1) for i in range(n)]
        s2_f = [np.exp(2 * np.clip(np.asarray(o[i, ..., 3:]), -5, 3))
                .reshape(-1, 3).T.reshape(-1) for i in range(n)]
        point = slope_from_fields(c_f, mu_f, s2_f, range(n))
        boots = [slope_from_fields(c_f, mu_f, s2_f, rng.integers(0, n, n))
                 for _ in range(N_BOOT)]
        se = float(np.std(boots, ddof=1))
        out[f"arm{arm}_oct{j}"] = {"ceiling": point, "se": se}
        print(f"arm {arm} oct {j}: ceiling = {point:.3f} +- {se:.3f}")

json.dump(out, open(os.path.join(REPO, "results", "ceiling_4a.json"), "w"), indent=1)
print("wrote results/ceiling_4a.json")
