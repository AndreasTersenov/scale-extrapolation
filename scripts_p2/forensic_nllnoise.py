"""NLL-head noise forensic (R5): regenerate end-to-end from the FROZEN 4a
checkpoints with the sampler's noise term removed (detail = mu only).

Replicates run_two_arms' generation protocol exactly (held-out stack, recursion
from octave 4, ckpt std_by_j, arm-B coords) except the sampler. No training; the
frozen checkpoints are opened read-only.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

try:
    os.sched_setaffinity(0, set(range(4)))
except Exception:
    pass
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

from wfm import haar
from wfm.cfm import G_CLIP
from wfm.dataset import field_to_octaves, normalize_tiles
from wfm.model import ConditionalUNet

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(REPO, "data_cache", "ckpt_aug")
OUT_NPZ = os.path.join(REPO, "results_p2", "forensic_nllnoise.npz")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sample_mean_path(apply_fn, params, coarse, out_channels, cond_vec=None):
    """sample_nll minus its noise term: mu = model(x_t=0, t=0 | coarse, cond)."""
    B, H, W, _ = coarse.shape
    zeros = jnp.zeros((B, H, W, out_channels))
    out0 = apply_fn({"params": params}, zeros, jnp.zeros((B,)), coarse, cond_vec)
    return out0[..., :out_channels]


def main():
    tiles = np.load(os.path.join(REPO, "data_cache", "tiles_pnull.npz"))["gowerstreet"]
    heldout = tiles.astype(np.float32)[-64:]
    fields = normalize_tiles(heldout)
    pools, _ = field_to_octaves(heldout, [4])
    results = {"real": np.asarray(fields[..., 0])}

    coords_raw = json.load(open(os.path.join(
        REPO, "data_cache", "running_couplings.json")))["gowerstreet"]

    for arm in ("A", "B"):
        with open(os.path.join(CKPT_DIR, f"arm{arm}_gowerstreet.pkl"), "rb") as fh:
            ck = pickle.load(fh)
        assert ck["nll"], "forensic targets the NLL-head checkpoints"
        model = ConditionalUNet(out_channels=3, channels=tuple(ck["channels"]),
                                bottleneck=ck["channels"][-1] * 2,
                                cond_dim=ck["cond_dim"], cond_mode=ck["cond_mode"],
                                variance_head=True)
        coord_norm = np.asarray(ck["coord_norm"])
        coords = {int(j): (np.asarray(v) / coord_norm) for j, v in coords_raw.items()}
        std = {int(k): float(v) for k, v in ck["std_by_j"].items()}
        coarse = pools[4][1]
        B = coarse.shape[0]
        for j in range(4, 0, -1):
            cv = None
            if arm == "B" and ck["cond_dim"] > 0:
                cv = jnp.broadcast_to(jnp.asarray(coords[j], jnp.float32),
                                      (B, ck["cond_dim"]))
            det = sample_mean_path(model.apply, ck["params"], coarse, 3, cond_vec=cv)
            det = det * std[j]
            coarse = haar.idwt2(coarse, (det[..., 0:1], det[..., 1:2], det[..., 2:3]))
            log(f"arm {arm} octave {j}: assembled {coarse.shape}")
        results[f"gen_{arm}"] = np.asarray(coarse[..., 0])
    np.savez(OUT_NPZ, **results)
    log(f"wrote {OUT_NPZ}")


if __name__ == "__main__":
    main()
