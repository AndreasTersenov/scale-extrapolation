#!/usr/bin/env python
"""Rung (v): zero-retrain transfer (P13).

Take the gowerstreet-trained arm A and arm B checkpoints (from run_two_arms.py) and, with
NO retraining, generate hf_pm_1024 fields coarse-to-fine from held-out real hf_pm coarse.
Arm B uses hf_pm's OWN stage-0 running-coupling coordinate and hf_pm's OWN per-octave detail
amplitude (both are measurements, allowed; no gradient steps). Arm A uses no coordinate.
Saves gen_A/gen_B/real for `measure_generated.py`.

P13: arm B (gowerstreet-learned coordinate->conditional map, applied at hf_pm coordinates)
repairs > 40% of arm A's octave-1 var_slope error on hf_pm generation.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

try:
    os.sched_setaffinity(0, set(range(4)))
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

from wfm.dataset import field_to_octaves, normalize_tiles
from wfm.generate import generate_recursive
from wfm.model import ConditionalUNet

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_arm(ckpt_path):
    ck = pickle.load(open(ckpt_path, "rb"))
    model = ConditionalUNet(out_channels=3, channels=tuple(ck["channels"]),
                            bottleneck=ck["channels"][-1] * 2, cond_dim=ck["cond_dim"],
                            cond_mode=ck.get("cond_mode", "add"))
    params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
    return model, params, ck


def std_for_field(tiles, train_octaves, gen_from):
    """hf_pm's own per-octave detail amplitude (measurement allowed), extrapolated finer."""
    _, std = field_to_octaves(tiles, sorted(set(train_octaves + [gen_from])))
    js = np.array(sorted(std))
    a, b = np.polyfit(js, np.log([std[j] for j in js]), 1)
    for j in range(1, min(train_octaves)):
        std[j] = float(np.exp(a * j + b))
    return std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="hf_pm_1024")
    ap.add_argument("--ckpt-dir", default=os.path.join(REPO, "data_cache", "ckpt"))
    ap.add_argument("--src", default="gowerstreet", help="field the checkpoints were trained on")
    ap.add_argument("--train-octaves", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--gen-from", type=int, default=4)
    ap.add_argument("--n-heldout", type=int, default=64)
    ap.add_argument("--sample-steps", type=int, default=80)
    ap.add_argument("--churn", type=float, default=0.0,
                    help="SDE churn eps0 (0=deterministic ODE); variance-faithful sampling")
    ap.add_argument("--data", default=os.path.join(REPO, "data_cache", "tiles_pnull.npz"))
    ap.add_argument("--coords-file",
                    default=os.path.join(REPO, "data_cache", "running_couplings.json"))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "npz", "transfer_generated.npz"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(f"[transfer] devices={jax.devices()} src={args.src} -> field={args.field}", flush=True)

    tiles = np.load(args.data)[args.field].astype(np.float32)
    heldout = tiles[-args.n_heldout:]
    std = std_for_field(heldout, args.train_octaves, args.gen_from)
    coord_norm = np.array([1.5, 13.0])
    raw = json.load(open(args.coords_file))[args.field]
    coords = {int(j): (np.asarray(v) / coord_norm).tolist() for j, v in raw.items()}
    pools, _ = field_to_octaves(heldout, [args.gen_from])
    coarse = pools[args.gen_from][1]

    results, t0 = {}, time.time()
    for arm in ("A", "B"):
        model, params, ck = load_arm(
            os.path.join(args.ckpt_dir, f"arm{arm}_{args.src}.pkl"))
        cond_fn = None if ck["cond_dim"] == 0 else (
            lambda j: jnp.broadcast_to(jnp.asarray(coords[j], jnp.float32),
                                       (coarse.shape[0], 2)))
        gen = generate_recursive(model.apply, params, coarse, args.gen_from,
                                 jax.random.PRNGKey(args.seed + 1), std, cond_fn=cond_fn,
                                 n_steps=args.sample_steps, churn=args.churn)
        results[f"gen_{arm}"] = np.asarray(gen[..., 0])
        print(f"[transfer] arm {arm} (cond_dim={ck['cond_dim']}) gen {gen.shape} "
              f"std1={std[1]:.3f}", flush=True)

    results["real"] = np.asarray(normalize_tiles(heldout)[..., 0])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, config=json.dumps(vars(args)), **results)
    print(f"[transfer] wrote {args.out} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
