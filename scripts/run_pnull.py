#!/usr/bin/env python
"""Rung (iii) P-null runner: train arms A/B on GRF, extrapolate to a finer octave, save.

Trains the shared conditional generator on GRF octaves ``--train-octaves`` (default 2 3 4),
then generates fields coarse-to-fine from HELD-OUT real GRF coarse at the coarsest train
octave down to octave 0 — so octave 1 is the *extrapolated* (untrained) scale. Saves the
generated fields (both arms) and the matched real held-out fields to an npz for the
scaledrift measurement bridge (`measure_generated.py`), which decides P-null.

CPU-runnable for validation (`--steps 400`); the real run is a GPU SLURM job
(`scripts/train_gpu.slurm`). The per-octave detail amplitude at the extrapolated octave is
log-linearly extrapolated from the trained octaves (the power-spectrum amplitude, P4).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

try:
    os.sched_setaffinity(0, set(range(4)))          # harmless on a GPU node
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

from wfm.dataset import field_to_octaves
from wfm.generate import generate_recursive
from wfm.train import train_generator

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extrapolate_std(std_by_j, target_j):
    """Log-linear extrapolation of the per-octave detail amplitude to an untrained octave."""
    js = np.array(sorted(std_by_j))
    ys = np.log(np.array([std_by_j[j] for j in js]))
    a, b = np.polyfit(js, ys, 1)
    return float(np.exp(a * target_j + b))


def scale_coord(j, train_octaves):
    """Toy 2-D scale coordinate for arm B (placeholder for the stage-0 running-coupling
    coordinate): [normalized octave index, 0]. For GRF the couplings are flat, so any
    smooth coordinate must leave the null intact."""
    jmin, jmax = min(train_octaves), max(train_octaves)
    return [(j - jmin) / max(1, jmax - jmin), 0.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="grf")
    ap.add_argument("--train-octaves", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--gen-from", type=int, default=4)     # start recursion here (coarsest)
    ap.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-heldout", type=int, default=64)
    ap.add_argument("--sample-steps", type=int, default=80)
    ap.add_argument("--nll-head", action="store_true",
                    help="phase-1c option 2: Gaussian-NLL log-sigma head, mean-path + "
                         "explicit-variance sampling (both arms symmetrically)")
    ap.add_argument("--data", default=os.path.join(REPO, "data_cache", "tiles_pnull.npz"))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "pnull_generated.npz"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = vars(args).copy()
    cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
    print(f"[run_pnull] config_hash={cfg_hash} devices={jax.devices()}", flush=True)
    print(f"[run_pnull] config={json.dumps(cfg)}", flush=True)

    data = np.load(args.data)
    tiles = data[args.field].astype(np.float32)
    heldout = tiles[-args.n_heldout:]
    train = tiles[:-args.n_heldout]
    print(f"[run_pnull] {args.field}: {train.shape[0]} train / {heldout.shape[0]} heldout tiles",
          flush=True)

    cond = {j: scale_coord(j, args.train_octaves)
            for j in range(1, max(args.train_octaves) + 1)}
    results = {}
    t0 = time.time()
    for arm in ("A", "B"):
        state, meta = train_generator(
            train, args.train_octaves, arm=arm,
            cond_by_octave=(cond if arm == "B" else None),
            channels=tuple(args.channels), steps=args.steps, batch=args.batch,
            lr=args.lr, seed=args.seed, nll=args.nll_head)
        std = dict(meta["std_by_j"])
        for j in range(1, min(args.train_octaves)):          # extrapolated finer octaves
            std[j] = extrapolate_std(meta["std_by_j"], j)
        pools, _ = field_to_octaves(heldout, [args.gen_from])
        coarse = pools[args.gen_from][1]
        cond_fn = None if arm == "A" else (
            lambda j: jnp.broadcast_to(jnp.asarray(cond[j], jnp.float32),
                                       (coarse.shape[0], 2)))
        gen = generate_recursive(state.apply_fn, state.params, coarse, args.gen_from,
                                 jax.random.PRNGKey(args.seed + 1), std, cond_fn=cond_fn,
                                 n_steps=args.sample_steps, nll=args.nll_head)
        results[f"gen_{arm}"] = np.asarray(gen[..., 0])
        print(f"[run_pnull] arm {arm}: loss {meta['loss0']:.3f}->{meta['lossN']:.4f} "
              f"gen {results['gen_'+arm].shape} std_by_j={ {k: round(v,3) for k,v in std.items()} }",
              flush=True)

    # matched real held-out fields (unit-variance normalized, as generation is)
    from wfm.dataset import normalize_tiles
    results["real"] = np.asarray(normalize_tiles(heldout)[..., 0])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, config=json.dumps(cfg), config_hash=cfg_hash, **results)
    print(f"[run_pnull] wrote {args.out} in {time.time()-t0:.0f}s "
          f"(gen_A, gen_B, real) config_hash={cfg_hash}", flush=True)


if __name__ == "__main__":
    main()
