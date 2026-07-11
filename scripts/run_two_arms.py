#!/usr/bin/env python
"""Rung (iv): the break-and-repair experiment (arms A and B) on a real field.

Trains arm A (naive weight-tying, no scale input) and arm B (conditioned on the stage-0
2-D running-coupling coordinate [var_slope_j, kurtosis_j]) on ``--field`` octaves
``--train-octaves``, then generates the finer UNTRAINED octave coarse-to-fine from held-out
real coarse. Saves gen_A/gen_B/real fields AND both trained checkpoints (for the rung-(v)
zero-retrain transfer). Score with `measure_generated.py`:

  P5 (break): arm A's conditional non-Gaussianity (var_slope) at the extrapolated octave is
              wrong (>3sigma, >10%).
  P6 (repair): arm B closes >=70% of arm A's error there, without hurting trained octaves.
  P4: detail amplitude (power) extrapolates within a few %, both arms.

GPU SLURM (`scripts/train_gowerstreet.slurm`); CPU-runnable at low --steps for validation.
"""
from __future__ import annotations

import argparse
import hashlib
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
from wfm.train import train_generator

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# fixed coordinate normalization (free periphery): map stage-0 couplings to ~[0,1]
COORD_NORM = np.array([1.5, 13.0])


def load_coords(coords_file, field):
    raw = json.load(open(coords_file))[field]
    return {int(j): (np.asarray(v) / COORD_NORM).tolist() for j, v in raw.items()}


def extrapolate_std(std_by_j, target_j):
    js = np.array(sorted(std_by_j))
    a, b = np.polyfit(js, np.log([std_by_j[j] for j in js]), 1)
    return float(np.exp(a * target_j + b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="gowerstreet")
    ap.add_argument("--train-octaves", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--gen-from", type=int, default=4)
    ap.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--cond-mode", default="add", choices=["add", "film"],
                    help="how arm B's scale coordinate enters (add=embedding, film=modulation)")
    ap.add_argument("--lambda-disp", type=float, default=0.0,
                    help="weight of the conditional-dispersion regularizer (step-c objective)")
    ap.add_argument("--disp-t-lo", type=float, default=0.0,
                    help="late-t window lower bound for the dispersion penalty (c' option 1; 0.6)")
    ap.add_argument("--nll-head", action="store_true",
                    help="phase-1c option 2: Gaussian-NLL log-sigma head, mean-path + "
                         "explicit-variance sampling (both arms symmetrically)")
    ap.add_argument("--ckpt-steps", type=int, nargs="*", default=[],
                    help="also save arm checkpoints at these step counts (params only)")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-heldout", type=int, default=64)
    ap.add_argument("--sample-steps", type=int, default=80)
    ap.add_argument("--data", default=os.path.join(REPO, "data_cache", "tiles_pnull.npz"))
    ap.add_argument("--coords-file",
                    default=os.path.join(REPO, "data_cache", "running_couplings.json"))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "arms_generated.npz"))
    ap.add_argument("--ckpt-dir", default=os.path.join(REPO, "data_cache", "ckpt"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = vars(args).copy()
    cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
    print(f"[run_two_arms] config_hash={cfg_hash} devices={jax.devices()}", flush=True)
    print(f"[run_two_arms] config={json.dumps(cfg)}", flush=True)

    tiles = np.load(args.data)[args.field].astype(np.float32)
    heldout, train = tiles[-args.n_heldout:], tiles[:-args.n_heldout]
    coords = load_coords(args.coords_file, args.field)
    print(f"[run_two_arms] {args.field}: {train.shape[0]} train / {heldout.shape[0]} heldout; "
          f"coords(norm) j1={np.round(coords[1],3).tolist()} j4={np.round(coords[4],3).tolist()}",
          flush=True)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    results, t0 = {}, time.time()
    for arm in ("A", "B"):
        def save_ckpt(step_i, st, loss_i, _arm=arm):
            path = os.path.join(args.ckpt_dir, f"arm{_arm}_{args.field}_s{step_i}.pkl")
            with open(path, "wb") as fh:
                pickle.dump({"params": jax.tree_util.tree_map(np.asarray, st.params),
                             "step": step_i, "loss": loss_i}, fh)
            print(f"[run_two_arms] arm {_arm} ckpt @{step_i} loss={loss_i:.4f}", flush=True)

        state, meta = train_generator(
            train, args.train_octaves, arm=arm,
            cond_by_octave=(coords if arm == "B" else None),
            channels=tuple(args.channels), steps=args.steps, batch=args.batch,
            lr=args.lr, seed=args.seed, cond_mode=args.cond_mode,
            lambda_disp=args.lambda_disp, disp_t_lo=args.disp_t_lo, nll=args.nll_head,
            ckpt_steps=tuple(args.ckpt_steps),
            on_checkpoint=(save_ckpt if args.ckpt_steps else None))
        std = dict(meta["std_by_j"])
        for j in range(1, min(args.train_octaves)):
            std[j] = extrapolate_std(meta["std_by_j"], j)
        pools, _ = field_to_octaves(heldout, [args.gen_from])
        coarse = pools[args.gen_from][1]
        cond_fn = None if arm == "A" else (
            lambda j: jnp.broadcast_to(jnp.asarray(coords[j], jnp.float32),
                                       (coarse.shape[0], 2)))
        gen = generate_recursive(state.apply_fn, state.params, coarse, args.gen_from,
                                 jax.random.PRNGKey(args.seed + 1), std, cond_fn=cond_fn,
                                 n_steps=args.sample_steps, nll=args.nll_head)
        results[f"gen_{arm}"] = np.asarray(gen[..., 0])
        ckpt = {"params": jax.tree_util.tree_map(np.asarray, state.params),
                "channels": list(args.channels), "cond_dim": meta["cond_dim"],
                "cond_mode": meta["cond_mode"], "lambda_disp": meta["lambda_disp"],
                "disp_t_lo": meta["disp_t_lo"], "nll": meta["nll"], "std_by_j": std,
                "coord_norm": COORD_NORM.tolist(),
                "train_octaves": list(args.train_octaves), "field": args.field}
        with open(os.path.join(args.ckpt_dir, f"arm{arm}_{args.field}.pkl"), "wb") as fh:
            pickle.dump(ckpt, fh)
        print(f"[run_two_arms] arm {arm}: loss {meta['loss0']:.3f}->{meta['lossN']:.4f} "
              f"gen {results['gen_'+arm].shape} std1={std[1]:.3f} ckpt saved", flush=True)

    results["real"] = np.asarray(normalize_tiles(heldout)[..., 0])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, config=json.dumps(cfg), config_hash=cfg_hash, **results)
    print(f"[run_two_arms] wrote {args.out} in {time.time()-t0:.0f}s config_hash={cfg_hash}",
          flush=True)


if __name__ == "__main__":
    main()
