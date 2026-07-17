#!/usr/bin/env python
"""Arm C3: energy-score-trained direct detail sampler (arms A and B), sandbox-first.

Pre-registered log/2026-07-17-prereg-c3-sandbox.md; approved with conditions
log/2026-07-17-reconvene-forensic-c3.md (R10). Single variable vs the C1 control
= the training objective + the sampler it entails (patched energy score + direct
noise-conditioned sampling replaces CFM + ODE pushforward). Everything else is the
C1 configuration verbatim: same UNet backbone / FiLM, weight tying, per-octave std
standardization, same data + seeds, 8x D4 augmentation, arms A/B, recursion from
octave 4, checkpoint grid. NO NLL head (retired, R3). Output npz has the same
gen_A/gen_B/real contract as run_two_arms.py so the frozen scorers apply unchanged.
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

from arms_p2.c3.sampler import generate_recursive_direct
from arms_p2.c3.train import train_c3_generator
from wfm.dataset import field_to_octaves, normalize_tiles

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    ap.add_argument("--field", default="sandbox")
    ap.add_argument("--train-octaves", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--gen-from", type=int, default=4)
    ap.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--cond-mode", default="film", choices=["add", "film"])
    ap.add_argument("--m-samples", type=int, default=8)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--ckpt-steps", type=int, nargs="*", default=[])
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-heldout", type=int, default=64)
    ap.add_argument("--data", default=os.path.join(REPO, "data_cache",
                                                   "tiles_sandbox.npz"))
    ap.add_argument("--coords-file", default=os.path.join(
        REPO, "data_cache", "running_couplings_sandbox.json"))
    ap.add_argument("--out", default=os.path.join(REPO, "results_p2",
                                                  "arms_c3_sandbox.npz"))
    ap.add_argument("--ckpt-dir", default=os.path.join(REPO, "data_cache",
                                                       "ckpt_c3_sandbox"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = vars(args).copy()
    cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
    print(f"[run_c3_arms] config_hash={cfg_hash} devices={jax.devices()}", flush=True)
    print(f"[run_c3_arms] config={json.dumps(cfg)}", flush=True)

    tiles = np.load(args.data)[args.field].astype(np.float32)
    heldout, train = tiles[-args.n_heldout:], tiles[:-args.n_heldout]
    coords = load_coords(args.coords_file, args.field)
    print(f"[run_c3_arms] {args.field}: {train.shape[0]} train / "
          f"{heldout.shape[0]} heldout", flush=True)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    results, t0 = {}, time.time()
    for arm in ("A", "B"):
        def save_ckpt(step_i, st, loss_i, _arm=arm):
            path = os.path.join(args.ckpt_dir,
                                f"arm{_arm}_{args.field}_s{step_i}.pkl")
            with open(path, "wb") as fh:
                pickle.dump({"params": jax.tree_util.tree_map(np.asarray, st.params),
                             "step": step_i, "loss": loss_i}, fh)
            print(f"[run_c3_arms] arm {_arm} ckpt @{step_i} loss={loss_i:.4f}",
                  flush=True)

        state, meta = train_c3_generator(
            train, args.train_octaves, arm=arm,
            cond_by_octave=(coords if arm == "B" else None),
            channels=tuple(args.channels), steps=args.steps, batch=args.batch,
            lr=args.lr, seed=args.seed, cond_mode=args.cond_mode,
            augment=args.augment, m=args.m_samples, patch=args.patch,
            stride=args.stride, ckpt_steps=tuple(args.ckpt_steps),
            on_checkpoint=(save_ckpt if args.ckpt_steps else None))
        std = dict(meta["std_by_j"])
        for j in range(1, min(args.train_octaves)):
            std[j] = extrapolate_std(meta["std_by_j"], j)
        pools, _ = field_to_octaves(heldout, [args.gen_from])
        coarse = pools[args.gen_from][1]
        B = coarse.shape[0]

        cond_fn = None
        if arm == "B":
            def cond_fn(j):
                return jnp.broadcast_to(jnp.asarray(coords[j], jnp.float32), (B, 2))
        gen = generate_recursive_direct(state.apply_fn, state.params, coarse,
                                        args.gen_from,
                                        jax.random.PRNGKey(args.seed + 1), std,
                                        cond_fn=cond_fn)
        results[f"gen_{arm}"] = np.asarray(gen[..., 0])
        ckpt = {"params": jax.tree_util.tree_map(np.asarray, state.params),
                "channels": list(args.channels), "cond_dim": meta["cond_dim"],
                "cond_mode": meta["cond_mode"], "objective": meta["objective"],
                "m_samples": meta["m_samples"], "patch": meta["patch"],
                "stride": meta["stride"], "augment": meta["augment"],
                "std_by_j": std, "coord_norm": COORD_NORM.tolist(),
                "train_octaves": list(args.train_octaves), "field": args.field}
        with open(os.path.join(args.ckpt_dir, f"arm{arm}_{args.field}.pkl"),
                  "wb") as fh:
            pickle.dump(ckpt, fh)
        print(f"[run_c3_arms] arm {arm}: loss {meta['loss0']:.3f}->"
              f"{meta['lossN']:.4f} gen {results['gen_'+arm].shape} "
              f"std1={std[1]:.3f} ckpt saved", flush=True)

    results["real"] = np.asarray(normalize_tiles(heldout)[..., 0])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, config=json.dumps(cfg), config_hash=cfg_hash, **results)
    print(f"[run_c3_arms] wrote {args.out} in {time.time()-t0:.0f}s "
          f"config_hash={cfg_hash}", flush=True)


if __name__ == "__main__":
    main()
