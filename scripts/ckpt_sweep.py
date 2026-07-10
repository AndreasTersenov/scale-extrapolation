#!/usr/bin/env python
"""Step (b): checkpoint / early-stopping sweep for dispersion fidelity.

Trains arm A on gowerstreet octaves 2-4 ONCE to ``--max-steps``, snapshotting at
``--ckpt-steps`` milestones, and generates held-out fields from each snapshot. The goal is
the var_slope-vs-training-steps curve (the "conditional-FM dispersion collapse") and the
peak-dispersion checkpoint, selected by trained-octave var_slope (NOT loss). Saves one npz
per milestone (gen_A + real) for `measure_generated.py`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

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
from wfm.train import train_generator

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="gowerstreet")
    ap.add_argument("--train-octaves", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--gen-from", type=int, default=4)
    ap.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--ckpt-steps", type=int, nargs="+", default=[2000, 4000, 6000, 8000, 12000])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-heldout", type=int, default=64)
    ap.add_argument("--sample-steps", type=int, default=80)
    ap.add_argument("--churn", type=float, default=0.0)
    ap.add_argument("--data", default=os.path.join(REPO, "data_cache", "tiles_pnull.npz"))
    ap.add_argument("--out-prefix", default=os.path.join(REPO, "results", "sweep"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(f"[sweep] devices={jax.devices()} field={args.field} ckpt_steps={args.ckpt_steps}",
          flush=True)

    tiles = np.load(args.data)[args.field].astype(np.float32)
    heldout, train = tiles[-args.n_heldout:], tiles[:-args.n_heldout]
    _, std = field_to_octaves(heldout, sorted(set(args.train_octaves + [args.gen_from])))
    js = np.array(sorted(std)); a, b = np.polyfit(js, np.log([std[j] for j in js]), 1)
    for j in range(1, min(args.train_octaves)):
        std[j] = float(np.exp(a * j + b))

    snaps = []          # (step, params, loss)
    def on_ckpt(step, state, loss):
        snaps.append((step, jax.tree_util.tree_map(np.asarray, state.params), loss))
        print(f"[sweep] snapshot step={step} loss={loss:.4f}", flush=True)

    train_generator(train, args.train_octaves, arm="A", channels=tuple(args.channels),
                    steps=max(args.ckpt_steps), batch=args.batch, lr=args.lr,
                    seed=args.seed, ckpt_steps=args.ckpt_steps, on_checkpoint=on_ckpt)

    model = ConditionalUNet(out_channels=3, channels=tuple(args.channels),
                            bottleneck=args.channels[-1] * 2, cond_dim=0)
    pools, _ = field_to_octaves(heldout, [args.gen_from])
    coarse = pools[args.gen_from][1]
    real = np.asarray(normalize_tiles(heldout)[..., 0])
    for step, params, loss in snaps:
        gen = generate_recursive(model.apply, jax.tree_util.tree_map(jnp.asarray, params),
                                 coarse, args.gen_from, jax.random.PRNGKey(args.seed + 1),
                                 std, cond_fn=None, n_steps=args.sample_steps,
                                 churn=args.churn)
        out = f"{args.out_prefix}_step{step}.npz"
        np.savez(out, gen_A=np.asarray(gen[..., 0]), gen_B=np.asarray(gen[..., 0]),
                 real=real, step=step, loss=loss)
        print(f"[sweep] step={step} loss={loss:.4f} wrote {out}", flush=True)


if __name__ == "__main__":
    main()
