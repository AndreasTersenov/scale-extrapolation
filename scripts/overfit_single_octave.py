#!/usr/bin/env python
"""Rung (i): a single-octave conditional FM must overfit one field.

Take one field, extract (detail_j, coarse_j) with the generator's Haar transform, train
v(detail_t, t | coarse_j) to memorize it, then sample from noise conditioned on coarse_j
and check the sampled detail reproduces the true detail. This is the Karpathy-ladder
sanity gate: if the conditional FM cannot memorize one example, nothing downstream works.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Pin CPU affinity BEFORE importing jax so XLA's CPU threadpool fits the login-node limit.
try:
    os.sched_setaffinity(0, set(range(4)))
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wfm.train import overfit_octave

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--octave", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--channels", type=int, nargs="+", default=[48, 96])
    ap.add_argument("--field", default="gowerstreet")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tiles = np.load(os.path.join(REPO, "data_cache", "tiles_128.npz"))[args.field]
    field = tiles[args.idx].astype(np.float32)
    rel, x = overfit_octave(field, j=args.octave, channels=tuple(args.channels),
                            steps=args.steps, seed=args.seed)
    print(f"field={args.field}[{args.idx}] octave={args.octave} detail={x['detail_shape']} "
          f"steps={x['steps']} loss {x['loss0']:.3f}->{x['lossN']:.4f} "
          f"| sampled-vs-true relative L2 = {rel:.4f}")
    print("OVERFIT GATE:", "PASS" if rel < 0.15 else "FAIL", "(threshold 0.15)")


if __name__ == "__main__":
    main()
