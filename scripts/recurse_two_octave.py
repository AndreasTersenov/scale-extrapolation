#!/usr/bin/env python
"""Rung (ii): two-octave coarse-to-fine recursion on one field (CLI wrapper)."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    os.sched_setaffinity(0, set(range(4)))
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wfm.train import overfit_field_recursive

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--j-max", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--channels", type=int, nargs="+", default=[48, 96])
    ap.add_argument("--field", default="gowerstreet")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tiles = np.load(os.path.join(REPO, "data_cache", "tiles_128.npz"))[args.field]
    rel, info = overfit_field_recursive(tiles[args.idx].astype(np.float32),
                                        j_max=args.j_max, channels=tuple(args.channels),
                                        steps=args.steps, seed=args.seed)
    print(f"field={args.field}[{args.idx}] octaves={info['octaves']} "
          f"loss {info['loss0']:.3f}->{info['lossN']:.4f} "
          f"| recursive field relative L2 = {rel:.4f} deterministic={info['deterministic']}")
    print("RUNG ii GATE:", "PASS" if rel < 0.2 and info["deterministic"] else "FAIL")


if __name__ == "__main__":
    main()
