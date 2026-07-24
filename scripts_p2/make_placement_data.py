"""Phase-B training data for the placement experiment (prereg 2026-07-24, R30).

Builds three tile sets, each with the SAME 64 held-out tiles appended (the frozen
seed-20260719 stream, identical to tiles_sandbox.npz — selection/adjudication
tiles shared across every arm, incl. the committed N1-seed0 baseline):

  N8    : 2576 fresh independent lognormal fields   (seed 20260725)
  N32   : 10304 fresh independent lognormal fields  (seed 20260726)
  C-ENS : 322 fresh parent coarses (seed 20260727) x 8 EXACT conditional detail
          redraws each (spawned from seed 20260728) = 2576 fields, matched total
          to N8. Multiplicity is exact at the level-4 GAUSSIAN Haar coarse
          (recipe.COND_LEVEL — the lognormal.py pre-declared subtlety) and
          partial at finer rungs; pre-stated in the prereg's Phase-A append.

N1 (322 independent fields) is tiles_sandbox.npz, unchanged (fresh SEED for the
arm, not fresh data — the rider replicates the committed regime).

Outputs -> $SCRATCH/scale-extrap-p2/tiles_placement_{n8,n32,cens}.npz
(field key 'sandbox' so run_c1t_arms.py consumes them unchanged).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from sandbox import recipe
from sandbox.lognormal import (
    coarse_spectrum, conditional_redraw, haar_coarse, lognormal_map, sample_grf,
)

SCRATCH = os.path.expanduser("~/links/scratch/scale-extrap-p2")
SEED_HELDOUT = 20260719          # the frozen held-out stream (make_c1_data.py)
SEED_N8, SEED_N32 = 20260725, 20260726
SEED_CENS_PARENTS, SEED_CENS_REDRAWS = 20260727, 20260728
N8, N32, CENS_P, CENS_R = 2576, 10304, 322, 8


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def heldout64(sp):
    rng = np.random.default_rng(SEED_HELDOUT)
    return np.stack([lognormal_map(sample_grf(sp, rng), recipe.SIGMA_G)
                     for _ in range(64)]).astype(np.float32)


def fresh(sp, n, seed):
    rng = np.random.default_rng(seed)
    return np.stack([lognormal_map(sample_grf(sp, rng), recipe.SIGMA_G)
                     for _ in range(n)]).astype(np.float32)


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    sp = recipe.spec()
    held = heldout64(sp)
    ref = np.load(os.path.join(REPO, "data_cache", "tiles_sandbox.npz"))["sandbox"]
    assert np.allclose(held, ref[-64:]), "held-out reproduction mismatch"
    log("held-out 64 reproduced byte-consistent with tiles_sandbox.npz")

    for name, n, seed in (("n8", N8, SEED_N8), ("n32", N32, SEED_N32)):
        path = os.path.join(SCRATCH, f"tiles_placement_{name}.npz")
        if os.path.exists(path):
            log(f"{name}: exists, skipping")
            continue
        tiles = np.concatenate([fresh(sp, n, seed), held])
        np.savez(path, sandbox=tiles)
        log(f"wrote {path} {tiles.shape}")

    path = os.path.join(SCRATCH, "tiles_placement_cens.npz")
    if not os.path.exists(path):
        lam = coarse_spectrum(sp, recipe.COND_LEVEL)
        rng_par = np.random.default_rng(SEED_CENS_PARENTS)
        child = np.random.SeedSequence(SEED_CENS_REDRAWS).spawn(CENS_P)
        out = np.empty((CENS_P * CENS_R, 128, 128), np.float32)
        for p in range(CENS_P):
            g = sample_grf(sp, rng_par)
            cstar = haar_coarse(g, recipe.COND_LEVEL)
            rng_c = np.random.default_rng(child[p])
            for r in range(CENS_R):
                gc = conditional_redraw(cstar, sp, recipe.COND_LEVEL, rng_c,
                                        lam=lam)
                out[p * CENS_R + r] = lognormal_map(gc, recipe.SIGMA_G)
            if (p + 1) % 64 == 0:
                log(f"C-ENS parents {p+1}/{CENS_P}")
        tiles = np.concatenate([out, held])
        np.savez(path, sandbox=tiles)
        log(f"wrote {path} {tiles.shape}")
    log("done")


if __name__ == "__main__":
    main()
