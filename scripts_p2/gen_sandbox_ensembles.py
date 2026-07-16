"""Stage A runner: exact conditional ensembles -> TRUE statistics -> Gate A inputs.

Steps (incremental; each skipped if its output exists, so the job is resumable):
  1. generate 256 parents x 64 exact conditional redraws (COND_LEVEL Gaussian coarse
     fixed), lognormal-mapped; + 322 unconditional training tiles; arrays -> $SCRATCH.
  2. TRUE estimand values per octave (sandbox.truth_stats, batch-means SE) -> repo.
  3. frozen instrument (scripts/measure_generated.couplings) on the 256 parent fields
     (PRIMARY, pre-declared) and on the first 64 (descriptive) -> repo.
Run under the env.sh stack (pywt needed for step 3 only).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from sandbox import recipe
from sandbox.lognormal import (
    coarse_spectrum, conditional_redraw, haar_coarse, lognormal_map, sample_grf,
)
from sandbox.truth_stats import truth_couplings

SCRATCH = os.path.expanduser("~/links/scratch/scale-extrap-p2")
ENS_PATH = os.path.join(SCRATCH, "sandbox_ens_f32.npy")          # (P, R, 128, 128)
PARENTS_PATH = os.path.join(SCRATCH, "sandbox_parents_f32.npy")  # (P, 128, 128)
CSTARS_PATH = os.path.join(SCRATCH, "sandbox_cstars_f64.npy")    # (P, 8, 8)
TRAIN_PATH = os.path.join(SCRATCH, "sandbox_train_f32.npy")      # (322, 128, 128)
TRUTH_OUT = os.path.join(REPO, "results_p2", "sandbox_truth.json")
INST_OUT = os.path.join(REPO, "results_p2", "gateA_instrument.json")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def step1_generate():
    if os.path.exists(ENS_PATH) and os.path.exists(TRAIN_PATH):
        log("step 1: outputs exist, skipping")
        return
    os.makedirs(SCRATCH, exist_ok=True)
    sp = recipe.spec()
    lam = coarse_spectrum(sp, recipe.COND_LEVEL)
    P, R = recipe.N_PARENTS, recipe.N_REDRAWS

    rng_par = np.random.default_rng(recipe.SEED_PARENTS)
    child_seeds = np.random.SeedSequence(recipe.SEED_REDRAWS).spawn(P)

    ens = np.lib.format.open_memmap(ENS_PATH, mode="w+", dtype=np.float32,
                                    shape=(P, R) + recipe.SHAPE)
    parents = np.empty((P,) + recipe.SHAPE, dtype=np.float32)
    cstars = np.empty((P, recipe.SHAPE[0] // 2 ** recipe.COND_LEVEL,
                       recipe.SHAPE[1] // 2 ** recipe.COND_LEVEL))
    t0 = time.time()
    for i in range(P):
        g_parent = sample_grf(sp, rng_par)
        c_star = haar_coarse(g_parent, recipe.COND_LEVEL)
        parents[i] = lognormal_map(g_parent, recipe.SIGMA_G).astype(np.float32)
        cstars[i] = c_star
        rng_i = np.random.default_rng(child_seeds[i])
        for r in range(R):
            g = conditional_redraw(c_star, sp, recipe.COND_LEVEL, rng_i, lam=lam)
            ens[i, r] = lognormal_map(g, recipe.SIGMA_G).astype(np.float32)
        if (i + 1) % 32 == 0:
            log(f"step 1: parent {i+1}/{P} ({time.time()-t0:.0f}s)")
    ens.flush()
    np.save(PARENTS_PATH, parents)
    np.save(CSTARS_PATH, cstars)

    rng_tr = np.random.default_rng(recipe.SEED_TRAIN)
    train = np.stack([lognormal_map(sample_grf(sp, rng_tr), recipe.SIGMA_G)
                      for _ in range(recipe.N_TRAIN_TILES)]).astype(np.float32)
    np.save(TRAIN_PATH, train)
    log(f"step 1 done in {time.time()-t0:.0f}s")


def step2_truth():
    if os.path.exists(TRUTH_OUT):
        log("step 2: output exists, skipping")
        return
    ens = np.load(ENS_PATH, mmap_mode="r")
    P, R = ens.shape[:2]
    fields = [np.asarray(ens[i, r], dtype=np.float64)
              for i in range(P) for r in range(R)]
    groups = np.repeat(np.arange(P), R)
    log(f"step 2: truth estimand on {len(fields)} exact fields")
    t0 = time.time()
    truth = truth_couplings(fields, groups, recipe.OCTAVES, n_blocks=16)
    meta = {"recipe": {k: getattr(recipe, k) for k in
                       ("SHAPE", "ALPHA", "SIGMA_G", "COND_LEVEL", "N_PARENTS",
                        "N_REDRAWS", "SEED_PARENTS", "SEED_REDRAWS", "SEED_TRAIN")},
            "n_fields": len(fields), "n_blocks": 16,
            "octaves": list(recipe.OCTAVES)}
    with open(TRUTH_OUT, "w") as f:
        json.dump({"truth": {str(j): truth[j] for j in truth}, "meta": meta}, f,
                  indent=1, default=str)
    log(f"step 2 done in {time.time()-t0:.0f}s -> {TRUTH_OUT}")


def step3_instrument():
    if os.path.exists(INST_OUT):
        log("step 3: output exists, skipping")
        return
    from measure_generated import couplings  # frozen production path (needs pywt)
    parents = np.load(PARENTS_PATH)
    fields256 = [np.asarray(parents[i], dtype=np.float64) for i in range(len(parents))]
    log("step 3: frozen instrument on 256 parent fields (PRIMARY)")
    t0 = time.time()
    inst256 = couplings(fields256, list(recipe.OCTAVES), n_boot=200, seed=0)
    log(f"step 3: primary done ({time.time()-t0:.0f}s); descriptive N=64 next")
    inst64 = couplings(fields256[:64], list(recipe.OCTAVES), n_boot=200, seed=0)
    with open(INST_OUT, "w") as f:
        json.dump({"primary_n256": {str(j): inst256[j] for j in inst256},
                   "descriptive_n64": {str(j): inst64[j] for j in inst64}}, f, indent=1)
    log(f"step 3 done in {time.time()-t0:.0f}s -> {INST_OUT}")


if __name__ == "__main__":
    step1_generate()
    step2_truth()
    step3_instrument()
    log("all steps complete")
