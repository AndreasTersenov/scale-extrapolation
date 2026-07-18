#!/usr/bin/env python
"""R18 required replication: the C1-t sandbox verdict re-adjudicated on 64 FRESH
test fields (new seed, exact law) at the FROZEN selected checkpoints — no
selection, no retraining, pure test.

Fresh tiles: data_cache/tiles_sandbox_repl.npz (seed 20260720, recipe in
make_c1t_repl_data.py). Checkpoints: data_cache/ckpt_c1t_sandbox arm A @7500 /
arm B @2500 (the committed picks in c1t_selection_sandbox.json — read from there,
not hard-coded). std_by_j is recomputed deterministically from the training tiles
exactly as the trainer did (the dense ckpts store params only). Bars identical to
the prereg (kurtosis 15%/3SE primary, var_slope 10%/3SE), now with ~sqrt(2)
tighter SEs. Writes results_p2/c1t_repl64.json.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np

try:
    os.sched_setaffinity(0, set(range(16)))
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts_p2"))

import jax
import jax.numpy as jnp

from arms_p2.c1t.flow import sample_tbase
from arms_p2.c1t.train import generate_recursive_tbase
from run_c1t_arms import pooled_wc, score_full
from sandbox.haar import octave_wc_pooled
from sandbox.truth_stats import estimand_scalars, tail_q999
from wfm import haar
from wfm.dataset import d4_augment, field_to_octaves, normalize_tiles
from wfm.model import ConditionalUNet

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COORD_NORM = np.array([1.5, 13.0])
TRUTH = json.load(open(os.path.join(REPO, "results_p2",
                                    "sandbox_truth_normconv.json")))["truth"]
SEL = json.load(open(os.path.join(REPO, "results_p2",
                                  "c1t_selection_sandbox.json")))
BARS = {"var_slope": 0.10, "kurtosis": 0.15}
TRAINED = [2, 3, 4]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def check(metric, val, se, j):
    t, tse = TRUTH[str(j)][metric], TRUTH[str(j)][metric + "_se"]
    rel = abs(val - t) / abs(t)
    bar = max(BARS[metric], 3 * float(np.hypot(se, tse) / abs(t)))
    return {"value": val, "se": se, "truth": t, "rel_err": rel, "bar": bar,
            "pass": bool(rel <= bar)}


tiles = np.load(os.path.join(REPO, "data_cache", "tiles_sandbox.npz"))["sandbox"]
train = tiles[:-64].astype(np.float32)
fresh = np.load(os.path.join(REPO, "data_cache",
                             "tiles_sandbox_repl.npz"))["sandbox"].astype(np.float32)
assert fresh.shape[0] == 64, fresh.shape
fresh_n = normalize_tiles(fresh)
coords_raw = json.load(open(os.path.join(
    REPO, "data_cache", "running_couplings_sandbox.json")))["sandbox"]
coords = {int(j): (np.asarray(v) / COORD_NORM) for j, v in coords_raw.items()}

_, std_by_j = field_to_octaves(d4_augment(train), TRAINED)
js = np.array(sorted(std_by_j))
a_, b_ = np.polyfit(js, np.log([std_by_j[j] for j in js]), 1)
std = dict(std_by_j)
std[1] = float(np.exp(a_ * 1 + b_))

gen_stacks = {}
out = {"meta": {"n_fields": 64, "seed_tiles": 20260720,
                              "selected": {a: SEL[a]["selected_step"]
                                           for a in ("A", "B")}}}
for arm in ("A", "B"):
    sel_step = SEL[arm]["selected_step"]
    with open(os.path.join(REPO, "data_cache", "ckpt_c1t_sandbox",
                           f"arm{arm}_sandbox_s{sel_step}.pkl"), "rb") as fh:
        params = pickle.load(fh)["params"]
    cond_dim = 0 if arm == "A" else 2
    model = ConditionalUNet(out_channels=3, channels=(32, 64, 128),
                            bottleneck=256, cond_dim=cond_dim, cond_mode="film",
                            variance_head=False)
    key = jax.random.PRNGKey(500)
    hc = {}
    for j in (2, 3, 4, 1):
        det_real, coarse = haar.octave_pair(fresh_n, j)
        cv = None if cond_dim == 0 else jnp.broadcast_to(
            jnp.asarray(coords[j], jnp.float32), (coarse.shape[0], 2))
        key, k = jax.random.split(key)
        det = sample_tbase(model.apply, params, k, coarse, 3, n_steps=80,
                           cond_vec=cv)
        s = score_full(pooled_wc(det, coarse), n_boot=200, seed=0)
        hc[str(j)] = s
        log(f"arm {arm} REPL hc oct{j}: vs={s['var_slope']:.3f} "
            f"kurt={s['kurtosis']:.2f} q999={s['q999']:.2f}")
    pools, _ = field_to_octaves(fresh, [4])
    coarse4 = pools[4][1]
    B = coarse4.shape[0]
    cond_fn = None
    if arm == "B":
        def cond_fn(j):
            return jnp.broadcast_to(jnp.asarray(coords[j], jnp.float32), (B, 2))
    gen = generate_recursive_tbase(model.apply, params, coarse4, 4,
                                   jax.random.PRNGKey(501), std,
                                   cond_fn=cond_fn, n_steps=80)
    gen_stacks[arm] = np.asarray(gen[..., 0])
    out.setdefault("hc_full", {})[arm] = hc

# SAMPLE phase output (wl-challenge-env, JAX): the SCORE phase (env.sh stack,
# pywt for the frozen production scorer) adjudicates — the two stacks cannot
# share a process (job 16669983 failed exactly there).
np.savez(os.path.join(REPO, "results_p2", "c1t_repl64_gen.npz"),
         gen_A=gen_stacks["A"], gen_B=gen_stacks["B"])
with open(os.path.join(REPO, "results_p2", "c1t_repl64_hc.json"), "w") as f:
    json.dump(out, f, indent=1)
log("SAMPLE phase done -> c1t_repl64_gen.npz + c1t_repl64_hc.json; "
    "run c1t_replication_score.py under env.sh to adjudicate")
