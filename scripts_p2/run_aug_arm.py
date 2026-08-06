#!/usr/bin/env python
"""AUG arm runner (prereg 2026-08-05-night3 §AUG): run_c1t_arms.py
mirrored, arm A only, with fractional band-limited training copies
(build_aug_stacks on TRAIN tiles only). Selection rule, TEST eval, and
the e2e recursion are VERBATIM baseline (original geometry; copies are
training-only; std convention from the original stack)."""
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

from arms_p2.aug.build import build_aug_stacks
from arms_p2.aug.train import train_aug_generator
from arms_p2.c1t.flow import sample_tbase
from arms_p2.c1t.train import generate_recursive_tbase
from sandbox.truth_stats import estimand_scalars, tail_q999
from wfm import haar
from wfm.dataset import field_to_octaves, normalize_tiles
from wfm.model import ConditionalUNet

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pooled_wc(det, coarse):
    det = np.asarray(det)
    coarse = np.asarray(coarse)[..., 0]
    per_field = []
    for i in range(det.shape[0]):
        w = np.concatenate([det[i, :, :, k].reshape(-1) for k in range(3)])
        c = np.tile(coarse[i].reshape(-1), 3)
        per_field.append((w, c))
    return per_field


def score_full(per_field, n_boot=200, seed=0):
    w = np.concatenate([p[0] for p in per_field])
    c = np.concatenate([p[1] for p in per_field])
    s = estimand_scalars(w, c)
    q = tail_q999(w)
    rng = np.random.default_rng(seed)
    n = len(per_field)
    boot = np.empty((n_boot, 3))
    for t in range(n_boot):
        idx = rng.integers(0, n, n)
        wb = np.concatenate([per_field[i][0] for i in idx])
        cb = np.concatenate([per_field[i][1] for i in idx])
        sb = estimand_scalars(wb, cb)
        boot[t] = (sb["var_slope"], sb["kurtosis"], tail_q999(wb))
    return {"var_slope": s["var_slope"],
            "var_slope_se": float(np.nanstd(boot[:, 0], ddof=1)),
            "kurtosis": s["kurtosis"],
            "kurtosis_se": float(np.nanstd(boot[:, 1], ddof=1)),
            "q999": q, "q999_se": float(np.nanstd(boot[:, 2], ddof=1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="sandbox")
    ap.add_argument("--gen-from", type=int, default=4)
    ap.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--cond-mode", default="film")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-heldout", type=int, default=64)
    ap.add_argument("--sample-steps", type=int, default=80)
    ap.add_argument("--sel-K", type=int, default=4)
    ap.add_argument("--sel-octave", type=int, default=2)
    ap.add_argument("--data", default=os.path.join(REPO, "data_cache",
                                                   "tiles_sandbox.npz"))
    ap.add_argument("--coords-file", default=os.path.join(
        REPO, "data_cache", "running_couplings_sandbox.json"))
    ap.add_argument("--truth", default=os.path.join(
        REPO, "results_p2", "sandbox_truth_normconv.json"))
    ap.add_argument("--out", default=os.path.join(REPO, "results_p2",
                                                  "night3_aug_arms.npz"))
    ap.add_argument("--sel-out", default=os.path.join(
        REPO, "results_p2", "night3_aug_selection.json"))
    ap.add_argument("--ckpt-dir", default=os.path.join(REPO, "data_cache",
                                                       "ckpt_night3_aug"))
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    cfg = vars(args).copy()
    cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
    log(f"config_hash={cfg_hash} devices={jax.devices()}")
    log(f"config={json.dumps(cfg)}")

    tiles = np.load(args.data)[args.field].astype(np.float32)
    heldout, train = tiles[-args.n_heldout:], tiles[:-args.n_heldout]
    nv = args.n_heldout // 2
    val, test = heldout[:nv], heldout[nv:]
    val_n, test_n = normalize_tiles(val), normalize_tiles(test)
    truth2 = json.load(open(args.truth))["truth"][str(args.sel_octave)]
    aug = build_aug_stacks(train)
    # slot sizes must be = 0 mod 8 (UNet halvings; A-N3-4): g96 oct-3
    # (12) and g80 oct-2 (20) excluded -> 6 slots.
    groups = [(train, [2, 3, 4]), (aug["g96"], [1, 2]),
              (aug["g80"], [1])]
    log(f"{args.field}: {train.shape[0]} train (+{aug['g96'].shape[0]} g96 "
        f"+{aug['g80'].shape[0]} g80) / {nv} val / {len(test)} test; "
        f"selection reference oct{args.sel_octave} "
        f"vs={truth2['var_slope']:.3f} kurt={truth2['kurtosis']:.3f}")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_steps = list(range(args.ckpt_every, args.steps + 1,
                            args.ckpt_every))
    out = {"meta": cfg | {"config_hash": cfg_hash}}
    t0 = time.time()

    def save_ckpt(step_i, st, loss_i):
        path = os.path.join(args.ckpt_dir,
                            f"armA_{args.field}_s{step_i}.pkl")
        with open(path, "wb") as fh:
            pickle.dump({"params": jax.tree_util.tree_map(np.asarray,
                                                          st.params),
                         "step": step_i, "loss": loss_i}, fh)

    state, meta = train_aug_generator(
        groups, channels=tuple(args.channels), steps=args.steps,
        batch=args.batch, lr=args.lr, seed=args.seed,
        cond_mode=args.cond_mode, augment=True,
        ckpt_steps=tuple(ckpt_steps), on_checkpoint=save_ckpt)
    log(f"trained, loss {meta['loss0']:.3f}->{meta['lossN']:.4f}; "
        f"slot_visits={meta['slot_visits']}")
    std = dict(meta["std_by_j"])
    js = np.array(sorted(std))
    a_, b_ = np.polyfit(js, np.log([std[j] for j in js]), 1)
    for j in range(1, min(meta["train_octaves"])):
        std[j] = float(np.exp(a_ * j + b_))

    model = ConditionalUNet(out_channels=3, channels=tuple(args.channels),
                            bottleneck=args.channels[-1] * 2, cond_dim=0,
                            cond_mode=args.cond_mode, variance_head=False)

    def sample_oct(params, j, fields_n, key, K=1):
        det_real, coarse = haar.octave_pair(fields_n, j)
        pf = []
        for k in range(K):
            key, kk = jax.random.split(key)
            det = sample_tbase(model.apply, params, kk, coarse, 3,
                               n_steps=args.sample_steps, cond_vec=None)
            pf += pooled_wc(det, coarse)
        return pf

    # selection sweep on VALIDATION (frozen rule, original stack)
    curve = {}
    key = jax.random.PRNGKey(args.seed + 100)
    for si in ckpt_steps:
        with open(os.path.join(args.ckpt_dir,
                               f"armA_{args.field}_s{si}.pkl"), "rb") as fh:
            cks = pickle.load(fh)
        key, k = jax.random.split(key)
        pf = sample_oct(cks["params"], args.sel_octave, val_n, k,
                        K=args.sel_K)
        w = np.concatenate([p[0] for p in pf])
        c = np.concatenate([p[1] for p in pf])
        s = estimand_scalars(w, c)
        rv = abs(s["var_slope"] - truth2["var_slope"]) / abs(truth2["var_slope"])
        rk = abs(s["kurtosis"] - truth2["kurtosis"]) / abs(truth2["kurtosis"])
        sc = max(rv / 0.10, rk / 0.15) if np.isfinite(rv + rk) else np.inf
        curve[str(si)] = {"var_slope": s["var_slope"],
                          "kurtosis": s["kurtosis"],
                          "q999": tail_q999(w), "rel_vs": rv,
                          "rel_kurt": rk, "score": sc}
        log(f"sel @{si}: vs={s['var_slope']:.3f} kurt={s['kurtosis']:.2f} "
            f"score={sc:.3f}")
    sel = min(ckpt_steps, key=lambda si: (curve[str(si)]["score"], si))
    log(f"SELECTED checkpoint: step {sel} "
        f"(score {curve[str(sel)]['score']:.3f})")

    arm_out = {"curve_val": curve, "selected_step": sel, "final": {},
               "final_at_end": {}}
    for tag, si in (("final", sel), ("final_at_end", args.steps)):
        with open(os.path.join(args.ckpt_dir,
                               f"armA_{args.field}_s{si}.pkl"), "rb") as fh:
            cks = pickle.load(fh)
        key = jax.random.PRNGKey(args.seed + 200)
        for j in (2, 3, 4, 1):
            key, k = jax.random.split(key)
            pf = sample_oct(cks["params"], j, test_n, k, K=1)
            s = score_full(pf, n_boot=200, seed=args.seed)
            arm_out[tag][str(j)] = s
            if tag == "final":
                log(f"TEST oct{j} @sel: vs={s['var_slope']:.3f} "
                    f"kurt={s['kurtosis']:.2f} q999={s['q999']:.2f}")

    with open(os.path.join(args.ckpt_dir,
                           f"armA_{args.field}_s{sel}.pkl"), "rb") as fh:
        cks = pickle.load(fh)
    pools, _ = field_to_octaves(test, [args.gen_from])
    coarse = pools[args.gen_from][1]
    gen = generate_recursive_tbase(model.apply, cks["params"], coarse,
                                   args.gen_from,
                                   jax.random.PRNGKey(args.seed + 1), std,
                                   n_steps=args.sample_steps)
    results = {"gen_A": np.asarray(gen[..., 0]),
               "real": np.asarray(test_n[..., 0])}
    out["A"] = arm_out
    np.savez(args.out, config=json.dumps(cfg), config_hash=cfg_hash,
             **results)
    with open(args.sel_out, "w") as f:
        json.dump({"A": arm_out, "meta": {"slots": meta["slots"],
                                          "slot_visits": meta["slot_visits"],
                                          "aug_std": meta["aug_std"]}},
                  f, indent=1, default=str)
    log(f"wrote {args.out} and {args.sel_out} in {time.time()-t0:.0f}s "
        f"config_hash={cfg_hash}")


if __name__ == "__main__":
    main()
