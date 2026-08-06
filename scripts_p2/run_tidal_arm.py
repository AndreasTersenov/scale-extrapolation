#!/usr/bin/env python
"""Arm TIDAL: eigenframe conditioning — t(5)-base CFM + pre-registered
validation-selected early stop, the C1-t recipe verbatim through the 4-channel
(coarse, H(coarse)) conditioning (prereg log/2026-08-05-prereg-night3.md
"TIDAL — design"; trains iff GATE-T).

run_c1t_arms.py mirrored: one job per leg, arm A (no cond vector), DENSE
checkpoints (every 500 steps), the PRE-REGISTERED selection rule on VALIDATION
fields only (sel-octave 2, score = max(rel_err_var_slope / 0.10,
rel_err_kurtosis / 0.15), argmin, ties -> earlier), TEST evaluation at the
selected (and 20k) checkpoint, e2e recursion via the tidal sampler
(generate_recursive_tidal). Every ckpt pickle embeds feat_std_by_j + sigma_h
so the night3 sampling scripts need only the ckpt.
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

from arms_p2.c1t.flow import sample_tbase
from arms_p2.tidal.features import SIGMA_H, hessian_features
from arms_p2.tidal.sample import feat_std_ladder
from arms_p2.tidal.train import (generate_recursive_tidal, tidal_feat_std,
                                 train_tidal_generator)
from sandbox.truth_stats import estimand_scalars, tail_q999
from wfm import haar
from wfm.dataset import field_to_octaves, normalize_tiles
from wfm.model import ConditionalUNet

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COORD_NORM = np.array([1.5, 13.0])


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
    ap.add_argument("--train-octaves", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--gen-from", type=int, default=4)
    ap.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--cond-mode", default="film", choices=["add", "film"])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-heldout", type=int, default=64,
                    help="split: first half VALIDATION (selection), last half TEST")
    ap.add_argument("--sample-steps", type=int, default=80)
    ap.add_argument("--sel-K", type=int, default=4)
    ap.add_argument("--sel-octave", type=int, default=2,
                    help="octave the selection rule scores on validation "
                         "(unchanged from the C1-t rule; must be a TRAINED octave)")
    ap.add_argument("--data", default=os.path.join(REPO, "data_cache",
                                                   "tiles_sandbox.npz"))
    ap.add_argument("--coords-file", default=os.path.join(
        REPO, "data_cache", "running_couplings_sandbox.json"))
    ap.add_argument("--truth", default=os.path.join(
        REPO, "results_p2", "sandbox_truth_normconv.json"))
    ap.add_argument("--out", default=os.path.join(REPO, "results_p2",
                                                  "arms_tidal_sandbox.npz"))
    ap.add_argument("--sel-out", default=os.path.join(
        REPO, "results_p2", "tidal_selection_sandbox.json"))
    ap.add_argument("--ckpt-dir", default=os.path.join(REPO, "data_cache",
                                                       "ckpt_tidal_sandbox"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", nargs="+", default=["A"],
                    choices=["A", "B"],
                    help="TIDAL prereg trains arm A only (features replace "
                         "the cond vector); B kept for parity with the "
                         "trainer contract")
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
    coords_raw = json.load(open(args.coords_file))[args.field]
    coords = {int(j): (np.asarray(v) / COORD_NORM) for j, v in coords_raw.items()}
    truth2 = json.load(open(args.truth))["truth"][str(args.sel_octave)]
    log(f"{args.field}: {train.shape[0]} train / {nv} val / {len(test)} test; "
        f"selection reference oct{args.sel_octave} vs={truth2['var_slope']:.3f} "
        f"kurt={truth2['kurtosis']:.3f}")

    # feature stds are deterministic in the train split — computed ONCE here so
    # every ckpt pickle carries them; asserted against the trainer's meta below
    feat_std_pre = tidal_feat_std(train, args.train_octaves)
    log("feat_std_by_j: " + " ".join(
        f"j{j}=[" + ",".join(f"{v:.4g}" for v in feat_std_pre[j]) + "]"
        for j in args.train_octaves))

    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_steps = list(range(args.ckpt_every, args.steps + 1, args.ckpt_every))
    results, out = {}, {"meta": cfg | {"config_hash": cfg_hash}}
    t0 = time.time()
    for arm in args.arms:
        def save_ckpt(step_i, st, loss_i, _arm=arm):
            path = os.path.join(args.ckpt_dir,
                                f"arm{_arm}_{args.field}_s{step_i}.pkl")
            with open(path, "wb") as fh:
                pickle.dump({"params": jax.tree_util.tree_map(np.asarray, st.params),
                             "step": step_i, "loss": loss_i,
                             "feat_std_by_j": feat_std_pre,
                             "sigma_h": SIGMA_H}, fh)

        state, meta = train_tidal_generator(
            train, args.train_octaves, arm=arm,
            cond_by_octave=(coords if arm == "B" else None),
            channels=tuple(args.channels), steps=args.steps, batch=args.batch,
            lr=args.lr, seed=args.seed, cond_mode=args.cond_mode, augment=True,
            ckpt_steps=tuple(ckpt_steps), on_checkpoint=save_ckpt)
        log(f"arm {arm}: trained, loss {meta['loss0']:.3f}->{meta['lossN']:.4f}")
        for j in args.train_octaves:                      # convention-drift gate
            assert np.array_equal(feat_std_pre[j], meta["feat_std_by_j"][j]), \
                f"feat std drift at octave {j} (ckpt embedding vs trainer)"
        fstd = feat_std_ladder(meta["feat_std_by_j"])
        std = dict(meta["std_by_j"])
        js = np.array(sorted(std))
        a_, b_ = np.polyfit(js, np.log([std[j] for j in js]), 1)
        for j in range(1, min(args.train_octaves)):
            std[j] = float(np.exp(a_ * j + b_))

        model = ConditionalUNet(out_channels=3, channels=tuple(args.channels),
                                bottleneck=args.channels[-1] * 2,
                                cond_dim=meta["cond_dim"],
                                cond_mode=args.cond_mode, variance_head=False)

        def sample_oct(params, j, fields_n, key, K=1):
            det_real, coarse = haar.octave_pair(fields_n, j)
            feats = hessian_features(coarse) / jnp.asarray(fstd[j])
            cond4 = jnp.concatenate([coarse, feats], axis=-1)
            cv = None if meta["cond_dim"] == 0 else jnp.broadcast_to(
                jnp.asarray(coords[j], jnp.float32),
                (coarse.shape[0], meta["cond_dim"]))
            pf = []
            for k in range(K):
                key, kk = jax.random.split(key)
                det = sample_tbase(model.apply, params, kk, cond4, 3,
                                   n_steps=args.sample_steps, cond_vec=cv)
                pf += pooled_wc(det, coarse)
            return pf

        # ---- selection sweep on VALIDATION (the pre-registered rule) ----------
        curve = {}
        key = jax.random.PRNGKey(args.seed + 100)
        for si in ckpt_steps:
            with open(os.path.join(args.ckpt_dir,
                                   f"arm{arm}_{args.field}_s{si}.pkl"), "rb") as fh:
                cks = pickle.load(fh)
            key, k = jax.random.split(key)
            pf = sample_oct(cks["params"], args.sel_octave, val_n, k, K=args.sel_K)
            w = np.concatenate([p[0] for p in pf])
            c = np.concatenate([p[1] for p in pf])
            s = estimand_scalars(w, c)
            rv = abs(s["var_slope"] - truth2["var_slope"]) / abs(truth2["var_slope"])
            rk = abs(s["kurtosis"] - truth2["kurtosis"]) / abs(truth2["kurtosis"])
            sc = max(rv / 0.10, rk / 0.15) if np.isfinite(rv + rk) else np.inf
            curve[str(si)] = {"var_slope": s["var_slope"], "kurtosis": s["kurtosis"],
                              "q999": tail_q999(w), "rel_vs": rv, "rel_kurt": rk,
                              "score": sc}
            log(f"arm {arm} sel @{si}: vs={s['var_slope']:.3f} "
                f"kurt={s['kurtosis']:.2f} score={sc:.3f}")
        sel = min(ckpt_steps,
                  key=lambda si: (curve[str(si)]["score"], si))
        log(f"arm {arm} SELECTED checkpoint: step {sel} "
            f"(score {curve[str(sel)]['score']:.3f})")

        # ---- adjudication numbers on TEST at the selected (and final) ckpt ----
        arm_out = {"curve_val": curve, "selected_step": sel, "final": {},
                   "final_at_20k": {}}
        for tag, si in (("final", sel), ("final_at_20k", args.steps)):
            with open(os.path.join(args.ckpt_dir,
                                   f"arm{arm}_{args.field}_s{si}.pkl"), "rb") as fh:
                cks = pickle.load(fh)
            key = jax.random.PRNGKey(args.seed + 200)
            for j in (2, 3, 4, 1):
                key, k = jax.random.split(key)
                pf = sample_oct(cks["params"], j, test_n, k, K=1)
                s = score_full(pf, n_boot=200, seed=args.seed)
                arm_out[tag][str(j)] = s
                if tag == "final":
                    log(f"arm {arm} TEST oct{j} @sel: vs={s['var_slope']:.3f} "
                        f"kurt={s['kurtosis']:.2f} q999={s['q999']:.2f}")

        # ---- end-to-end recursion npz from the SELECTED checkpoint ------------
        with open(os.path.join(args.ckpt_dir,
                               f"arm{arm}_{args.field}_s{sel}.pkl"), "rb") as fh:
            cks = pickle.load(fh)
        pools, _ = field_to_octaves(test, [args.gen_from])
        coarse = pools[args.gen_from][1]
        B = coarse.shape[0]
        cond_fn = None
        if arm == "B":
            def cond_fn(j):
                return jnp.broadcast_to(jnp.asarray(coords[j], jnp.float32), (B, 2))
        gen = generate_recursive_tidal(model.apply, cks["params"], coarse,
                                       args.gen_from,
                                       jax.random.PRNGKey(args.seed + 1), std,
                                       fstd, cond_fn=cond_fn,
                                       n_steps=args.sample_steps)
        results[f"gen_{arm}"] = np.asarray(gen[..., 0])
        out[arm] = arm_out
        log(f"arm {arm}: e2e recursion from selected ckpt done "
            f"[{time.time()-t0:.0f}s]")

    results["real"] = np.asarray(test_n[..., 0])
    np.savez(args.out, config=json.dumps(cfg), config_hash=cfg_hash, **results)
    with open(args.sel_out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    log(f"wrote {args.out} and {args.sel_out} in {time.time()-t0:.0f}s "
        f"config_hash={cfg_hash}")


if __name__ == "__main__":
    main()
