"""C1 head-conditional scoring (JAX; runs in the GPU job after run_two_arms).

For each arm and checkpoint step: sample standardized detail at octave 2 given REAL
held-out coarse (plain-CFM ODE sampler, nll=False) and compute the estimand
var_slope of (generated detail, real coarse) — the dispersion-vs-training curve the
collapse signature is read from. At the final checkpoint: octaves 2,3,4 (+1
descriptive) with bootstrap-over-fields SEs, var_slope AND kurtosis.

Estimand code is the truth-side implementation (sandbox.truth_stats — numpy, no
pywt), so this runs in wl-challenge-env; the estimand's definition is identical to
the frozen instrument's (validated at Gate A).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

try:
    os.sched_setaffinity(0, set(range(4)))
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

from wfm import haar
from wfm.cfm import sample
from wfm.dataset import normalize_tiles
from wfm.model import ConditionalUNet
from sandbox.truth_stats import estimand_scalars

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COORD_NORM = np.array([1.5, 13.0])


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pooled_wc(det, coarse):
    """(B,H,W,3) generated detail + (B,H,W,1) real coarse -> pooled (w, c) per field."""
    det = np.asarray(det)
    coarse = np.asarray(coarse)[..., 0]
    per_field = []
    for i in range(det.shape[0]):
        w = np.concatenate([det[i, :, :, k].reshape(-1) for k in range(3)])
        c = np.tile(coarse[i].reshape(-1), 3)
        per_field.append((w, c))
    return per_field


def score(per_field, n_boot=200, seed=0):
    w = np.concatenate([p[0] for p in per_field])
    c = np.concatenate([p[1] for p in per_field])
    s = estimand_scalars(w, c)
    rng = np.random.default_rng(seed)
    n = len(per_field)
    boot = np.empty((n_boot, 2))
    for t in range(n_boot):
        idx = rng.integers(0, n, n)
        wb = np.concatenate([per_field[i][0] for i in idx])
        cb = np.concatenate([per_field[i][1] for i in idx])
        sb = estimand_scalars(wb, cb)
        boot[t] = (sb["var_slope"], sb["kurtosis"])
    return {"var_slope": s["var_slope"],
            "var_slope_se": float(np.nanstd(boot[:, 0], ddof=1)),
            "kurtosis": s["kurtosis"],
            "kurtosis_se": float(np.nanstd(boot[:, 1], ddof=1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="sandbox")
    ap.add_argument("--data", default=os.path.join(REPO, "data_cache",
                                                   "tiles_sandbox.npz"))
    ap.add_argument("--coords-file", default=os.path.join(
        REPO, "data_cache", "running_couplings_sandbox.json"))
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--ckpt-steps", type=int, nargs="+",
                    default=[1000, 2000, 4000, 6000, 8000, 12000, 16000])
    ap.add_argument("--n-heldout", type=int, default=64)
    ap.add_argument("--sample-steps", type=int, default=80)
    ap.add_argument("--octaves-final", type=int, nargs="+", default=[2, 3, 4, 1])
    ap.add_argument("--out", default=os.path.join(REPO, "results_p2",
                                                  "c1_conditional_sandbox.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tiles = np.load(args.data)[args.field].astype(np.float32)
    heldout = normalize_tiles(tiles[-args.n_heldout:])
    coords_raw = json.load(open(args.coords_file))[args.field]
    coords = {int(j): (np.asarray(v) / COORD_NORM) for j, v in coords_raw.items()}

    out = {"meta": vars(args) | {"data": args.data}}
    for arm in ("A", "B"):
        final_path = os.path.join(args.ckpt_dir, f"arm{arm}_{args.field}.pkl")
        with open(final_path, "rb") as fh:
            ck = pickle.load(fh)
        model = ConditionalUNet(out_channels=3, channels=tuple(ck["channels"]),
                                bottleneck=ck["channels"][-1] * 2,
                                cond_dim=ck["cond_dim"], cond_mode=ck["cond_mode"],
                                variance_head=bool(ck.get("nll", False)))
        assert not ck.get("nll", False), "C1 is the plain-CFM arm (no NLL head)"
        key = jax.random.PRNGKey(args.seed)
        res = {"curve_oct2": {}, "final": {}}

        def sample_octave(params, j, k):
            det_real, coarse = haar.octave_pair(heldout, j)
            cv = None if ck["cond_dim"] == 0 else jnp.broadcast_to(
                jnp.asarray(coords[j], jnp.float32),
                (coarse.shape[0], ck["cond_dim"]))
            det = sample(model.apply, params, k, coarse, 3,
                         n_steps=args.sample_steps, cond_vec=cv)
            return pooled_wc(det, coarse)

        for step in args.ckpt_steps:
            path = os.path.join(args.ckpt_dir,
                                f"arm{arm}_{args.field}_s{step}.pkl")
            with open(path, "rb") as fh:
                cks = pickle.load(fh)
            key, k = jax.random.split(key)
            pf = sample_octave(cks["params"], 2, k)
            s = score(pf, n_boot=50, seed=args.seed)
            res["curve_oct2"][str(step)] = s
            log(f"arm {arm} @{step}: oct2 conditional var_slope={s['var_slope']:.3f}")
        for j in args.octaves_final:
            key, k = jax.random.split(key)
            pf = sample_octave(ck["params"], j, k)
            s = score(pf, n_boot=200, seed=args.seed)
            res["final"][str(j)] = s
            log(f"arm {arm} final oct{j}: var_slope={s['var_slope']:.3f} "
                f"kurt={s['kurtosis']:.2f}")
        out[arm] = res
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
