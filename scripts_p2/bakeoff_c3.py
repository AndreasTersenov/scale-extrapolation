#!/usr/bin/env python
"""R13 objective bake-off runner: one candidate x one toy gate per invocation.

Candidates (pinned in log/2026-07-17-c3-bakeoff.md BEFORE any result):
  twcrps — patched ES with the chain_tail threshold weighting (tau=2, a=4)
  beta05 — patched ES with beta=0.5
  tbase  — C1 CFM with unit-variance Student-t(5) base, heun-80 pushforward

Toys (the validated C3 gate pair; production net + patch config, 768 train /
256 held-out fields, 4k steps, lr 1e-3, held-out eval):
  t5flat    — flat sigma x symmetric t(5) noise (truth: skew 0, excess kurt 6)
  composite — modulated sigma x t(5) (truth measured in-job from the data)

Writes results_p2/bakeoff_<candidate>_<toy>.json with the full trajectory; every
number the verdict table quotes comes from these artifacts (R12 rule).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

try:
    os.sched_setaffinity(0, set(range(16)))
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax

from arms_p2.c1t.flow import make_tcfm_step, sample_tbase
from arms_p2.c3.energy import chain_tail
from arms_p2.c3.sampler import sample_direct
from arms_p2.c3.train import make_es_step
from arms_p2.toys import make_data, resid_stats, true_mean
from wfm.cfm import make_train_state
from wfm.model import ConditionalUNet

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True,
                    choices=["twcrps", "beta05", "tbase"])
    ap.add_argument("--toy", required=True, choices=["t5flat", "composite"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    flat = args.toy == "t5flat"
    detail, coarse = make_data(jax.random.PRNGKey(9), "t5", flat_sigma=flat, n=768)
    d_ho, c_ho = make_data(jax.random.PRNGKey(99), "t5", flat_sigma=flat, n=256)
    truth_ref = resid_stats(d_ho, c_ho)
    print(f"[bakeoff {args.candidate}/{args.toy}] DATA heldout truth reference: "
          f"skew {truth_ref['skew']:.3f} kurt {truth_ref['kurt']:.2f} "
          f"q999 {truth_ref['q999']:.2f}", flush=True)

    model = ConditionalUNet(out_channels=3, channels=(32, 64, 128), bottleneck=256,
                            embed_dim=128, cond_dim=0)
    state = make_train_state(model, jax.random.PRNGKey(args.seed),
                             (16,) + detail.shape[1:], (16,) + coarse.shape[1:],
                             0, args.lr, total_steps=args.steps,
                             warmup=args.steps // 10)
    if args.candidate == "twcrps":
        step = make_es_step(None, m=8, chain=chain_tail)
    elif args.candidate == "beta05":
        step = make_es_step(None, m=8, beta=0.5)
    else:
        step = make_tcfm_step(None, nu=5.0)

    def evaluate(st, K=4, seed=11):
        outs, ccs = [], []
        for k in range(K):
            kk = jax.random.PRNGKey(seed + k)
            if args.candidate == "tbase":
                g = sample_tbase(st.apply_fn, st.params, kk, c_ho, 3, n_steps=80)
            else:
                g = sample_direct(st.apply_fn, st.params, kk, c_ho, 3)
            outs.append(np.asarray(g))
            ccs.append(np.asarray(np.broadcast_to(c_ho, g.shape)))
        import jax.numpy as jnp
        return resid_stats(jnp.asarray(np.concatenate(outs)),
                           jnp.asarray(np.concatenate(ccs)))

    every = 1000 if args.candidate == "tbase" else 500
    rng = np.random.default_rng(args.seed)
    traj = {}
    t0 = time.time()
    for i in range(args.steps):
        idx = rng.integers(0, detail.shape[0], 16)
        state, loss = step(state, detail[idx], coarse[idx])
        if (i + 1) % every == 0:
            s = evaluate(state)
            traj[str(i + 1)] = s | {"loss": float(loss)}
            print(f"[bakeoff {args.candidate}/{args.toy}] step {i+1:5d} "
                  f"loss {float(loss):.4f} HELDOUT skew {s['skew']:.3f} "
                  f"kurt {s['kurt']:.2f} q999 {s['q999']:.2f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    out = {"candidate": args.candidate, "toy": args.toy,
           "config": vars(args), "truth_ref_heldout": truth_ref,
           "trajectory": traj, "final": traj[str(args.steps)]}
    path = os.path.join(REPO, "results_p2",
                        f"bakeoff_{args.candidate}_{args.toy}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[bakeoff {args.candidate}/{args.toy}] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
