#!/usr/bin/env python
"""R15 tail-dynamics diagnosis runner (one candidate x toy x data-size per job).

Serves all three authorized probes from one trajectory log:
  P1 (rung-4 causal test): 1x (768) vs 8x (6144) training fields under the
     IDENTICAL 12k-step schedule — data size is the only difference; decay onset
     compared within-schedule (the 4k bake-off runs are not valid onset baselines:
     a 12k window reshapes the cosine anneal).
  P2 (checkpoint viability): every eval logs dispersion (binned_sigma_maxrel vs
     the held-out data) AND skew/kurt/q999 — joint near-truth checkpoints are read
     off the same trajectory.
  P3 (t-base attribution): for tbase runs, every eval ALSO pushes a GAUSSIAN base
     through the SAME trained flow — the (t-base out) minus (N-base out) tail gap
     is the base's surviving contribution; nu is architecturally FIXED (=5, no
     learnable df), so any drift of realized tail order is the flow's doing.

Prereg log/2026-07-17-prereg-tail-dynamics.md; artifacts
results_p2/taildyn_<candidate>_<toy>_n<ntrain>.json (R12: verdicts quote these).
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
from arms_p2.toys import binned_sigma_maxrel, make_data, resid_stats
from wfm.cfm import make_train_state
from wfm.model import ConditionalUNet

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, choices=["twcrps", "tbase"])
    ap.add_argument("--toy", required=True, choices=["t5flat", "composite"])
    ap.add_argument("--n-train", type=int, required=True, choices=[768, 6144])
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    flat = args.toy == "t5flat"
    detail, coarse = make_data(jax.random.PRNGKey(9), "t5", flat_sigma=flat,
                               n=args.n_train)
    d_ho, c_ho = make_data(jax.random.PRNGKey(99), "t5", flat_sigma=flat, n=256)
    truth_ref = resid_stats(d_ho, c_ho)
    tag = f"{args.candidate}/{args.toy}/n{args.n_train}"
    print(f"[taildyn {tag}] DATA heldout truth reference: "
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
    else:
        step = make_tcfm_step(None, nu=5.0)

    def gen_stack(st, sampler, K=4, seed=11):
        outs, ccs = [], []
        for k in range(K):
            g = sampler(st, jax.random.PRNGKey(seed + k))
            outs.append(np.asarray(g))
            ccs.append(np.asarray(np.broadcast_to(c_ho, g.shape)))
        return np.concatenate(outs), np.concatenate(ccs)

    def evaluate(st):
        if args.candidate == "tbase":
            main_s = lambda s, k: sample_tbase(s.apply_fn, s.params, k, c_ho, 3,
                                               n_steps=80)
        else:
            main_s = lambda s, k: sample_direct(s.apply_fn, s.params, k, c_ho, 3)
        gen, cc = gen_stack(st, main_s)
        out = resid_stats(gen, cc)
        out["disp_maxrel"] = binned_sigma_maxrel(gen, cc, d_ho, c_ho)
        if args.candidate == "tbase":            # P3: N-base through the SAME flow
            from wfm.cfm import sample as sample_gaussbase
            gb = lambda s, k: sample_gaussbase(s.apply_fn, s.params, k, c_ho, 3,
                                               n_steps=80, solver="heun")
            geng, ccg = gen_stack(st, gb)
            sg = resid_stats(geng, ccg)
            out["gaussbase_kurt"] = sg["kurt"]
            out["gaussbase_q999"] = sg["q999"]
        return out

    rng = np.random.default_rng(args.seed)
    traj = {}
    t0 = time.time()
    for i in range(args.steps):
        idx = rng.integers(0, detail.shape[0], 16)
        state, loss = step(state, detail[idx], coarse[idx])
        if (i + 1) % args.eval_every == 0:
            s = evaluate(state)
            traj[str(i + 1)] = s | {"loss": float(loss)}
            extra = (f" Nbase-kurt {s['gaussbase_kurt']:.2f}"
                     if "gaussbase_kurt" in s else "")
            print(f"[taildyn {tag}] step {i+1:6d} loss {float(loss):.4f} "
                  f"skew {s['skew']:+.3f} kurt {s['kurt']:.2f} "
                  f"q999 {s['q999']:.2f} disp {s['disp_maxrel']:.3f}{extra} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    out = {"candidate": args.candidate, "toy": args.toy, "n_train": args.n_train,
           "config": vars(args), "truth_ref_heldout": truth_ref,
           "trajectory": traj}
    path = os.path.join(
        REPO, "results_p2",
        f"taildyn_{args.candidate}_{args.toy}_n{args.n_train}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[taildyn {tag}] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
