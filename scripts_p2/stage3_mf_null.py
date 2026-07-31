"""A4 (R42): the Minkowski-judge split-half null as a COMMITTED artifact.

Reference-side null calibration for the Stage-3 one-shot's MF bar (the
judge freeze quarantines GENERATIONS; real tiles are the reference side —
R42's reading). Twenty seeded 16v16 half-splits of the 32 committed
stage-D real test tiles, judge_T at n_boot=400, in the native and the
declared-resolution (0.5-px-smoothed, both halves) domains. Every split's
seed and T recorded. The prereg's quoted values (native 1.43/0.47/2.65;
smoothed 1.48/0.55/2.87) must match this artifact EXACTLY (A4 rule) for
the one-shot to be pre-cleared.

Writes results_p2/stage3_mf_null.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts_p2"))

from minkowski_judge import judge_T

real = [np.asarray(f, np.float64) for f in
        np.load(os.path.join(RES, "arms_stageD.npz"),
                allow_pickle=True)["real"]]
out = {"convention": "20 seeded 16v16 half-splits of the 32 stage-D real "
                     "tiles; split rng default_rng(100+s).permutation(32); "
                     "judge_T(n_boot=400, seed=500+s); smoothed domain = "
                     "gaussian_filter(0.5) on both halves", "domains": {}}
for tag, stack in (("native", real),
                   ("smoothed", [gaussian_filter(f, 0.5) for f in real])):
    Ts = []
    for s in range(20):
        rng = np.random.default_rng(100 + s)
        idx = rng.permutation(32)
        a = [stack[i] for i in idx[:16]]
        b = [stack[i] for i in idx[16:]]
        T, _ = judge_T(a, b, n_boot=400, seed=500 + s)
        Ts.append({"split_seed": 100 + s, "judge_seed": 500 + s,
                   "T": float(T)})
    vals = np.array([t["T"] for t in Ts])
    out["domains"][tag] = {
        "splits": Ts,
        "mean": round(float(vals.mean()), 2),
        "sd": round(float(vals.std(ddof=1)), 2),
        "max": round(float(vals.max()), 2),
        "mean_raw": float(vals.mean()), "sd_raw": float(vals.std(ddof=1)),
        "max_raw": float(vals.max())}
    print(f"{tag}: null T mean {vals.mean():.2f} sd {vals.std(ddof=1):.2f} "
          f"max {vals.max():.2f}")
with open(os.path.join(RES, "stage3_mf_null.json"), "w") as f:
    json.dump(out, f, indent=1)
print("wrote stage3_mf_null.json")
