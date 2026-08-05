"""NIGHT-3 CKPT-SWEEP scorer (env.sh; prereg 2026-08-05-night3 — grant-2
probe, descriptive, adjudicates NO branch). Per (dir, step): MF(dev)
declared vs the matching real leg, fragmentation-profile max |z|
(ncomp+holes rows), native peak excesses; joined with each dir's cage
selection-score curve; Spearman rank corr (cage score vs MF(dev)) = the
registered watched line. Caveat printed: sweep maps use fresh PRNG keys,
so MF readings carry seed jitter ~ the split-half null sd (0.55) around
any committed same-ckpt reading. Writes
results_p2/night3_ckpt_sweep_scores.json."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts_p2"))

from audit_peak_ci import bootstrap_excess
from minkowski_judge import judge_T
from oct1_texture import stack_fragmentation

STEPS = list(range(2500, 20001, 2500))
NCOMP_HOLES = [0, 1, 4, 5, 8, 9]  # ncomp+holes rows of the 12-vector
SEL = {"oracle": "oct1fix_oracle_selection.json",
       "prod": "c1t_selection_gowerstreet.json",
       "blind": "stage3_blind_selection.json"}
REAL = {"oracle": ("arms_c1t_gowerstreet.npz", "real"),
        "prod": ("arms_c1t_gowerstreet.npz", "real"),
        "blind": ("arms_stageD.npz", "real")}

G = np.load(os.path.join(RES, "night3_ckpt_sweep.npz"))
reals = {}
for tag, (fn, key) in REAL.items():
    reals[tag] = [np.asarray(f, np.float64) for f in
                  np.load(os.path.join(RES, fn), allow_pickle=True)[key]]
sm = lambda st: [gaussian_filter(np.asarray(f, np.float64), 0.5)  # noqa: E731
                 for f in st]
frag_real = {tag: stack_fragmentation(reals[tag], seed=20260930)
             for tag in reals}

out = {}
for di, tag in enumerate(("oracle", "prod", "blind")):
    sel = json.load(open(os.path.join(RES, SEL[tag])))["A"]
    curve = sel["curve_val"]
    rows = {}
    for si, step in enumerate(STEPS):
        gen = [np.asarray(f, np.float64) for f in G[f"{tag}_{step}"]]
        T_dec, _ = judge_T(sm(gen), sm(reals[tag]),
                           seed=20260940 + di * 20 + si)
        mf, sf = stack_fragmentation(gen, seed=20260970 + di * 20 + si)
        mr, sr_ = frag_real[tag]
        fz = (mf - mr) / np.hypot(sf, sr_)
        pk = {nu: bootstrap_excess(gen, reals[tag], float(nu),
                                   seed=20261000 + di * 40 + si * 2 + k)
              for k, nu in enumerate(("2.5", "3.0"))}
        rows[str(step)] = {
            "T_mf_dev_declared": float(T_dec),
            "frag_maxabs_z": float(np.max(np.abs(fz[NCOMP_HOLES]))),
            "peak_excess": {nu: pk[nu]["excess"] for nu in pk},
            "cage_score": curve.get(str(step), {}).get("score"),
        }
        print(f"{tag} s{step}: MF(dev)={T_dec:.2f} "
              f"frag={rows[str(step)]['frag_maxabs_z']:.1f} "
              f"pk2.5={pk['2.5']['excess']:+.1%} "
              f"cage={rows[str(step)]['cage_score']}")
    ts = [rows[str(s)]["T_mf_dev_declared"] for s in STEPS]
    cs = [rows[str(s)]["cage_score"] for s in STEPS]
    rho = spearmanr(cs, ts) if all(c is not None for c in cs) else None
    out[tag] = {
        "selected_step": sel["selected_step"],
        "rows": rows,
        "mf_spread": float(max(ts) - min(ts)),
        "mf_min": float(min(ts)), "mf_min_step": STEPS[int(np.argmin(ts))],
        "n_pass_category": int(sum(t < 3.25 for t in ts)),
        "spearman_cage_vs_mf": None if rho is None else
        {"rho": float(rho.statistic), "p": float(rho.pvalue)},
    }
    print(f"== {tag}: selected@{sel['selected_step']} "
          f"spread={out[tag]['mf_spread']:.2f} "
          f"min={out[tag]['mf_min']:.2f}@{out[tag]['mf_min_step']} "
          f"pass-cat={out[tag]['n_pass_category']}/8 "
          f"rho={out[tag]['spearman_cage_vs_mf']}")
with open(os.path.join(RES, "night3_ckpt_sweep_scores.json"), "w") as f:
    json.dump(out, f, indent=1)
print("NOTE: fresh-PRNG maps; per-reading seed jitter ~0.55 (null sd).")
