#!/usr/bin/env python
"""NIGHT-3 CASC scorer (env.sh CPU; NO JAX — the envs cannot share a
process; prereg 2026-08-05-prereg-night3 §CASC, line exec P = 22).

REGISTERED line (mechanical): casc pooled 96 (night3_casc_gen.npz) vs the
COMMITTED adj baseline (l1pp_main_gen.npz adj1-3 pooled 96) — for each
fragmentation-profile row (oct1_texture.stack_fragmentation) and the
alignment stat: Delta = (mean_casc - mean_adj) / hypot(se_casc, se_adj);
line FIRES iff any |Delta| >= 3 (base phase structure measurably carried
to output texture).
Descriptive: same instrument rows vs real (arms_c1t_gowerstreet "real",
32); MF(dev) declared+native (minkowski_judge.judge_T, oct1fix_o_score
seeds/smoothing) for casc-vs-real AND adj-vs-real recomputed side by side;
coloring C of the casc stack (octave 1) WATCHED against l1pp_pt.json's
interval (adj numbers copied from that json, R12 numbers-by-copy).
Writes results_p2/night3_casc_verdict.json."""
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

from coloring_index import stack_coloring
from minkowski_judge import judge_T
from oct1_texture import FRAG_NUS, stack_alignment, stack_fragmentation

ROWS = [f"nu{nu}_{name}" for nu in FRAG_NUS
        for name in ("ncomp", "holes", "chi", "small_frac")]

casc = np.load(os.path.join(RES, "night3_casc_gen.npz"))
casc_pool = [np.asarray(f, np.float64) for k in ("casc1", "casc2", "casc3")
             for f in casc[k]]
adj = np.load(os.path.join(RES, "l1pp_main_gen.npz"))
adj_pool = [np.asarray(f, np.float64) for k in ("adj1", "adj2", "adj3")
            for f in adj[k]]
real = [np.asarray(f, np.float64) for f in
        np.load(os.path.join(RES, "arms_c1t_gowerstreet.npz"),
                allow_pickle=True)["real"]]
out = {"n": {"casc": len(casc_pool), "adj": len(adj_pool),
             "real": len(real)}}

# --- (a) REGISTERED line: casc vs adj, fragmentation rows + alignment ---
mc, sc = stack_fragmentation(casc_pool, seed=7300)
ma, sa = stack_fragmentation(adj_pool, seed=7301)
d_frag = (mc - ma) / np.hypot(sc, sa)
alc, salc = stack_alignment(casc_pool, seed=7302)
ala, sala = stack_alignment(adj_pool, seed=7303)
d_al = (alc - ala) / float(np.hypot(salc, sala))
deltas = {r: {"casc": [float(mc[i]), float(sc[i])],
              "adj": [float(ma[i]), float(sa[i])],
              "delta": float(d_frag[i])} for i, r in enumerate(ROWS)}
deltas["alignment"] = {"casc": [alc, salc], "adj": [ala, sala],
                       "delta": d_al}
line_fired = bool(max(np.abs(d_frag).max(), abs(d_al)) >= 3)
out["registered_line"] = {"rule": "any |Delta| >= 3, casc vs adj",
                          "deltas": deltas, "line_fired": line_fired,
                          "max_abs_delta": float(max(np.abs(d_frag).max(),
                                                     abs(d_al)))}

# --- descriptive: same instrument rows vs real ---
mr, sr = stack_fragmentation(real, seed=7304)
alr, salr = stack_alignment(real, seed=7305)
out["descriptive_vs_real"] = {
    "frag_z_casc": ((mc - mr) / np.hypot(sc, sr)).tolist(),
    "frag_z_adj": ((ma - mr) / np.hypot(sa, sr)).tolist(),
    "alignment_z_casc": (alc - alr) / float(np.hypot(salc, salr)),
    "alignment_z_adj": (ala - alr) / float(np.hypot(sala, salr)),
    "rows": ROWS}

# --- (b) MF(dev) declared + native, casc and adj vs real, same seeds ---
sm = lambda st: [gaussian_filter(f, 0.5) for f in st]  # noqa: E731
mf = {}
for tag, pool in (("casc", casc_pool), ("adj_recomputed", adj_pool)):
    T_dec, _ = judge_T(sm(pool), sm(real), seed=20260890)
    T_nat, _ = judge_T(pool, real, seed=20260891)
    mf[tag] = {"declared": float(T_dec), "native": float(T_nat)}
out["MF_dev"] = mf

# --- (c) coloring C of the casc stack, WATCHED vs l1pp_pt interval ---
PT = json.load(open(os.path.join(RES, "l1pp_pt.json")))
c_casc, se_c = stack_coloring(casc_pool, 1, seed=7000)
out["coloring"] = {"C_casc": c_casc, "SE": se_c,
                   "interval": PT["interval"],
                   "in_interval": bool(PT["interval"][0] <= c_casc
                                       <= PT["interval"][1]),
                   "C_adj_committed": PT["C_pooled"],
                   "SE_adj_committed": PT["SE_pooled"]}

with open(os.path.join(RES, "night3_casc_verdict.json"), "w") as f:
    json.dump(out, f, indent=1)

print("REGISTERED LINE:", "FIRED" if line_fired else "not fired",
      f"(max |Delta| = {out['registered_line']['max_abs_delta']:.2f})")
worst = sorted(ROWS, key=lambda r: -abs(deltas[r]["delta"]))[:3]
for r in worst + ["alignment"]:
    d = deltas[r]
    print(f"  {r:>16}: casc {d['casc'][0]:.3f}±{d['casc'][1]:.3f}  "
          f"adj {d['adj'][0]:.3f}±{d['adj'][1]:.3f}  "
          f"Delta {d['delta']:+.2f}")
print(f"MF(dev) casc: declared {mf['casc']['declared']:.2f} "
      f"native {mf['casc']['native']:.2f}  |  adj recomputed: declared "
      f"{mf['adj_recomputed']['declared']:.2f} native "
      f"{mf['adj_recomputed']['native']:.2f}")
print(f"C(casc) = {c_casc:.4f}±{se_c:.4f} vs interval "
      f"[{PT['interval'][0]:.4f}, {PT['interval'][1]:.4f}] -> "
      f"{'IN' if out['coloring']['in_interval'] else 'OUT'} "
      f"(adj committed {PT['C_pooled']:.4f})")
print("wrote", os.path.join(RES, "night3_casc_verdict.json"))
