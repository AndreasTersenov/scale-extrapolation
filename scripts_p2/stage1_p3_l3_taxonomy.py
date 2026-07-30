"""L3 — the coloring-vs-count-bias taxonomy curve (R38 order 2; CPU,
DESCRIPTIVE, committed stacks only; does not touch the one-lever rule).

For every committed (gen stack, matched real stack) pair: the oct-1
whiteness z = (C_real - C_gen)/hypot(SEs) (tests-first coloring index) and
the peak-count excess at nu=2.5 (frozen bootstrap_excess). Question: does C
predict the sign/size of the count bias across the taxonomy? Excluded:
phase-1 aug/forensic stacks (different real-stack conventions), checkpoint/
validation curves (within-model sweeps, not final picks).

Writes results_p2/stage1_p3_l3_taxonomy.json + .png (eye-interpretable
scatter, per the figures convention).
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts_p2"))

from audit_peak_ci import bootstrap_excess
from coloring_index import stack_coloring


def L(path, key):
    return [np.asarray(f, np.float64) for f in
            np.load(os.path.join(RES, path), allow_pickle=True)[key]]


SB_REAL = [np.asarray(f, np.float64) for f in np.load(
    os.path.join(REPO, "data_cache", "tiles_sandbox.npz"))["sandbox"][-32:]]
SBR_REAL = [np.asarray(f, np.float64) for f in np.load(
    os.path.join(REPO, "data_cache", "tiles_sandbox_repl.npz"))["sandbox"]]
GOW_REAL = L("arms_c1t_gowerstreet.npz", "real")
SD_REAL = L("arms_stageD.npz", "real")

PAIRS = [
    # (label, field, gen, real)
    ("sb_c1_A(gauss)", "sandbox", L("arms_c1_sandbox.npz", "gen_A"), SB_REAL),
    ("sb_c1t_A", "sandbox", L("arms_c1t_sandbox.npz", "gen_A"), SB_REAL),
    ("sb_c1t_B", "sandbox", L("arms_c1t_sandbox.npz", "gen_B"), SB_REAL),
    ("sb_F2_A", "sandbox", L("f2_test_gen.npz", "F2_A_e2e"), SB_REAL),
    ("sb_F2_B", "sandbox", L("f2_test_gen.npz", "F2_B_e2e"), SB_REAL),
    ("sb_repl64_A", "sandbox", L("c1t_repl64_gen.npz", "gen_A"), SBR_REAL),
    ("sb_pcure_n1s1", "sandbox", L("arms_pcure_n1s1.npz", "gen_A"), SB_REAL),
    ("sb_pcure_n8", "sandbox", L("arms_pcure_n8.npz", "gen_A"), SB_REAL),
    ("sb_pcure_n32", "sandbox", L("arms_pcure_n32.npz", "gen_A"), SB_REAL),
    ("sb_pcure_cens", "sandbox", L("arms_pcure_cens.npz", "gen_A"), SB_REAL),
    ("gow_c1_A(gauss)", "gowerstreet", L("arms_c1_gowerstreet.npz", "gen_A"),
     GOW_REAL),
    ("gow_c1t_A", "gowerstreet", L("arms_c1t_gowerstreet.npz", "gen_A"),
     GOW_REAL),
    ("gow_F_A", "gowerstreet", L("fj_test_gen.npz", "F_gowA_e2e"), GOW_REAL),
    ("gow_F2_A", "gowerstreet", L("f2_test_gen.npz", "F2_gowA_e2e"), GOW_REAL),
    ("gow_rep1", "gowerstreet", L("stage1_p3_replicates.npz", "rep1"),
     GOW_REAL),
    ("gow_rep2", "gowerstreet", L("stage1_p3_replicates.npz", "rep2"),
     GOW_REAL),
    ("sd_A(edge)", "stage-D", L("arms_stageD.npz", "gen_A"), SD_REAL),
    ("sd_B(edge)", "stage-D", L("arms_stageD.npz", "gen_B"), SD_REAL),
    ("sd_F2_A(edge)", "stage-D", L("stage0_p3_gen.npz", "gen_A"), SD_REAL),
    ("sd_F2_B(edge)", "stage-D", L("stage0_p3_gen.npz", "gen_B"), SD_REAL),
]

rows = {}
for i, (label, field, gen, real) in enumerate(PAIRS):
    cr, sr = stack_coloring(real, 1, seed=200 + i)
    cg, sg = stack_coloring(gen, 1, seed=400 + i)
    zw = (cr - cg) / float(np.hypot(sr, sg))
    ex = bootstrap_excess(gen, real, 2.5, seed=600 + i)
    rows[label] = {"field": field, "C_real": cr, "C_gen": cg,
                   "z_whiteness": zw,
                   "excess_nu2.5": ex["excess"], "excess_se": ex["se"]}
    print(f"{label:>18} [{field:>11}] z_white={zw:+6.2f} "
          f"excess={ex['excess']:+7.2%}±{ex['se']:.2%}")

with open(os.path.join(RES, "stage1_p3_l3_taxonomy.json"), "w") as f:
    json.dump({"note": "L3 descriptive; excluded phase-1 aug/forensic + "
                       "checkpoint curves", "rows": rows}, f, indent=1)

COLORS = {"sandbox": "tab:blue", "gowerstreet": "tab:red",
          "stage-D": "tab:orange"}
fig, ax = plt.subplots(figsize=(9, 6.5))
for label, r in rows.items():
    ax.errorbar(r["z_whiteness"], 100 * r["excess_nu2.5"],
                yerr=100 * r["excess_se"], fmt="o",
                color=COLORS[r["field"]], capsize=3)
    ax.annotate(label, (r["z_whiteness"], 100 * r["excess_nu2.5"]),
                fontsize=7, xytext=(4, 4), textcoords="offset points")
ax.axhline(0, color="gray", lw=0.7)
ax.axvline(0, color="gray", lw=0.7)
ax.axvline(3, color="gray", lw=0.7, ls="--")
ax.set_xlabel("oct-1 whiteness z = (C_real − C_gen) / hypot(SE)  "
              "[>0: gen details whiter than real]")
ax.set_ylabel("peak-count excess at ν=2.5 [%]")
ax.set_title("L3: oct-1 detail whiteness vs peak-count bias — "
             "all committed final-pick stacks")
for fld, c in COLORS.items():
    ax.plot([], [], "o", color=c, label=fld)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(RES, "stage1_p3_l3_taxonomy.png"), dpi=150)
print("wrote stage1_p3_l3_taxonomy.json + .png")
