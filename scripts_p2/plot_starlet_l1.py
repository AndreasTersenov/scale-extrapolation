"""Starlet-l1 readout figure (SPEC-starlet-l1). Parses committed jsons only
(R12): starlet_l1_{sandbox,gowerstreet,edge,taxonomy}.json.

4 rows x 4 cols: rows = legs (sandbox truth-referenced / gowerstreet trained
octaves / Stage-D edge / generator taxonomy), cols = starlet detail scales
2/4/8/16 px (octaves 1-4). Curves: mean per-map binned l1 (sum |SNR| per bin)
with +-1 bootstrap-SE bands, log-y. Scored scales carry the arm-A rel. error
vs its 1sigma-w-10%-floor bar; 2px is descriptive everywhere.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")

LEGS = [
    ("sandbox", "SANDBOX truth-referenced\n(32 truth fields vs 32 e2e samples)"),
    ("gowerstreet", "GOWERSTREET trained octaves\n(32 real test fields)"),
    ("edge", "STAGE-D EDGE (octave 2 held out)\ndeployment-protocol maps"),
    ("taxonomy", "GENERATOR TAXONOMY (descriptive)\n64-field real reference"),
]
SCALES = ["2px", "4px", "8px", "16px"]

data = {leg: json.load(open(os.path.join(RES, f"starlet_l1_{leg}.json")))
        for leg, _ in LEGS}

fig, axes = plt.subplots(4, 4, figsize=(16.5, 13.5), sharex="col")

ARM_STYLE = {"gen_A": ("tab:purple", "arm A"), "gen_B": ("tab:red", "arm B")}
TAX_STYLE = {"4a_nll": ("tab:orange", "-", "4a NLL-head"),
             "4a_mu_only": ("tab:gray", "--", "4a mu-only skeleton"),
             "c1": ("tab:blue", "-", "C1 (Gaussian base)"),
             "c1t": ("tab:purple", "-", "C1-t")}

for row, (leg, rowtitle) in enumerate(LEGS):
    d = data[leg]
    for col, lab in enumerate(SCALES):
        ax = axes[row][col]
        bins = np.array(d["bins"][col])
        real = d["sets"]["real"][lab]
        m, se = np.array(real["curve_mean"]), np.array(real["curve_se"])
        ax.fill_between(bins, np.maximum(m - se, 1e-3), m + se,
                        color="k", alpha=0.18)
        ax.plot(bins, m, "k-", lw=1.6, label="real")
        styles = TAX_STYLE if leg == "taxonomy" else ARM_STYLE
        for key, st in styles.items():
            if key not in d["sets"]:
                continue
            g = d["sets"][key][lab]
            gm = np.array(g["curve_mean"])
            if leg == "taxonomy":
                color, ls, name = st
                ax.plot(bins, gm, ls, color=color, lw=1.3, label=name)
            else:
                color, name = st
                ax.plot(bins, gm, "-", color=color, lw=1.3, label=name)
        ax.set_yscale("log")
        octv = real["octave"]
        if leg == "taxonomy":
            note = "descriptive"
        else:
            chk = d["checks"]["gen_A"][lab]
            tag = "scored" if chk["scored"] else "descriptive"
            note = (f"{tag} · A {100*chk['rel']:+.1f}% "
                    f"(bar {100*chk['bar_rel']:.0f}%) "
                    f"{'PASS' if chk['pass'] else 'FAIL'}")
        head = " · THE EDGE" if (leg == "edge" and lab == "4px") else ""
        ax.set_title(f"{lab} (octave {octv}){head}\n{note}", fontsize=9)
        if col == 0:
            ax.set_ylabel(rowtitle + "\n\nbinned $\\ell_1$ (sum |SNR|)",
                          fontsize=8.5)
        if row == 3:
            ax.set_xlabel("SNR bin center", fontsize=9)
        ax.grid(alpha=0.2, which="both")
        if row == 0 and col == 0:
            ax.legend(fontsize=7.5, loc="lower center")
        if row == 3 and col == 0:
            ax.legend(fontsize=7, loc="lower center")

fig.suptitle(
    "Held-out starlet $\\ell_1$-norm (wl_stats_torch; basis and statistic never used in any design/selection loop) — "
    "arm-A totals pass the 1$\\sigma$-w-10%-floor at ALL scored scales on all three legs; the position-blind statistic "
    "does NOT flag the known peak-placement excess (the tier question fires). Convention: fixed global $\\sigma$ = real-ensemble std, "
    "plateau noise normalization, shared per-scale bins.", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.955])
out = os.path.join(RES, "starlet_l1.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
