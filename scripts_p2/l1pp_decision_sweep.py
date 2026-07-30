"""R41 decision package probe 2 (env.sh; prereg 2026-07-30-decision-package):
the resolution-scoped peak audit.

Five config x leg combos, Gaussian sigma_s in {0, 0.25, 0.5, 0.75, 1.0} px
(both stacks smoothed before counting), nu in {2.5, 3.0}, 5000-boot CIs
(frozen bootstrap_excess), per-parent panel at every sigma. Mechanical
reads per the prereg: zero crossing by linear interpolation; declared-
resolution pass at sigma = both nu ci95 include 0 AND panel not sign-
consistent at either nu; the candidate declared resolution = smallest grid
sigma where ALL FIVE combos pass. Writes results_p2/l1pp_decision_sweep.json
+ .png (eye-interpretable curves).
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts_p2"))

from audit_peak_ci import bootstrap_excess, per_parent_excess

SIGMAS = (0.0, 0.25, 0.5, 0.75, 1.0)
NUS = ("2.5", "3.0")
BLOCKS = [(0, 10), (10, 21), (21, 32)]


def stack(*arrs):
    return [np.asarray(f, np.float64) for a in arrs for f in a]


R1 = np.load(os.path.join(RES, "stage1_p3_replicates.npz"))
LPP = np.load(os.path.join(RES, "l1pp_main_gen.npz"))
gow_real = stack(np.load(os.path.join(RES, "arms_c1t_gowerstreet.npz"),
                         allow_pickle=True)["real"])
sd = np.load(os.path.join(RES, "stage0_p3_gen.npz"), allow_pickle=True)
sd_real = stack(sd["real"])
COMBOS = {
    "trained_white": (stack(np.load(os.path.join(
        RES, "f2_test_gen.npz"))["F2_gowA_e2e"], R1["rep1"], R1["rep2"]),
        gow_real),
    "trained_l1pp": (stack(LPP["adj1"], LPP["adj2"], LPP["adj3"]), gow_real),
    "trained_oracle": (stack(LPP["oracle_oct1meas"]), gow_real),
    "edge_white": (stack(sd["gen_A"]), sd_real),
    "edge_deconv": (stack(np.load(os.path.join(
        RES, "l1pp_decision_edge_gen.npz"))["gen_edge_deconv"]), sd_real),
}

out = {"sigmas": list(SIGMAS), "combos": {}}
for ci, (name, (gen, real)) in enumerate(COMBOS.items()):
    rows = {}
    for si, sig in enumerate(SIGMAS):
        g = gen if sig == 0 else [gaussian_filter(f, sig) for f in gen]
        r = real if sig == 0 else [gaussian_filter(f, sig) for f in real]
        entry = {}
        for ni, nu in enumerate(NUS):
            ex = bootstrap_excess(g, r, float(nu),
                                  seed=3000 + 100 * ci + 10 * si + ni)
            entry[nu] = {k: ex[k] for k in ("excess", "se", "ci95")}
        # per-parent panel: parent blocks index REAL tiles; pooled gen
        # stacks repeat the 32-tile conditioning per stream. First emission
        # pixel-AVERAGED the streams' maps before counting — averaging
        # independent realizations suppresses fine-scale structure and
        # produced impossible all-negative panels next to positive pooled
        # excesses (caught by its own output; disclosed in the readout).
        # Corrected: per-stream panels, averaged at the EXCESS level.
        n_rep = len(g) // 32
        pps = [per_parent_excess(g[s * 32:(s + 1) * 32], r, BLOCKS)
               for s in range(n_rep)]
        pp = {nu: [float(np.mean([p[nu][b] for p in pps]))
                   for b in range(len(BLOCKS))] for nu in pps[0]}
        entry["per_parent"] = {nu: pp[nu] for nu in NUS}
        entry["pass"] = bool(all(
            entry[nu]["ci95"][0] <= 0 <= entry[nu]["ci95"][1] for nu in NUS)
            and not any(all(v > 0 for v in pp[nu]) or
                        all(v < 0 for v in pp[nu]) for nu in NUS))
        rows[str(sig)] = entry
    # zero crossings
    zc = {}
    for nu in NUS:
        xs, ys = list(SIGMAS), [rows[str(s)][nu]["excess"] for s in SIGMAS]
        z = None
        for a, b, ya, yb in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
            if ya == 0 or ya * yb < 0:
                z = a + (b - a) * (0 - ya) / (yb - ya)
                break
        zc[nu] = z
    out["combos"][name] = {"rows": rows, "zero_crossing": zc}
    print(f"{name}: " + "  ".join(
        f"s={s}: {rows[str(s)]['2.5']['excess']:+.1%}/"
        f"{rows[str(s)]['3.0']['excess']:+.1%}"
        f"{'*' if rows[str(s)]['pass'] else ''}" for s in SIGMAS)
        + f"  zc={ {nu: (round(v, 2) if v is not None else None) for nu, v in zc.items()} }")

declared = None
for s in SIGMAS[1:]:
    if all(out["combos"][n]["rows"][str(s)]["pass"] for n in COMBOS):
        declared = s
        break
out["declared_resolution_candidate"] = declared
print(f"declared-resolution candidate: {declared}")
with open(os.path.join(RES, "l1pp_decision_sweep.json"), "w") as f:
    json.dump(out, f, indent=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, nu in zip(axes, NUS):
    for name in COMBOS:
        ys = [100 * out["combos"][name]["rows"][str(s)][nu]["excess"]
              for s in SIGMAS]
        es = [100 * out["combos"][name]["rows"][str(s)][nu]["se"]
              for s in SIGMAS]
        ax.errorbar(SIGMAS, ys, yerr=es, marker="o", capsize=3, label=name)
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_xlabel("Gaussian smoothing σ_s [px]")
    ax.set_title(f"peak-count excess vs resolution, ν={nu}")
axes[0].set_ylabel("excess [%]")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(RES, "l1pp_decision_sweep.png"), dpi=150)
print("wrote l1pp_decision_sweep.json + .png")
