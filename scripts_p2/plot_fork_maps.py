"""DESCRIPTIVE figures for the scope-fork decision (committed maps only;
no new verdicts). Three panels:

  A l1pp_decision_maps_trained.png — one representative trained-leg tile
    (real / white-base / L1'' deconvolved / oracle share the SAME coarse
    conditioning), native resolution vs 0.5-px smoothed, peaks (nu>=2.5,
    frozen convention) marked, counts annotated. The declared-resolution
    story on actual maps.
  B l1pp_decision_maps_edge.png — the same layout at the stage-D edge
    (real / white / deconvolved edge stream).
  C l1pp_decision_mechanism.png — octave-1 detail planes (H channel) for
    real / white / deconvolved + the measured ring spectra with instrument-
    C values: the whiteness defect and its cure, on the coefficients
    themselves.

Tile choice (deterministic, stated): the tile whose REAL nu=2.5 peak count
is closest to the stack median.
"""
from __future__ import annotations

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

from audit_peak_ci import peaks_count
from colored_base import ring_spectrum, ring_table
from parity_localization import dwt_levels

NU = 2.5


def peak_xy(field, nu=NU):
    f = (field - field.mean()) / field.std()
    c = f[1:-1, 1:-1]
    is_max = np.ones(c.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            is_max &= c > f[1 + dy:f.shape[0] - 1 + dy,
                            1 + dx:f.shape[1] - 1 + dx]
        # (same strict-interior convention as the frozen peaks_count)
    ys, xs = np.nonzero(is_max & (c > nu))
    return xs + 1, ys + 1


def map_grid(stacks, tile, path, title):
    """stacks: ordered {label: (N,128,128)}; rows = native / 0.5px."""
    ref = np.asarray(list(stacks.values())[0][tile], np.float64)
    vmin, vmax = np.percentile((ref - ref.mean()) / ref.std(), [1, 99.5])
    ncol = len(stacks)
    fig, axes = plt.subplots(2, ncol, figsize=(3.2 * ncol, 7.4),
                             gridspec_kw={"hspace": 0.18})
    for col, (label, st) in enumerate(stacks.items()):
        f = np.asarray(st[tile], np.float64)
        for row, sig in enumerate((0.0, 0.5)):
            g = f if sig == 0 else gaussian_filter(f, sig)
            gs = (g - g.mean()) / g.std()
            ax = axes[row, col]
            ax.imshow(gs, cmap="magma", vmin=vmin, vmax=vmax,
                      origin="lower", interpolation="nearest")
            xs, ys = peak_xy(g)
            ax.plot(xs, ys, "o", ms=5, mfc="none", mec="cyan", mew=1.1)
            n = peaks_count(g, NU)
            ax.set_title(f"{label}\n{n} peaks (ν≥{NU})" if row == 0
                         else f"{n} peaks", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0, col].set_xlabel("")
    axes[0, 0].set_ylabel("native resolution", fontsize=11)
    axes[1, 0].set_ylabel("σ = 0.5 px smoothed", fontsize=11)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


# ---- trained leg -------------------------------------------------------------
real = np.load(os.path.join(RES, "arms_c1t_gowerstreet.npz"),
               allow_pickle=True)["real"]
white = np.load(os.path.join(RES, "f2_test_gen.npz"))["F2_gowA_e2e"]
LPP = np.load(os.path.join(RES, "l1pp_main_gen.npz"))
counts = [peaks_count(np.asarray(f, np.float64), NU) for f in real]
tile = int(np.argmin(np.abs(np.array(counts) - np.median(counts))))
print(f"trained tile {tile} (real count {counts[tile]}, "
      f"median {np.median(counts):.0f})")
map_grid({"REAL": real, "white base": white,
          "L1'' deconvolved": LPP["adj1"], "oracle": LPP["oracle_oct1meas"]},
         tile, os.path.join(RES, "l1pp_decision_maps_trained.png"),
         f"Trained leg, tile {tile} — same coarse conditioning across "
         "columns; peak excess lives at native resolution only")

# ---- edge leg ---------------------------------------------------------------
sd = np.load(os.path.join(RES, "stage0_p3_gen.npz"), allow_pickle=True)
edge_dc = np.load(os.path.join(RES, "l1pp_decision_edge_gen.npz"))
ec = [peaks_count(np.asarray(f, np.float64), NU) for f in sd["real"]]
etile = int(np.argmin(np.abs(np.array(ec) - np.median(ec))))
print(f"edge tile {etile} (real count {ec[etile]})")
map_grid({"REAL": sd["real"], "white base": sd["gen_A"],
          "deconvolved": edge_dc["gen_edge_deconv"]},
         etile, os.path.join(RES, "l1pp_decision_maps_edge.png"),
         f"Stage-D EDGE leg, tile {etile} — the extrapolated octave; "
         "same layout")

# ---- mechanism panel --------------------------------------------------------
def oct1_H(field):
    return np.asarray(dwt_levels(np.asarray(field, np.float64), 1)[1][0])


def spec_of(stack):
    planes = []
    for f in stack:
        for c in dwt_levels(np.asarray(f, np.float64), 1)[1]:
            planes.append(np.asarray(c, np.float64))
    planes = np.array(planes)
    return ring_spectrum(planes / planes.std())


rings = ring_table(64)
cnt = np.array([(rings == r).sum() for r in range(rings.max() + 1)])
pos = cnt > 0
CVAL = {"REAL": 0.7864, "white base": 0.7237, "L1'' deconvolved": 0.7414,
        "oracle": 0.7815}
fig = plt.figure(figsize=(14, 4.2))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.6])
stacks3 = {"REAL": real, "white base": white, "L1'' deconvolved": LPP["adj1"]}
for i, (label, st) in enumerate(stacks3.items()):
    ax = fig.add_subplot(gs[0, i])
    h = oct1_H(st[tile])
    h = (h - h.mean()) / h.std()
    ax.imshow(h, cmap="RdBu_r", vmin=-2.5, vmax=2.5, origin="lower",
              interpolation="nearest")
    ax.set_title(f"{label}\noct-1 detail (H), C={CVAL[label]:.3f}",
                 fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
ax = fig.add_subplot(gs[0, 3])
LP = np.load(os.path.join(RES, "l1p_main_gen.npz"))
for label, st, c in (("real", real, "k"),
                     ("white base", white, "tab:orange"),
                     ("L1' wrong-direction", [LP[k][i] for k in
                      ("adj1", "adj2", "adj3") for i in range(32)], "tab:red"),
                     ("L1'' deconvolved", [LPP[k][i] for k in
                      ("adj1", "adj2", "adj3") for i in range(32)],
                      "tab:blue")):
    s = spec_of(st)
    s = s / s[pos].mean()
    k = np.arange(len(s))
    ax.plot(k[1:33], s[1:33], color=c, label=label,
            lw=2 if label == "real" else 1.4)
ax.set_xlabel("|k| (ring)")
ax.set_ylabel("normalized ring power")
ax.set_title("oct-1 detail spectra — the defect and the cure", fontsize=11)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(RES, "l1pp_decision_mechanism.png"), dpi=150)
print("wrote l1pp_decision_mechanism.png")
