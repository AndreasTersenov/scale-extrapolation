"""Clean map galleries (no annotations — by-eye inspection; descriptive
re-render of committed maps). Native resolution, shared per-row color scale
set by the real tile. Tiles chosen for variety: the real stack's min /
median / max ν≥2.5-count tiles.

Writes results_p2/gallery_clean_trained.png + gallery_clean_edge.png.
"""
from __future__ import annotations

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

from audit_peak_ci import peaks_count


def pick_tiles(real):
    c = np.array([peaks_count(np.asarray(f, np.float64), 2.5) for f in real])
    return [int(np.argmin(c)), int(np.argsort(c)[len(c) // 2]),
            int(np.argmax(c))], c


def gallery(stacks, tiles, path, title):
    ncol = len(stacks)
    fig, axes = plt.subplots(len(tiles), ncol,
                             figsize=(3.1 * ncol, 3.25 * len(tiles)))
    for row, t in enumerate(tiles):
        ref = np.asarray(list(stacks.values())[0][t], np.float64)
        ref = (ref - ref.mean()) / ref.std()
        vmin, vmax = np.percentile(ref, [1, 99.5])
        for col, (label, st) in enumerate(stacks.items()):
            f = np.asarray(st[t], np.float64)
            fs = (f - f.mean()) / f.std()
            ax = axes[row, col]
            ax.imshow(fs, cmap="magma", vmin=vmin, vmax=vmax,
                      origin="lower", interpolation="nearest")
            if row == 0:
                ax.set_title(label, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"tile {t}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


# trained scales
greal = np.load(os.path.join(RES, "arms_c1t_gowerstreet.npz"),
                allow_pickle=True)["real"]
tiles, _ = pick_tiles(greal)
gallery({"REAL": greal,
         "final config (shipped seed)": np.load(
             os.path.join(RES, "l1pp_main_gen.npz"))["adj1"],
         "white base (pre-calibration)": np.load(
             os.path.join(RES, "f2_test_gen.npz"))["F2_gowA_e2e"]},
        tiles, os.path.join(RES, "gallery_clean_trained.png"),
        "Trained scales — native resolution, no annotations "
        "(columns share each tile's coarse conditioning)")

# edge
sreal = np.load(os.path.join(RES, "arms_stageD.npz"),
                allow_pickle=True)["real"]
tiles, _ = pick_tiles(sreal)
B = np.load(os.path.join(RES, "stage3_blind_final.npz"))
gallery({"REAL": sreal,
         "blind fresh seed (one-shot)": B["final1"],
         "shipped lineage (edge)": np.load(
             os.path.join(RES, "l1pp_decision_edge_gen.npz"))
         ["gen_edge_deconv"]},
        tiles, os.path.join(RES, "gallery_clean_edge.png"),
        "Held-out octave (the extrapolation edge) — native resolution, "
        "no annotations")
