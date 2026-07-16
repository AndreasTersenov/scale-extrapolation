"""C1 maps panel: real held-out field vs both arms' end-to-end generations, sandbox
and gowerstreet legs. Index 0 (first held-out — no cherry-picking), one shared color
scale per row. Generated panels are conditional SAMPLES from the same octave-4
coarse — pixel agreement at fine scales is not expected."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

rows = []
for leg in ("sandbox", "gowerstreet"):
    path = os.path.join(REPO, "results_p2", f"arms_c1_{leg}.npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        rows.append((leg, d["real"][0], d["gen_A"][0], d["gen_B"][0]))

fig, axes = plt.subplots(len(rows), 3, figsize=(10.5, 3.6 * len(rows)))
axes = np.atleast_2d(axes)
for r, (leg, real, ga, gb) in enumerate(rows):
    vmin, vmax = np.percentile(np.stack([real, ga, gb]), [1, 99])
    for c, (m, t) in enumerate(((real, "REAL held-out (index 0)"),
                                (ga, "arm A sample (same oct-4 coarse)"),
                                (gb, "arm B sample (same oct-4 coarse)"))):
        ax = axes[r, c]
        ax.imshow(m, cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(f"{leg}: {t}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("C1 (plain CFM + augmentation): conditional samples vs truth — "
             "shared color scale per row (1–99 pct)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(REPO, "results_p2", "c1_maps.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
