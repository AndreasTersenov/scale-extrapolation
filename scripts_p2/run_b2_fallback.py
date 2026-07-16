"""B2 fallback (submit ONLY if job 16491648 times out before writing stageB2_crops):
bounded-IO variant — 12 parents from the first 3 shards (physical-diversity caveat
documented in the readout if used). Same estimator as run_stageB.b2_crops.
"""
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from scaledrift.data import iter_parent_maps
from scripts_p2.run_stageB import _crop_summary, OUT_CROPS


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


root = "/project/rrg-lplevass/shared/wl_chall_data/gowerstreet-train"
parents = list(iter_parent_maps(root, 12, seed=0, max_shards=3, per_shard=4))
log(f"B2 fallback: {len(parents)} parents (first 3 shards only)")
out = {"fallback": "12 parents / 3 shards (bounded IO); diversity caveat applies"}
for stride in (128, 64, 32):
    rows = []
    for pi, pmap in enumerate(parents):
        H, W = pmap.shape
        for y in range(0, H - 128 + 1, stride):
            for x in range(0, W - 128 + 1, stride):
                tile = pmap[y:y + 128, x:x + 128]
                if not np.isfinite(tile).all():
                    continue
                vs, ds = _crop_summary(tile)
                rows.append((pi, y, x, vs, ds))
    rows = np.array(rows)
    n = len(rows)
    neffs = {}
    for si, sname in ((3, "var_slope"), (4, "detail_std")):
        s = rows[:, si]
        s = (s - s.mean()) / s.std()
        rho_sum = float(n)
        pairs = {}
        for p in np.unique(rows[:, 0]):
            sel = rows[:, 0] == p
            sub = rows[sel]
            v = s[sel]
            for a in range(len(sub)):
                for b in range(a + 1, len(sub)):
                    d = (abs(sub[a, 1] - sub[b, 1]), abs(sub[a, 2] - sub[b, 2]))
                    pairs.setdefault(d, []).append(v[a] * v[b])
        for d, vals in pairs.items():
            rho = max(0.0, float(np.mean(vals)))
            rho_sum += 2 * rho * len(vals)
        neffs[sname] = n * n / rho_sum
    out[str(stride)] = {"n_crops": int(n),
                        "n_eff_var_slope": round(neffs["var_slope"], 1),
                        "n_eff_detail_std": round(neffs["detail_std"], 1),
                        "n_eff_min": round(min(neffs.values()), 1)}
    log(f"stride {stride}: n={n} n_eff_min={out[str(stride)]['n_eff_min']}")
with open(OUT_CROPS, "w") as f:
    json.dump(out, f, indent=1)
log(f"wrote {OUT_CROPS}")
