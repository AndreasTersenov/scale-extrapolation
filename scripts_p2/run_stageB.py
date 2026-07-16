"""Stage B runner: B1 predictability curves + shape test; B2 crops inventory.

Protocol pinned in log/2026-07-16-stageB-prereg.md (committed before this runs on
gowerstreet). Runs under the env.sh stack on a compute node (data dir needed for B2).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from depmeasure.predictability import predictability_curve
from depmeasure.shape import shape_test

SCRATCH = os.path.expanduser("~/links/scratch/scale-extrap-p2")
OUT_CURVES = os.path.join(REPO, "results_p2", "stageB1_curves.json")
OUT_SHAPE = os.path.join(REPO, "results_p2", "stageB1_shape.json")
OUT_CROPS = os.path.join(REPO, "results_p2", "stageB2_crops.json")

R_GRIDS = {1: (0, 1, 2, 3, 4, 6, 8, 12), 2: (0, 1, 2, 3, 4, 6, 8),
           3: (0, 1, 2, 3, 4), 4: (0, 1, 2)}
MAXPOS = {1: 768, 2: 1024, 3: 4096, 4: 4096}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rstar(curve, r_grid):
    rmax = r_grid[-1]
    thresh = curve[rmax]["ridge"] + 3 * curve[rmax]["ridge_se"]
    for r in r_grid:
        if curve[r]["ridge"] <= thresh:
            return r
    return rmax


def b1_curves():
    if os.path.exists(OUT_CURVES):
        log("B1 curves exist, skipping")
        return
    gow = np.load(os.path.join(REPO, "data_cache", "tiles_pnull.npz"))["gowerstreet"]
    sand = np.load(os.path.join(SCRATCH, "sandbox_parents_f32.npy"))
    out = {}
    for name, fields, periodic in (("gowerstreet", gow, False),
                                   ("sandbox", sand, True)):
        fl = [np.asarray(f, dtype=np.float64) for f in fields]
        out[name] = {}
        for j, grid in R_GRIDS.items():
            row = {}
            for target in ("w", "absw"):
                ests = ("ridge", "knn") if target == "w" else ("ridge",)
                t0 = time.time()
                curve = predictability_curve(
                    fl, j, grid, periodic=periodic, seed=0,
                    max_pos_per_field=MAXPOS[j], estimators=ests, target=target)
                row[target] = {str(r): curve[r] for r in grid}
                row[f"rstar_{target}"] = rstar(curve, grid)
                log(f"B1 {name} oct{j} {target}: rstar={row[f'rstar_{target}']} "
                    f"({time.time()-t0:.0f}s)")
            out[name][str(j)] = row
    with open(OUT_CURVES, "w") as f:
        json.dump(out, f, indent=1)
    log(f"wrote {OUT_CURVES}")


def b1_shape():
    if os.path.exists(OUT_SHAPE):
        log("B1 shape exists, skipping")
        return
    gow = np.load(os.path.join(REPO, "data_cache", "tiles_pnull.npz"))["gowerstreet"]
    sand = np.load(os.path.join(SCRATCH, "sandbox_parents_f32.npy"))
    out = {}
    for name, fields, periodic, octs in (("gowerstreet", gow, False, (2, 1)),
                                         ("sandbox", sand, True, (2,))):
        fl = [np.asarray(f, dtype=np.float64) for f in fields]
        out[name] = {}
        for j in octs:
            row = {}
            for target in ("absw", "w"):
                t0 = time.time()
                res = shape_test(fl, j, r_iso=4, aspect=4.0, periodic=periodic,
                                 max_pos_per_field=2048, seed=0, target=target)
                res["z_align"] = (res["delta_align"] / res["delta_align_se"]
                                  if res["delta_align_se"] > 0 else float("nan"))
                row[target] = res
                log(f"shape {name} oct{j} {target}: z={res['z_align']:.2f} "
                    f"({time.time()-t0:.0f}s)")
            out[name][str(j)] = row
    with open(OUT_SHAPE, "w") as f:
        json.dump(out, f, indent=1)
    log(f"wrote {OUT_SHAPE}")


def _crop_summary(tile):
    """(var_slope, detail_std) of a single 128^2 crop at octave 2."""
    from sandbox.haar import octave_wc_pooled
    from sandbox.truth_stats import estimand_scalars
    w, c = octave_wc_pooled(tile, 2)
    s = estimand_scalars(w, c)
    return s["var_slope"], s["detail_std"]


def b2_crops():
    if os.path.exists(OUT_CROPS):
        log("B2 exists, skipping")
        return
    from scaledrift.data import iter_parent_maps, tile_map
    root = "/project/rrg-lplevass/shared/wl_chall_data/gowerstreet-train"
    parents = list(iter_parent_maps(root, 30, seed=0))
    log(f"B2: {len(parents)} parent maps loaded")
    out = {}
    for stride in (128, 64, 32):
        rows = []          # (parent, y, x, var_slope, detail_std)
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
            # measured correlation vs offset, pooled over parents
            rho_sum = float(n)   # self-pairs
            pairs = {}
            for p in np.unique(rows[:, 0]):
                sel = rows[:, 0] == p
                sub = rows[sel]
                v = s[sel]
                for a in range(len(sub)):
                    for b in range(a + 1, len(sub)):
                        d = (abs(sub[a, 1] - sub[b, 1]), abs(sub[a, 2] - sub[b, 2]))
                        pairs.setdefault(d, []).append(v[a] * v[b])
            rho_by_offset = {d: float(np.mean(vals)) for d, vals in pairs.items()}
            for d, vals in pairs.items():
                rho = max(0.0, rho_by_offset[d])   # negative sampling noise -> 0
                rho_sum += 2 * rho * len(vals)
            neffs[sname] = n * n / rho_sum
        out[str(stride)] = {"n_crops": int(n),
                            "n_eff_var_slope": round(neffs["var_slope"], 1),
                            "n_eff_detail_std": round(neffs["detail_std"], 1),
                            "n_eff_min": round(min(neffs.values()), 1)}
        log(f"B2 stride {stride}: n={n} n_eff_min={out[str(stride)]['n_eff_min']}")
    with open(OUT_CROPS, "w") as f:
        json.dump(out, f, indent=1)
    log(f"wrote {OUT_CROPS}")


if __name__ == "__main__":
    b1_curves()
    b1_shape()
    b2_crops()
    log("stage B complete")
