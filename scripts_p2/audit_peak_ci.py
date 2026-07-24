"""Takeover-audit descriptive analysis (2026-07-24): bootstrap error bars for the
peak-count excesses quoted in the campaign record, from COMMITTED artifacts only
(no new training, no new generation).

Motivation: the ~15% joint-structure residual (stage-D peak audit) is quoted
without error bars anywhere in the record, yet it motivates the placement
experiment (SPEC-placement-test.md) and the paper's frontier section. The
CLAUDE.md hard rule — a drift claim without error bars doesn't count — applies.

Legs (peaks_count is copied VERBATIM from score_stageD.py — same convention,
per-field standardization, strict 8-neighbour local maxima, nu in {1, 2.5, 3}):
  stage_d   : arms_stageD.npz gen vs its own real edge fields (n=32 each)
  sandbox32 : arms_c1t_sandbox.npz gen vs the 32 held-out truth tiles
  repl64    : c1t_repl64_gen.npz gen vs the 64 fresh tiles (seed 20260720,
              data_cache/tiles_sandbox_repl.npz — rebuild via make_c1t_repl_data.py)
Each leg also carries a truth-vs-truth split-half NULL (instrument noise floor).

Writes results_p2/audit_peak_ci.json. CPU, env.sh stack, deterministic seeds.
"""
from __future__ import annotations

import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")
NUS = (1.0, 2.5, 3.0)
N_BOOT = 5000


def peaks_count(field, nu):
    f = (field - field.mean()) / field.std()
    c = f[1:-1, 1:-1]
    is_max = np.ones(c.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            is_max &= c > f[1 + dy:f.shape[0] - 1 + dy, 1 + dx:f.shape[1] - 1 + dx]
    return int((is_max & (c > nu)).sum())


def bootstrap_excess(gen_fields, real_fields, nu, n_boot=N_BOOT, seed=0):
    """Excess = (mean gen count - mean real count) / mean real count, both sides
    resampled over fields."""
    g = np.array([peaks_count(np.asarray(f, np.float64), nu) for f in gen_fields], float)
    r = np.array([peaks_count(np.asarray(f, np.float64), nu) for f in real_fields], float)
    rng = np.random.default_rng(seed)
    ex = np.empty(n_boot)
    for b in range(n_boot):
        gb = g[rng.integers(0, len(g), len(g))].mean()
        rb = r[rng.integers(0, len(r), len(r))].mean()
        ex[b] = (gb - rb) / rb
    point = float((g.mean() - r.mean()) / r.mean())
    return {"excess": point,
            "se": float(ex.std()),
            "ci68": [float(v) for v in np.percentile(ex, [16, 84])],
            "ci95": [float(v) for v in np.percentile(ex, [2.5, 97.5])],
            "z": float(point / ex.std()),
            "gen_mean": float(g.mean()), "real_mean": float(r.mean()),
            "n_gen": len(g), "n_real": len(r)}


def leg(gen_by_arm, real, seed0):
    out = {}
    for i, (arm, gen) in enumerate(sorted(gen_by_arm.items())):
        out[arm] = {str(nu): bootstrap_excess(gen, real, nu, seed=seed0 + i)
                    for nu in NUS}
    # truth-vs-truth split-half null: first half as "gen", second as "real"
    h = len(real) // 2
    out["null_splithalf"] = {str(nu): bootstrap_excess(real[:h], real[h:], nu,
                                                       seed=seed0 + 90)
                             for nu in NUS}
    return out


def per_parent_excess(gen, real, blocks):
    """Stage-D real edge tiles come from 3 parent maps (tiles_pnull.npz is
    parent-ordered, 11 tiles/parent; test slice 298..329 = parents 27/28/29 as
    blocks [0:10], [10:21], [21:32]). Gen field i is conditioned on real tile
    i's coarse, so blocks pair. Field-bootstrap SEs understate reference
    uncertainty when N_parents is this small (the campaign's own N_eff ~ parents
    accounting); per-parent excesses show whether the signal is parent-robust."""
    res = {}
    for nu in NUS:
        g = np.array([peaks_count(np.asarray(f, np.float64), nu) for f in gen], float)
        r = np.array([peaks_count(np.asarray(f, np.float64), nu) for f in real], float)
        res[str(nu)] = [float((g[a:b].mean() - r[a:b].mean()) / r[a:b].mean())
                        for a, b in blocks]
    return res


def main():
    out = {"convention": "peaks_count verbatim from score_stageD.py; "
                         "excess bootstrap-resampled over fields on both sides",
           "n_boot": N_BOOT, "legs": {}}

    d = np.load(os.path.join(RES, "arms_stageD.npz"), allow_pickle=True)
    out["legs"]["stage_d"] = leg({"A": d["gen_A"], "B": d["gen_B"]}, d["real"], 0)
    blocks = [(0, 10), (10, 21), (21, 32)]
    out["legs"]["stage_d"]["per_parent"] = {
        arm: per_parent_excess(d[f"gen_{arm}"], d["real"], blocks)
        for arm in ("A", "B")}

    d = np.load(os.path.join(RES, "arms_c1t_sandbox.npz"), allow_pickle=True)
    out["legs"]["sandbox32"] = leg({"A": d["gen_A"], "B": d["gen_B"]}, d["real"], 100)

    g = np.load(os.path.join(RES, "c1t_repl64_gen.npz"), allow_pickle=True)
    real = np.load(os.path.join(REPO, "data_cache",
                                "tiles_sandbox_repl.npz"))["sandbox"]
    out["legs"]["repl64"] = leg({"A": g["gen_A"], "B": g["gen_B"]}, real, 200)

    path = os.path.join(RES, "audit_peak_ci.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    for name, l in out["legs"].items():
        print(f"== {name} ==")
        for arm in sorted(l):
            if arm == "per_parent":
                for a, pp in l[arm].items():
                    print(f"  per-parent {a}: " + "  ".join(
                        f"nu={nu}: [" + ", ".join(f"{v:+.0%}" for v in pp[str(nu)]) + "]"
                        for nu in NUS))
                continue
            row = "  ".join(
                f"nu={nu}: {l[arm][str(nu)]['excess']:+.1%}"
                f"±{l[arm][str(nu)]['se']:.1%} (z={l[arm][str(nu)]['z']:+.1f})"
                for nu in NUS)
            print(f"  {arm:>14}: {row}")
    print("wrote", path)


if __name__ == "__main__":
    main()
