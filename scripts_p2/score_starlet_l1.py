"""Score the four starlet-l1 legs (SPEC-starlet-l1) from frozen npz stacks.

Conventions per log/2026-07-18-prereg-starlet-l1.md. Runs under
~/wl-challenge-env (torch), CPU. Writes results_p2/starlet_l1_{sandbox,
gowerstreet,edge,taxonomy}.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts_p2.starlet_l1_lib import (  # noqa: E402
    add_noise,
    combined_ranges,
    curve_boot,
    l1_curves,
    tail_share_boot,
    totals_boot,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")

SCALES = ["2px", "4px", "8px", "16px"]  # detail scale s=0..3 <-> octave 1..4
OCTAVE = {"2px": 1, "4px": 2, "8px": 3, "16px": 4}
SCORED = ["4px", "8px", "16px"]  # octaves 2-4; 2px descriptive
NOISE_SEED = 20260718


def measure_sets(sets, sigma, leg_idx):
    """sets: ordered {name: (B,H,W)}. Shared ranges; per-set curves/totals."""
    ranges = combined_ranges(list(sets.values()), sigma)
    out = {"sigma": sigma, "ranges": ranges, "bins": None, "sets": {}}
    for k, (name, maps) in enumerate(sets.items()):
        bins, l1 = l1_curves(maps, sigma, ranges)
        if out["bins"] is None:
            out["bins"] = [b.tolist() for b in bins]
        entry = {}
        for s, lab in enumerate(SCALES):
            seed = 1000 * leg_idx + 10 * k + s
            t, t_se = totals_boot(l1[s], seed=seed)
            mean, se = curve_boot(l1[s], seed=seed)
            ts, ts_se = tail_share_boot(l1[s], bins[s], seed=seed)
            entry[lab] = {
                "octave": OCTAVE[lab],
                "total": t, "total_se": t_se,
                "tail_share_ge3": ts, "tail_share_se": ts_se,
                "curve_mean": mean.tolist(), "curve_se": se.tolist(),
            }
        out["sets"][name] = entry
    return out


def adjudicate(meas, real_key="real"):
    """Pass rule per prereg: |T_gen-T_real| <= max(hypot(SEs), 0.10*T_real)."""
    checks = {}
    real = meas["sets"][real_key]
    for arm in ("gen_A", "gen_B"):
        if arm not in meas["sets"]:
            continue
        checks[arm] = {}
        for lab in SCALES:
            tr, ser = real[lab]["total"], real[lab]["total_se"]
            tg, seg = meas["sets"][arm][lab]["total"], meas["sets"][arm][lab]["total_se"]
            bar = max(float(np.hypot(ser, seg)), 0.10 * abs(tr))
            checks[arm][lab] = {
                "octave": OCTAVE[lab],
                "scored": lab in SCORED,
                "T_real": tr, "T_gen": tg,
                "rel": (tg - tr) / tr,
                "bar_rel": bar / abs(tr),
                "pass": bool(abs(tg - tr) <= bar),
            }
    return checks


def secondary(sets, sigma_real, leg_idx):
    """Survey-like noise (descriptive): sigma_n = 2*sigma_real, fixed seeds."""
    sigma_n = 2.0 * sigma_real
    noisy = {name: add_noise(maps, sigma_n, NOISE_SEED + i)
             for i, (name, maps) in enumerate(sets.items())}
    ranges = combined_ranges(list(noisy.values()), sigma_n)
    out = {"sigma_n": sigma_n, "totals": {}}
    for k, (name, maps) in enumerate(noisy.items()):
        _, l1 = l1_curves(maps, sigma_n, ranges)
        out["totals"][name] = {}
        for s, lab in enumerate(SCALES):
            t, t_se = totals_boot(l1[s], seed=5000 + 1000 * leg_idx + 10 * k + s)
            out["totals"][name][lab] = {"total": t, "total_se": t_se}
    for arm in ("gen_A", "gen_B"):
        for lab in SCALES:
            tr = out["totals"]["real"][lab]["total"]
            tg = out["totals"][arm][lab]["total"]
            out["totals"][arm][lab]["rel"] = (tg - tr) / tr
    return out


def run_leg(name, leg_idx, npz_path, out_json, with_secondary):
    d = np.load(npz_path, allow_pickle=True)
    sets = {"real": np.asarray(d["real"], dtype=np.float64),
            "gen_A": np.asarray(d["gen_A"], dtype=np.float64),
            "gen_B": np.asarray(d["gen_B"], dtype=np.float64)}
    sigma = float(sets["real"].std())
    meas = measure_sets(sets, sigma, leg_idx)
    meas["leg"] = name
    meas["source_npz"] = os.path.relpath(npz_path, REPO)
    meas["n_maps"] = {k: int(v.shape[0]) for k, v in sets.items()}
    meas["checks"] = adjudicate(meas)
    for arm in meas["checks"]:
        meas["checks"][arm]["all_scored_pass"] = bool(all(
            meas["checks"][arm][lab]["pass"] for lab in SCORED))
    if with_secondary:
        meas["secondary_survey_noise"] = secondary(sets, sigma, leg_idx)
    with open(out_json, "w") as fh:
        json.dump(meas, fh, indent=1)
    print(f"[{name}] sigma={sigma:.6g}  " + "  ".join(
        f"{arm}:{'PASS' if meas['checks'][arm]['all_scored_pass'] else 'FAIL'}"
        for arm in ("gen_A", "gen_B")))
    return meas


def run_taxonomy():
    c1 = np.load(os.path.join(RES, "arms_c1_gowerstreet.npz"), allow_pickle=True)
    aug = np.load(os.path.join(REPO, "results", "npz", "arms_aug.npz"), allow_pickle=True)
    forensic = np.load(os.path.join(RES, "forensic_nllnoise.npz"), allow_pickle=True)
    c1t = np.load(os.path.join(RES, "arms_c1t_gowerstreet.npz"), allow_pickle=True)
    sets = {
        "real": np.asarray(c1["real"], dtype=np.float64),
        "4a_nll": np.asarray(aug["gen_A"], dtype=np.float64),
        "4a_mu_only": np.asarray(forensic["gen_A"], dtype=np.float64),
        "c1": np.asarray(c1["gen_A"], dtype=np.float64),
        "c1t": np.asarray(c1t["gen_A"], dtype=np.float64),
    }
    sigma = float(sets["real"].std())
    meas = measure_sets(sets, sigma, leg_idx=4)
    meas["leg"] = "taxonomy"
    meas["note"] = ("descriptive, arm A of each generation; real = 64-field "
                    "held-out stack; c1t conditions on its test-32 subset")
    meas["n_maps"] = {k: int(v.shape[0]) for k, v in sets.items()}
    for gen in ("4a_nll", "4a_mu_only", "c1", "c1t"):
        for lab in SCALES:
            tr = meas["sets"]["real"][lab]["total"]
            tg = meas["sets"][gen][lab]["total"]
            meas["sets"][gen][lab]["rel_vs_real"] = (tg - tr) / tr
    out_json = os.path.join(RES, "starlet_l1_taxonomy.json")
    with open(out_json, "w") as fh:
        json.dump(meas, fh, indent=1)
    print("[taxonomy] written")


def main():
    run_leg("sandbox", 1, os.path.join(RES, "arms_c1t_sandbox.npz"),
            os.path.join(RES, "starlet_l1_sandbox.json"), with_secondary=False)
    run_leg("gowerstreet", 2, os.path.join(RES, "arms_c1t_gowerstreet.npz"),
            os.path.join(RES, "starlet_l1_gowerstreet.json"), with_secondary=True)
    run_leg("edge", 3, os.path.join(RES, "arms_stageD.npz"),
            os.path.join(RES, "starlet_l1_edge.json"), with_secondary=True)
    run_taxonomy()


if __name__ == "__main__":
    main()
