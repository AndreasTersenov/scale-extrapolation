#!/usr/bin/env python
"""Measurement bridge: score generated fields against real with the scaledrift instrument.

Runs in the scale-extrap env (scaledrift needs pywt). Loads an npz with `gen_A`, `gen_B`,
`real` field stacks (from `run_pnull.py`) and, per octave, measures the running-coupling
scalars (var_slope, kurtosis) and the detail amplitude (power proxy, P4) with scaledrift.
Reports each arm vs real; the extrapolated octave (the finest, untrained) is the P5/P-null
discriminator. Emits results/<name>_score.json and a verdict line.

Usage:  python scripts/measure_generated.py --npz results/pnull_generated.npz
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from scaledrift import coupling_scalars, octave_wc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def couplings(fields, octaves):
    """Per-octave (var_slope, kurtosis, detail_std) pooled over a field stack."""
    out = {}
    for j in octaves:
        ws, cs = [], []
        for f in fields:
            w, c = octave_wc(np.asarray(f, float), j)
            ws.append(w); cs.append(c)
        w = np.concatenate(ws); c = np.concatenate(cs)
        cs_ = coupling_scalars(w, c, n_bins=10)
        out[j] = {"var_slope": cs_["var_slope"], "kurtosis": cs_["kurtosis"],
                  "detail_std": float(w.std())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(REPO, "results", "pnull_generated.npz"))
    ap.add_argument("--octaves", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--extrapolated", type=int, default=1)
    ap.add_argument("--tol", type=float, default=0.10, help="max |Δvar_slope| for null")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    real = couplings(d["real"], args.octaves)
    scores = {"config_hash": str(d["config_hash"]) if "config_hash" in d else None,
              "real": real, "arms": {}}
    print(f"octaves {args.octaves}  extrapolated octave = {args.extrapolated}\n")
    print(f"{'octave':>6} {'metric':>10} {'real':>9} {'armA':>9} {'armB':>9}")
    verdict = {}
    for arm in ("A", "B"):
        g = couplings(d[f"gen_{arm}"], args.octaves)
        scores["arms"][arm] = g
        # null discriminator: var_slope match at the extrapolated octave
        dv = abs(g[args.extrapolated]["var_slope"] - real[args.extrapolated]["var_slope"])
        verdict[arm] = dv
    for j in args.octaves:
        for m in ("var_slope", "kurtosis", "detail_std"):
            a = scores["arms"]["A"][j][m]; b = scores["arms"]["B"][j][m]
            print(f"{j:>6} {m:>10} {real[j][m]:>9.3f} {a:>9.3f} {b:>9.3f}")
    print()
    for arm in ("A", "B"):
        ok = verdict[arm] < args.tol
        print(f"P-null arm {arm}: |Δvar_slope| at octave {args.extrapolated} = "
              f"{verdict[arm]:.3f} -> {'consistent' if ok else 'DEVIATES'} (tol {args.tol})")
    both = all(verdict[a] < args.tol for a in ("A", "B"))
    print(f"\nP-NULL: {'PASS (both arms extrapolate GRF)' if both else 'CHECK — a gen arm deviates from real GRF'}")

    out = args.npz.replace(".npz", "_score.json")
    with open(out, "w") as f:
        json.dump(scores, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
