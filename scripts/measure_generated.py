#!/usr/bin/env python
"""Measurement bridge: score generated fields against real with the scaledrift instrument.

Runs in the scale-extrap env (scaledrift needs pywt). Loads an npz with `gen_A`, `gen_B`,
`real` field stacks and, per octave, measures the running-coupling scalars (var_slope,
kurtosis) and detail amplitude (P4), each with a bootstrap-over-fields SE. At the
extrapolated (finest, untrained) octave it reports, for each arm vs real: the var_slope
z-score and relative error (the P5 break test) and the arm-B repair fraction (P6). The same
scorer covers the GRF null (arm A ~ consistent) and the gowerstreet break/repair.

var_slope is the primary discriminator; kurtosis is reported but confounded by the
inter-tile-amplitude pooling artifact (see the P-null log).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from scaledrift import coupling_scalars, octave_wc


def _scalars(fields, j):
    ws, cs = [], []
    for f in fields:
        w, c = octave_wc(np.asarray(f, float), j)
        ws.append(w); cs.append(c)
    w = np.concatenate(ws); c = np.concatenate(cs)
    s = coupling_scalars(w, c, n_bins=10)
    return s["var_slope"], s["kurtosis"], float(w.std())


def couplings(fields, octaves, n_boot=200, seed=0):
    """Per-octave (var_slope, kurtosis, detail_std) with bootstrap-over-fields SE."""
    rng = np.random.default_rng(seed)
    n = len(fields)
    out = {}
    for j in octaves:
        vs, ku, ds = _scalars(fields, j)
        bs = np.array([_scalars([fields[i] for i in rng.integers(0, n, n)], j)
                       for _ in range(n_boot)])
        out[j] = {"var_slope": vs, "var_slope_se": float(np.nanstd(bs[:, 0], ddof=1)),
                  "kurtosis": ku, "kurtosis_se": float(np.nanstd(bs[:, 1], ddof=1)),
                  "detail_std": ds}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results",
        "arms_generated.npz"))
    ap.add_argument("--octaves", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--extrapolated", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    real = couplings(d["real"], args.octaves, args.n_boot)
    A = couplings(d["gen_A"], args.octaves, args.n_boot)
    B = couplings(d["gen_B"], args.octaves, args.n_boot)
    je = args.extrapolated

    print(f"octaves {args.octaves}  extrapolated (untrained) octave = {je}\n")
    print(f"{'oct':>3} {'metric':>10} {'real':>16} {'armA':>16} {'armB':>16}")
    for j in args.octaves:
        for m, se in (("var_slope", "var_slope_se"), ("kurtosis", "kurtosis_se")):
            print(f"{j:>3} {m:>10} {real[j][m]:>8.3f}±{real[j][se]:<7.3f} "
                  f"{A[j][m]:>8.3f}±{A[j][se]:<7.3f} {B[j][m]:>8.3f}±{B[j][se]:<7.3f}")
        print(f"{j:>3} {'detail_std':>10} {real[j]['detail_std']:>16.3f} "
              f"{A[j]['detail_std']:>16.3f} {B[j]['detail_std']:>16.3f}")

    def z_rel(arm):
        dv = arm[je]["var_slope"] - real[je]["var_slope"]
        se = np.hypot(arm[je]["var_slope_se"], real[je]["var_slope_se"])
        return abs(dv) / se if se > 0 else np.nan, abs(dv) / abs(real[je]["var_slope"])

    zA, relA = z_rel(A)
    zB, relB = z_rel(B)
    errA = abs(A[je]["var_slope"] - real[je]["var_slope"])
    errB = abs(B[je]["var_slope"] - real[je]["var_slope"])
    repair = 1.0 - errB / errA if errA > 0 else np.nan
    p4 = {a: abs(x[je]["detail_std"] - real[je]["detail_std"]) / real[je]["detail_std"]
          for a, x in (("A", A), ("B", B))}

    print(f"\n--- octave {je} (extrapolated) verdicts ---")
    print(f"P4  detail amplitude rel err: armA {p4['A']*100:.1f}%  armB {p4['B']*100:.1f}%  "
          f"-> {'PASS' if max(p4.values())<0.10 else 'CHECK'} (both within 10%)")
    print(f"P5  arm A var_slope: {A[je]['var_slope']:.3f} vs real {real[je]['var_slope']:.3f} "
          f"| z={zA:.1f} rel={relA*100:.0f}%  -> "
          f"{'BREAK (P5 holds)' if zA>3 and relA>0.10 else 'not significant'}")
    print(f"P6  arm B var_slope: {B[je]['var_slope']:.3f} | z={zB:.1f} rel={relB*100:.0f}%  "
          f"| repair = {repair*100:.0f}%  -> "
          f"{'REPAIR (P6 holds)' if repair>=0.70 else ('partial' if repair>0.30 else 'FAIL')}")

    out = args.npz.replace(".npz", "_score.json")
    json.dump({"real": real, "armA": A, "armB": B,
               "extrapolated_octave": je,
               "zA": zA, "relA": relA, "zB": zB, "relB": relB,
               "repair": repair, "p4": p4,
               "config_hash": str(d["config_hash"]) if "config_hash" in d else None},
              open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
