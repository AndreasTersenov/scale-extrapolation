#!/usr/bin/env python
"""Print the RESULTS-relevant summary from results/measurement.json:
GRF null verdict, P9a (adjacent-octave drift significance & size), P9b (effective
dimensionality), running couplings, and cross-octave couplings.
"""
import argparse
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pct(x, y):
    return 100.0 * x / y if y else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(REPO, "results", "measurement.json"))
    args = ap.parse_args()
    b = json.load(open(args.json))
    cfg = b["config"]
    print(f"config: n_parents={cfg['n_parents']} tile={cfg['tile']} octaves={cfg['octaves']} "
          f"n_bins={cfg['n_bins']} n_boot={cfg['n_boot']} wavelet={cfg['wavelet']}\n")

    for r in b["fields"]:
        print(f"================ {r['key']}  ({r['role']}) ================")
        print(f" parents={r['n_parents']} tiles={r['n_tiles']}")
        print(" adjacent-octave excess drift  (excess ± se, z, excess/measured %):")
        for d in r["adjacent"]:
            print(f"   {d['j_fine']}->{d['j_coarse']}: excess={d['excess']:+.4f} "
                  f"se={d['excess_se']:.4f} z={d['z']:+.2f} "
                  f"meas={d['measured']:.4f} ({pct(d['excess'], d['measured']):.0f}%)")
        # drift vs separation from finest octave
        sep = ", ".join(f"sep{d['separation']}:{d['excess']:+.4f}(z={d['z']:+.1f})"
                        for d in r["separation"])
        print(f" drift vs separation from j={r['octaves'][0]}: {sep}")
        # running couplings
        print(" running couplings g(j)  [var_slope, kurtosis]:")
        for j in r["octaves"]:
            c = r["couplings"][str(j)]
            print(f"   j={j}: var_slope={c['var_slope']:+.3f}±{c['var_slope_se']:.3f}  "
                  f"kurtosis={c['kurtosis']:+.2f}±{c['kurtosis_se']:.2f}")
        # cross-octave couplings
        cx = ", ".join(f"{d['j']}-{d['j']+1}:{d['rho']:.3f}" for d in r["cross_octave"])
        print(f" cross-octave |w| coupling rho: {cx}")
        # P9b
        p = r["pca"]
        cum = ", ".join(f"{v:.2f}" for v in p["cumulative"][:3])
        print(f" P9b PCA: eff_dim_80={p['eff_dim_80']}  cumulative[1:3]={cum}")
        print()

    # ---- verdicts ----
    print("================ VERDICTS ================")
    fields = {r["key"]: r for r in b["fields"]}
    if "GRF_HF" in fields:
        g = fields["GRF_HF"]
        clean = [d for d in g["adjacent"] if d["j_fine"] in (2, 3)]
        zmax = max(abs(d["z"]) for d in clean)
        print(f"GRF null (clean octaves 2->3,3->4): max|z|={zmax:.2f} "
              f"-> {'PASS (consistent with null)' if zmax < 3 else 'CHECK'}")
    for key in ("gowerstreet", "hf_pm_1024"):
        if key not in fields:
            continue
        r = fields[key]
        sig = [d for d in r["adjacent"]
               if d["z"] > 3 and pct(d["excess"], d["measured"]) > 10 and d["j_fine"] <= 3]
        print(f"P9a [{key}]: {len(sig)}/{sum(1 for d in r['adjacent'] if d['j_fine']<=3)} "
              f"fine adjacent pairs are >3sigma AND >10% -> "
              f"{'TRUE' if sig else 'FALSE'}")
    dims = {k: fields[k]["pca"]["eff_dim_80"] for k in fields}
    print(f"P9b eff_dim_80 per field: {dims} -> "
          f"{'TRUE (<=3 for the drifting fields)' if all(v<=3 for k,v in dims.items() if fields[k]['role']!='null') else 'CHECK'}")


if __name__ == "__main__":
    main()
