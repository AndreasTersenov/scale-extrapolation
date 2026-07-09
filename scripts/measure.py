#!/usr/bin/env python
"""Measurement M1 on the control ladder: scale-drift of conditional wavelet statistics.

Runs GRF_HF (null) -> lognormal (control) -> gowerstreet (N-body) -> hf_pm (cross-check),
computing for each field:
  * adjacent-octave excess conditional-W1 drift  (P9a evidence)
  * excess drift vs octave separation from the finest scale
  * conditional-moment profiles Var/Skew(w | coarse bin) per octave
  * interpretable running couplings g(j): var-slope, var hi/lo, marginal kurtosis
    (bootstrap-over-parent-maps CI)
  * cross-octave |w| couplings rho(j)
  * P9b effective dimensionality of the cross-octave drift (PCA)

Writes results/measurement.json, results/profiles.npz and PNG plots. Fields, octaves,
map counts, and wavelet are CLI-configurable; --quick shrinks everything for a smoke run.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scaledrift import (collect_wc_grouped, coupling_scalars, cross_octave_coupling,
                        drift_estimate, marginal_pdf, octave_conditional_moments,
                        octave_wc, running_coupling_pca)
from scaledrift.data import load_parent_tiles

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")

# (key, dataset path under DATA_ROOT, column, role) -- GRF null FIRST.
FIELDS = [
    ("GRF_HF",      "GRF_HF",             "kappa", "null"),
    ("lognormal",   "lognormal",          "kappa", "control"),
    ("gowerstreet", "gowerstreet-train",  "kappa", "measurement"),
    ("hf_pm_1024",  "hf_pm_1024/train",   "maps",  "cross-check"),
]


def pooled_wc(parents, j, wavelet):
    ws, cs = [], []
    for tiles in parents:
        for t in tiles:
            w, c = octave_wc(t, j, wavelet)
            ws.append(w); cs.append(c)
    return np.concatenate(ws), np.concatenate(cs)


def running_couplings(parents, octaves, n_bins, wavelet, n_boot, seed):
    """Per-octave coupling scalars with bootstrap-over-parents CIs."""
    rng = np.random.default_rng(seed)
    n = len(parents)
    out = {}
    # pre-pool per parent per octave for fast bootstrap
    per = {j: [pooled_wc([p], j, wavelet) for p in parents] for j in octaves}
    for j in octaves:
        def scal(idx):
            w = np.concatenate([per[j][i][0] for i in idx])
            c = np.concatenate([per[j][i][1] for i in idx])
            return coupling_scalars(w, c, n_bins)
        point = scal(range(n))
        boots = [scal(rng.integers(0, n, n)) for _ in range(n_boot)]
        rec = {}
        for k in ("var_slope", "var_hi_lo", "kurtosis"):
            vals = np.array([b[k] for b in boots], float)
            rec[k] = point[k]
            rec[k + "_se"] = float(np.nanstd(vals, ddof=1))
        out[j] = rec
    return out


def measure_field(key, path, column, role, args):
    print(f"\n=== {key} ({role}) ===", flush=True)
    parents = load_parent_tiles(path, n_parents=args.n_parents, tile=args.tile,
                                seed=args.seed, max_shards=args.max_shards, column=column)
    n_tiles = sum(len(p) for p in parents)
    print(f"  loaded {len(parents)} parents, {n_tiles} tiles", flush=True)
    octs = args.octaves
    data = collect_wc_grouped(parents, octs, wavelet=args.wavelet)

    adjacent = [drift_estimate(data, a, b, n_bins=args.n_bins, n_boot=args.n_boot,
                               seed=args.seed) for a, b in zip(octs[:-1], octs[1:])]
    ref = octs[0]
    separation = [drift_estimate(data, ref, j, n_bins=args.n_bins, n_boot=args.n_boot,
                                 seed=args.seed) for j in octs[1:]]

    moments = {j: octave_conditional_moments(
        [t for p in parents for t in p], j, n_bins=args.n_bins_prof,
        wavelet=args.wavelet) for j in octs}
    couplings = running_couplings(parents, octs, args.n_bins_prof, args.wavelet,
                                  args.n_boot, args.seed)
    cross = [cross_octave_coupling([t for p in parents for t in p], j,
                                   wavelet=args.wavelet, n_boot=args.n_boot,
                                   seed=args.seed) for j in octs[:-1]]
    pca = running_coupling_pca([t for p in parents for t in p], octs,
                               n_bins=args.n_bins_prof, wavelet=args.wavelet)
    pdfs = {j: marginal_pdf([t for p in parents for t in p], j, wavelet=args.wavelet)
            for j in octs}

    return {
        "key": key, "role": role, "n_parents": len(parents), "n_tiles": n_tiles,
        "octaves": list(octs), "wavelet": args.wavelet,
        "adjacent": adjacent, "separation": separation,
        "moments": {int(j): {k: v.tolist() for k, v in m.items()}
                    for j, m in moments.items()},
        "couplings": {int(j): c for j, c in couplings.items()},
        "cross_octave": cross,
        "pca": {"eff_dim_80": pca["eff_dim_80"],
                "explained_var_ratio": pca["explained_var_ratio"].tolist(),
                "cumulative": pca["cumulative"].tolist()},
        "pdfs": {int(j): {"x": x.tolist(), "p": p.tolist()} for j, (x, p) in pdfs.items()},
    }


# ---------------------------------------------------------------- plotting ----
def plot_all(results, outdir):
    keys = [r["key"] for r in results]
    colors = dict(zip(keys, plt.cm.viridis(np.linspace(0, 0.85, len(keys)))))

    # 1. adjacent-octave excess drift with CI
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in results:
        seps = [d["j_fine"] for d in r["adjacent"]]
        exc = [d["excess"] for d in r["adjacent"]]
        lo = [d["excess"] - d["excess_ci"][0] for d in r["adjacent"]]
        hi = [d["excess_ci"][1] - d["excess"] for d in r["adjacent"]]
        ax.errorbar(seps, exc, yerr=[np.abs(lo), np.abs(hi)], marker="o", capsize=3,
                    color=colors[r["key"]], label=f"{r['key']} ({r['role']})")
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("finer octave j of the adjacent pair (j -> j+1)")
    ax.set_ylabel("excess conditional-W1 drift")
    ax.set_title("Adjacent-octave scale-drift (excess over finite-sample floor)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "drift_adjacent.png"), dpi=130); plt.close(fig)

    # 2. drift vs separation from finest octave
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in results:
        sep = [d["separation"] for d in r["separation"]]
        exc = [d["excess"] for d in r["separation"]]
        se = [d["excess_se"] for d in r["separation"]]
        ax.errorbar(sep, exc, yerr=se, marker="s", capsize=3, color=colors[r["key"]],
                    label=r["key"])
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("octave separation from finest scale")
    ax.set_ylabel("excess conditional-W1 drift")
    ax.set_title("Cumulative scale-drift vs separation")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "drift_vs_separation.png"), dpi=130); plt.close(fig)

    # 3. running couplings g(j): var_slope, kurtosis
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, k, ttl in zip(axes, ["var_slope", "var_hi_lo", "kurtosis"],
                          ["conditional-variance slope", "Var(top)/Var(bottom)",
                           "marginal excess kurtosis"]):
        for r in results:
            js = r["octaves"]
            y = [r["couplings"][int(j)][k] for j in js]
            se = [r["couplings"][int(j)][k + "_se"] for j in js]
            ax.errorbar(js, y, yerr=se, marker="o", capsize=2, color=colors[r["key"]],
                        label=r["key"])
        ax.axhline(0 if k != "var_hi_lo" else 1, color="k", lw=0.7, ls=":")
        ax.set_xlabel("octave j (1=finest)"); ax.set_title(ttl)
    axes[0].set_ylabel("coupling value"); axes[0].legend(fontsize=8)
    fig.suptitle("Empirical running couplings g(j)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "running_couplings.png"), dpi=130); plt.close(fig)

    # 4. conditional variance profiles per octave (measurement + null side by side)
    show = [r for r in results if r["role"] in ("null", "measurement")]
    fig, axes = plt.subplots(1, len(show), figsize=(5.5 * len(show), 4), squeeze=False)
    for ax, r in zip(axes[0], show):
        js = r["octaves"]
        cmap = plt.cm.plasma(np.linspace(0, 0.85, len(js)))
        for j, col in zip(js, cmap):
            m = r["moments"][int(j)]
            ax.plot(m["c_center"], m["var"], marker=".", color=col, label=f"j={j}")
        ax.set_xlabel("coarse field (std units)")
        ax.set_ylabel("Var(detail | coarse)")
        ax.set_title(f"{r['key']} ({r['role']})"); ax.legend(fontsize=7)
    fig.suptitle("Conditional variance profiles per octave (drift = octave spread)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "conditional_variance_profiles.png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-parents", type=int, default=40)
    ap.add_argument("--max-shards", type=int, default=14)
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--octaves", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--n-bins", type=int, default=8)
    ap.add_argument("--n-bins-prof", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--wavelet", default="haar")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fields", nargs="+", default=[f[0] for f in FIELDS])
    ap.add_argument("--out", default="measurement")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n_parents, args.n_boot, args.max_shards = 6, 100, 4

    os.makedirs(RESULTS, exist_ok=True)
    results = []
    for key, path, column, role in FIELDS:
        if key not in args.fields:
            continue
        results.append(measure_field(key, path, column, role, args))

    bundle = {"config": vars(args), "fields": results}
    with open(os.path.join(RESULTS, f"{args.out}.json"), "w") as f:
        json.dump(bundle, f, indent=1)
    # profiles npz (arrays), keyed field/octave/stat
    np.savez(os.path.join(RESULTS, "profiles.npz"),
             **{f"{r['key']}_j{j}_{k}": np.array(v)
                for r in results for j, m in r["moments"].items()
                for k, v in m.items()})
    if not args.quick or True:
        plot_all(results, RESULTS)
    print(f"\nwrote {args.out}.json, profiles.npz, and plots to {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
