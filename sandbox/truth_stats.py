"""TRUE conditional statistics of the sandbox law — the estimand, computed directly.

The estimand is the population functional the frozen instrument
(scaledrift.coupling_scalars, n_bins=10) estimates:

  * pool (w, c) over fields at octave j (orientation-pooled details, tiled coarse);
  * standardize w and c by their POOLED mean/std;
  * 10 equal-count (quantile) bins in c; per-bin Var(w);
  * var_slope = OLS slope of per-bin variance against per-bin mean c;
  * kurtosis = marginal excess kurtosis of pooled standardized w.

Binning is part of the estimand's definition, so using the same n_bins here is not
circular: the truth is the SAME functional evaluated on the exact law via a much
larger exact sample (the pooled conditional ensembles), with an INDEPENDENT
implementation (sandbox.haar, this module — no scaledrift imports).

Error bars: disjoint parent-block batch means (K blocks of whole parents; each block
scored by the full estimand with its own edges/standardization; SE of the full-pool
point = std(blocks)/sqrt(K)). Parents are i.i.d., so blocks are independent; this is
the standard batch-means estimator and avoids a 200-fold bootstrap over ~4e8-point
pools. The truth SE enters Gate A only in quadrature with the (much larger)
instrument SE.
"""
from __future__ import annotations

import numpy as np

from .haar import octave_wc_pooled

N_BINS = 10


def estimand_scalars(w, c, n_bins=N_BINS):
    """var_slope + kurtosis of pooled (w, c) — the estimand's definition, from scratch."""
    w = np.asarray(w)
    c = np.asarray(c)
    wm = float(np.mean(w, dtype=np.float64))
    wsd = float(np.std(w, dtype=np.float64))
    cm = float(np.mean(c, dtype=np.float64))
    csd = float(np.std(c, dtype=np.float64))
    ws = (w - wm) / wsd
    cs = (c - cm) / csd
    edges = np.quantile(cs, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(cs, edges) - 1, 0, n_bins - 1)
    cc = np.full(n_bins, np.nan)
    var = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() < 8:
            continue
        cc[b] = float(np.mean(cs[sel], dtype=np.float64))
        var[b] = float(np.var(ws[sel], dtype=np.float64))
    ok = ~np.isnan(cc) & ~np.isnan(var)
    var_slope = float(np.polyfit(cc[ok], var[ok], 1)[0]) if ok.sum() >= 2 else np.nan
    kurtosis = float(np.mean(ws.astype(np.float64) ** 4) - 3.0)
    return {"var_slope": var_slope, "kurtosis": kurtosis, "detail_std": wsd}


def tail_q999(w):
    """99.9th percentile of |standardized w| — the R10 condition-1 extreme-tail
    instrument (descriptive, never adjudicating). Standardization is by the POOLED
    mean/std, the estimand's convention, so the value is scale-invariant and reads
    directly as 'tail extent beyond the second moment' (N(0,1): 3.29; t(5): 5.32)."""
    w = np.asarray(w)
    ws = (w - np.mean(w, dtype=np.float64)) / np.std(w, dtype=np.float64)
    return float(np.quantile(np.abs(ws), 0.999))


def _pool(fields, j, dtype=np.float32):
    ws, cs = [], []
    for f in fields:
        w, c = octave_wc_pooled(f, j)
        ws.append(w.astype(dtype))
        cs.append(c.astype(dtype))
    return np.concatenate(ws), np.concatenate(cs)


def truth_couplings(fields, groups, octaves, n_blocks=16):
    """Estimand on the full exact ensemble + batch-means SE over parent blocks.

    fields : sequence of 2-D arrays (all ensemble members, any order)
    groups : integer array, len(fields) — parent index of each member
    """
    groups = np.asarray(groups)
    parents = np.unique(groups)
    blocks = np.array_split(parents, n_blocks)
    out = {}
    for j in octaves:
        w, c = _pool(fields, j)
        point = estimand_scalars(w, c)
        del w, c
        bstats = []
        for blk in blocks:
            sel = np.isin(groups, blk)
            wb, cb = _pool([f for f, s in zip(fields, sel) if s], j)
            s = estimand_scalars(wb, cb)
            bstats.append((s["var_slope"], s["kurtosis"]))
        bstats = np.asarray(bstats)
        k = len(blocks)
        out[j] = {
            "var_slope": point["var_slope"],
            "var_slope_se": float(np.nanstd(bstats[:, 0], ddof=1) / np.sqrt(k)),
            "kurtosis": point["kurtosis"],
            "kurtosis_se": float(np.nanstd(bstats[:, 1], ddof=1) / np.sqrt(k)),
            "detail_std": point["detail_std"],
            "n_blocks": k,
        }
    return out
