"""Predictability-saturation estimators (B1) + exact GRF truth.

ESTIMAND (pre-declared): for octave j and detail band o, the residual variance of the
best predictor IN THE ESTIMATOR'S CLASS of the detail coefficient w_o(p) from the
coarse values {c(p+d) : |d| <= r} (same-level Haar coarse, offsets d on the coarse
grid). Averaged over the three bands. Every practical estimator conditions on a
function of the patch, so each V(r) is an UPPER bound on the true conditional
variance given the patch; the ridge curve measures LINEAR predictability, the k-NN
curve adds nonlinear dependence on annulus summaries. Saturation radius r* is defined
on the pre-declared r-grid as the smallest r with V(r) <= V(r_max) + 3*SE(V(r_max)).

Fit/eval discipline: fields are split in half BY FIELD (even/odd index); predictors
are fit on the train half and V is the residual variance on the eval half (no
in-sample optimism). k-NN uses k=32 with the k/(k+1) correction for neighbor-mean
estimation error (exact when the conditional mean is locally constant).

Exact GRF truth: for a Gaussian field, (w_o(p), context) are jointly Gaussian with
covariances <atom_a, Sigma atom_b> (Haar atoms, stationary Sigma) -> the conditional
variance is closed-form. Both estimators are gated against this in
tests_p2/test_depmeasure.py before any real-field use (NIGHT-ORDERS B1).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from sandbox.haar import haar_atom, haar_level

R_GRID_DEFAULT = (0, 1, 2, 3, 4, 6, 8, 12)
ANNULUS_EDGES = (0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 8.5, 12.5)
KNN_K = 32


# ------------------------------------------------------------------ geometry helpers

def disk_offsets(r):
    """Integer offsets (dy, dx) with dy^2+dx^2 <= r^2, center included, fixed order."""
    rr = int(np.ceil(r))
    out = []
    for dy in range(-rr, rr + 1):
        for dx in range(-rr, rr + 1):
            if dy * dy + dx * dx <= r * r + 1e-9:
                out.append((dy, dx))
    return sorted(out)


def _decompose(field, j):
    cA = np.asarray(field, dtype=np.float64)
    bands = None
    for _ in range(j):
        cA, bands = haar_level(cA)
    return cA, bands  # coarse at level j, details at level j


def _gather(fields, j, offsets, periodic, rng, max_pos_per_field=2048,
            target="w"):
    """(X, Y) samples: context vectors and per-band detail targets, pooled by band.

    target: 'w' (detail coefficient — mean-channel predictability) or 'w2' (squared
    coefficient — variance-channel predictability, the var_slope-relevant one).
    Returns X (n, n_off), Y (n,), band index B (n,), field index F (n,).
    """
    Xs, Ys, Bs, Fs = [], [], [], []
    offs = np.asarray(offsets)
    rmax = int(np.max(np.abs(offs))) if len(offs) else 0
    for fi, f in enumerate(fields):
        cA, (cH, cV, cD) = _decompose(f, j)
        H, W = cA.shape
        if periodic:
            ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        else:
            if H - 2 * rmax < 2 or W - 2 * rmax < 2:
                continue
            ys, xs = np.meshgrid(np.arange(rmax, H - rmax),
                                 np.arange(rmax, W - rmax), indexing="ij")
        ys, xs = ys.ravel(), xs.ravel()
        if ys.size > max_pos_per_field:
            sel = rng.choice(ys.size, size=max_pos_per_field, replace=False)
            ys, xs = ys[sel], xs[sel]
        cols = []
        for dy, dx in offsets:
            if periodic:
                cols.append(cA[(ys + dy) % H, (xs + dx) % W])
            else:
                cols.append(cA[ys + dy, xs + dx])
        X = np.stack(cols, axis=1) if cols else np.empty((ys.size, 0))
        if target in ("w2", "absw") and X.shape[1]:
            X = np.concatenate([X, X * X], axis=1)   # signed + amplitude channels
        for b, band in enumerate((cH, cV, cD)):
            y = band[ys, xs]
            if target == "w2":
                y = y * y
            elif target == "absw":
                y = np.abs(y)
            Xs.append(X)
            Ys.append(y)
            Bs.append(np.full(ys.size, b))
            Fs.append(np.full(ys.size, fi))
    return (np.concatenate(Xs), np.concatenate(Ys),
            np.concatenate(Bs), np.concatenate(Fs))


# ----------------------------------------------------------------------- estimators

def ridge_vr(X, Y, B, F, n_fields, lam_rel=1e-4):
    """Held-out residual variance of per-band ridge regression. Returns (V, SE).

    Train = even field indices, eval = odd. SE via per-eval-field batch means.
    """
    train = (F % 2 == 0)
    ev = ~train
    resid_by_field = {}
    v_num, v_cnt = 0.0, 0
    for b in range(3):
        tb = train & (B == b)
        eb = ev & (B == b)
        Xt, Yt = X[tb], Y[tb]
        mu_x, mu_y = Xt.mean(axis=0), Yt.mean()
        Xc, Yc = Xt - mu_x, Yt - mu_y
        if Xc.shape[1] == 0:
            pred = np.full(eb.sum(), mu_y)
        else:
            G = Xc.T @ Xc
            lam = lam_rel * np.trace(G) / max(G.shape[0], 1)
            beta = np.linalg.solve(G + lam * np.eye(G.shape[0]), Xc.T @ Yc)
            pred = (X[eb] - mu_x) @ beta + mu_y
        r2 = (Y[eb] - pred) ** 2
        v_num += r2.sum()
        v_cnt += r2.size
        for fi, val in zip(F[eb], r2):
            resid_by_field.setdefault(fi, []).append(val)
    V = v_num / v_cnt
    per_field = np.array([np.mean(v) for v in resid_by_field.values()])
    SE = per_field.std(ddof=1) / np.sqrt(len(per_field))
    return float(V), float(SE)


def annulus_features(X, offsets, edges=ANNULUS_EDGES):
    """Reduce raw patch pixels to center + annulus means (linear functionals)."""
    offs = np.asarray(offsets, dtype=float)
    d = np.hypot(offs[:, 0], offs[:, 1])
    feats = []
    center = d < 0.5
    if center.any():
        feats.append(X[:, center].mean(axis=1))
    lo = 0.5
    for hi in edges:
        sel = (d >= lo) & (d < hi)
        if sel.any():
            feats.append(X[:, sel].mean(axis=1))
        lo = hi
    return np.stack(feats, axis=1) if feats else np.empty((X.shape[0], 0))


def knn_vr(X, Y, B, F, offsets, k=KNN_K):
    """Held-out k-NN residual variance on annulus features. Returns (V, SE)."""
    train = (F % 2 == 0)
    ev = ~train
    resid_by_field = {}
    v_num, v_cnt = 0.0, 0
    for b in range(3):
        tb = train & (B == b)
        eb = ev & (B == b)
        Ft = annulus_features(X[tb], offsets)
        Fe = annulus_features(X[eb], offsets)
        if Ft.shape[1] == 0:
            pred = np.full(eb.sum(), Y[tb].mean())
            corr = 1.0
        else:
            sd = Ft.std(axis=0)
            sd[sd == 0] = 1.0
            tree = cKDTree(Ft / sd)
            _, idx = tree.query(Fe / sd, k=k, workers=-1)
            pred = Y[tb][idx].mean(axis=1)
            corr = k / (k + 1.0)
        r2 = (Y[eb] - pred) ** 2 * corr
        v_num += r2.sum()
        v_cnt += r2.size
        for fi, val in zip(F[eb], r2):
            resid_by_field.setdefault(fi, []).append(val)
    V = v_num / v_cnt
    per_field = np.array([np.mean(v) for v in resid_by_field.values()])
    SE = per_field.std(ddof=1) / np.sqrt(len(per_field))
    return float(V), float(SE)


def predictability_curve(fields, j, r_grid=R_GRID_DEFAULT, periodic=True, seed=0,
                         max_pos_per_field=2048, estimators=("ridge", "knn"),
                         target="w"):
    """V(r) for both estimators on a stack of fields. Returns dict r -> results."""
    if target != "w":
        estimators = tuple(e for e in estimators if e != "knn")  # knn features are
        # defined on raw patches only; amplitude channels use ridge on [raw, sq]
    out = {}
    for r in r_grid:
        rng = np.random.default_rng(seed)  # same positions per r for comparability
        offsets = disk_offsets(r)
        X, Y, B, F = _gather(fields, j, offsets, periodic, rng, max_pos_per_field,
                             target=target)
        n_fields = len(fields)
        row = {"n_samples": int(Y.size), "n_features": len(offsets)}
        if "ridge" in estimators:
            row["ridge"], row["ridge_se"] = ridge_vr(X, Y, B, F, n_fields)
        if "knn" in estimators:
            row["knn"], row["knn_se"] = knn_vr(X, Y, B, F, offsets)
        out[r] = row
    return out


# ------------------------------------------------------------------ exact GRF truth

def analytic_grf_vr(spec, j, r_grid=R_GRID_DEFAULT, ridge_class=True):
    """Exact V(r) for a GRF: per-band conditional variance given the disk context.

    For jointly Gaussian variables the conditional variance is
    Var(w) - cov^T Cov_ctx^{-1} cov, identical for the linear (ridge) class and the
    true conditional. Position-independent by stationarity (periodic geometry).
    """
    from sandbox.lognormal import sigma_apply
    shape = spec.shape
    Hj = shape[0] // 2 ** j
    p0 = (Hj // 2, Hj // 2)
    out = {}
    for r in r_grid:
        offsets = disk_offsets(r)
        ctx_atoms = [haar_atom(shape, j, "c", 0,
                               ((p0[0] + dy) % Hj, (p0[1] + dx) % Hj))
                     for dy, dx in offsets]
        ctx_sig = [sigma_apply(a, spec) for a in ctx_atoms]
        C = np.array([[float(np.sum(a * s)) for s in ctx_sig] for a in ctx_atoms])
        vs = []
        for b in range(3):
            wa = haar_atom(shape, j, "w", b, p0)
            wsig = sigma_apply(wa, spec)
            var_w = float(np.sum(wa * wsig))
            cov = np.array([float(np.sum(a * wsig)) for a in ctx_atoms])
            if len(offsets):
                sol = np.linalg.solve(C + 1e-12 * np.eye(len(C)), cov)
                vs.append(var_w - float(cov @ sol))
            else:
                vs.append(var_w)
        out[r] = float(np.mean(vs))
    return out


def analytic_grf_vr_annulus(spec, j, r_grid=R_GRID_DEFAULT, edges=ANNULUS_EDGES):
    """Exact V(r) for a GRF given the ANNULUS-SUMMARY features (the k-NN class).

    Annulus means are linear functionals of the field, so the conditional variance
    given them is closed-form: same formula with annulus-mean atoms.
    """
    from sandbox.lognormal import sigma_apply
    shape = spec.shape
    Hj = shape[0] // 2 ** j
    p0 = (Hj // 2, Hj // 2)
    out = {}
    for r in r_grid:
        offsets = np.asarray(disk_offsets(r), dtype=float)
        groups = []
        if len(offsets):
            d = np.hypot(offsets[:, 0], offsets[:, 1])
            sel = d < 0.5
            if sel.any():
                groups.append(offsets[sel])
            lo = 0.5
            for hi in edges:
                s = (d >= lo) & (d < hi)
                if s.any():
                    groups.append(offsets[s])
                lo = hi
        atoms = []
        for g in groups:
            a = np.zeros(shape)
            for dy, dx in g.astype(int):
                a += haar_atom(shape, j, "c", 0,
                               ((p0[0] + dy) % Hj, (p0[1] + dx) % Hj))
            atoms.append(a / len(g))
        sig = [sigma_apply(a, spec) for a in atoms]
        C = np.array([[float(np.sum(a * s)) for s in sig] for a in atoms]) \
            if atoms else np.empty((0, 0))
        vs = []
        for b in range(3):
            wa = haar_atom(shape, j, "w", b, p0)
            wsig = sigma_apply(wa, spec)
            var_w = float(np.sum(wa * wsig))
            if atoms:
                cov = np.array([float(np.sum(a * wsig)) for a in atoms])
                sol = np.linalg.solve(C + 1e-12 * np.eye(len(C)), cov)
                vs.append(var_w - float(cov @ sol))
            else:
                vs.append(var_w)
        out[r] = float(np.mean(vs))
    return out
