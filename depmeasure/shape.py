"""B1 shape test: does ALIGNING an elongated context to the local structure buy
predictability? (aligned vs misaligned, matched shape+area — the clean estimand.)

Design lesson (measured during estimator development, kept as the null test): the
naive disk-vs-elongated comparison is SHAPE-CONFOUNDED — under exact isotropy an
elongated mask at matched area systematically LOSES ~0.2% to the disk, so it cannot
serve as an anisotropy null. The primary estimand is therefore

    Delta_align = V_misaligned - V_aligned

(the SAME elongated mask, class assignment rotated 90 degrees), which is exactly zero
in population under isotropy (exchangeability) and positive when the conditional law
is transported along structures. V_iso (disk, matched area) is reported as a
descriptive secondary ("does orientation beat even the compact context").

Per field and octave: local orientation class from the coarse structure tensor
(masks.orientation_class), positions restricted to coherence above the per-field
median and to interior margins. Two canonical elongated masks (axis-aligned 0-deg and
diagonal 45-deg, aspect 4:1, exactly area-matched to the disk); classes 90/135 map to
them by EXACT 90-degree offset rotation, so ridge features stay comparable within
each class pair. Held-out ridge (fit even fields / eval odd), paired per-field SEs.
"""
from __future__ import annotations

import numpy as np

from .masks import elongated_offsets, orientation_class
from .predictability import disk_offsets, _decompose


def _ridge_heldout(X, Y, F, lam_rel=1e-4):
    """Fit on even fields, return per-sample squared residuals on odd fields."""
    train = (F % 2 == 0)
    ev = ~train
    Xt, Yt = X[train], Y[train]
    mu_x, mu_y = Xt.mean(axis=0), Yt.mean()
    Xc, Yc = Xt - mu_x, Yt - mu_y
    G = Xc.T @ Xc
    lam = lam_rel * np.trace(G) / max(G.shape[0], 1)
    beta = np.linalg.solve(G + lam * np.eye(G.shape[0]), Xc.T @ Yc)
    pred = (X[ev] - mu_x) @ beta + mu_y
    return (Y[ev] - pred) ** 2, F[ev]


def shape_test(fields, j, r_iso=4, aspect=4.0, sigma_st=2.0, periodic=False,
               max_pos_per_field=2048, seed=0, target="w"):
    """Returns V_iso / V_aligned / V_misaligned + paired deltas with by-field SEs.

    target='w' probes mean-channel transport (ridge on the coefficient); target='w2'
    probes VARIANCE-channel transport (ridge on the squared coefficient — the
    var_slope-relevant channel; a linear fit in context captures the leading
    amplitude modulation).
    """
    rng = np.random.default_rng(seed)
    iso_mask = disk_offsets(r_iso)
    n_area = len(iso_mask)
    canon = {0: elongated_offsets(n_area, aspect, 0.0),
             1: elongated_offsets(n_area, aspect, 45.0)}
    rot90 = {c: [(dx, -dy) for dy, dx in canon[c]] for c in (0, 1)}
    # class -> offsets, aligned and misaligned (element order preserved: comparable)
    aligned = {0: canon[0], 1: canon[1], 2: rot90[0], 3: rot90[1]}
    misaligned = {0: rot90[0], 1: rot90[1], 2: canon[0], 3: canon[1]}
    all_offs = iso_mask + [d for c in aligned.values() for d in c] \
        + [d for c in misaligned.values() for d in c]
    rmax = int(max(max(abs(dy), abs(dx)) for dy, dx in all_offs))

    Xi, Xa, Xm, Yv, Fv, Pair = [], [], [], [], [], []
    for fi, f in enumerate(fields):
        cA, (cH, cV, cD) = _decompose(f, j)
        H, W = cA.shape
        if not periodic and (H - 2 * rmax < 2 or W - 2 * rmax < 2):
            continue
        cls, coh = orientation_class(cA, sigma=sigma_st)
        if periodic:
            ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        else:
            ys, xs = np.meshgrid(np.arange(rmax, H - rmax),
                                 np.arange(rmax, W - rmax), indexing="ij")
        ys, xs = ys.ravel(), xs.ravel()
        good = coh[ys, xs] > np.median(coh[ys, xs])
        if good.sum() < 16:          # degenerate coherence (e.g. constant): keep all
            good = np.ones(ys.size, dtype=bool)
        ys, xs = ys[good], xs[good]
        if ys.size > max_pos_per_field:
            sel = rng.choice(ys.size, size=max_pos_per_field, replace=False)
            ys, xs = ys[sel], xs[sel]
        pcls = cls[ys, xs]

        def gather_at(sub_ys, sub_xs, offsets):
            cols = []
            for dy, dx in offsets:
                if periodic:
                    cols.append(cA[(sub_ys + dy) % H, (sub_xs + dx) % W])
                else:
                    cols.append(cA[sub_ys + dy, sub_xs + dx])
            X = np.stack(cols, axis=1)
            if target in ("w2", "absw"):
                X = np.concatenate([X, X * X], axis=1)
            return X

        Xiso = gather_at(ys, xs, iso_mask)
        Xal = np.empty_like(Xiso)
        Xmi = np.empty_like(Xiso)
        for c in range(4):
            m = pcls == c
            if m.any():
                Xal[m] = gather_at(ys[m], xs[m], aligned[c])
                Xmi[m] = gather_at(ys[m], xs[m], misaligned[c])
        for b, band in enumerate((cH, cV, cD)):
            y = band[ys, xs]
            if target == "w2":
                y = y * y
            elif target == "absw":
                y = np.abs(y)
            Xi.append(Xiso)
            Xa.append(Xal)
            Xm.append(Xmi)
            Yv.append(y)
            Fv.append(np.full(ys.size, fi))
            Pair.append(pcls < 2)   # True = axis class pair, False = diagonal pair

    Xi = np.concatenate(Xi)
    Xa = np.concatenate(Xa)
    Xm = np.concatenate(Xm)
    Yv = np.concatenate(Yv)
    Fv = np.concatenate(Fv)
    Pair = np.concatenate(Pair)

    r2_iso, F_ev = _ridge_heldout(Xi, Yv, Fv)
    ev = (Fv % 2 == 1)

    def per_pair_residuals(X):
        r2 = np.empty_like(r2_iso)
        filled = np.zeros(ev.sum(), dtype=bool)
        for flag in (True, False):
            selp = Pair == flag
            if not selp.any():
                continue
            r2p, _ = _ridge_heldout(X[selp], Yv[selp], Fv[selp])
            sub = selp[ev]
            r2[sub] = r2p
            filled |= sub
        assert filled.all()
        return r2

    r2_al = per_pair_residuals(Xa)
    r2_mi = per_pair_residuals(Xm)

    def field_se(diff):
        per_field = {}
        for fi, d in zip(F_ev, diff):
            per_field.setdefault(fi, []).append(d)
        means = np.array([np.mean(v) for v in per_field.values()])
        return float(means.std(ddof=1) / np.sqrt(len(means))), len(per_field)

    d_align = r2_mi - r2_al
    d_disk = r2_iso - r2_al
    se_align, nf = field_se(d_align)
    se_disk, _ = field_se(d_disk)
    return {"V_iso": float(r2_iso.mean()), "V_aligned": float(r2_al.mean()),
            "V_misaligned": float(r2_mi.mean()),
            "delta_align": float(d_align.mean()), "delta_align_se": se_align,
            "delta_disk": float(d_disk.mean()), "delta_disk_se": se_disk,
            "n_eval": int(r2_iso.size), "n_fields_eval": nf}
