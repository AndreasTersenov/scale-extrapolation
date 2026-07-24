"""Phase-A' parity-localization estimators (R31 order 1). Validated tests-first
in tests_p2/test_parity_localization.py before touching any committed artifact.

All wavelet operations use pywt 'haar' with mode='periodization' (the repo's
documented match to sandbox.haar). Conventions are never assumed: the 2x2
corner weight matrix W is derived NUMERICALLY from pywt synthesis at import
time, so every corner statement is convention-proof by construction.

Estimators:
  peak_parity_profile(field, k, level)   4 shares: position of each top-K peak
        within its 2^level synthesis block, at half-block granularity
        (level 1 = fine-pixel parity (y%2, x%2)).
  coef_stats_profile(field, octave)      9 components, all exactly zero in
        expectation under a D4-symmetric law: normalized channel means
        E[H]/sd, E[V]/sd, E[D]/sd; sign-rate excesses P(H>0)-1/2 etc.; and
        cross-channel correlations corr(H,V), corr(H,D), corr(V,D).
  corner_argmax_profile(field, octave)   4 shares: which corner of each 2x2
        synthesis block at that octave wins (argmax over the W-projected
        (a,h,v,d)), restricted to the top-decile |a| blocks (where peaks live).
  hybrid_with_gen_octave(real, gen, octave)   real field with the generated
        field's octave-j detail triple transplanted (all else real), exactly
        resynthesized — the octave whose transplant reproduces the output
        parity bias is the originating octave.
"""
from __future__ import annotations

import numpy as np
import pywt

from placement_instruments import topk_peaks_xy


# ------------------------------------------------------------- wavelet plumbing --

def dwt_levels(field, levels):
    return pywt.wavedec2(np.asarray(field, np.float64), "haar",
                         mode="periodization", level=levels)


def idwt_levels(coeffs):
    return pywt.waverec2(coeffs, "haar", mode="periodization")


def corner_weights():
    """W (4x4): corner values [(0,0),(0,1),(1,0),(1,1)] of a 2x2 synthesis
    block as a linear map of (a, h, v, d) — derived numerically from pywt."""
    W = np.zeros((4, 4))
    for i in range(4):
        unit = np.zeros(4)
        unit[i] = 1.0
        coeffs = [np.array([[unit[0]]]),
                  (np.array([[unit[1]]]), np.array([[unit[2]]]),
                   np.array([[unit[3]]]))]
        blk = idwt_levels(coeffs)
        W[:, i] = [blk[0, 0], blk[0, 1], blk[1, 0], blk[1, 1]]
    return W


_W = None


def _get_W():
    global _W
    if _W is None:
        _W = corner_weights()
    return _W


# ----------------------------------------------------------------- estimators --

def peak_parity_profile(field, k, level=1):
    """Share of top-K peaks per half-block position within their 2^level block."""
    ys, xs, _ = topk_peaks_xy(field, k)
    if len(ys) == 0:
        return None
    half = 2 ** (level - 1)
    cls = ((ys // half) % 2) * 2 + ((xs // half) % 2)
    return np.bincount(cls, minlength=4) / len(ys)


def coef_stats_profile(field, octave):
    """9 D4-null components of the octave's detail coefficients (see module doc)."""
    coeffs = dwt_levels(field, octave)
    H, V, D = (np.asarray(c, np.float64).ravel() for c in coeffs[1])
    out = []
    for c in (H, V, D):
        out.append(c.mean() / (c.std() + 1e-30))
    for c in (H, V, D):
        out.append((c > 0).mean() - 0.5)
    for a, b in ((H, V), (H, D), (V, D)):
        sa, sb = a.std(), b.std()
        out.append(((a - a.mean()) * (b - b.mean())).mean() /
                   (sa * sb + 1e-30))
    return np.array(out)


COEF_STAT_NAMES = ["meanH", "meanV", "meanD", "signH", "signV", "signD",
                   "corrHV", "corrHD", "corrVD"]


def corner_argmax_profile(field, octave, top_frac=0.1):
    """Share of high-amplitude 2x2 synthesis blocks (top-decile |a|) whose
    argmax corner is [(0,0),(0,1),(1,0),(1,1)] under the octave's (a,h,v,d)."""
    coeffs = dwt_levels(field, octave)
    a = np.asarray(coeffs[0], np.float64)
    H, V, D = (np.asarray(c, np.float64) for c in coeffs[1])
    W = _get_W()
    stack = np.stack([a, H, V, D], axis=-1)              # (hc, wc, 4)
    corners = stack @ W.T                                 # (hc, wc, 4)
    keep = np.abs(a) >= np.quantile(np.abs(a), 1 - top_frac)
    if keep.sum() == 0:
        return None
    cls = np.argmax(corners[keep], axis=-1)
    return np.bincount(cls, minlength=4) / keep.sum()


def hybrid_with_gen_octave(real, gen, octave, levels=4):
    """Real field with the generated field's octave-j detail triple swapped in."""
    cr = dwt_levels(real, levels)
    cg = dwt_levels(gen, levels)
    # wavedec2 detail index: coeffs[1] is the COARSEST octave (= levels),
    # coeffs[-1] the finest (= 1)
    idx = levels - octave + 1
    cr = list(cr)
    cr[idx] = cg[idx]
    return idwt_levels(cr)


def inject_channel_bias(field, octave, eps):
    """Test helper: add eps*sd to the H and V channels (and -eps*sd to D) of one
    octave — a synthetic parity-breaking defect with a known signature."""
    coeffs = list(dwt_levels(field, octave))
    H, V, D = (np.asarray(c, np.float64) for c in coeffs[1])
    coeffs[1] = (H + eps * H.std(), V + eps * V.std(), D - eps * D.std())
    return idwt_levels(coeffs)
