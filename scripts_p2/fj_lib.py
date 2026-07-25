"""Arm F/J machinery (prereg 2026-07-25-parity-cure, approved R32; env.sh side).
Validated tests-first in tests/test_fj_machinery.py.

Offset conventions: estimated with pywt ('haar', periodization) on generated
FIELDS in their saved units, per octave j (pywt decomposition level j) and
channel in PYWT ORDER (cH, cV, cD). The JAX recursion consumes them after
mapping through WFM_FROM_PYWT (fixed exact permutation/sign, test-established:
wfm's H and pywt's V are the same channel — the naive identity mapping is
WRONG, which is exactly what the test caught).
"""
from __future__ import annotations

import numpy as np
import pywt

# wfm channel name -> (pywt channel name, sign); established by
# tests/test_fj_machinery.py::test_channel_mapping_wfm_vs_pywt.
# NOTE: H and V are SWAPPED between the two transforms (wfm's "H" is
# high-along-width = pywt's cV); signs agree. Applying offsets without this
# mapping would correct the wrong channels.
WFM_FROM_PYWT = {"H": ("V", 1.0), "V": ("H", 1.0), "D": ("D", 1.0)}
PYWT_ORDER = ("H", "V", "D")


def _detail_at(field, j):
    coeffs = pywt.wavedec2(np.asarray(field, np.float64), "haar",
                           mode="periodization", level=j)
    return coeffs[1]          # the level-j (coarsest of this decomposition) triple


def estimate_offsets(fields, octaves=(1, 2, 3, 4)):
    """Per-octave, per-channel mean coefficient offset over fields (pywt order).

    Returns {"mean": {j: array(3)}, "se": {j: array(3)}} — the SE is over
    fields of the per-field channel means."""
    mean, se = {}, {}
    for j in octaves:
        per_field = np.array([[c.mean() for c in _detail_at(f, j)]
                              for f in fields])          # (n, 3)
        mean[j] = per_field.mean(axis=0)
        se[j] = per_field.std(axis=0, ddof=1) / np.sqrt(len(per_field))
    return {"mean": mean, "se": se}


def stability_gate(fields, octaves=(1, 2, 3, 4), n_folds=4):
    """Prereg gate: offset estimates unstable across contiguous K-fold splits
    by >2x their fold SEs (max over octaves/channels adjudicates)."""
    folds = np.array_split(np.arange(len(fields)), n_folds)
    worst = 0.0
    for j in octaves:
        est = [estimate_offsets([fields[i] for i in fo], octaves=(j,))
               for fo in folds]
        means = np.array([e["mean"][j] for e in est])        # (folds, 3)
        ses = np.array([e["se"][j] for e in est])
        spread = means.max(axis=0) - means.min(axis=0)
        typical_se = ses.mean(axis=0)
        worst = max(worst, float(np.max(spread / (2.0 * typical_se + 1e-30))))
    return {"fires": bool(worst > 2.0), "worst_ratio": worst}


def offsets_pywt_to_wfm(offsets):
    """Map {j: array(3) in pywt order} to wfm channel order (H, V, D) with
    the test-established signs."""
    out = {}
    for j, vec in offsets.items():
        mapped = np.empty(3)
        for i, wfm_name in enumerate(("H", "V", "D")):
            pywt_name, sign = WFM_FROM_PYWT[wfm_name]
            mapped[i] = sign * vec[PYWT_ORDER.index(pywt_name)]
        out[int(j)] = mapped
    return out


def apply_offsets_field(field, offsets, levels=4):
    """Post-hoc field-space correction: subtract the constant channel offsets
    in coefficient space (exact by DWT linearity). offsets: {j: array(3),
    pywt order}; octaves absent from the dict are untouched."""
    coeffs = list(pywt.wavedec2(np.asarray(field, np.float64), "haar",
                                mode="periodization", level=levels))
    for j, vec in offsets.items():
        idx = levels - int(j) + 1
        H, V, D = coeffs[idx]
        coeffs[idx] = (H - vec[0], V - vec[1], D - vec[2])
    return pywt.waverec2(coeffs, "haar", mode="periodization")


def joint_pick(curve):
    """The frozen J criterion: argmin over checkpoints of
    max(rel_vs/0.10, rel_kurt/0.15, T_coef/6.0); ties -> earlier step."""
    def score(e):
        return max(e["rel_vs"] / 0.10, e["rel_kurt"] / 0.15,
                   e["T_coef"] / 6.0)
    steps = sorted(int(s) for s in curve)
    best = min(steps, key=lambda s: (score(curve[str(s)]), s))
    return best
