"""Tests-first validation for arms F and J (prereg 2026-07-25-parity-cure,
approved R32). Lives in tests/ (env.sh stack): pywt is required, and pywt
cannot share a process with JAX (the standing env split).

Gates:
- the wfm-JAX <-> pywt detail-channel mapping is a fixed exact permutation
  with signs (established empirically here; the F correction uses it);
- the offset estimator recovers a known injected coefficient offset;
- the K-fold stability gate passes on homogeneous input and fires on a
  two-population (unstable) input;
- post-hoc field-space correction removes an injected offset exactly
  (linearity of the DWT);
- the joint selection criterion is the frozen mechanical rule (max of three
  normalized scores; argmin; ties -> earlier step).
"""
import os
import sys

import numpy as np
import pywt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts_p2"))

from fj_lib import (
    WFM_FROM_PYWT, apply_offsets_field, estimate_offsets, joint_pick,
    offsets_pywt_to_wfm, stability_gate,
)

RNG = np.random.default_rng(20260726)


def _wfm_dwt2_numpy(x):
    """Pure-numpy re-implementation of wfm.haar.dwt2 (channel-last dropped)."""
    s = np.sqrt(2.0)
    e, o = x[:, 0::2], x[:, 1::2]
    lo, hi = (e + o) / s, (e - o) / s
    le, lo_ = lo[0::2], lo[1::2]
    he, ho = hi[0::2], hi[1::2]
    cA = (le + lo_) / s
    cV = (le - lo_) / s
    cH = (he + ho) / s
    cD = (he - ho) / s
    return cA, (cH, cV, cD)


def test_channel_mapping_wfm_vs_pywt():
    x = RNG.standard_normal((16, 16))
    _, (wH, wV, wD) = _wfm_dwt2_numpy(x)
    _, (pH, pV, pD) = pywt.wavedec2(x, "haar", mode="periodization", level=1)
    p = {"H": pH, "V": pV, "D": pD}
    for wfm_name, wfm_arr in (("H", wH), ("V", wV), ("D", wD)):
        pywt_name, sign = WFM_FROM_PYWT[wfm_name]
        assert np.allclose(wfm_arr, sign * p[pywt_name]), (wfm_name, pywt_name)


def test_offset_estimator_recovers_injection():
    eps = {"H": 0.07, "V": -0.05, "D": 0.03}
    fields = []
    for _ in range(48):
        f = RNG.standard_normal((32, 32))
        coeffs = list(pywt.wavedec2(f, "haar", mode="periodization", level=1))
        H, V, D = coeffs[1]
        coeffs[1] = (H + eps["H"], V + eps["V"], D + eps["D"])
        fields.append(pywt.waverec2(coeffs, "haar", mode="periodization"))
    est = estimate_offsets(fields, octaves=(1,))
    for i, ch in enumerate(("H", "V", "D")):
        assert abs(est["mean"][1][i] - eps[ch]) < 4 * est["se"][1][i] + 0.01


def test_stability_gate():
    homog = [RNG.standard_normal((32, 32)) + 0.0 for _ in range(48)]
    assert not stability_gate(homog, octaves=(1,), n_folds=4)["fires"]
    # two-population offsets: folds disagree far beyond their SEs
    hetero = []
    for i in range(48):
        f = RNG.standard_normal((32, 32))
        coeffs = list(pywt.wavedec2(f, "haar", mode="periodization", level=1))
        H, V, D = coeffs[1]
        shift = 0.5 if i < 24 else -0.5
        coeffs[1] = (H + shift, V, D)
        hetero.append(pywt.waverec2(coeffs, "haar", mode="periodization"))
    # NOTE: folds must interleave populations to see instability? No — the
    # gate uses contiguous folds by design (val fields are exchangeable in
    # the real pipeline); contiguous folds on this input maximally disagree.
    assert stability_gate(hetero, octaves=(1,), n_folds=4)["fires"]


def test_apply_offsets_field_removes_injection():
    f = RNG.standard_normal((32, 32))
    coeffs = list(pywt.wavedec2(f, "haar", mode="periodization", level=1))
    H, V, D = coeffs[1]
    coeffs[1] = (H + 0.2, V - 0.1, D + 0.05)
    dirty = pywt.waverec2(coeffs, "haar", mode="periodization")
    clean = apply_offsets_field(dirty, {1: np.array([0.2, -0.1, 0.05])},
                                levels=1)
    assert np.max(np.abs(clean - f)) < 1e-10


def test_offsets_pywt_to_wfm_roundtrip_shape():
    off = {1: np.array([0.2, -0.1, 0.05]), 2: np.array([0.0, 0.1, 0.0])}
    w = offsets_pywt_to_wfm(off)
    assert set(w) == {1, 2} and all(v.shape == (3,) for v in w.values())


def test_joint_pick_rule():
    curve = {
        "1000": {"rel_vs": 0.05, "rel_kurt": 0.15, "T_coef": 12.0},
        "2000": {"rel_vs": 0.02, "rel_kurt": 0.075, "T_coef": 3.0},
        "3000": {"rel_vs": 0.02, "rel_kurt": 0.075, "T_coef": 3.0},
        "4000": {"rel_vs": 0.001, "rel_kurt": 0.01, "T_coef": 30.0},
    }
    # scores: 1000 -> max(.5, 1.0, 2.0)=2.0; 2000/3000 -> max(.2,.5,.5)=0.5;
    # 4000 -> max(.01,.067,5.0)=5.0; tie 2000 vs 3000 -> earlier
    assert joint_pick(curve) == 2000
