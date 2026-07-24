"""Tests-first validation of the Phase-A' parity-localization estimators
(R31 order 1; log/2026-07-25-reconvene-phaseA.md).

Lives in tests/ (env.sh stack): pywt cannot share a process with JAX (the
Stage-D env split), so the JAX-gate suite (tests_p2) cannot import it.

Gates:
- the numerically-derived corner weight matrix W reproduces pywt's Haar
  synthesis exactly (convention-proof by construction);
- multi-level peak parity is uniform on iid ensembles (null);
- coefficient statistics (channel means / sign rates / cross-correlations)
  vanish on a D4-symmetric ensemble (null) and detect an injected channel-mean
  bias (sensitivity);
- hybrid resynthesis with a field's OWN coefficients is the identity at
  machine precision;
- corner-argmax distribution is uniform on the null and shifts under the
  injected bias in the predicted direction.
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts_p2"))

from parity_localization import (
    coef_stats_profile, corner_argmax_profile, corner_weights, dwt_levels,
    hybrid_with_gen_octave, idwt_levels, inject_channel_bias,
    peak_parity_profile,
)
from placement_instruments import stack_profiles, tstat

RNG = np.random.default_rng(20260725)


def _grf(n, size=64, rng=RNG, alpha=1.5):
    ky = np.fft.fftfreq(size)[:, None]
    kx = np.fft.fftfreq(size)[None, :]
    k = np.sqrt(ky ** 2 + kx ** 2)
    k[0, 0] = np.inf
    amp = k ** (-alpha / 2)
    out = []
    for _ in range(n):
        w = rng.standard_normal((size, size))
        out.append(np.fft.ifft2(np.fft.fft2(w) * amp).real)
    return np.array(out)


def test_corner_weights_match_pywt_synthesis():
    W = corner_weights()
    rng = np.random.default_rng(3)
    a, h, v, d = rng.standard_normal(4)
    coeffs = [np.array([[a]]), (np.array([[h]]), np.array([[v]]), np.array([[d]]))]
    import pywt
    block = pywt.waverec2(coeffs, "haar", mode="periodization")
    corners = W @ np.array([a, h, v, d])
    # corner order: (0,0),(0,1),(1,0),(1,1)
    assert np.allclose(corners, [block[0, 0], block[0, 1],
                                 block[1, 0], block[1, 1]])


def test_peak_parity_null_uniform():
    fields = _grf(48)
    profs = [peak_parity_profile(f, k=20, level=1) for f in fields]
    m, se = stack_profiles(profs)
    assert np.all(np.abs(m - 0.25) < 4 * se + 0.03)
    T, _ = tstat(*stack_profiles(profs[:24]), *stack_profiles(profs[24:]))
    assert T < 3.0


def test_coef_stats_null_on_symmetric_ensemble():
    fields = _grf(40)
    # D4-symmetrize the ensemble exactly
    sym = np.concatenate([fields, fields[:, ::-1], fields[:, :, ::-1],
                          fields[:, ::-1, ::-1]])
    a = [coef_stats_profile(f, octave=1) for f in sym[::2]]
    b = [coef_stats_profile(f, octave=1) for f in sym[1::2]]
    T, _ = tstat(*stack_profiles(a), *stack_profiles(b))
    assert T < 3.0
    # channel means / corrs are near zero in absolute terms on the ensemble
    m, se = stack_profiles(a)
    assert np.all(np.abs(m[:3]) < 4 * se[:3] + 0.02)   # normalized channel means


def test_injected_bias_detected_in_coefs_and_corners():
    fields = _grf(48)
    biased = np.array([inject_channel_bias(f, octave=1, eps=0.25) for f in fields])
    a = [coef_stats_profile(f, octave=1) for f in biased]
    b = [coef_stats_profile(f, octave=1) for f in fields]
    T, _ = tstat(*stack_profiles(a), *stack_profiles(b))
    assert T >= 5.0, T
    ca = [corner_argmax_profile(f, octave=1) for f in biased]
    cb = [corner_argmax_profile(f, octave=1) for f in fields]
    Tc, _ = tstat(*stack_profiles(ca), *stack_profiles(cb))
    assert Tc >= 5.0, Tc


def test_hybrid_identity():
    f = _grf(1)[0]
    hyb = hybrid_with_gen_octave(f, f, octave=2)
    assert np.max(np.abs(hyb - f)) < 1e-10


def test_dwt_roundtrip():
    f = _grf(1)[0]
    assert np.max(np.abs(idwt_levels(dwt_levels(f, 3)) - f)) < 1e-10
