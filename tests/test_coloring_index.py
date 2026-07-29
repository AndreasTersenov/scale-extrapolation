"""Tests-first validation of the coloring index (N1 gate probe; prereg
log/2026-07-29-prereg-n1-mechanism.md). Runs BEFORE the estimator touches
any real data. Bands are the prereg-fixed ones (low 1<=|k|<=N/8, high
N/4<|k|<=N/2); the three prereg'd properties:

(a) discrimination: fields whose DETAIL PLANES are planted white vs
    power-law-colored (built by inverse DWT — the generator's own synthesis
    direction) separate at z > 10 at every probed octave (1, 2, 3). A first
    version of this test used power-law-colored FIELDS and FAILED
    (oct1 colored C 0.890 < white 1.003): level-1 Haar details of a red
    field are near-white/slightly blue (the Haar high-pass ~k^2 response
    cancels P~k^-2 and decimation aliases near-Nyquist power into low
    plane-k). The gate compares real-vs-generated DETAIL planes directly,
    so detail-space coloring is the correct validation target — finding
    documented in the N1 prereg note, made BEFORE any real data was
    touched;
(b) D4 invariance: C is invariant under flips/rotations of the field
    (permutation-exact up to float roundoff);
(c) consistency: doubling the number of maps shrinks the bootstrap SE ~sqrt(2).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_p2"))

from coloring_index import stack_coloring  # noqa: E402
from parity_localization import idwt_levels  # noqa: E402


def _white(rng, n, size=128):
    return rng.standard_normal((n, size, size))


def _colored_plane(rng, size, alpha):
    """One power-law-colored plane, unit variance; alpha=0 -> white."""
    w = rng.standard_normal((size, size))
    if alpha == 0:
        return w
    k = np.fft.fftfreq(size) * size
    kk = np.hypot(k[:, None], k[None, :])
    filt = np.where(kk > 0, np.maximum(kk, 1e-9) ** -alpha, 0.0)
    f = np.fft.ifft2(np.fft.fft2(w) * filt).real
    return f / f.std()


def _fields_with_planted_details(rng, n, alpha, size=128, levels=3):
    """Fields synthesized by inverse Haar DWT from planted detail planes
    (white coarse; per-level detail channels colored by |k|^-alpha)."""
    out = []
    for _ in range(n):
        cs = size >> levels
        coeffs = [rng.standard_normal((cs, cs))]
        for lv in range(levels, 0, -1):
            ps = size >> lv
            coeffs.append(tuple(_colored_plane(rng, ps, alpha)
                                for _ in range(3)))
        out.append(idwt_levels(coeffs))
    return np.array(out)


def test_discrimination_white_vs_colored_details():
    rng = np.random.default_rng(20260729)
    white = _fields_with_planted_details(rng, 32, alpha=0.0)
    colored = _fields_with_planted_details(rng, 32, alpha=1.0)
    for octave in (1, 2, 3):
        cw, sw = stack_coloring(white, octave, n_boot=500, seed=1)
        cc, sc = stack_coloring(colored, octave, n_boot=500, seed=2)
        z = (cc - cw) / np.hypot(sw, sc)
        assert cc > cw, f"oct{octave}: colored C {cc:.3f} <= white {cw:.3f}"
        assert z > 10, f"oct{octave}: discrimination z {z:.1f} <= 10"


def test_white_baseline_near_unity():
    rng = np.random.default_rng(7)
    white = _white(rng, 32)
    for octave in (1, 2, 3):
        c, s = stack_coloring(white, octave, n_boot=500, seed=3)
        assert abs(c - 1.0) < max(5 * s, 0.05), \
            f"oct{octave}: white C {c:.3f} not ~1 (se {s:.3f})"


def test_d4_invariance():
    rng = np.random.default_rng(11)
    fields = _fields_with_planted_details(rng, 4, alpha=1.0)
    for octave in (1, 2, 3):
        c0, _ = stack_coloring(fields, octave, n_boot=10, seed=0)
        for g in (lambda f: np.rot90(f, 1), lambda f: np.rot90(f, 2),
                  lambda f: f[::-1], lambda f: f[:, ::-1],
                  lambda f: np.rot90(f, 1)[::-1]):
            cg, _ = stack_coloring([g(f) for f in fields], octave,
                                   n_boot=10, seed=0)
            assert abs(cg - c0) < 1e-9 * max(1.0, abs(c0)), \
                f"oct{octave}: C not D4-invariant ({c0} vs {cg})"


def test_bootstrap_se_scaling():
    rng = np.random.default_rng(13)
    fields = _fields_with_planted_details(rng, 64, alpha=1.0)
    _, se_32 = stack_coloring(fields[:32], 1, n_boot=2000, seed=5)
    _, se_64 = stack_coloring(fields, 1, n_boot=2000, seed=6)
    ratio = se_32 / se_64
    assert 1.15 < ratio < 1.75, f"SE scaling ratio {ratio:.2f} not ~sqrt(2)"
