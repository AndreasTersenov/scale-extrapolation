"""Tests-first validation of the oct-1 phase-texture instrument
(synthetics only; prereg 2026-08-05-oct1fix §I)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_p2"))

from oct1_texture import (  # noqa: E402
    alignment_stat,
    contribution,
    fragmentation_profile,
    stack_alignment,
    stack_fragmentation,
)


def _phase_randomize(u, rng):
    """Same power spectrum, scrambled phases — the corrector's null."""
    F = np.fft.fft2(u)
    ph = np.exp(2j * np.pi * rng.random(u.shape))
    v = np.fft.ifft2(np.abs(F) * ph).real
    return v / v.std()


def _grf(rng, size=128, alpha=1.0):
    k = np.fft.fftfreq(size) * size
    kk = np.hypot(k[:, None], k[None, :])
    filt = np.where(kk > 0, np.maximum(kk, 1e-9) ** -alpha, 0.0)
    f = np.fft.ifft2(np.fft.fft2(rng.standard_normal((size, size)))
                     * filt).real
    return f / f.std()


def _organized_field(rng):
    """Multiplicative field: fine texture modulated by (and aligned to)
    coarse structure — organized u by construction."""
    base = _grf(rng, alpha=1.3)
    return np.exp(0.8 * base) * (1 + 0.4 * _grf(rng, alpha=0.3))


def _scrambled_field(rng):
    """Same coarse part, u phase-randomized at fixed spectrum."""
    f = _organized_field(rng)
    from oct1_texture import bandlimit
    u = f - bandlimit(f)
    return bandlimit(f) + u.std() * _phase_randomize(u, rng)


def test_fragmentation_discriminates_scrambled():
    rng = np.random.default_rng(20260806)
    org = [_organized_field(rng) for _ in range(24)]
    scr = [_scrambled_field(rng) for _ in range(24)]
    mo, so = stack_fragmentation(org, n_boot=400, seed=1)
    ms, ss = stack_fragmentation(scr, n_boot=400, seed=2)
    z = (ms - mo) / np.hypot(so, ss)
    assert np.max(np.abs(z)) > 5, f"max |z| {np.max(np.abs(z)):.1f} <= 5"


def test_alignment_discriminates():
    rng = np.random.default_rng(9)
    org = [_organized_field(rng) for _ in range(24)]
    scr = [_scrambled_field(rng) for _ in range(24)]
    ma, sa = stack_alignment(org, n_boot=400, seed=3)
    mb, sb = stack_alignment(scr, n_boot=400, seed=4)
    assert ma > mb, f"organized alignment {ma:.3f} <= scrambled {mb:.3f}"
    assert (ma - mb) / np.hypot(sa, sb) > 5


def test_d4_invariance():
    rng = np.random.default_rng(11)
    f = _organized_field(rng)
    p0 = fragmentation_profile(f)
    a0 = alignment_stat(f)
    for g in (lambda x: np.rot90(x, 1), lambda x: x[::-1],
              lambda x: np.rot90(x, 2)[:, ::-1]):
        assert np.allclose(fragmentation_profile(g(f)), p0, atol=1e-9)
        assert abs(alignment_stat(g(f)) - a0) < 1e-9


def test_se_scaling():
    rng = np.random.default_rng(13)
    fields = [_organized_field(rng) for _ in range(48)]
    _, s24 = stack_alignment(fields[:24], n_boot=2000, seed=5)
    _, s48 = stack_alignment(fields, n_boot=2000, seed=6)
    assert 1.15 < s24 / s48 < 1.75
