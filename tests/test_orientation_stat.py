"""Tests-first validation of the GATE-T orientation-decoherence instrument
(synthetics only; no real or generated maps touch this file).

Planted construction: coarse GRF + fine oct-1-band texture whose local
stripe direction follows a Hessian eigenframe — the map's OWN frame
(coupled) or an INDEPENDENT map's frame (decoupled: same texture spectrum
and same local anisotropy, no coupling — the orientation-decoherence null).
Per the instrument's sign convention (module docstring), stripes ALONG the
coarse frame put the texture gradients ACROSS it, so the coupled stack is
strongly NEGATIVE, not positive."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_p2"))

from oct1_texture import bandlimit  # noqa: E402
from orientation_stat import (  # noqa: E402
    coarse_frame,
    orientation_alignment,
    stack_orientation,
)

_ANGLES = np.pi * np.arange(8) / 8


def _grf(rng, size=128, alpha=1.0):
    k = np.fft.fftfreq(size) * size
    kk = np.hypot(k[:, None], k[None, :])
    filt = np.where(kk > 0, np.maximum(kk, 1e-9) ** -alpha, 0.0)
    f = np.fft.ifft2(np.fft.fft2(rng.standard_normal((size, size)))
                     * filt).real
    return f / f.std()


def _oriented_texture(rng, theta, size=128, k0=2.4, dk=0.35, ell=2.0, p=3):
    """Fine noise whose stripe direction follows theta(x): one white draw,
    a bank of 8 directional filters (ring at |k| ~ k0, inside the oct-1
    band; along-stripe wavevector suppressed at scale ell), blended per
    pixel with pi-periodic cos^(2p) weights."""
    W = np.fft.fft2(rng.standard_normal((size, size)))
    k = 2.0 * np.pi * np.fft.fftfreq(size)
    ky, kx = k[:, None], k[None, :]
    ring = np.exp(-0.5 * ((np.hypot(ky, kx) - k0) / dk) ** 2)
    tex = np.zeros((size, size))
    wsum = np.zeros((size, size))
    for phi in _ANGLES:
        kpar = kx * np.cos(phi) + ky * np.sin(phi)
        t = np.fft.ifft2(W * ring * np.exp(-0.5 * (kpar * ell) ** 2)).real
        m = np.cos(theta - phi) ** (2 * p)
        tex += m * t
        wsum += m
    tex /= wsum
    return tex / tex.std()


def _coarse(rng):
    """Coarse GRF with no oct-1 content (bandlimit is idempotent)."""
    return bandlimit(_grf(rng, alpha=1.3))


def _planted_field(rng, coupled=True):
    c = _coarse(rng)
    src = c if coupled else _coarse(rng)
    theta, _ = coarse_frame(src)
    return c + 0.5 * c.std() * _oriented_texture(rng, theta)


def _isotropic_field(rng, size=128, k0=2.4, dk=0.35):
    """Coarse GRF + isotropic ring noise at the texture band (plain null)."""
    c = _coarse(rng)
    k = 2.0 * np.pi * np.fft.fftfreq(size)
    kk = np.hypot(k[:, None], k[None, :])
    ring = np.exp(-0.5 * ((kk - k0) / dk) ** 2)
    t = np.fft.ifft2(np.fft.fft2(rng.standard_normal((size, size)))
                     * ring).real
    return c + 0.5 * c.std() * t / t.std()


def test_oriented_discriminates_decoupled():
    rng = np.random.default_rng(20260805)
    coup = [_planted_field(rng, coupled=True) for _ in range(24)]
    deco = [_planted_field(rng, coupled=False) for _ in range(24)]
    mc, sc = stack_orientation(coup, n_boot=2000, seed=1)
    md, sd = stack_orientation(deco, n_boot=2000, seed=2)
    z = abs(mc - md) / np.hypot(sc, sd)
    assert z > 5, f"discrimination z {z:.1f} <= 5"
    assert mc < -0.5, f"coupled stack {mc:.3f} not strongly negative"
    assert abs(md) < 3 * sd, f"decoupled null {md:.4f} > 3*SE {sd:.4f}"


def test_isotropic_null():
    rng = np.random.default_rng(7)
    iso = [_isotropic_field(rng) for _ in range(24)]
    m, s = stack_orientation(iso, n_boot=2000, seed=3)
    assert abs(m) < 3 * s, f"isotropic null |{m:.4f}| > 3*SE {s:.4f}"


def test_d4_invariance():
    rng = np.random.default_rng(11)
    f = _planted_field(rng, coupled=True)
    a0 = orientation_alignment(f)
    for g in (lambda x: np.rot90(x, 1), lambda x: np.rot90(x, 2),
              lambda x: np.rot90(x, 3), lambda x: x[::-1],
              lambda x: np.rot90(x, 1)[:, ::-1]):
        assert abs(orientation_alignment(g(f)) - a0) < 1e-9


def test_se_scaling():
    rng = np.random.default_rng(13)
    fields = [_planted_field(rng, coupled=True) for _ in range(48)]
    _, s24 = stack_orientation(fields[:24], n_boot=2000, seed=5)
    _, s48 = stack_orientation(fields, n_boot=2000, seed=6)
    assert 1.15 < s24 / s48 < 1.75
