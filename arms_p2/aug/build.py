"""AUG fractional band-limited copy builder (prereg 2026-08-05-night3
§AUG). Copies are functions of bandlimit(f) ALONE (no oct-1 leak by
construction) and are NOT Haar-nested with the original pyramid, so they
create genuinely new (coarse, detail) pairs at fractional octave-role
offsets (×0.75 → 0.415 oct, ×0.625 → 0.678 oct). The literal ×2 copy is
a Haar-nesting NO-OP (exhibit in tests_p2/test_aug_build.py) — that is
why the fractional design exists.

fourier_resample: symmetric FFT-crop with the new Nyquist row/col ZEROED
(even-size grids: the −n/2 mode has no +n/2 partner after cropping;
keeping it breaks both realness and D4 exactness — the colored_base
frequency-index lesson). Kept modes {|k| ≤ n/2−1} are D4-symmetric, so
the resample commutes with rot90/flips (in-test)."""
from __future__ import annotations

import numpy as np

import jax.numpy as jnp

from wfm import haar

SIZES = (96, 80)


def bandlimit_haar(tiles):
    """(N,H,W) -> (N,H,W) with the oct-1 Haar detail triple zeroed."""
    x = jnp.asarray(tiles, jnp.float32)[..., None]
    cA, (h, v, d) = haar.dwt2(x)
    z = jnp.zeros_like(h)
    return np.asarray(haar.idwt2(cA, (z, z, z))[..., 0], np.float64)


def fourier_resample(f, n):
    """(N,N) real -> (n,n) real; symmetric spectral crop, Nyquist zeroed."""
    N = f.shape[0]
    Fs = np.fft.fftshift(np.fft.fft2(np.asarray(f, np.float64)))
    c = (N - n) // 2
    Fc = Fs[c:c + n, c:c + n].copy()
    Fc[0, :] = 0.0
    Fc[:, 0] = 0.0
    g = np.fft.ifft2(np.fft.ifftshift(Fc)).real * (n * n) / (N * N)
    # restore the (real, D4-invariant) mean lost with the k=0 row shift:
    # the crop keeps k=0 at index c -> it stays; nothing further needed.
    return g


def build_aug_stacks(tiles):
    """(N,128,128) -> {"g96": (N,96,96) f32, "g80": (N,80,80) f32}."""
    bl = bandlimit_haar(np.asarray(tiles))
    out = {}
    for n in SIZES:
        out[f"g{n}"] = np.array([fourier_resample(f, n) for f in bl],
                                np.float32)
    return out
