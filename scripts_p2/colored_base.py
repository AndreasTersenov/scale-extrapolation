"""Colored heavy-tailed base — copula construction (R38 L1; NIGHT-ORDERS-2
N2 shape). Built and validated BEFORE any run; which filter SOURCE is used
(oct-1-measured vs oct-2-shape-rescaled, see the L1 delta memo) is a
reconvene decision — the machinery supports both.

Construction (per octave where a filter is given; white-t elsewhere):
  g   ~ white N(0,1), shape (B,H,W,C)
  gc  = Re ifft2( fft2(g) * filt )            filt: (H,W) ring-lookup map,
                                              normalized so E[gc^2] = 1
  x   = T5( gc )                              pointwise monotone quantile map
                                              N(0,1) -> unit-variance t(5),
                                              via a frozen z->x interp table
D4-invariance: filt is built by integer-|k| ring lookup => EXACTLY invariant
under the grid D4 action; g is iid; the map is pointwise => the base law is
exactly D4-invariant (proved in tests_p2/test_colored_base.py by asserting
filt == g(filt) for all 8 elements + orbit statistics).

The quantile map attenuates spatial correlation (Hermite mixing), so the
Gaussian-stage filter is CALIBRATED by fixed-point iteration in
l1_fit_filter.py until the FINAL (post-map) annular spectrum matches the
measured target within tolerance. The committed filter npz stores the
calibrated filt map + the z->x table; this module only applies them.
"""
from __future__ import annotations

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    HAVE_JAX = True
except Exception:  # env.sh stack (fit-time numpy path only)
    HAVE_JAX = False


def ring_table(N):
    """Integer-|k| ring index per FFT-grid mode; exactly D4-symmetric."""
    k = np.fft.fftfreq(N) * N
    kk = np.hypot(k[:, None], k[None, :])
    return np.rint(kk).astype(int)


def ring_spectrum(planes, N=None):
    """Mean power per ring over a stack of 2-D planes (numpy, fit/score)."""
    planes = np.asarray(planes, np.float64)
    N = planes.shape[-1] if N is None else N
    rings = ring_table(N)
    nmax = rings.max()
    P = np.abs(np.fft.fft2(planes)) ** 2 / N**2
    Pm = P.mean(axis=0) if P.ndim == 3 else P
    spec = np.zeros(nmax + 1)
    for r in range(nmax + 1):
        m = rings == r
        spec[r] = Pm[m].mean()
    return spec


def filter_from_ring_amps(ring_amps, N):
    """(H,W) amplitude map from per-ring amplitudes, unit output variance."""
    rings = ring_table(N)
    amps = np.asarray(ring_amps, np.float64)
    filt = amps[np.minimum(rings, len(amps) - 1)]
    filt = filt / np.sqrt((filt**2).mean())
    return filt


def make_z_table(nu=5.0, zmax=9.0, n=4097):
    """Frozen z->x table: N(0,1) quantiles to unit-variance t(nu)."""
    from scipy.stats import norm, t
    z = np.linspace(-zmax, zmax, n)
    x = t.ppf(norm.cdf(z), nu) / np.sqrt(nu / (nu - 2.0))
    return z, x


def colored_t_base_np(rng, shape, filt, z_grid, x_grid):
    """Numpy reference sampler (fit-time calibration path)."""
    g = rng.standard_normal(shape)
    gc = np.fft.ifft2(np.fft.fft2(g, axes=(-3, -2))
                      * filt[..., None], axes=(-3, -2)).real
    return np.interp(gc, z_grid, x_grid)


if HAVE_JAX:
    def colored_t_base(key, shape, filt, z_grid, x_grid):
        """JAX sampler: white normal -> ring-filter color -> t(5) marginals.

        shape (B,H,W,C); filt (H,W) calibrated unit-variance map;
        z_grid/x_grid the frozen quantile table."""
        g = jax.random.normal(key, shape)
        gc = jnp.fft.ifft2(jnp.fft.fft2(g, axes=(1, 2))
                           * jnp.asarray(filt)[None, :, :, None],
                           axes=(1, 2)).real
        return jnp.interp(gc, jnp.asarray(z_grid), jnp.asarray(x_grid))
