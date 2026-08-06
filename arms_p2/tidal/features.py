"""TIDAL eigenframe features H(c) (NIGHT-ORDERS-3; prereg
log/2026-08-05-prereg-night3.md "TIDAL — design").

Gaussian-smoothed Hessian of the conditioning coarse at sigma_H = 2.0 px,
computed via Fourier multipliers on the periodic torus. Channels
(tr, a1, a2) = (Hxx + Hyy, Hxx - Hyy, 2 Hxy) with x along axis 2 (W) and
y along axis 1 (H). The Nyquist row/col of the ik (derivative) multipliers
is ZEROED on even sizes: the Nyquist mode is its own negative, so the sign
of ik there is ambiguous and any nonzero choice breaks exact D4
commutation of the mixed term (the Gaussian needs no such treatment —
|k|^2 is sign-blind). With the zeroing, covariance is exact up to FFT
roundoff, gate-tested in tests_p2/test_tidal_features.py.

Channel action under f2_group.apply_g's convention (rot90 by k over axes
(1,2), THEN W-flip if f) — derived from rot90: G(y,x) = F(y=x, x=N-1-y)
=> G_xx = F_yy, G_yy = F_xx, G_xy = -F_xy; W-flip: G(y,x) = F(y, N-1-x)
=> G_xx = F_xx, G_yy = F_yy, G_xy = -F_xy. Composing k rotations then f
flips (all channels also spatially transported by apply_g):

    g=(k,f)   tr    a1    a2          sign(a1) = (-1)^k
    (0,0)     +     +     +           sign(a2) = (-1)^(k+f)
    (1,0)     +     -     -
    (2,0)     +     +     +
    (3,0)     +     -     -
    (0,1)     +     +     -
    (1,1)     +     -     +
    (2,1)     +     +     -
    (3,1)     +     -     +

tr is D4-invariant; (a1, a2) is spin-2: both negate under a 90-degree
rotation; under the W-mirror a1 is fixed and a2 negates.
"""
from __future__ import annotations

import numpy as np

import jax.numpy as jnp

SIGMA_H = 2.0


def _hessian_multipliers(H, W, sigma):
    """Real (H, W) Fourier multipliers for (Hxx, Hyy, Hxy) at smoothing sigma."""
    ky = 2.0 * np.pi * np.fft.fftfreq(H)      # axis 1 (rows, y), rad/px
    kx = 2.0 * np.pi * np.fft.fftfreq(W)      # axis 2 (cols, x), rad/px
    kyd, kxd = ky.copy(), kx.copy()
    if H % 2 == 0:
        kyd[H // 2] = 0.0                     # zero the ik Nyquist (see module doc)
    if W % 2 == 0:
        kxd[W // 2] = 0.0
    gauss = np.exp(-0.5 * sigma * sigma * (ky[:, None] ** 2 + kx[None, :] ** 2))
    KY, KX = kyd[:, None], kxd[None, :]
    mxx = -(KX * KX) * gauss                  # (i kx)^2 * gaussian
    myy = -(KY * KY) * gauss
    mxy = -(KX * KY) * gauss
    return mxx, myy, mxy


def hessian_features(coarse, sigma=SIGMA_H):
    """(B,H,W,1) coarse -> (B,H,W,3) features (tr, a1, a2). Deterministic."""
    x = jnp.asarray(coarse)[..., 0]
    mxx, myy, mxy = _hessian_multipliers(x.shape[1], x.shape[2], sigma)
    F = jnp.fft.fft2(x, axes=(1, 2))
    hxx = jnp.fft.ifft2(F * mxx, axes=(1, 2)).real
    hyy = jnp.fft.ifft2(F * myy, axes=(1, 2)).real
    hxy = jnp.fft.ifft2(F * mxy, axes=(1, 2)).real
    feats = jnp.stack([hxx + hyy, hxx - hyy, 2.0 * hxy], axis=-1)
    return feats.astype(jnp.asarray(coarse).dtype)
