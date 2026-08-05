"""T — the oct-1 orientation-decoherence instrument (GATE-T; tests-first in
tests/test_orientation_stat.py on synthetics before any real data).

Measures whether the oct-1 texture's LOCAL ORIENTATION is locked to the
coarse field's Hessian eigenframe (complement of oct1_texture.alignment_stat,
which measures amplitude organization only).

Definitions (u = f − bandlimit(f), standardized per map):
  fine frame    J = G_rho * (∇u ⊗ ∇u)          structure tensor of u
                theta_u = 0.5·atan2(2 J12, J11 − J22)
                w_u = sqrt((J11−J22)² + 4 J12²) / (J11 + J22 + eps)  (coherence)
  coarse frame  H = Hess( G_sigma_c * bandlimit(f) )
                theta_c = 0.5·atan2(2 H12, H11 − H22)
                w_c = sqrt((H11−H22)² + 4 H12²) / mean(·)  (anisotropy,
                normalized per map, scale-free)
  A_or = sum(w_u w_c cos 2(theta_u − theta_c)) / sum(w_u w_c)  ∈ [−1, 1]

SIGN CONVENTION (read before interpreting): theta_u is the dominant-GRADIENT
direction of the texture (structure-tensor major eigenvector); theta_c is the
coarse Hessian major-EIGENVALUE direction, which points ALONG ridges of
positive structures. Texture ELONGATED along the coarse frame has gradients
perpendicular to theta_c, so it gives A_or < 0; texture whose gradients lock
to theta_c gives A_or > 0; orientation decoherence gives A_or ≈ 0. Any
locking to the eigenframe moves |A_or| away from 0.

Scale choices (tuned on planted synthetics 2026-08-05, then FIXED — design
periphery per the GATE-T order; 3×3 scan, 16 frame-coupled vs 16
frame-decoupled planted maps per cell):
  RHO = 1.5 px      structure-tensor smoothing. Scanned {1.0, 1.5, 2.0}:
                    discrimination z varies <10% with rho at every sigma_c
                    (107/103/96 at sigma_c = 3), all z >> 5; 1.5 px averages
                    a few stripe periods of oct-1 texture (wavelength
                    ~2.5–4 px) without blurring across coarse-frame
                    variation. Kept the pre-registered value.
  SIGMA_C = 3.0 px  coarse-Hessian smoothing. Scanned {2.0, 3.0, 4.0}: the
                    coupled signal strengthens with sigma_c (−0.78 → −0.93)
                    but at 4.0 the decoupled control drifts slightly
                    positive (+0.024 ± 0.012), and at 2.0 the k²-weighted
                    frame response (peak wavelength 2π·sigma_c/√2 ≈ 8.9 px)
                    sits too close to the next-to-finest band to count as
                    coarse. At 3.0 the frame peaks at ~13 px, the decoupled
                    control is null (+0.008 ± 0.009), the isotropic-ring
                    null is clean (−0.001 ± 0.008), and z ≈ 103. Kept the
                    pre-registered value.

Numerics: ALL derivatives and smoothings are Fourier multipliers on the
periodic torus, so they commute EXACTLY with the grid D4 action. The
first-derivative multipliers i·k have the Nyquist row/col ZEROED: on
even-size FFT grids the Nyquist bin maps to itself under k → −k, so an odd
multiplier that keeps it breaks rot90/flip exactness (same class of bug as
the frequency-index-map fix in colored_base's D4 test history). All higher
derivatives (Hessian) are built as products of the zeroed first-derivative
multipliers, so the whole tensor pipeline inherits the exactness. Gaussian
multipliers are even in k and need no zeroing.

D4-invariance of A_or: bandlimit (Haar level-1 projection, even grid) maps
2×2 blocks to 2×2 blocks under D4, so u and bandlimit(f) transform with the
field. Under a rotation both tensors transform as T → R T Rᵀ, so theta_u and
theta_c shift by the SAME angle and the difference is preserved (mod π);
under a flip both off-diagonals negate, so theta_u and theta_c both negate
and cos 2(theta_u − theta_c) is even in the difference. Both weights are
rotation/flip invariants of their tensors (eigenvalue functions). Hence A_or
is exactly D4-invariant (machine precision; asserted at 1e-9 in tests).

stack_orientation follows the oct1_texture._stack tile-bootstrap pattern:
per-map A_or values, bootstrap over MAPS for the SE.
"""
from __future__ import annotations

import numpy as np

from oct1_texture import bandlimit

RHO = 1.5        # px — structure-tensor smoothing (FIXED, see docstring)
SIGMA_C = 3.0    # px — coarse-Hessian smoothing (FIXED, see docstring)
EPS = 1e-12


def _multipliers(shape):
    """(ikx, iky, k2): i·k first-derivative multipliers with the Nyquist
    row/col zeroed (D4 exactness on even grids — see module docstring) and
    the full |k|² grid for Gaussian smoothing. Radians per pixel."""
    ny, nx = shape
    ky = 2.0 * np.pi * np.fft.fftfreq(ny)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx)
    k2 = ky[:, None] ** 2 + kx[None, :] ** 2
    if ny % 2 == 0:
        ky[ny // 2] = 0.0
    if nx % 2 == 0:
        kx[nx // 2] = 0.0
    return (1j * kx)[None, :], (1j * ky)[:, None], k2


def fine_frame(field, rho=RHO):
    """(theta_u, w_u): structure-tensor orientation and coherence weight of
    the standardized oct-1 contribution u = f − bandlimit(f)."""
    f = np.asarray(field, np.float64)
    u = f - bandlimit(f)
    u = u / (u.std() + EPS)
    ikx, iky, k2 = _multipliers(u.shape)
    U = np.fft.fft2(u)
    ux = np.fft.ifft2(U * ikx).real
    uy = np.fft.ifft2(U * iky).real
    g = np.exp(-0.5 * rho * rho * k2)
    J11 = np.fft.ifft2(np.fft.fft2(ux * ux) * g).real
    J22 = np.fft.ifft2(np.fft.fft2(uy * uy) * g).real
    J12 = np.fft.ifft2(np.fft.fft2(ux * uy) * g).real
    theta = 0.5 * np.arctan2(2.0 * J12, J11 - J22)
    w = np.sqrt((J11 - J22) ** 2 + 4.0 * J12 ** 2) / (J11 + J22 + EPS)
    return theta, w


def coarse_frame(field, sigma_c=SIGMA_C):
    """(theta_c, w_c): Hessian eigenframe orientation and per-map-normalized
    anisotropy weight of the smoothed coarse part bandlimit(f)."""
    f = np.asarray(field, np.float64)
    c = bandlimit(f)
    ikx, iky, k2 = _multipliers(c.shape)
    C = np.fft.fft2(c) * np.exp(-0.5 * sigma_c * sigma_c * k2)
    H11 = np.fft.ifft2(C * ikx * ikx).real
    H22 = np.fft.ifft2(C * iky * iky).real
    H12 = np.fft.ifft2(C * ikx * iky).real
    theta = 0.5 * np.arctan2(2.0 * H12, H11 - H22)
    w = np.sqrt((H11 - H22) ** 2 + 4.0 * H12 ** 2)
    return theta, w / (w.mean() + EPS)


def orientation_alignment(field, rho=RHO, sigma_c=SIGMA_C):
    """A_or ∈ [−1, 1]: coherence-and-anisotropy-weighted mean of
    cos 2(theta_u − theta_c). See module docstring for the sign convention."""
    theta_u, w_u = fine_frame(field, rho)
    theta_c, w_c = coarse_frame(field, sigma_c)
    w = w_u * w_c
    return float(np.sum(w * np.cos(2.0 * (theta_u - theta_c)))
                 / (np.sum(w) + EPS))


def stack_orientation(fields, n_boot=5000, seed=0):
    """Per-map A_or values → stack mean ± bootstrap-over-maps SE
    (oct1_texture._stack pattern)."""
    vals = np.asarray([orientation_alignment(f) for f in fields], np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), (n_boot, len(vals)))
    boot = vals[idx].mean(axis=1)
    return float(vals.mean()), float(boot.std(ddof=1))
