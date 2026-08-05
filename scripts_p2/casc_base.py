"""CASC base — multiplicative-cascade-modulated white seeds (NIGHT-3 prereg
2026-08-05-prereg-night3.md §CASC; inference-only probe, NO training).

Construction (log-normal MRW family):
  omega ~ log-correlated GRF: Fourier synthesis with P_omega(k) prop. k^-2
          (the 2-D log-correlated exponent), k=0 mode zeroed, amplitude map
          even under k -> -k so the output is exactly real (house pattern:
          Re ifft2(fft2(white) * amp)); amp normalized so E[omega^2] = 1.
  M     = exp(lam * omega)                lam controls intermittency.
  eps   = M * z, z iid N(0,1) from the SAME key split; per-map (batch,
          channel plane) unit-variance normalization.
eps is exactly white in second order (z iid and independent of M => all
lagged covariances vanish, flat spectrum in expectation) but variance-
clustered / multifractal in higher order: corr(eps^2, eps^2 shifted) =
(e^{4 lam^2 (1+rho)} - e^{4 lam^2}) / (3 e^{8 lam^2} - e^{4 lam^2}), rho =
1-px omega correlation (~0.8 on a 64 grid).

LAM = 0.4 (documented choice): predicted 1-px eps^2 correlation ~0.13 =>
separates from plain white at z ~ 15 over 24-map stacks (test bar z > 5,
comfortable margin), while M's per-map dynamic range stays sane
(exp(lam * (max-min) omega) ~ e^2.8 ~ 16 typical; M in [~0.2, ~5] for
|omega| <= 4). lam = 0.3 would still pass the bar (~0.10) but with less
margin; lam >= 0.6 pushes M ratios past ~e^4 for no diagnostic gain.

casc_colored_base: the committed copula path of colored_base.colored_t_base
with the initial white Gaussian draw replaced by casc_seed — downstream
steps (spectral filter, z -> t(5) quantile table) reimplemented VERBATIM
here (do-not-edit rule on colored_base.py); everything else identical.
Validated tests-first in tests_p2/test_casc_base.py (whiteness, isotropy,
determinism, multifractal discrimination, splice smoke) before any run.

JAX-env module (wl-challenge-env); the CPU scorer never imports this.
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

LAM = 0.4


def log_corr_amp(H, W):
    """(H,W) Fourier amplitude map for the log-correlated GRF: amp prop.
    1/|k| (=> P prop. k^-2), k=0 zeroed, normalized so E[omega^2] = 1.
    Even under k -> -k and symmetric under kx <-> ky (D4-in-law)."""
    ky = np.fft.fftfreq(H) * H
    kx = np.fft.fftfreq(W) * W
    kk = np.hypot(ky[:, None], kx[None, :])
    amp = np.zeros_like(kk)
    amp[kk > 0] = 1.0 / kk[kk > 0]
    amp = amp / np.sqrt((amp**2).mean())
    return amp


def log_corr_field(key, shape):
    """Unit-variance log-correlated GRF, shape (B,H,W,C), FFT over (H,W)."""
    _, H, W, _ = shape
    g = jax.random.normal(key, shape)
    amp = jnp.asarray(log_corr_amp(H, W))
    return jnp.fft.ifft2(jnp.fft.fft2(g, axes=(1, 2))
                         * amp[None, :, :, None], axes=(1, 2)).real


def casc_seed(key, shape, lam=LAM):
    """Cascade-modulated white seed eps = exp(lam*omega) * z, unit variance
    per (batch, channel) map. White in second order; multifractal beyond."""
    k_om, k_z = jax.random.split(key)
    omega = log_corr_field(k_om, shape)
    z = jax.random.normal(k_z, shape)
    eps = jnp.exp(lam * omega) * z
    return eps / eps.std(axis=(1, 2), keepdims=True)


def casc_colored_base(filt, z_grid, x_grid, lam=LAM):
    """base_fn(key, shape) for l1p_lib.gen_groupavg_base(base_by_j=...):
    colored_base.colored_t_base's pipeline with the white draw replaced by
    casc_seed. Filter + quantile-table steps below are VERBATIM from
    colored_base.colored_t_base (splice point = the seed only)."""
    def base_fn(key, shape):
        eps = casc_seed(key, shape, lam)
        gc = jnp.fft.ifft2(jnp.fft.fft2(eps, axes=(1, 2))
                           * jnp.asarray(filt)[None, :, :, None],
                           axes=(1, 2)).real
        return jnp.interp(gc, jnp.asarray(z_grid), jnp.asarray(x_grid))
    return base_fn
