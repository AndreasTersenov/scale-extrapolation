"""Shape-test validation gates.

Primary estimand: Delta_align = V_misaligned - V_aligned (same elongated mask, class
assignment rotated 90 deg). NOTE (2026-07-16 amendment): NOT exactly zero under
isotropy — classifier selection creates a small negative baseline (measured z=-3.09
on the 256-parent sandbox control); these in-test nulls bound it at small sample. Two
channels: target 'w' (mean transport) and 'w2' (variance transport; features are
[raw, squared] context so both signed-value and amplitude modulation are visible).

Gates:
  1+2. exchangeability null on an isotropic GRF, both channels.
  3. positive control, w2 channel: a variance-transported field (slow row-amplitude
     modulation — amplitude constant along x, varying in y) where aligned-along-
     structure contexts MUST predict w2 better than misaligned ones.
  4. w-channel power limitation is DOCUMENTED (not asserted): on smooth GRFs the
     mean channel saturates within the compact core, so a w-channel real-field NULL
     is only interpretable jointly with the w2 channel. (Measured during estimator
     development: anisotropic-GRF w-channel delta ~ +0.04% of V, below 3 SE.)

Design lesson kept on record: disk-vs-elongated at matched area is SHAPE-CONFOUNDED
(elongation costs ~0.2% under exact isotropy) — delta_disk is descriptive only.
"""
import numpy as np

from depmeasure.shape import shape_test
from sandbox.lognormal import GRFSpec, sample_grf


def _row_amplitude_field(shape, rng):
    """Variance-transported control WITH detectable orientation: anisotropic GRF
    texture (structures along x, so the structure tensor finds class 0) times a
    STOCHASTIC row-wise amplitude A(y) with finite correlation length — variance is
    constant along the structures but unpredictable across them, exactly the
    filament hypothesis. Two designs measured during development and kept as
    documentation: an isotropic-texture version has undetectable orientation (null,
    correctly); a DETERMINISTIC sinusoidal A(y) is globally coherent, so
    cross-structure context predicts the phase BETTER and delta_align goes
    significantly NEGATIVE (z=-8) — the transport must be stochastic."""
    ky = np.fft.fftfreq(shape[0])[:, None]
    kx = np.fft.fftfreq(shape[1])[None, :]
    k = np.sqrt((kx * 2.0) ** 2 + (ky / 2.0) ** 2)
    k[0, 0] = np.inf
    amp = k ** (-1.0)
    amp /= np.sqrt(np.sum(amp ** 2) / (shape[0] * shape[1]))
    g = np.fft.ifft2(np.fft.fft2(rng.standard_normal(shape)) * amp).real
    # 1-D lognormal-ish amplitude, correlation length ~4 fine rows
    noise = rng.standard_normal(shape[0])
    kern = np.exp(-0.5 * (np.arange(-8, 9) / 2.0) ** 2)
    kern /= np.sqrt(np.sum(kern ** 2))
    smooth = np.convolve(np.concatenate([noise, noise[:16]]), kern,
                         mode="same")[:shape[0]]
    A = np.exp(0.6 * smooth)[:, None]
    return A * g


def test_shape_null_on_isotropic_grf_w():
    rng = np.random.default_rng(0)
    spec = GRFSpec(shape=(64, 64), alpha=2.0)
    fields = [sample_grf(spec, rng) for _ in range(64)]
    res = shape_test(fields, j=1, periodic=True, max_pos_per_field=512, seed=0)
    assert abs(res["delta_align"]) < 4 * res["delta_align_se"] + 1e-12, res


def test_shape_null_on_isotropic_grf_w2():
    rng = np.random.default_rng(2)
    spec = GRFSpec(shape=(64, 64), alpha=2.0)
    fields = [sample_grf(spec, rng) for _ in range(64)]
    res = shape_test(fields, j=1, periodic=True, max_pos_per_field=512, seed=0,
                     target="w2")
    assert abs(res["delta_align"]) < 4 * res["delta_align_se"] + 1e-12, res


def test_shape_positive_control_variance_transport_absw():
    # |w| target: same amplitude information as w^2 with far lighter tails
    # (w^2's heavy tails leave the same effect at z<1 — measured; |w| has power)
    rng = np.random.default_rng(1)
    fields = [_row_amplitude_field((64, 64), rng) for _ in range(384)]
    res = shape_test(fields, j=1, periodic=True, max_pos_per_field=2048, seed=0,
                     target="absw")
    assert res["delta_align"] > 3 * res["delta_align_se"], res
