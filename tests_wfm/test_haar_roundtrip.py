"""Backpressure (a) — the generator's own Haar transform round-trips at machine precision.

The coarse-to-fine recursion reconstructs fields by inverting the transform; if it is not
an exact inverse, every generated field is transform error, not physics. Tested in float32
(the generator's working precision): the error is pure float32 rounding.
"""
import jax
import jax.numpy as jnp

from wfm import haar

TOL = 1e-5   # float32 machine precision for this transform


def _rand(shape, seed=0):
    return jax.random.normal(jax.random.PRNGKey(seed), shape)


def test_single_level_roundtrip():
    x = _rand((2, 64, 64, 1))
    cA, det = haar.dwt2(x)
    assert cA.shape == (2, 32, 32, 1)
    assert all(d.shape == (2, 32, 32, 1) for d in det)
    rec = haar.idwt2(cA, det)
    assert float(jnp.max(jnp.abs(rec - x))) < TOL


def test_multi_level_roundtrip():
    x = _rand((1, 128, 128, 1), seed=1)
    for levels in (1, 3, 5):
        coeffs = haar.wavedec2(x, levels)
        assert coeffs[0].shape == (1, 128 // 2 ** levels, 128 // 2 ** levels, 1)
        rec = haar.waverec2(coeffs)
        assert float(jnp.max(jnp.abs(rec - x))) < TOL


def test_orthonormality_preserves_energy():
    """Orthonormal Haar preserves L2 energy: sum of squared coeffs == field energy."""
    x = _rand((1, 64, 64, 1), seed=2)
    coeffs = haar.wavedec2(x, 3)
    energy_x = float(jnp.sum(x ** 2))
    energy_c = float(sum(jnp.sum(c ** 2) for c in [coeffs[0]]
                         + [d for det in coeffs[1:] for d in det]))
    assert abs(energy_c - energy_x) / energy_x < TOL


def test_octave_pair_shapes():
    x = _rand((3, 128, 128, 1), seed=3)
    for j in (1, 2, 3, 4):
        detail, coarse = haar.octave_pair(x, j)
        n = 128 // 2 ** j
        assert detail.shape == (3, n, n, 3)     # 3 stacked sub-bands
        assert coarse.shape == (3, n, n, 1)
