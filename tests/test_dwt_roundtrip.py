"""Validation gate 1 -- DWT round-trip at machine precision.

reconstruct(dwt2(field)) must return the field to ~machine precision. Without this,
every downstream coefficient statistic is measuring transform error, not the field.
Checked for the primary wavelet (haar) and its robustness alternatives (db4, sym4).
"""
import numpy as np
import pytest

from scaledrift import dwt2, octave_pair, reconstruct
from conftest import make_grf


@pytest.mark.parametrize("wavelet,tol", [("haar", 1e-12), ("db4", 1e-11), ("sym4", 1e-9)])
def test_roundtrip_machine_precision(wavelet, tol):
    rng = np.random.default_rng(0)
    for field in [rng.standard_normal((128, 128)), make_grf(1, seed=3)[0]]:
        coeffs = dwt2(field, wavelet, level=4, mode="periodization")
        recon = reconstruct(coeffs, wavelet, mode="periodization")
        assert recon.shape == field.shape
        err = np.abs(recon - field).max()
        assert err < tol, f"{wavelet}: round-trip error {err:.2e} exceeds {tol:.0e}"


def test_haar_is_essentially_exact():
    """Haar (the analysis default) round-trips at floating-point epsilon."""
    field = np.random.default_rng(1).standard_normal((256, 256))
    recon = reconstruct(dwt2(field, "haar", level=6, mode="periodization"), "haar",
                        mode="periodization")
    assert np.abs(recon - field).max() < 1e-13


def test_octave_pair_shapes_aligned():
    """Detail triple and coarse field at an octave live on the same grid."""
    field = make_grf(1, seed=5)[0]
    for j in (1, 2, 3, 4):
        details, coarse = octave_pair(field, j, "haar", mode="periodization")
        assert details.shape[0] == 3
        assert details.shape[1:] == coarse.shape
        assert coarse.shape == (128 // 2 ** j, 128 // 2 ** j)
