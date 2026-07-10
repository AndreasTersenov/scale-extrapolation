"""Rung (ii) gate — two-octave coarse-to-fine recursion on one field.

A weight-tied (shared) conditional model is overfit on octaves 1 and 2 of one gowerstreet
field; the full field is then generated coarse-to-fine from its true coarsest coarse
(sample detail | coarse -> invert one Haar level -> repeat). Gates: the reconstructed
field matches the true field, and the recursion is deterministic under a fixed seed.
"""
import os

import numpy as np

from wfm.train import overfit_field_recursive

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_two_octave_recursion_reconstructs_field():
    tiles = np.load(os.path.join(REPO, "data_cache", "tiles_128.npz"))["gowerstreet"]
    rel, info = overfit_field_recursive(tiles[0].astype(np.float32), j_max=2,
                                        channels=(48, 96), steps=2000, seed=0)
    assert info["deterministic"], "recursion not reproducible under a fixed seed"
    assert info["octaves"] == [1, 2]
    assert rel < 0.2, f"two-octave recursion field relative L2 {rel:.3f} >= 0.2"
