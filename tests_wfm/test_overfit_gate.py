"""Backpressure (c) / rung (i) gate — a single-octave conditional FM overfits one field.

If the conditional velocity model cannot memorize one (detail_j | coarse_j) pair, no
downstream extrapolation claim is meaningful. Trains on ONE real gowerstreet field and
requires the sampled detail to reproduce the true detail (relative L2 < 0.15).
"""
import os

import numpy as np

from wfm.train import overfit_octave

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_single_octave_overfit():
    tiles = np.load(os.path.join(REPO, "data_cache", "tiles_128.npz"))["gowerstreet"]
    field = tiles[0].astype(np.float32)
    rel, info = overfit_octave(field, j=2, channels=(48, 96), steps=2000, seed=0)
    assert info["lossN"] < 0.5 * info["loss0"]          # it actually learned
    assert rel < 0.15, f"overfit relative L2 {rel:.3f} >= 0.15 (did not memorize)"
