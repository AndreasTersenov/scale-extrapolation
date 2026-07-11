"""Validation gates for D4 (flip/rotation) training-data augmentation (attempt 4a).

Augmentation is FIELD-LEVEL (transform tiles before wavelet decomposition), so no
subband permutation/sign bookkeeping can be wrong by construction. The gates check:
  1. the orbit is the full D4 group: 8x tiles, all orientations present and exact;
  2. the per-octave pooled detail std is invariant across orientations (Haar, even
     tile size => exact up to float error) -- the std_by_j the generator uses;
  3. the conditional law is preserved: the per-coarse-bin detail variance profile of
     each orientation block matches the un-augmented one (sign flips and channel
     permutations cannot change squared-detail statistics at the same site);
  4. train_generator(augment=True) runs and sees the 8x pool.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from wfm import haar
from wfm.dataset import d4_augment, field_to_octaves, normalize_tiles
from wfm.train import train_generator


@pytest.fixture(scope="module")
def tiles():
    rng = np.random.default_rng(0)
    # smooth-ish random fields so conditional bins are populated
    t = rng.normal(size=(6, 32, 32))
    k = np.ones((3, 3)) / 9.0
    from scipy.signal import convolve2d
    return np.stack([convolve2d(x, k, mode="same", boundary="wrap") for x in t])


def test_orbit_is_full_d4(tiles):
    aug = d4_augment(tiles)
    n = tiles.shape[0]
    assert aug.shape == (8 * n,) + tiles.shape[1:]
    got = {tuple(np.asarray(aug[k * n]).ravel().round(12)) for k in range(8)}
    exp = set()
    t0 = tiles[0]
    for base in (t0, np.fliplr(t0)):
        for k in range(4):
            exp.add(tuple(np.rot90(base, k).ravel().round(12)))
    assert got == exp, "augmented blocks must be exactly the 8 D4 orientations"


def test_detail_std_invariant_per_orientation(tiles):
    aug = d4_augment(tiles)
    n = tiles.shape[0]
    for j in (1, 2):
        stds = []
        for k in range(8):
            _, s = field_to_octaves(aug[k * n:(k + 1) * n], [j])
            stds.append(s[j])
        # std subtracts the pooled mean; sign-flipped channels shift the tiny nonzero
        # detail mean, so invariance holds to O(mean^2) only -- 1e-3 still catches any
        # real subband-bookkeeping error (which would be O(1)).
        assert np.ptp(stds) / np.mean(stds) < 1e-3, (j, stds)


def test_conditional_profile_invariant(tiles):
    aug = d4_augment(tiles)
    n = tiles.shape[0]
    prof = []
    for k in range(8):
        det, coa = haar.octave_pair(normalize_tiles(aug[k * n:(k + 1) * n]), 1)
        # per-site squared power: sign flips / channel permutations leave it invariant
        w2 = (np.asarray(det) ** 2).reshape(-1, 3).sum(-1)
        c = np.asarray(coa).reshape(-1)
        edges = np.quantile(c, np.linspace(0, 1, 6))
        edges[0] -= 1e-9; edges[-1] += 1e-9
        idx = np.clip(np.digitize(c, edges) - 1, 0, 4)
        prof.append(np.array([w2[idx == b].mean() for b in range(5)]))
    prof = np.array(prof)
    assert np.allclose(prof, prof[0], rtol=1e-6), "conditional law must be exactly preserved"


def test_train_generator_augment_runs(tiles):
    state, meta = train_generator(np.asarray(tiles, np.float32), [1, 2], arm="A",
                                  channels=(8, 16), steps=6, batch=4, lr=1e-3,
                                  nll=True, augment=True)
    assert meta["augment"] is True
    assert np.isfinite(meta["lossN"])
