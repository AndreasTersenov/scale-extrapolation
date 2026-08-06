"""Tests-first validation of the AUG fractional-copy pipeline (prereg
2026-08-05-night3 §AUG; JAX env). Includes the registered ×2 NO-OP
exhibit: Haar-approx ×2 downsampled copies duplicate the original's
deeper pairs (up to per-tile normalization scalars), which is WHY the
arm uses fractional band-limited copies instead."""
import os
import sys

import numpy as np

try:
    os.sched_setaffinity(0, set(range(4)))
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp  # noqa: E402

from arms_p2.aug.build import bandlimit_haar, build_aug_stacks, fourier_resample  # noqa: E402
from arms_p2.aug.train import train_aug_generator  # noqa: E402
from wfm import haar  # noqa: E402


def _tiles(rng, n=6, size=128, alpha=1.2):
    k = np.fft.fftfreq(size) * size
    kk = np.hypot(k[:, None], k[None, :])
    filt = np.where(kk > 0, np.maximum(kk, 1e-9) ** -alpha, 0.0)
    out = []
    for _ in range(n):
        f = np.fft.ifft2(np.fft.fft2(rng.standard_normal((size, size)))
                         * filt).real
        out.append(f / f.std())
    return np.array(out, np.float32)


def test_x2_noop_exhibit():
    """The registered theorem: octave-1 pairs of the Haar-approx ×2 copy
    ARE the original's octave-2 pairs (identical after per-stack detail
    standardization; per-tile normalize scalars are the only novelty)."""
    rng = np.random.default_rng(1)
    f = jnp.asarray(_tiles(rng, n=4))[..., None]
    cA1, _ = haar.dwt2(f)
    g = cA1  # the ×2 downsampled copy, Haar convention
    cA2_direct, d2 = haar.dwt2(cA1)
    cA_g, d1g = haar.dwt2(g)
    assert np.allclose(np.asarray(cA_g), np.asarray(cA2_direct), atol=1e-6)
    for a, b in zip(d1g, d2):
        assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-6)
    # with per-tile normalization the pairs differ ONLY by a scalar:
    sa = np.asarray(f)[..., 0].std(axis=(1, 2))
    sg = np.asarray(g)[..., 0].std(axis=(1, 2))
    ratio = sa / sg
    assert (ratio > 0.5).all() and (ratio < 1.5).all()


def test_no_oct1_leak():
    rng = np.random.default_rng(2)
    tiles = _tiles(rng, n=3)
    x = jnp.asarray(tiles)[..., None]
    cA, (h, v, d) = haar.dwt2(x)
    noise = [jnp.asarray(rng.standard_normal(np.asarray(h).shape),
                         jnp.float32) for _ in range(3)]
    x2 = haar.idwt2(cA, tuple(noise))
    a = build_aug_stacks(tiles)
    b = build_aug_stacks(np.asarray(x2[..., 0]))
    for k in ("g96", "g80"):
        assert np.allclose(a[k], b[k], atol=1e-4), \
            f"{k} leaks oct-1 content"


def test_shapes_finite():
    rng = np.random.default_rng(3)
    a = build_aug_stacks(_tiles(rng, n=5))
    assert a["g96"].shape == (5, 96, 96) and a["g80"].shape == (5, 80, 80)
    assert a["g96"].dtype == np.float32
    assert np.isfinite(a["g96"]).all() and np.isfinite(a["g80"]).all()


def test_d4_commutation_in_law():
    """Array-centered rot90/flips do NOT commute pointwise with the
    Fourier crop: the rotation center sits half a pixel off the DFT
    origin, and that offset is a different pixel-fraction on the two
    grids — the discrepancy is a pure TRANSLATION (A-N3-4). The correct
    invariant: magnitude spectra identical (translation-invariant), so
    the copies' law is D4-symmetrized exactly by the trainer's own
    d4_augment applied per stack."""
    rng = np.random.default_rng(4)
    tiles = _tiles(rng, n=2)
    a = build_aug_stacks(tiles)
    r = build_aug_stacks(np.rot90(tiles, 1, axes=(1, 2)).copy())
    m = build_aug_stacks(tiles[:, :, ::-1].copy())
    for k in ("g96", "g80"):
        A = np.abs(np.fft.fft2(np.rot90(a[k], 1, axes=(1, 2))))
        B = np.abs(np.fft.fft2(r[k]))
        assert np.abs(A - B).max() / A.max() < 1e-5
        A2 = np.abs(np.fft.fft2(a[k][:, :, ::-1]))
        B2 = np.abs(np.fft.fft2(m[k]))
        assert np.abs(A2 - B2).max() / A2.max() < 1e-5


def test_resample_real_and_bandlimit():
    rng = np.random.default_rng(5)
    f = _tiles(rng, n=1)[0]
    g = fourier_resample(bandlimit_haar(f[None])[0], 96)
    assert g.shape == (96, 96) and np.isfinite(g).all()
    assert abs(g.mean() - f.mean()) < 0.1


def test_trainer_smoke_all_slots():
    rng = np.random.default_rng(6)
    tiles = _tiles(rng, n=24)
    aug = build_aug_stacks(tiles)
    # UNet constraint (found in-test, A-N3-4): sizes must be = 0 mod 8
    # (three exact halvings); 12 (g96 oct-3) and 20 (g80 oct-2) break the
    # skip-connection shapes -> slots are originals {2,3,4} + g96 {1,2}
    # + g80 {1} = 6.
    groups = [(tiles, [2, 3, 4]), (aug["g96"], [1, 2]),
              (aug["g80"], [1])]
    state, meta = train_aug_generator(groups, steps=12, batch=4, lr=1e-3,
                                      seed=0, ckpt_steps=(12,),
                                      on_checkpoint=lambda *_: None)
    assert np.isfinite(meta["lossN"])
    assert len(meta["slots"]) == 6
    assert all(v > 0 for v in meta["slot_visits"].values()), meta["slot_visits"]
    assert set(meta["std_by_j"]) == {2, 3, 4}
