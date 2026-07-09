"""2D DWT octave decomposition (Measurement M1 substrate).

Conventions
-----------
* Octave index ``j`` is 1-based, ``j=1`` the FINEST detail scale. A ``j``-level
  ``pywt.wavedec2`` returns ``[cA_j, (cH_j, cV_j, cD_j), (cH_{j-1}, ...), ...]``;
  we read ``cA_j`` (approximation = "coarse field at that scale") and the level-``j``
  detail triple, which share the same grid.
* Default wavelet ``haar`` with ``mode="periodization"``: exact reconstruction at
  machine precision and power-of-two octave sizes, so a periodic (FFT-generated) GRF
  is decomposed self-similarly with no boundary leakage. Real (non-periodic) tiles get
  small, octave-common edge effects only. Haar is chosen because it is exactly
  symmetric (linear phase), so the transform commutes with flips/90-deg rotations and
  the drift metric is isotropy-invariant (the symmetry gate); ``sym4``/``db4`` are the
  robustness alternatives (``sym4`` near-symmetric, ``db4`` asymmetric).
* "Detail" at an octave pools the three orientation sub-bands (H, V, D). Pooling makes
  every downstream statistic invariant (in distribution) under flips / 90-deg rotations
  of the field — the isotropy the symmetry gate checks.
"""
from __future__ import annotations

import numpy as np
import pywt

DEFAULT_WAVELET = "haar"   # exactly symmetric -> transform commutes with flips/rotations
DEFAULT_MODE = "periodization"


def max_octaves(shape, wavelet=DEFAULT_WAVELET, mode=DEFAULT_MODE):
    """Largest usable octave count for a field of this shape."""
    n = min(shape)
    if mode == "periodization":
        # periodization halves cleanly; cap so the coarsest grid stays >= 4 px/side.
        return int(np.floor(np.log2(n))) - 2
    return pywt.dwtn_max_level(shape, wavelet)


def dwt2(field, wavelet=DEFAULT_WAVELET, level=None, mode=DEFAULT_MODE):
    """Full multi-level 2D DWT coefficient list (see pywt.wavedec2)."""
    return pywt.wavedec2(np.asarray(field, dtype=np.float64), wavelet,
                         mode=mode, level=level)


def reconstruct(coeffs, wavelet=DEFAULT_WAVELET, mode=DEFAULT_MODE):
    """Inverse transform of a pywt.wavedec2 coefficient list."""
    return pywt.waverec2(coeffs, wavelet, mode=mode)


def octave_pair(field, j, wavelet=DEFAULT_WAVELET, mode=DEFAULT_MODE):
    """Detail triple and aligned coarse field at octave ``j``.

    Returns
    -------
    details : (3, Hj, Wj) array   -- stacked (cH, cV, cD) detail sub-bands
    coarse  : (Hj, Wj) array      -- approximation coeffs cA_j on the same grid
    """
    coeffs = pywt.wavedec2(np.asarray(field, dtype=np.float64), wavelet,
                           mode=mode, level=j)
    cA = coeffs[0]
    cH, cV, cD = coeffs[1]
    details = np.stack([cH, cV, cD], axis=0)
    return details, cA


def octave_wc(field, j, wavelet=DEFAULT_WAVELET, mode=DEFAULT_MODE):
    """Paired (detail, coarse) samples at octave ``j``, orientation-pooled.

    ``w`` stacks the three detail sub-bands; ``c`` is the coarse value at the same
    grid position, repeated once per sub-band so ``w`` and ``c`` are aligned 1-D
    arrays of equal length (3 * Hj * Wj). Raw (unnormalized); callers standardize.
    """
    details, coarse = octave_pair(field, j, wavelet, mode)
    w = details.reshape(-1)
    c = np.tile(coarse.reshape(-1), 3)
    return w, c
