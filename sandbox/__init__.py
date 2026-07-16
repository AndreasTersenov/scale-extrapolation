"""Phase-2 Stage A: the lognormal sandbox — exact conditional truth by construction.

Numpy-only (runs under both the env.sh stack and ~/wl-challenge-env). The frozen
scaledrift instruments are NEVER imported here (independence of truth from instrument).
"""
from .lognormal import (
    GRFSpec,
    conditional_mean_map,
    conditional_redraw,
    coarse_spectrum,
    haar_coarse,
    haar_coarse_adjoint,
    lognormal_map,
    sample_grf,
    sigma_apply,
)
from .haar import haar_level, octave_wc_pooled
from .truth_stats import estimand_scalars, truth_couplings
