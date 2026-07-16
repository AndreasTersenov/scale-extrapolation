"""FROZEN Stage-A sandbox recipe (2026-07-16 overnight run; NIGHT-ORDERS Stage A).

Selected by the descriptive pre-freeze pilot (logged in
log/2026-07-16-stageA-prereg.md): alpha=2.0, sigma_g=0.6 gives per-octave
(var_slope, kurtosis) of ~(1.25, 7.6) -> (0.81, 2.6) across octaves 1-4 on 128^2 —
a gowerstreet-like drifting, non-Gaussian regime inside the phase-1 COORD_NORM.

Seeds are part of the recipe: everything downstream is reproducible from this file.
"""
from __future__ import annotations

import numpy as np

from .lognormal import GRFSpec

SHAPE = (128, 128)
ALPHA = 2.0
SIGMA_G = 0.6
COND_LEVEL = 4          # fix the 8x8 Gaussian Haar coarse; redraw everything finer
N_PARENTS = 256
N_REDRAWS = 64
N_TRAIN_TILES = 322     # mirrors the phase-1 data regime (Arm C1)

SEED_PARENTS = 20260716     # parent GRF stream
SEED_REDRAWS = 20260717     # spawned per-parent for conditional redraws
SEED_TRAIN = 20260718       # unconditional training tiles (disjoint stream)

OCTAVES = (1, 2, 3, 4)


def spec() -> GRFSpec:
    return GRFSpec(shape=SHAPE, alpha=ALPHA)
