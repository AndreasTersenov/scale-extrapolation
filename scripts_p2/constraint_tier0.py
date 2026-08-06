"""Constraint tier-0: the empty-beam floor violation instrument
(BRIEF-foundations idea 7, tier 0; R48 D1 free battery row). Physical
convergence cannot be more negative than an empty line of sight; the
emptiest OBSERVED real pixel is a conservative (upper-bound) estimate of
that standardized floor. A faithful generator should not place mass
below it. All fields per-field standardized (the generation convention)
so gen and real share one axis; the statistic is pixel-pooled ⇒ exactly
D4-invariant. tests-first: tests/test_constraint_tier0.py.

floor(real): pooled minimum over per-field-standardized real fields.
violation_rate(fields, floor): per-field fraction of pixels < floor.
stack_violation(fields, floor, ...): mean ± tile-bootstrap SE.
real_loo_null(real, ...): leave-one-field-out crossing rate — for each
    real field, the fraction of its pixels below the pooled min of the
    OTHER fields; the sampling-expected exceedance the generator is
    compared against (real data DOES cross the min-of-others).
adjudicate(gen, real): z = (v_gen − v_null)/hypot(SEs); VIOLATION if
    z ≥ 3 (generator undershoots the floor more than real does).
"""
from __future__ import annotations

import numpy as np


def _std(field):
    f = np.asarray(field, np.float64)
    return (f - f.mean()) / f.std()


def floor(real_fields):
    return float(min(_std(f).min() for f in real_fields))


def violation_rate(fields, flr):
    """Per-field fraction of (standardized) pixels strictly below flr."""
    return np.array([float((_std(f) < flr).mean()) for f in fields])


def _boot(vals, n_boot, seed):
    vals = np.asarray(vals, np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), (n_boot, len(vals)))
    boot = vals[idx].mean(axis=1)
    return float(vals.mean()), float(boot.std(ddof=1))


def stack_violation(fields, flr, n_boot=5000, seed=0):
    return _boot(violation_rate(fields, flr), n_boot, seed)


def real_loo_null(real_fields, n_boot=5000, seed=0):
    """Leave-one-field-out crossing rate: each field's fraction below the
    pooled min of the others — the null exceedance under the real law."""
    sr = [_std(f) for f in real_fields]
    mins = np.array([f.min() for f in sr])
    v = []
    for i, f in enumerate(sr):
        other_min = float(np.min(np.delete(mins, i)))
        v.append(float((f < other_min).mean()))
    return _boot(v, n_boot, seed)


def adjudicate(gen_fields, real_fields, n_boot=5000, seed=0):
    flr = floor(real_fields)
    v_gen, se_gen = stack_violation(gen_fields, flr, n_boot, seed)
    v_null, se_null = real_loo_null(real_fields, n_boot, seed + 1)
    denom = float(np.hypot(se_gen, se_null))
    z = (v_gen - v_null) / denom if denom > 0 else 0.0
    return {"floor": flr, "v_gen": v_gen, "se_gen": se_gen,
            "v_null": v_null, "se_null": se_null, "z": float(z),
            "violation": bool(z >= 3.0)}
