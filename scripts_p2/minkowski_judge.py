"""FROZEN STAGE-3 JUDGE — Minkowski functionals (R38 order 5, binding).

*** FREEZE NOTICE ***
Committed and frozen BEFORE any Stage-2 lever run (R38 sequencing). This
scorer must be applied to NO generation — committed or new — until Stage 3.
Its validation (tests/test_minkowski_judge.py) uses in-test synthetic GRFs
ONLY. Any edit to this file after the freeze commit is a protocol violation
absent a reconvene order.
*** END FREEZE NOTICE ***

Estimators (per-field standardization, thresholds nu in NUS):
  V0(nu): excursion-set area fraction, mean(f > nu).
  V1(nu): boundary-length proxy — count of 4-neighbour pixel pairs that
          straddle the threshold, per interior pixel.
  V2(nu): Euler characteristic per pixel via 2x2 quad counts:
          chi = (Q1 - Q3) / 4, with Q1/Q3 = quads with exactly one/three
          pixels ON. This is the average of the 4- and 8-connectivity
          Gray formulas ((Q1-Q3+/-2*Qd)/4): the diagonal-saddle ambiguity
          cancels, which the validation caught — the pure 4-connectivity
          form read V2(0) = +0.010 on synthetic GRFs where symmetry
          demands 0 (fixed pre-freeze, before the judge touched anything).
All three are D4-invariant by construction (pattern classes closed under
rotations/flips). Gaussian references used ONLY in the validation tests:
V0 = 1 - Phi(nu); V1 and V2 shapes prop. to exp(-nu^2/2) and
nu*exp(-nu^2/2) (Tomita), tested via threshold ratios so lattice constants
cancel.

Stack scoring convention (frozen now): per-map functional vectors ->
stack mean +- tile-bootstrap SE per (functional, nu); comparison metric
z per entry and T = max |z| over the scored grid.
"""
from __future__ import annotations

import numpy as np

NUS = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)


def _standardize(field):
    f = np.asarray(field, np.float64)
    return (f - f.mean()) / f.std()


def minkowski_vector(field, nus=NUS):
    """Concatenated [V0(nu...), V1(nu...), V2(nu...)] for one field."""
    f = _standardize(field)
    v0, v1, v2 = [], [], []
    for nu in nus:
        m = f > nu
        v0.append(m.mean())
        edges = (m[1:, :] != m[:-1, :]).sum() + (m[:, 1:] != m[:, :-1]).sum()
        v1.append(edges / m.size)
        a, b = m[:-1, :-1], m[:-1, 1:]
        c, d = m[1:, :-1], m[1:, 1:]
        s = (a.astype(int) + b.astype(int) + c.astype(int) + d.astype(int))
        q1 = (s == 1).sum()
        q3 = (s == 3).sum()
        v2.append((q1 - q3) / 4.0 / m.size)
    return np.array(v0 + v1 + v2)


def stack_minkowski(fields, nus=NUS, n_boot=2000, seed=0):
    """Stack mean vector and tile-bootstrap SE vector."""
    vecs = np.array([minkowski_vector(f, nus) for f in fields])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vecs), (n_boot, len(vecs)))
    boot = vecs[idx].mean(axis=1)
    return vecs.mean(axis=0), boot.std(axis=0, ddof=1)


def judge_T(gen_fields, real_fields, nus=NUS, n_boot=2000, seed=0):
    """The frozen comparison: per-entry z and T = max |z|."""
    mg, sg = stack_minkowski(gen_fields, nus, n_boot, seed)
    mr, sr = stack_minkowski(real_fields, nus, n_boot, seed + 1)
    z = (mg - mr) / np.hypot(sg, sr)
    return float(np.max(np.abs(z))), z
