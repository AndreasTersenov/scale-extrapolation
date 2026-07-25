# 2026-07-25 — PREREG: arm F2 — the D4 group-averaged sampler (R33 order 1)
# Committed BEFORE any run. Inference-only, zero training.

## Design (frozen on this commit)

Sampling becomes exactly D4-equivariant in law: per field, per octave draw, a
group element g is drawn uniformly from the 8-element D4 group (seeded,
recorded); the detail is sampled in the transformed frame and mapped back:
assemble f' = idwt2(g·c, d'·std) with d' = model(g·c), return f = g⁻¹·f'.
The synthesis–transform–analysis route avoids any hand-written channel action
table (the F mapping lesson); correctness rests on the exact Haar/D4
commutation, TESTS-FIRST in tests_wfm/test_d4_action.py:
 1. commutation: dwt2(g·x).coarse = g·(dwt2(x).coarse) exactly, all 8 g;
 2. identity equivalence: all-identity assignment reproduces plain assembly
    exactly (runtime gate on the real model as in F: max abs diff must be 0);
 3. symmetrization: a deterministic biased pseudo-model (constant channel-mean
    defect), averaged over the 8 elements, yields exactly zero channel means
    and uniform corner preference.
No fitted parameters — F's offset constant is NOT applied (group-averaging
nulls the mean layer by construction; subtracting the constant afterwards
would REINTRODUCE a bias). F2 is evaluated standalone; if F2-CLEAN, which
calibration ships (F constant vs F2 sampler) is the reconvene's call at
adjudication.

Legs: sandbox arm A at the committed pick 7500 (ADJUDICATING), arm B @2500
descriptive; corrected real-field echo re-scored: gowerstreet arm A @16000
group-averaged, nn WATCHED (the named tier-3 frontier statistic). Test-side
generations on the frozen test tiles; identical hc conventions to F.

## Adjudication (mechanical; sandbox arm A)

- Defect statistics, all three: T_coef oct-1 < 3 AND T_coef oct-2 < 3 AND
  level-1 parity T < 3 (vs the 32 test tiles, A'/F frozen conventions).
- Marginals: all standing bars pass (hc + e2e, octaves 2–4, the c1t
  convention — the F criterion); additionally each entry's |Δ| vs the F run
  is reported against combined SEs (the "unchanged within SE" texture,
  descriptive).

| branch | rule | reconvene (R33) | executor |
|---|---|---|---|
| F2-CLEAN | all three defect stats < 3 AND all marginal bars pass | 70 | 65 |
| F2-PARTIAL | any defect stat ≥ 3 (marginals pass) — the defect is NOT purely an equivariance violation: a D4-INVARIANT component exists (the group average cannot null an invariant), which would be the first genuinely architecture-flavored layer | 15 | 20 |
| F2-BREAKS | any marginal bar fails — group-averaging interacts with the conditional law beyond symmetry (would contradict the invariance argument; investigate before any adoption) | (in gates) | 5 |
| gates | commutation/identity test failure; infra resubmission | 15 | 10 |

Null meanings stated inline above. Echo (descriptive, load-bearing): parity
and T_coef expected clean; **nn is the watched quantity** — F2 cannot cure a
D4-invariant spacing anomaly (nn distances are D4-invariant), so nn is
EXPECTED to survive ≈ 4; if group-averaging moves nn materially, the spacing
residual has an equivariance component we mislocalized (executor: P(nn moves
below 3) = 10 — a genuine surprise branch).

## Alongside (R33 order 2, descriptive, committed artifacts)

corr(H,V) (and the full 9-component z-vector) across the existing M5
checkpoint curve (parity_ckpt_gen.npz, both arms) — is the second layer also
an optimization transient? Zero new compute; extends parity_ckpt_curve.json.

## Cost & discipline

One MIG job (≈ minutes; sample phase) + env.sh scoring. Artifacts →
results_p2/f2_*. R12 throughout; STOP at the F2 readout (R33); D proceeds
after, per R32 unchanged.
