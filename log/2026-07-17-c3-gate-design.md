# 2026-07-17 — C3 validation-gate design record: the R10 condition-2 investigation

Context: R10 approved the C3 prereg with two conditions; condition 2 requires the
direct noise-conditioned sampler to pass a two-moment + skewness recovery toy
BEFORE any training. Writing that gate surfaced a real, instructive chain of
measurements. All numbers below are held-out-eval unless marked train-eval; toy =
exp-noise conditional (y = m(c) + 0.85·(Exp(1)−1), resid skew 2.0, excess kurt 6.0)
on 16² fields with the production patch config (8×8, stride 4, dim 192, fair m=8)
unless stated.

## The chain of measurements

1. **First gate draft (tiny conv net (8,16), n=192 fields, train-eval, 1500 steps):
   skew 0.80 — FAIL.** Diagnostic v1 (same setup to 8k steps): skew crosses truth
   at ~2–2.5k then OVERSHOOTS monotonically to 3.40 (kurt 17) while the loss falls.
   Transit, not learning — at 660 sightings/field the net memorizes tail patterns
   through the z-channel (the ES's empirical minimizer at one observation per
   conditioning IS the per-conditioning delta).
2. **Diagnostic v2 (4× data n=768, held-out eval): no train/held-out gap; converged
   skew 0.33, kurt 1.9** — the tiny net's POPULATION answer badly under-recovers
   shape. (So v1's "pass window" was pure overfit transit; a gate tuned to stop
   there would have been self-deception.)
3. **Patch scan (tiny net, held-out): patch 1 (per-pixel CRPS) skew 4.7 / kurt 46
   (wild overshoot, unstable); patch 4: 0.37/2.1; patch 8: 0.33/1.9.** At tiny
   capacity NO patch size is calibrated.
4. **Capacity resolution (SLURM 16605903, production net (32,64,128) + production
   patch config, n=768, held-out): skew 2.014, kurt 5.76 at 4k steps vs truth
   2.0/6.0.** The under-recovery was a CAPACITY confound of the toy, not an
   objective failure. The production configuration recovers third and fourth
   moments essentially exactly, on held-out conditionings.
5. **1-D isolation (MLP direct sampler through the identical production loss path,
   infinite fresh data): skewness recovers to 2.0 (gate PASSES); symmetric t(5)
   tails plateau at excess kurt ~1.1 (truth 6.0) for 20k steps at z-dim 1, 4, 8.**
   Skewness carries a first-order CRPS signal; symmetric heavy tails are
   second-order — the 1-D mechanism under-weights them, and z-dimensionality is
   not the binding factor.
6. **Production-config t(5) field run (SLURM 16606069): the field-scale
   tail-capability evidence** — see appended result below.

## Resulting gate structure (tests_p2/test_energy_score.py, all green)

- BINDING hook gates: ES↔Gaussian-CRPS unbiasedness; propriety direction (paired
  design); patch geometry (octave-4 = one patch); **1-D two-moment + skewness
  recovery (R10 condition 2)** through the exact production loss path; conv
  field-scale two-moment recovery (modulated-σ + flat-σ null, production patch
  config); trainer-pipeline identity with C1 (D4, std convention, hooks);
  determinism/multiscale; tail_q999 estimator vs closed forms (R10 condition 1).
- CHARACTERIZATION guard: 1-D t(5) kurtosis pinned to the measured regime
  [0.5, 3.0] (catches estimator regressions; NOT a capability claim — honestly
  labeled in-test).
- FIELD-SCALE SHAPE EVIDENCE (cannot fit the 900s Stop-hook budget: production net
  ≈ 15 min): the two SLURM runs above, recorded here pre-submission per R10
  ("BEFORE any training").

## Lessons (bar-design ledger)

- **Capacity is part of the toy's validity regime**: a mechanism gate run at
  toy capacity can fail (or pass!) for reasons the production configuration does
  not share — check the capacity axis before reading a toy as a mechanism verdict.
  (Fourth entry in the bar-design ledger; cf. the shape-null and normconv lessons.)
- **Train-eval toys with one observation per conditioning are memorization traps
  for proper-scoring-rule training** — held-out eval is mandatory (the ES's
  empirical optimum is the training delta).
- The prereg's own hedge ("β=1 may under-weight extreme tails") now has a measured
  1-D form: symmetric-tail signal is second-order in CRPS/ES. Whether it binds at
  production config is exactly what the arm measures at full scale (kurtosis
  PRIMARY), with the t(5) field run as the pre-training calibration point.

## APPENDED (pre-submission): production-config t(5) field result — BLOCKER FIRES

Job 16606069 COMPLETED (13:51 elapsed). Held-out trajectory (truth skew 0, excess
kurt 6.0): kurt 1.39 @500 → 1.04 @1k → 0.70 @1.5k → 0.55 @2k → 0.48 @2.5k →
0.48 @3k → 0.49 @3.5k → **0.49 @4k**; skew → 0.04 (correctly symmetric). A hard
plateau, flat for the final 1,500 annealed steps, BELOW even the 1-D limit (1.1) —
while the identical config on exp noise reached kurt 5.76/6.0 in the same budget.

The pre-registered rule above fires: this is a BLOCKER. The production
configuration recovers asymmetric (one-sided) tails and does NOT recover symmetric
heavy tails; Haar detail marginals are exactly symmetric (the D4-augmented
training law enforces w ~ −w) and heavy-tailed — the arm's PRIMARY object
(kurtosis) is precisely the statistic the toy shows the objective cannot deliver.
NO SUBMISSION. Full evidence and options for the reconvene:
log/2026-07-17-c3-blocker-symmetric-tails.md.
