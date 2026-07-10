# 2026-07-10 — PRE-REGISTRATION (before submission): rung (iv) break & repair, gowerstreet

Pre-registered BEFORE the training job is submitted (per phase-1 rule). This is the
P5 break test (85%) and the P6 repair test (55%) from PLAN.md FROZEN CORE.

## Job to be submitted
- Runner `scripts/run_two_arms.py` via `scripts/train_gowerstreet.slurm`.
- **config_hash `71a9fd7e6d`**: field=gowerstreet, train_octaves=[2,3,4], gen_from=4,
  channels=[32,64,128], steps=10000, batch=32, lr=1e-3, n_heldout=64, sample_steps=80,
  seed=0, data=tiles_pnull.npz (330 gowerstreet tiles), coords=running_couplings.json.
- Resources: full H100 (`--gpus-per-node=1`, `--time<=2:59` → fast b1), account rrg-lplevass.

## Setup
Both arms are weight-tied conditional FM generators trained on gowerstreet octaves 2–4;
the FINEST octave (j=1) is the untrained "first extrapolated octave". Arm A gets no scale
input; arm B is conditioned on the stage-0 2-D running-coupling coordinate
[var_slope_j, kurtosis_j] (normalized /[1.5,13]) — INCLUDING octave 1's measured value
(measurement allowed, no training). Generation starts from held-out real coarse at octave 4.

## Primary discriminator (chosen from the rung-iii caveat)
**var_slope** (conditional-variance slope) at octave 1 — the clean conditional
non-Gaussianity. Marginal **kurtosis** is reported but treated as SECONDARY: the generator
cannot reproduce the inter-tile-amplitude pooling kurtosis (shown on GRF, P-null log), so a
kurtosis miss is not by itself evidence of P5. Estimates use bootstrap over held-out fields
(N=64) for sigma; "wrong" = >3 sigma AND >10% relative, per stage-0 conventions.

Quantified expectation (stage-0): real octave-1 var_slope = **1.305**; trained-octave
(2,3,4) mean = **0.915**. Arm A, having learned ~the pooled trained conditional, should
produce var_slope ~0.9 at octave 1.

## Pre-registered predictions
- **P4 (70%):** detail amplitude (power) at octave 1 within a few % of real, BOTH arms.
- **P5 (85%):** arm A var_slope at octave 1 is wrong — |Δ| > 3 sigma AND > 10% (expected
  ~0.9 vs 1.305, a ~30% under-shoot). If arm A is already right, P5 fails → K-T1 reframe.
- **P6 (55%):** arm B repairs ≥ 70% of arm A's octave-1 var_slope error:
  `repair = 1 − |vB − vreal| / |vA − vreal| ≥ 0.70`, AND does not degrade trained octaves
  2–4 (arm B var_slope within ~1 sigma of arm A / real there). If P5 holds but P6 < 30%
  after honest tuning → K-T2 (2-D conditioning insufficient; reconvene on mechanism).

## Gates
G-null already passed (rung iii). Verdicts use bootstrap CIs. Harvest with
`scripts/measure_generated.py --npz results/arms_generated.npz` (bootstrap added).

## Result
(job id + P4/P5/P6 verdicts filled after completion)
