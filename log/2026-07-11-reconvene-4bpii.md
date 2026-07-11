# 2026-07-11 — RECONVENE RULING: 4b′-ii gate-stop adjudicated; attempt 5 (self-conditioning) authorized as the FINAL generator attempt of phase 1c

Inputs: log/2026-07-11-prereg-4bpii (terminated at gate), results/smatched_4bpii.*.
The executor's execution was exemplary: the pre-named out-of-range condition fired,
no generation job was submitted, and the measurement upgraded a risk note into a
result.

## Ruling 1 — the gate-stop stands, and its measurement is adopted as a finding.

The conditioning drift is NOT additive noise: mimicking the observed attenuation
would need white-noise s ≈ 0.5+ while training covered ≤ 0.3, and the aligned
residual (~1.0) shows the generated coarse is as far from paired real coarse as
real samples are from each other — the damage is carried by structured,
off-manifold texture mismatch, to which the heads are far more sensitive per unit
amplitude than to white noise. The 4b′ risk note ("additive corruption may
mis-model modulation-flattening drift") is now measured, not hypothesized.
Scorecard: P-4b′ii-lever 50% not adjudicable (gate-stop); the gate firing was an
outcome neither side pre-weighted — logged as an unpriced branch, a prereg-design
lesson (weight the gate branches too).

## Ruling 2 — attempt 5 AUTHORIZED: drift-shaped corruption = self-conditioning. LAST generator attempt of this phase.

The measurement points at exactly one well-posed lever: make the training-time
conditioning distribution match the generation-time one — train (or fine-tune) the
heads on GENERATED coarse (or on alternative conditional samples of the coarse),
the alternative explicitly named and deferred in the 4b′ prereg. Implementation is
executor periphery (from-scratch vs fine-tune of the s=0.3 checkpoints; how the
generated-coarse pool is refreshed), with these cores:

- Single variable: the conditioning distribution. Augmentation (4a) stays frozen
  in; no other changes.
- No leakage: generated-coarse conditioning pools must be built from training-tile
  starts only; held-out tiles never touch the pool.
- Bars unchanged, two-tier: lever bar vs the model's own given-real-coarse ceiling
  (report it for THIS model); frozen G-1c dispersion + kurtosis bars vs real.
  Bounded-OOD binds. Kurtosis reported at all octaves; student-t branch remains
  pre-named for variance-passes/kurtosis-fails, after this readout.
- Pre-register the gate branches WITH weights (the 4b′-ii lesson).

**Pre-commitment (the stopping rule, agreed at reconvene level):** attempt 5 is the
final generator attempt of phase 1c. If the lever bar fails, the generator freezes
at "calibrated heads + measured compounding limit" and the project reshapes around
what is already banked — the P5 break, the causally-confirmed collapse law, the
drift measurements, and the validation architecture — which is a coherent
contribution on its own. No attempt 6 without a new phase and Andreas's explicit
sign-off.

## Predictions (reconvene, before attempt 5 design)

- P-attempt5-lever: **55%** — the mechanism now targets the measured failure
  exactly (train-test conditioning mismatch is THE textbook cause of exposure-bias
  compounding, and self-conditioning is its textbook fix); risk: training on own
  samples can feed back the flattening instead of correcting it if the pool is
  refreshed carelessly.
- P-project-dispersion (all octaves vs real): **40%.**
- P-kurtosis at oct 2 after attempt 5 alone: **15%** (unchanged; expect student-t
  to be the next conversation either way).

Pull-before-preregister; one variable; grep-verify; STOP at the readout.
