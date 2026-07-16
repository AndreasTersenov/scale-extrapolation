# 2026-07-16 — PRE-REGISTRATION: Step 2 — the INVERTED validation pilot

Per the reconciliation ruling (phase-1d program, step 2). The pilot runs the paper's
validation architecture ON THE FROZEN FAILING GENERATOR (the 4a model — the frozen
reference; attempt-5 models are the tested negative, not the reference). Inversion:
the protocol's product is that it CATCHES the failure we know is there — and passes
where the generator is known-good. Committed before the job submission.

## Component 1 — slide-the-edge at a second position (one MIG job)

Train octaves 3–5 → generate from real octave-5 coarse (4×4) → the first extrapolated
octave is 2, where GROUND TRUTH exists. Frozen generator config verbatim (NLL head +
D4 augmentation, FiLM, 20k steps, seed 0; no corruption, no self-conditioning).
Octave-5 couplings already in `running_couplings.json` ([0.421, 1.440]).
Job `scripts/arms_edge2.slurm`, **config_hash 577d20bb5e** → `results/arms_edge2.npz`.
Scored at octaves 2–5 with extrapolated octave = 2.

**AMENDMENT (2026-07-16, before any readout — job 16401043 FAILED at startup):** the
frozen U-Net (3 pooling levels) cannot ingest octave-5 pairs (4×4 < 8×8 minimum; the
original config's smallest input, octave 4 at 8×8, just fits — an architecture
constraint I failed to check pre-submission). Second edge REDEFINED within the frozen
architecture: **train octaves {3,4} → generate from real octave-4 coarse → first
extrapolated octave = 2** (ground truth exists). Honest confound, stated: edge 2
trains on TWO octaves where edge 1 trained on three — the weight-tying pool differs;
P-edge therefore tests edge-position consistency up to that confound. All P-edge
numbers/criteria unchanged (rel-err agreement within factor 2 of edge-1's 36.5%).
Amended job: **config_hash f77178e6cf**, same outputs.

- **P-edge (the protocol claim):** the one-octave extrapolation error is a property of
  the OPERATOR, not the particular edge: arm B's relative var_slope error at its first
  extrapolated octave agrees between edges within a factor of 2.
  Edge 1 reference (4a run, frozen): arm B oct-1 rel err = |0.709−1.117|/1.117 = 36.5%
  (arm A: 39.1%). So edge 2 (oct-2, trained 3–5) predicts rel err in [18%, 73%] (B).
  **Confidence: 65%.** Also reported: arm A same check; trained-octave table.

## Component 2 — self-consistency of generated couplings (CPU, existing + new npz)

Ground-truth-free check (deployable where no truth exists): measure the couplings ON
the generated fields (the frozen scorer already does) and compare, per octave, with
the coupling value the generator was CONDITIONED on (arm B) / the real curve.
Deviation z = |vs_gen − vs_cond| / SE_gen.
- **Detection criterion:** the check FLAGS (z ≥ 3) at the extrapolated octave of the
  4a run (oct 1) and of the edge-2 run (oct 2). **Confidence: 90%** (known failure).
- **Specificity criterion:** at the best trained octave (oct 4 of the 4a run) it does
  NOT flag (z < 3). **Confidence: 60%** (oct-4 var_slope 0.475 vs conditioned 0.44 —
  wait, conditioned values are the REAL couplings 0.532·norm; measured-on-gen 0.475
  ± 0.019 → z ≈ 2.4 vs the real value — borderline; honest uncertainty).

### Pre-computation clarification (added before any component-2 number was computed)

The component-2 text conflates two references. Defined now, both computed:
- **SC-a (deployable):** z vs the CONDITIONING curve (`running_couplings.json`) — what
  a production pipeline has when no truth exists. Calibration check pre-registered:
  SC-a applied to the REAL held-out couplings themselves; if real fields flag, SC-a
  needs curve-vs-population recalibration and THAT is reported as a protocol finding
  (the stage-0 curve and the held-out population are known to differ, e.g. 1.305 vs
  1.117±0.051 at oct 1).
- **SC-b (truth-referenced):** z vs the held-out-real curve — this carries the
  pre-registered detection (≥3 at extrapolated octaves, 90%) and specificity
  (oct-4 < 3, 60%) criteria, matching the numbers quoted above.

## Component 3 — held-out statistics (never used in design; CPU)

On the 4a run's held-out-start generated fields vs real (64 maps), per octave where
applicable. Estimators get validation tests BEFORE use (`tests_wfm/test_heldout_stats.py`,
CLAUDE.md backpressure): identical-stack z≈0; GRF-vs-GRF null |z|<3; gross-difference
detection.
- **Wavelet-L1** per octave: mean |detail coefficient| (amplitude-like, close to what
  P4 tracks). Prediction: mostly PASSES (|z|<3) at trained octaves — **60%**.
- **Scattering coefficients** (kymatio Scattering2D J=4 L=4, the rival school's
  instrument family): per order-2 channel, mean log-coefficient over maps/positions,
  z via bootstrap-over-maps (200). Summary statistic (pre-registered): fraction of
  order-2 channels with |z| ≥ 3, and median |z|. Prediction: FLAGS (fraction ≥ 25%)
  — the under-dispersed fine-scale texture is exactly what order-2 scattering sees.
  **Confidence: 75%.**
- **Concordance claim (the pilot's headline):** the held-out battery flags the same
  failure the tracked couplings flag, without having been designed against it — i.e.
  a pipeline using ONLY our audit protocol would have rejected this generator.
  **Confidence: 80%.**

## RESULT (2026-07-16, harvested same day)

Edge job 16401105, hash f77178e6cf verified. Full numbers `results/pilot_validation.json`.
- **P-edge: PASS** (arm B rel err 36.5% → 59.1%, ratio 1.62; arm A 39.1% → 26.8%,
  ratio 1.46 — both within factor 2; prediction 65% HIT). The extrapolation error is a
  property of the operator, not the edge — up to the stated 2-vs-3-octave confound.
- **Self-consistency SC-b: detection HIT** (extrapolated octave z=7.9, ≥3 as
  predicted at 90%) and **specificity HIT** (oct 4 z=0.7 < 3; 60% prediction). It also
  flags the trained-octave compounding shortfall (oct 2 z=7.1) — a sensitivity bonus.
- **SC-a (deployable variant): calibration check FAILED as pre-flagged** — REAL fields
  flag against the stage-0 curve at every octave (z 3.4–4.0). Protocol lesson, now
  measured: a deployable self-consistency check must use population-calibrated
  reference bands (curve-fit + population variance), not bootstrap SEs alone.
- **Held-out battery: detection HIT, loudly** — scattering order-2: 98% (A) / 100% (B)
  of channels flag, median |z| ≈ 9–11 (predicted ≥25% at 75%). Wavelet-L1: flags oct
  1/2/4 for arm A (my "mostly passes" 60% prediction MISSED — L1's tight bootstrap
  bars flag even few-% amplitude offsets that P4's 10% criterion calls PASS).
  **Concordance headline (80%): HIT** — an audit-only pipeline (no ground truth at the
  extrapolated octave beyond the curve, plus held-out stats) would have rejected this
  generator.
- **Bonus catch:** arm B's exp-head OOD amplitude blow-up RETURNS at edge 2 (oct-1
  detail_std 2.198 vs real 0.743; oct-2 +37%) — the narrower trained coordinate range
  {3,4} re-triggers what augmentation had bounded at edge 1; the amplitude column of
  the battery catches it. Bounded-OOD is range-dependent, not solved once.
(Scattering note: one NaN-channel warning in the log; summary stats computed on valid
channels — nanmedian hardening deferred to the paper-grade version.)

## Deliverables

`results/pilot_validation.{json,png}` (one eye-readable panel per component),
RESULTS-phase1c.md §step-2, readout in chat. STOP-discipline: this step only; step 3
gets its own prereg.
