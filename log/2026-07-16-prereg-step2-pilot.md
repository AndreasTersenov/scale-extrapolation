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

## Deliverables

`results/pilot_validation.{json,png}` (one eye-readable panel per component),
RESULTS-phase1c.md §step-2, readout in chat. STOP-discipline: this step only; step 3
gets its own prereg.
