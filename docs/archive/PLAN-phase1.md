# PLAN — Toy phase: RG-consistent generation (break & repair)

**Phase 1.** Stage-0 passed (`PLAN-stage0.md`, `RESULTS.md`, 2026-07-10): N-body
conditional-wavelet statistics drift 3–7σ across adjacent octaves (GRF null clean at
|z|=0.15), the drift is **effectively 2-dimensional** (2 PCA components ≥85%), the
running-coupling coordinates are extracted (var_slope, kurtosis, cross-octave ρ — smooth,
monotonic), and the same drift shape appears in different sim physics (hf_pm_1024).
This phase runs the pre-registered break-and-repair experiment: P4/P5/P6 (+ new P13).
Context: `~/claude-notes/brainstorms/2026-07-09-dl-project-directions.md` (§D4, v1.1
predictions, v2/v3). **Positioning per novelty sweep:** the contribution is NOT
weight-tied wavelet FM (WFM arXiv:2605.16573 exists; 1D equivariance proof
arXiv:2605.17582; ancestor WC-RG arXiv:2207.04941) — it is the measured **non-Gaussian
break under extrapolation and its low-dimensional repair**.

**Stack (binding, per CLAUDE.md):** JAX, jax_flows FM core; `scaledrift/` (this repo,
gate-tested) is the measurement instrument — reuse, never reimplement; `wl_stats_torch`
for wavelet-L1/peaks validation at the numpy boundary. GPU via SLURM only
(rrg-lplevass GPU / def-lplevass CPU).

---

## FROZEN CORE — do not modify (pre-registered 2026-07-10)

### The system under test

Wavelet-factorized conditional flow-matching generator: per-octave conditional model
p(detail_j | coarse_j), **weights shared across octaves**. Two arms, identical except
one input:
- **Arm A (naive tying):** no scale information — the RG-fixed-point assumption taken
  literally.
- **Arm B (running couplings):** arm A + conditioning on the 2-D scale coordinate from
  stage-0 (the measured running-coupling values at octave j; exact parameterization is
  free periphery, the *dimensionality* ≤ 3 is frozen — that's the P9b bet being cashed).

Training: gowerstreet patches at ≤128² (octaves j ≤ j_train). Generation: recursive
coarse-to-fine to 256² and 512² (2 and 4 extrapolated octaves). Controls: GRF_HF
(null — both arms must extrapolate it perfectly), lognormal (analytic control).
Transfer: hf_pm_1024, zero retraining.

### Evaluation (frozen)

Per-octave, on generated vs held-out real fields, with bootstrap CIs, measured by the
gate-tested `scaledrift` instrument: (a) power-spectrum amplitude/slope; (b)
conditional-W1 drift profile; (c) running-coupling scalars (var_slope, kurtosis,
cross-octave ρ); (d) wavelet-L1 + peak counts (`wl_stats_torch`). "Wrong" = >3σ AND
>10% relative, per stage-0 conventions.

### Pre-registered predictions (updated confidences, logged 2026-07-10)

- **P4 (70%):** power-spectrum slope/amplitude extrapolate within a few % in the first
  extrapolated octave, BOTH arms (spectra are cheap; weight-tying enforces them).
- **P5 (85%, raised from 60% — stage-0 forces it):** arm A's non-Gaussian statistics
  are wrong (>3σ, >10%) in the first extrapolated octave, worsening with octave depth.
  A weight-tied net without scale input cannot represent conditionals that measurably
  drift.
- **P6 (55%):** arm B repairs ≥70% of arm A's non-Gaussian drift error in the first
  two extrapolated octaves, without degrading trained octaves (≤1σ change there).
  **This is the paper's load-bearing result.**
- **P13 (new, 55%):** zero-retrain transfer — arm B with couplings *measured on
  hf_pm_1024's coarse octaves* (measurement allowed; no training) repairs >40% of the
  drift on hf_pm generation. Tests "field-general method" vs "gowerstreet fit".
- **P-null (90%):** both arms extrapolate GRF_HF with all metrics consistent with real
  GRF (the end-to-end null gate — if this fails, the pipeline is buggy, not the physics).

### Gate / kill / reframe criteria

- **G-null:** P-null must pass before any real-field verdict is claimed (analog of
  K-M1a).
- **K-T1(D4) — reframe, not kill:** if P5 fails (arm A just works), the break-and-repair
  paper doesn't exist; the fallback claim is "certified extrapolation via weight-tying"
  — weaker vs WFM; STOP and reconvene on framing before more compute.
- **K-T2(D4):** if P5 holds but P6 fails (<30% repair after honest tuning), the 2-D
  conditioning hypothesis is wrong despite P9b — reconvene; next lever is conditioning
  *mechanism* (e.g. per-octave FiLM vs input concat), not architecture scale.
- **Budget: 15 H100-days cap.** Karpathy ladder, ordered and committed at each rung:
  (i) single-octave conditional FM overfits one field; (ii) two-octave recursion on one
  field; (iii) GRF end-to-end null (P-null); (iv) full arms A/B on gowerstreet;
  (v) transfer (P13). No multi-hour job before the previous rung is green.

### Out of scope (do NOT build)

D6b single-realization mode (explicitly deferred until P4–P6 verdicts); >512²;
non-cosmology demo fields (paper-stage decision); TRACE/typicality certificates;
comparisons against arXiv:2507.01707 beyond citing (paper-stage); any per-field
fine-tuning at eval time.

---

## FREE PERIPHERY — implementer's choice

Conditional-FM parameterization, how the 2-D coupling coordinate enters (embedding,
FiLM, concat), patch/boundary handling, recursion details (sampling per octave),
optimizer/schedule/curriculum, exact train-octave split, how many fields per batch,
Haar vs db4 for the *generator* (the instrument stays Haar per stage-0), checkpoint
cadence, SLURM shapes (consult rorqual-jobs; MIG slices were sufficient for D1 studies).

## Backpressure additions (existing gates stay)

New required tests before training: (a) wavelet synthesis round-trip through the
generator's transform at machine precision; (b) recursion determinism (fixed seed →
identical field); (c) single-octave overfit gate as executable test; (d) `scaledrift`
suite untouched and green (it is the instrument — any modification needs written
justification in log/). Every SLURM job logged pre-submission with config hash +
expected outcome.

## Logging & deliverable

Standard log discipline. Deliverable: `RESULTS-toy.md` — P4/P5/P6/P13/P-null verdicts
with numbers, the per-octave drift profiles of both arms (the fan-out figure, generated
version), the repair-fraction table, transfer results, honest limits. Written for the
reconvene; assumes PLAN.md, not the code.
