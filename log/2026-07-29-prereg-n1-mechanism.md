# 2026-07-29 — PREREG: Phase N1 mechanism profiling (NIGHT-ORDERS-2;
# committed BEFORE any probe computes)

Authorization: NIGHT-ORDERS-2.md (Andreas-authorized overnight run;
supersedes R37 order 2's training gate via pre-delegated GATE-N1). Framing:
R37's field-not-scale reframe — the excess is a field-level texture mismatch,
profiled FIELD-FIRST at trained scales on gowerstreet. Zero training in N1.

## Substrate (all committed artifacts)

- PRIMARY (trained scales): gen = f2_test_gen.npz F2_gowA_e2e (32 maps,
  c1t gowerstreet A@16000, F2-sampled, conditioned on the real test coarse —
  gen map i pairs with real tile i); real = arms_c1t_gowerstreet.npz real
  (32 test tiles). Committed references on these exact stacks: peak excess
  +14.48%±3.12 @ν2.5, +11.07%±2.76 @ν3.0; nn_T 4.045 (f2_verdict echo).
- Edge continuity leg: stage0_p3_gen.npz gen_A vs its real
  (+12.45%±3.09 / +12.94%±3.00 committed).

## Probes, with registered expectations (executor lines; reconvene's GATE
## line is already registered in the orders)

1. **Coloring index C (THE gate probe; new estimator → tests-first).**
   Per octave j ∈ {1,2,3}: pywt Haar detail planes (frozen
   parity_localization.dwt_levels), per map: 2-D FFT power of each of the 3
   detail channels, summed over channels, binned in integer-|k| annuli;
   C = mean power in LOW band / mean power in HIGH band, with bands FIXED
   IN THE VALIDATION TEST before real data is touched:
   low = 1 ≤ |k| ≤ N/8, high = N/4 < |k| ≤ N/2 (N = detail plane size:
   64/32/16 at j=1/2/3). SEs: tile bootstrap, n_boot 5000, seeded.
   Validation test (tests/, env.sh stack): (a) C discriminates synthetic
   white vs power-law-colored noise at z > 10 with the fixed bands at all
   three plane sizes; (b) C is invariant under flips/90° rotations
   (machine precision); (c) doubling tiles shrinks the bootstrap SE ~√2.
   **GATE-N1 (verbatim from the orders): CONFIRMED iff at ≥2 octaves of
   {1,2,3}: C_real − C_gen ≥ 3·hypot(SE_real, SE_gen) with C_real > C_gen.
   Ambiguous (exactly 1 octave, or within 0.5·SE of the 3σ line) = NOT
   CONFIRMED.** Reconvene line: P(CONFIRMED) = 60. **Executor line:
   P(CONFIRMED) = 55** — the ν=1 surplus (+22.9/+38.0% edge; grainy texture)
   points at high-k detail excess, but 3σ at ≥2 octaves from 32-tile
   bootstraps is demanding, and the sandbox sign flip warns the premise may
   be field-specific in strength.
2. **Peak morphology (descriptive).** Height-resolved excess at
   ν ∈ {2.5, 3, 3.5, 4} (bootstrap CIs, frozen peaks_count) + smoothing
   response: Gaussian smoothing σ_s ∈ {0.5, 1, 2} px applied to BOTH stacks
   before counting (standardization inside peaks_count), excess at
   ν=2.5/3.0 vs σ_s. Lines: P(1-px smoothing cuts the ν=2.5 excess to
   < half its unsmoothed value) = 60 (grainy-surplus reading);
   P(excess at ν=4 ≥ excess at ν=2.5, growing with height) = 45.
3. **Octave attribution — hybrid transplant (descriptive).** For each
   octave j ∈ {1,2,3,4}: take the real pyramid of tile i, substitute the
   GENERATED detail coefficients of map i at octave j only, reconstruct
   (frozen dwt/idwt machinery), peak excess vs real at ν=2.5/3.0 with
   bootstrap CIs. The per-octave excess contribution. Line: P(j=1 carries
   the largest single-octave ν=2.5 excess) = 55.
4. **Spacing co-location (descriptive).** nn_T (frozen instrument,
   K_real=174, phase-A edges) on each single-octave hybrid stack from
   probe 3: does the spacing residual concentrate at the same octave as the
   count excess? Line: P(argmax-octave of |count excess| = argmax-octave of
   hybrid nn_T) = 50 (genuinely uncertain; nn is the statistic four cures
   never moved).
5. **Reference replication (one MIG job, ~0.1 H100-h).** TWO fresh-PRNG F2
   regenerations of the trained leg (keys 2501/2502, group rngs
   20260731/20260732 — disjoint from every committed stream) + a REPLAY
   gate: the committed stream (key 2400, grng 20260728+2) must reproduce
   the committed F2_gowA_e2e under the corrected-G2 criterion (corr_min ≥
   0.99, amplitude-ratio mean within 5e-3 of 1, rel max-abs ≤ 5e-2 — the
   G2 lesson, now the standing cross-run criterion). Purpose: N3's
   reference becomes a 3-stream mean with priced generation-stream
   variance. Line: P(both replicate ν=2.5 excesses within 2·SE_committed
   of +14.48%) = 80.

## Discipline

CPU probes run in-session (score-phase precedent; minutes-scale on 32+32
maps — infra discretion per RIDER §4, documented); the replicate job is
SLURM MIG (JOBS.md entry pre-submission). Tests-first for C before any real
data touches it; probes 2–4 use frozen instruments only. All probe outputs →
results_p2/stage1_p3_*.json. Gate verdict appended to this log with the
mechanical rule applied verbatim; then N2 (CONFIRMED) or the
diagnosis-and-report path (NOT CONFIRMED), per the orders. R12 throughout.

## Validation note (appended BEFORE any real data was touched)

The first discrimination test used power-law-colored FIELDS and FAILED:
level-1 Haar details of a red field measured C = 0.890 vs white 1.003 — the
Haar high-pass response (~k²) cancels a P~k⁻² field spectrum and decimation
aliases near-Nyquist power into low plane-k, so "colored field ⇒ colored
details" is FALSE at fine octaves. Since the gate compares real-vs-generated
DETAIL planes directly (and the generative mechanism draws details from the
base directly — the N2 lever colors the base, i.e. detail space), the
validation synthetic was corrected to detail-space coloring: fields built by
inverse DWT from planted white vs |k|⁻¹-colored detail planes. All four
tests pass (discrimination z > 10 at octaves 1–3, D4 invariance, white
baseline ≈ 1, SE ~√2). Bands unchanged from the prereg. Lesson recorded for
the instrument ledger: coloring statements about detail coefficients must be
made IN detail space; field-space spectral intuition inverts under the
wavelet transform at fine octaves.
