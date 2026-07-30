# 2026-07-30 — PREREG: L1″ — the deconvolved base as a QUANTITATIVE
# PREDICTION TEST (R40; committed BEFORE any run; R12 throughout)

Instantiates R40 order 4 exactly: inference-only on committed checkpoints,
base filter = S_target/T_hat from the COMMITTED two-point measurement
(T_hat = geometric mean of the white- and colored-input transfer estimates;
per-ring spread as its uncertainty). Deployment split carried: ADJUDICATING
streams deconvolve the oct2-rescaled target (S_target and T both
deployment-available); ONE labeled oracle ablation deconvolves the measured
oct-1 real spectrum. Zero training. Artifacts → results_p2/l1pp_*.

## P-T — the registered premise (scored FIRST, before any peak number)

From l1pp_pt_prediction.json (computed mode-exactly from the ring model,
MC over the T uncertainty, committed with this prereg):

- **ADJUDICATING: C_pred = 0.7556, 3σ band ±0.0364 → interval
  [0.7192, 0.7920].** P-T LANDS iff the pooled 3-stream instrument C falls
  inside this interval. The NULL split keys on P-T (R40): NULL-PT-CONFIRMED
  vs NULL-PT-FAILED (a P-T failure = input-dependence beyond the measured
  range — itself a mechanism finding).
- Oracle ablation prediction (descriptive, not gated): C_pred = 0.7819
  ±0.0381 — the model says the oracle deconvolution should land AT the real
  value (0.7864).
- **Honest power note, stated now:** the deconvolved adjudicating base is
  nearly white (instrument C 0.995 — the transfer's blue tilt nearly
  cancels the target's blue tilt), so |C_pred − baseline| = 0.032 is 0.88×
  the band: P-T can land without proving movement. The directional
  question is carried by a REGISTERED DESCRIPTIVE LINE (not a gate):
  P(C_pooled − 0.7237 ≥ 3·0.0049) = 60 (executor).
- P(P-T lands): reconvene 70 (registered in R40); **executor 65** (the
  input-independence premise extrapolates from base colorings {0.74, 1.00}
  to ≈1.0-flat — the mildest possible extrapolation, but the two-point
  spread already carries ~3% wiggle).

## Streams and seeds (recorded now; keys/grngs disjoint from all committed)

| leg | ckpt | filter | key / grng | role |
|---|---|---|---|---|
| sandbox canary | sandbox A@7500 | filt_canary_sandbox (deconvolved, T from committed F2_A_e2e) | 3501 / 20260807 | must-not-regress |
| sandbox replay | sandbox A@7500 | WHITE | 2200 / 20260728 | identity gate |
| gow adj 1–3 | gow A@16000 | filt_adj (oct2rescaled ÷ T_hat) | 3601-3 / 20260808-10 | ADJUDICATING |
| gow oracle | gow A@16000 | filt_oracle (oct1meas ÷ T_hat) | 3605 / 20260811 | labeled DESCRIPTIVE |
| gow replay | gow A@16000 | WHITE | 2400 / 20260728+2 | identity gate |

No edge leg (not in the R40 order). Same plumbing (l1p_lib; white-base ==
sample_tbase equivalence already tested), same replay criterion (corr_min ≥
0.99, |ratio−1| ≤ 5e-3, rel ≤ 5e-2).

## Order of scoring (binding, as in L1′)

1. Canary marginals; KILL iff var_slope rel > 50% at any scored octave
   (2–4). WATCHED (context, not branch): oct2 kurtosis — L1′ read 17.3% vs
   the 15.5% bar; a redder base may push it further; still context unless
   catastrophe (R40). Result: PENDING.
2. Identity gates (replay criterion). Results: PENDING ×2.
3. **P-T first** (pooled 3-stream C, frozen stack_coloring, l1pp_pt.json
   written before any peak number): PENDING. Per-stream + oracle C
   alongside; directional line scored here.
4. Peaks vs A3 (verbatim: r = +15.290/+12.474%, SE_ref = 3.12/2.80%; pooled
   96-vs-32 bootstrap; Δν = (rν − eν)/hypot; #11 half-SE at-bar bands on
   the Δ=2 boundary; ν=1 descriptive #10; ONE fresh-PRNG disambiguation
   licensed). Result: PENDING.
5. Must-not-regress: marginal catastrophe / starlet-ℓ1 trained leg (frozen
   scorer, leg_idx 10) / parity_T < 3 / identity gates. nn_T WATCHED,
   descriptive, NOT interpreted (R40: the 4.045→3.33 movement on the
   wrong-direction coloring is noted and not interpreted). Result: PENDING.

## Branch table (mechanical; weights reconvene / executor)

| branch | rule | rec | exec |
|---|---|---|---|
| N-CURED | both pooled ci95 include 0; no regression | 12 | 8 |
| N-IMPROVED | not cured; both Δ ≥ 2; no regression | 38 | 27 |
| N-NULL | both Δ < 2 (mixed → worse); no regression; **split by P-T: NULL-PT-CONFIRMED / NULL-PT-FAILED** | 32 | 45 |
| N-REGRESSED | any must-not-regress failure | 8 | 10 |
| N-FLIPPED / gates | either ci95 entirely < 0; identity/infra (one resubmission per job) | 10 | 10 |

Executor's NULL-heavy reasoning, registered: L1′ was an inadvertent
dose-response test of the C→peaks link — ΔC = −0.12 (24.8σ of coloring
movement, the wrong direction) moved the peak excess by 0.02σ. If peak
counts were strongly spectrum-driven at octave 1, wrong-direction coloring
should have inflated the excess; it did not. The counter-consideration
(also registered): L1′ moved LOW-k content while the graininess mechanism
lives in the post-standardization Nyquist FRACTION, which L1″'s restoration
of low/mid-k would reduce — so a peak response remains possible. Hence 27
IMPROVED, not lower.

## Sequencing

Canary job → CPU kill check → main job (3 adj + oracle) → starlet leg →
score in the binding order → readout appended here → **STOP for reconvene
adjudication.** Budget ≲0.1 H100-h (two MIG jobs ≈0.05 expected).
