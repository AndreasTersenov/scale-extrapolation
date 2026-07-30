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

---

## READOUT (appended; verbatim from l1pp_* artifacts)

1. **Canary: PASS, all six bars** (var_slope 5.5/6.2/12.9%). The watched
   oct2 kurtosis IMPROVED under the deconvolved base: 13.8% vs bar 17.9%
   (L1′ context miss was 17.3% vs 15.5%).
2. **Identity gates: both PASS** (canary corr_min 0.99999999 / rel
   1.74e-3; main corr_min 0.99999999 / rel 6.14e-3).
3. **P-T (scored first): LANDS.** C_pooled = 0.7414 ± 0.0048, inside the
   committed interval [0.7192, 0.7919] (prediction 0.7556). The
   REGISTERED DIRECTIONAL LINE ALSO FIRES: C − baseline = +0.0177 =
   +3.6σ_instr — real movement toward real, in the predicted direction.
   Per-stream: 0.7405 / 0.7234 / 0.7609 (stream scatter ~0.019 — larger
   than the instrument SE; generation-stream variance in C is real;
   pooled adjudicates). **Oracle ablation: C = 0.7815 ± 0.0103 vs its
   committed prediction 0.7819 — 0.04σ.** The multiplicative transfer
   model survives its third input point and lands the oracle deconvolution
   ON the real value (0.7864, ~0.5σ). The system identification of R40 is
   quantitatively VALIDATED. Line scorecard: P(P-T) rec 70 / exec 65 —
   both HIT; directional 60 — HIT.
4. **Peaks vs A3: NULL again.** Pooled +13.82%±2.92 (ν2.5, Δ = 0.34) /
   +13.66%±2.68 (ν3.0, Δ = −0.31). Streams +10.58/+16.24/+14.64 (ν2.5).
   **Oracle stream: +15.75%±3.30 / +12.91%±2.98 — coloring restored to the
   real value, excess UNCHANGED.** ν=1 pooled +19.92% (#10). No at-bar, no
   flip.
5. **Must-not-regress: NO regression.** No catastrophe; parity_T 1.70;
   starlet trained leg PASS (leg 10); identity PASS. WATCHED, descriptive,
   not interpreted (R40): nn_T = 2.90 — first sub-3 reading on
   adjudicating trained-leg streams; the across-coloring sequence is
   4.045 (white) → 3.33 (wrong-direction) → 2.90 (corrected), reported as
   numbers only. coef_T_oct1 = 2.69.
6. **BRANCH (mechanical): N-NULL-PT-CONFIRMED.** Peak-weight scorecard:
   reconvene modal IMPROVED 38 → MISS; executor modal NULL 45 → HIT (the
   registered L1′-dose-response reasoning predicted exactly this).

### The finding (verdict-grade, stated plainly)

Three coloring doses now bracket the question: C_gen ∈ {0.603, 0.724,
0.741–0.782} all produce the SAME ~+14–15% peak excess (nine streams
across L1′/L1″ within ±2 SE of each other; the oracle stream at
real-value coloring included). **The oct-1 peak excess on the real field
is INDEPENDENT of the oct-1 detail power spectrum at fixed weights.**
M-GRAIN's spectral form is refuted as the peak mechanism; the pixel-scale
surplus (0.5-px smoothing kill, N1) must live in the PHASE/higher-order
structure of the generated fine details, which no base-spectrum lever
reaches through these weights. The inference-only spectral lever family is
EXHAUSTED for the peak tier — and, separately, it is now a validated,
deployment-available CALIBRATION TOOL for the coloring/whiteness defect
itself (P-T + oracle: the practitioner can set C to target without
retraining). Both facts are paper material: the first scopes the peak
tier's mechanism to training-side structure (the L2 family, or a declared
boundary); the second is the audit-guided-design exhibit completed.

### Execution accounting

Two MIG jobs (17951572 canary ~8 min, 17952979 main ~11 min) ≈ 0.05
H100-h; order-set spend ≈ 0.11 of ~4; zero training; suites green; the
P-T-before-peaks order enforced structurally (l1pp_pt.json written first).

**STOP — reconvene adjudication.**
