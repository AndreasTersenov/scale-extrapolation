# RESULTS-phase1c — Step 1: the Gaussian-NLL detail head (G-1c)

**For the reconvene; assumes PLAN.md (phase 1c).** Pre-registration:
`log/2026-07-11-prereg-1c-nllhead.md` (design, sampling procedure, bars, predictions —
committed before submission). Run: job 15738958, config_hash 4866f6e236 (matches),
gowerstreet arms A/B, FiLM, NLL head both arms, 10k steps, 2k-granularity checkpoints.
Scored with the untouched scaledrift instrument, bootstrap over 64 held-out fields.

## G-1c VERDICT: **FAILED — STOP, reconvene** (per the frozen gate)

- **Dispersion bar** (each arm, |var_slope − real| ≤ 1 combined σ at octaves 2, 3, 4
  simultaneously): **FAILED.** oct 2: arm A z≈7.9, arm B z≈7.3; oct 3: z≈5.3 / 6.9;
  oct 4: arm A z≈1.0 (passes alone), arm B z≈2.4.
- **Kurtosis check** (within 2σ at trained octaves): **FAILED**, z≈5 at oct 2–3 (real
  6.7 / 3.9; generated 1.6–3.0 — far too tame).
- **Student-t fallback: NOT triggered** — it is pre-named only for the
  variance-passes/kurtosis-fails branch; the variance bar itself failed.
- **GRF null** (job 15738957, hash dc676a1f01 verified): **also NOT cleanly
  preserved.** Extrapolated-octave var_slope: arm A −0.012±0.004 vs real 0.001±0.002
  (z≈3.2 — a small spurious modulation, ~1% of the real-field signal but outside the
  pre-registered bootstrap-consistency criterion; arm B passes, z≈1.1). Detail
  amplitude at the extrapolated octave is inflated on GRF too: arm A +14%, arm B
  **+43%** — arm B's placeholder coordinate is likewise out-of-range at j=1, so this
  independently corroborates the exp-head OOD instability (finding 3) on a different
  field. (`results/pnull_nll_score.json`.)
- Trained-octave detail amplitude: fine (≤6%, both arms) — the failure is purely in the
  conditional MODULATION, as before.

| oct | metric | real | arm A | arm B |
|---|---|---|---|---|
| 2 | var_slope | 1.020±0.038 | 0.711±0.009 | 0.737±0.008 |
| 3 | var_slope | 0.801±0.037 | 0.594±0.013 | 0.531±0.012 |
| 4 | var_slope | 0.532±0.033 | 0.492±0.024 | 0.444±0.017 |
| 2 | kurtosis | 6.69±0.80 | 2.63±0.11 | 2.99±0.09 |
| 3 | kurtosis | 3.94±0.68 | 2.04±0.14 | 1.56±0.12 |
| 1 (extrap) | var_slope | 1.117±0.051 | 0.612±0.005 | **0.021±0.003** |
| 1 (extrap) | detail_std | 0.743 | 0.775 | **1.275** |

Figures: `results/g1c_verdict.png` (the bars, visually), `results/nll_diagnosis.png`
(the mechanism), `results/nll_sigma_maps.png` (the learned σ map).

## Diagnosis (exact decomposition, no sampling — `scripts/diagnose_nll.py`)

Per coarse-quantile bin, generated variance = Var(μ|bin) + E[e^{2g}|bin] exactly. The
split at the final checkpoint:

1. **The variance head starved; the mean ate the variance.** At every trained octave
   the σ-term is tiny and FLAT (noise-part slope: 0.03 / 0.00 / 0.00 at oct 2/3/4)
   while the μ-part alone carries ~the entire real variance profile
   (Var(μ|bin) ≈ Var(real|bin) at moderate coarse). A true conditional mean cannot
   have the data's full variance — **μ(coarse) memorized the 322-tile coarse→detail
   mapping** (finite-data interpolation), residuals shrank, and the proper NLL dutifully
   drove e^{2g}→~0. The sampler is then effectively deterministic-texture-given-coarse,
   and its total modulation caps at the memorized map's (~70% of real; the deficit is
   concentrated in the brightest bins).
2. **The collapse-with-training law, third independent confirmation.** Implied
   var_slope at oct 2 vs steps: 0.97 (2k — nearly the real 1.02) → 0.75 (10k);
   monotone decay, same shape as the L2-CFM dispersion-collapse curve. The law
   generalizes: *it was never about where the variance lives (ODE pushforward, penalty
   target, or an explicit NLL head) — finite-data memorization of the conditional MEAN
   eats the conditional variance in whatever channel carries it.* Note: even the best
   early checkpoint would not pass the bar simultaneously (oct 3 at 2k: 0.63 vs real
   0.80, ~4–5σ).
3. **Exponential head is OOD-unstable (arm B, extrapolated octave).** With the
   octave-1 coordinate (outside the trained FiLM range), e^g anti-modulates
   (noise-part slope −0.29) and inflates amplitude +71%; arm B's oct-1 var_slope
   collapses to 0.02. Any future variance-explicit design must bound/regularize the
   OOD behavior of exp(g) under the conditioning it is supposed to extrapolate with.
4. **Nuance:** the σ map DOES modulate spatially in the right places (it tracks the
   environment; `nll_sigma_maps.png`) — it knows WHERE, but its overall level
   collapsed — it lost HOW MUCH.

## Prediction verdicts (pre-registered in the prereg)

P-NLL-var 65% → **failed**. P-kurt 55% → **failed** (and the Gaussianization
hypothesis remains UNTESTED: with σ collapsed, generated kurtosis measures the
memorized μ, not the conditional). GRF-null 90% → **failed** (arm A z≈3.2 spurious
slope; extrapolated-octave amplitude inflated both arms — the head's miscalibration is
not specific to non-Gaussian fields). Amplitude-undegraded 85% → passed at trained
octaves.

## Honest limits

Single seed / single λ-free config (per prereg); adjudicated on the final checkpoint as
pre-registered (the checkpoint curve is reported above and does not change the verdict);
the unit-scale toy passed while the real fields failed — the toy had no
memorization-capacity mismatch (its σ carried all signal; the real fields' mean can
substitute for variance), which is exactly the gap the diagnosis exposes.

## Attempt 4a — the diagnosis test (D4 augmentation alone; ruling of 2026-07-11)

Pre-registered `log/2026-07-11-prereg-4a-augment.md`; job 15744601, hash 1e61bd812a
verified. Figure: `results/signature_4a.png`.

**Readout: the collapse is GONE within the 20k horizon.** Implied var_slope (oct 2,
arm A): baseline 0.97@2k → 0.75@10k (collapse); augmented 0.94@1k → dip 0.85@2k →
**0.97–1.01 flat from 4k to 20k** (real: 1.02). The σ-head's share of generated
variance: baseline crashes 81%→17%; augmented holds **86–93% throughout** — the
variance channel stays alive. Training loss floors at 1.26 vs 0.43 un-augmented (less
memorization headroom), as the diagnosis predicts.

**Signature adjudication — honest rule caveat.** The LITERAL pre-registered onset rule
("first checkpoint ≥0.10 below the curve's maximum") fires at 2k — on the warm-up dip
that PRECEDES the curve's own peak (6k), which cannot be a collapse onset; the rule was
written against curves that peak first (true of the baseline, where both readings
coincide at 4k). Under the rule as written: REFUTED. Under its manifest intent (drop
from the RUNNING peak): censored >20k → CONFIRMED in the strongest form. Both readings
are computed and reported (`scripts/signature_4a.py`); the visual fact is not in
dispute. **Stopped at the readout; the reconvene adjudicates the rule.**

**Second finding (descriptive, frozen scorer on the 20k fields —
`results/arms_aug_score.json`): the head is fixed but the RECURSION now limits.**
End-to-end (coarse-to-fine from octave 4), oct-2 var_slope measures 0.746±0.014 vs
real 1.020±0.038 (~7σ) even though the head's conditional response GIVEN REAL COARSE
is 0.96. The deficit has moved: no longer the head's collapse, but octave-to-octave
compounding — each octave conditions on generated (slightly flattened) coarse, and the
flattening accumulates (oct 4: 0.475 vs 0.532 mild → oct 2: 0.75). Note for any 4b
decision: the two-stage residual-fit targets head calibration, not compounding.
Also descriptive: arm B's OOD amplitude blow-up is GONE under augmentation (oct-1
detail_std 0.689–0.729 vs real 0.743, was +71%/+43%); P4 at oct 1 PASSES (1.9%/7.2%);
P5 break intact (z≈8); kurtosis still fails (~3.0 vs 6.7 at oct 2).

## Attempt 4b′ — conditioning robustness (ruling of 2026-07-11, post-4a)

Pre-registered `log/2026-07-11-prereg-4bprime-condrobust.md`; jobs 15753842/15753843,
hashes 4f5bbe7b0f/9f56a059ad verified. Figure: `results/readout_4bp.png`.

**LEVER BAR: FAILED** (both s_max, both arms; binding at octave 2). End-to-end oct-2
var_slope vs the fixed 4a ceiling: arm A 0.709±0.012 (s=0.1) / 0.773±0.018 (s=0.3) vs
0.956±0.035 → z≈6.7 / 4.7; arm B 0.700/0.701 vs 1.016±0.039 → z≈7.6. Octaves 3–4 pass
(already at ceiling). **Project bar: also failed** (oct 2–3 vs real). Ceiling-binds
branch: NOT reached (it presupposes the lever passing). Bounded-OOD: still satisfied
descriptively (oct-1 amplitude 0.700–0.736 vs real 0.743).

**The decomposition is the finding.** The pre-registered attenuation check inverted:
corruption training RAISED the heads' own given-real-coarse ceilings — arm A oct 2:
0.956→0.996 (≈ real 1.020), oct 3: 0.683→0.748 (real 0.801); similar at s=0.1 — a free
regularization gain that also moved kurtosis (arm A s=0.3: oct-3 kurtosis 3.88±0.73 vs
real 3.94±0.68 — at real; oct 2: 3.90 vs 2.99 before, real 6.69). Yet end-to-end did
not move (s=0.1 slightly WORSE than no corruption: 0.709 vs 0.746): the
end-to-end-minus-ceiling gap widened. **The robustness was trained but is never
engaged: generation samples at s=0, the "trust the conditioning fully" mode.** The
recipe's own literature applies non-zero conditioning noise at INFERENCE matched to the
drift; the pre-registered s_gen=0 choice (made to avoid a hand-tuned knob) plausibly
discarded the operating mode the mechanism needs.

**Prediction verdicts:** P-lever 60% → failed. P-s0.3>s0.1 at oct 2 → correct
(0.773 vs 0.709, arm A). P-project 20% → failed as expected. P-ceiling-binds → n/a.

**For the reconvene (observations only, stopped at the readout):** (1) the cheapest
next probe reuses the EXISTING s=0.3 checkpoints with inference-time matched corruption
(corrupt the generated coarse at a pre-registered s_gen>0 and expose it) — zero
training, one generation job; principled per the cascaded-diffusion literature, though
it must be pre-registered as a sampling-procedure change, not tuned. (2) The ceiling is
NOT immovable (corruption training raised it toward real at every octave) — relevant to
the oct-3 ceiling-binds concern and possibly to the kurtosis branch. (3) Original-4b
revival remains pre-named but its premise (head under-response binding) is now weaker
at oct 2 (own-ceiling ≈ real) and the binding failure is squarely the recursion.

## Attempt 4b′-ii — inference-time matched corruption: STOPPED AT ITS OWN GATE

Ruling `log/2026-07-11-reconvene-4bprime.md` authorized engaging the trained
robustness at inference (existing s=0.3 checkpoints, zero training), gated on a
pre-generation drift measurement: primary adjudication only at s_gen = s_matched, and
**stop if s_matched > 0.3**. The measurement (`scripts/measure_drift_smatched.py`,
`results/smatched_4bpii.{json,png}`) fired the stop: white-noise corruption of real
coarse cannot reproduce the measured end-to-end attenuation anywhere in the trained
range (head reaches only 0.86/0.85 at s=0.35 vs targets 0.773/0.701 at oct 2; the only
crossing at all is arm A oct 3 at 0.335 > 0.3). Aligned generated-vs-real coarse
residual: 0.91–1.15. **No generation was run.**

**What this measured (the round's second methods datum):** the conditioning drift is
NOT additive noise — per unit amplitude, white noise is far less damaging to the
head's modulation than the actual drift (white noise averages out over the receptive
field; the drift's structure does not). Matched white-noise corruption is therefore
the wrong noise model for anti-compounding here; a drift-shaped corruption (self-
conditioning on generated coarse, or corruption by an alternative conditional sample)
is the measurement's direct suggestion — reconvene's call. The 4b′ regularization gain
stands (the s=0.3 heads' own ceilings ≈ real, now with SEs: A oct-2 0.996±0.041,
oct-3 0.748±0.026).

## Attempt 5 — self-conditioning (FINAL generator attempt): LEVER FAILED, branch B5 — GENERATOR FROZEN

Pre-registered `log/2026-07-11-prereg-attempt5-selfcond.md` (weighted branches); jobs
15762584/15762585, hashes ffdeac4d4b/ccc86ccf1b verified. Figure: `results/readout_a5.png`.

**Branch B5 (degradation; my weight 5%, largest single miss of my distribution —
though B4+B5 = 40% carried the mechanism).** Octave-2 end-to-end DROPPED: 4a reference
A 0.746 / B 0.734 → p=0.5: 0.482/0.410; p=1.0: 0.467/0.444. Extrapolated octave
collapsed to 0.24–0.39. Bounded-OOD amplitude still holds (oct-1 within 10%). p=1.0
also lost the real-coarse ceiling (0.611/0.502 at oct 2 — pure self-conditioning
withers the clean conditional); p=0.5 kept it (0.992/1.005).

**The mechanism, measured (the discriminator):** on the SAME drifted (4a-generated)
coarse at oct 2, the 4a model responds 0.742 — equal to its end-to-end 0.746 within
0.004 (end-to-end IS the drifted-input response, confirmed) — while the
self-conditioned models respond 0.530/0.524: they learned the HONEST,
information-limited conditional given drifted coarse, and honesty is LOWER than the
4a model's over-trusting extrapolation. Conclusion, closing the anti-compounding
program (wording per the 2026-07-16 reconciliation ruling): **a measured compounding
cap consistent with an informational limit; train-side rescue excluded by a tested
negative (this attempt); scope to be bounded by the self-similar control.** The
generated coarse carries only
~half the modulation; naive over-trust already extracts ~0.74 of it; no
conditioning-side or head-side training can exceed that without making the generated
coarse itself more faithful — and octaves 3–4 are already at their own ceilings on
var_slope, so the leak is in off-manifold texture the tracked couplings don't see.

**Per the pre-commitment (`log/2026-07-11-reconvene-4bpii.md`): the generator FREEZES
at "calibrated heads + measured compounding limit". No attempt 6.** The banked
contribution: the P5 break (robust, every config); the collapse law (causally
confirmed by intervention); calibrated heads (given real coarse: var_slope ≈ real at
oct 2, kurtosis at real at oct 3); the drift measurements (structured, not additive;
informational limit; end-to-end = drifted-input response); and the validation
architecture. Prediction verdicts: reconvene P-lever 55% → miss; my B-distribution
(B1 20 / B2 15 / B3 25 / B4 35 / B5 5) → B5 fired: directionally right about
no-recovery (B4+B5=40 vs reconvene's 45 on pass), wrong that it would be harmless.

## Phase 1d, step 2 — the INVERTED validation pilot (2026-07-16): the protocol catches everything we know is there

Prereg + result: `log/2026-07-16-prereg-step2-pilot.md`; figure
`results/pilot_validation.png`; numbers `results/pilot_validation.json`.
**P-edge PASS** (extrapolation error edge-consistent within factor 2: 1.62 arm B, 1.46
arm A). **Self-consistency (truth-referenced): detects the extrapolated-octave failure
(z=7.9) and the trained-octave compounding (z=7.1), passes where the generator is good
(oct 4, z=0.7).** The deployable curve-referenced variant FAILED its own calibration
check on real fields (z≈3.5 everywhere) — measured protocol lesson: deployable checks
need population-calibrated bands. **Held-out battery: scattering order-2 flags at 98–
100% of channels (median |z|≈10)** — the rival school's instrument rejects the
generator without having been designed against it; wavelet-L1 flags several octaves
(its tight bars catch few-% amplitude offsets P4 tolerates). Bonus: the battery also
catches arm B's OOD amplitude blow-up re-emerging at the narrower second edge.
Bottom line for the reshaped paper: **an audit-only pipeline would have rejected this
generator at every failure point we know about, with no access to extrapolated-octave
truth beyond the coupling curve.**

## Phase 1d, step 3 — the downstream-bias demo (2026-07-16): P(k) passes, peaks are tilted

Prereg + result: `log/2026-07-16-prereg-step3-downstream.md`; figure
`results/downstream_peaks.png`. On the frozen generator's fields (P4 amplitude ≤7%
everywhere — power-level checks PASS), the peak-count function is distorted with a
SIGN FLIP: +30%/+23% excess low peaks (ν=1, z=+14/+10 — spurious graininess from the
spatially-white conditional noise) and −14%/−23% missing extreme peaks (ν=3,
z=−6.6/−12 — the kurtosis deficit, flagged in advance by the tracked tables). A
threshold-dependent shape distortion is the worst case for peak-based inference. This
is the usefulness sentence of the reshaped paper: **power-spectrum-level validation
ships a generator whose standard higher-order observable is wrong at 6–14σ, in both
directions at once; our audit flagged both responsible mechanisms before a peak was
counted.**

## For the reconvene (observations, not actions — G-1c bars further variants)

The mechanism points at the MEAN's finite-data memorization as the single upstream
cause. Levers that target it directly, in rough order of cheapness: D4-symmetry data
augmentation (8× effective data; the training set is 322 tiles and un-augmented),
validation-NLL early stopping / capacity control on the mean path, and decoupling the
variance channel from the mean's features (separate trunk or frozen-mean two-stage
fit — the two-stage fit would also make e^{2g} regress the HELD-OUT residual, which is
the quantity the sampler actually needs). The (a)+(b) churn prototype's 90% repair
prior is unchanged. Whether any of these is worth a fourth attempt, and under what
bars, is the reconvene's call.
