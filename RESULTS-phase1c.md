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
- **GRF null**: job 15738957 still queued at writing (drain backlog); does not affect
  the verdict. Harvest per JOBS.md when it completes.
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
memorized μ, not the conditional). GRF-null 90% → pending. Amplitude-undegraded 85% →
passed at trained octaves.

## Honest limits

Single seed / single λ-free config (per prereg); adjudicated on the final checkpoint as
pre-registered (the checkpoint curve is reported above and does not change the verdict);
the unit-scale toy passed while the real fields failed — the toy had no
memorization-capacity mismatch (its σ carried all signal; the real fields' mean can
substitute for variance), which is exactly the gap the diagnosis exposes.

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
