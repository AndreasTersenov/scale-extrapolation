# 2026-07-16 — PRE-REGISTRATION: Step 3 — the downstream-bias demo

Per the reconciliation ruling (phase-1d step 3, "the usefulness centerpiece"): show a
parameter-relevant bias in a statistic the community actually uses, that P(k)-level
checks miss, on the frozen generator's fields — and show our diagnostics flagged it in
advance. Target statistic and bias criterion fixed HERE, before any computation on the
generated fields. CPU only; existing artifacts (`results/arms_aug.npz`, 64 held-out
starts).

## Target statistic (pre-registered)

**Peak counts** — the standard weak-lensing higher-order observable (used for
cosmological constraints alongside/beyond the power spectrum). Estimator: per-map
count of 8-neighbour local maxima above ν ∈ {1, 2, 3} (maps are unit-variance, so ν is
in σ units); `pilotstats.peak_counts`, validation-gated BEFORE this prereg
(identical-stack zero, GRF null, monotonicity, smoothing-suppression;
`tests_wfm/test_heldout_stats.py::test_peak_counts_gates`, green).

## Bias criterion (pre-registered)

For each arm of the frozen 4a generator vs the 64 held-out real maps:
- **Downstream bias established** if |z| ≥ 3 (bootstrap-over-maps, `z_stack`, 200
  resamples) at ν = 2 or ν = 3, TOGETHER with the already-recorded P(k)-level PASS
  (P4 amplitude: 1.9% arm A / 7.2% arm B at the extrapolated octave; within the 10%
  criterion at all octaves). That conjunction is the demo: a pipeline gating only on
  power-level checks would ship this generator; the peak observable is biased.
- Direction prediction: NEGATIVE bias at high ν (under-dispersed fine texture +
  kurtosis deficit 2.6 vs 9.8 ⇒ missing extreme peaks).
- "Flagged in advance": the pre-existing var_slope/kurtosis tables (G-1c and 4a
  readouts) flagged the responsible deficits before any peak was counted — cited, not
  recomputed.

## Predictions (confidences)

- P-bias at ν=3 (either arm, |z|≥3, negative): **75%.**
- P-bias at ν=2: **60%** (moderate peaks partly carried by the mean path).
- P-null-at-ν=1: |z| < 3: **55%** (low peaks ~ Gaussian bulk + correct amplitude).
- Falsifier acknowledged: if peaks come out unbiased at all ν, the demo FAILS and the
  memo says so — the usefulness argument would then rest on the scattering channel
  (step 2) alone.

## RESULT (2026-07-16, computed after this prereg was committed)

**Bias established, with a sign structure richer than predicted.** Peaks per map
(real / A / B, z): ν=1: 590 / 777 (+14.1) / 726 (+10.2); ν=2: 265 / 297 (+5.6) /
273 (+1.5); ν=3: 123 / 105 (−6.6) / 95 (−12.0). Criterion met (|z|≥3 at ν=2 and ν=3
with P4 amplitude passing at ≤7%): a pipeline gating on power-level checks ships this
generator; its peak FUNCTION is tilted — ~30% spurious excess of low peaks (the white
per-pixel conditional noise creates grainy spurious maxima; the fingerprint of the
spatially-uncorrelated fluctuation channel) and 14–23% missing extreme peaks (the
kurtosis deficit, flagged in advance by the tracked tables). A shape distortion, not
an offset — threshold-dependent bias is the worst case for peak-based parameter
inference. Prediction verdicts: P-ν3-negative 75% HIT; P-ν2 60% HIT (direction at
ν=2 positive — surprise, noted); P-null-ν1 55% **MISS** (strongly positive instead —
the graininess channel was not in my prediction model; it is now measured).

## Deliverables

`results/downstream_peaks.{json,png}` (peak-count curves real vs arms with error
bars + the z table), RESULTS-phase1c.md §step-3, readout in chat before step 4.
