# 2026-07-16 — C1 gowerstreet leg readout (DESCRIPTIVE per prereg; adjudication
# belongs to the morning reconvene against the G-1c project bars)

Job 16491989 COMPLETED, config_hash **6f9cdca680**. Identical recipe to the sandbox
leg (plain CFM + 8× D4 augmentation, no NLL head). Tables:
results_p2/c1_descriptive_gowerstreet.json; maps results_p2/c1_maps.png;
peaks (descriptive) results_p2/c1_peaks_gowerstreet_descriptive.json.
Pre-harvest expectation registered in JOBS.md: "compounding reappears ≥15% at
oct 2" 65% / "mild <10%" 20% / other 15%. **The 20% branch fired — my modal
expectation MISSED** (readout below; misses first, as always).

## The numbers (64 held-out fields, normalized convention, real = same-stack reference)

| oct | level | arm A vs (ratio) | arm B vs (ratio) | arm A kurt | arm B kurt | real vs / kurt |
|---|---|---|---|---|---|---|
| 2 | head-cond | 0.999 (0.98) | 0.994 (0.97) | 5.87 | 5.88 | 1.020 / 6.69 |
| 2 | end-to-end | 0.921 (0.90) | 0.927 (0.91) | 4.43 | 4.94 | 1.020 / 6.69 |
| 3 | head-cond | 0.741 (0.92) | 0.755 (0.94) | 2.69 | 3.18 | 0.801 / 3.94 |
| 3 | end-to-end | 0.752 (0.94) | 0.757 (0.94) | 3.01 | 3.54 | 0.801 / 3.94 |
| 4 | end-to-end | 0.509 (0.96) | 0.532 (1.00) | 1.38 | 1.27 | 0.532 / 1.89 |
| 1 (extrap) | head-cond | 1.097 (0.98) | 1.009 (0.90) | 9.38 | 7.00 | 1.117 / 9.78 |
| 1 (extrap) | end-to-end | 0.985 (0.88) | 0.937 (0.84) | 5.86 | 4.95 | 1.117 / 9.78 |

Checkpoint curves (oct-2 head-conditional): arm A rises 0.64→1.03 and holds (no
collapse); arm B has a transient 4k→6k dip of 0.11 that RECOVERS to ~1.0 — the
running-peak rule technically fires on it, the same oscillation-vs-collapse rule
weakness as the 4a literal-rule incident (descriptive here; flagged for the rule's
ledger).

## What moved vs phase 1 (the load-bearing comparison)

1. **The oct-2 end-to-end compounding deficit shrank −27% → −9/−10%** (4a NLL-head:
   0.746/0.734 vs real 1.020; C1 plain-CFM: 0.921/0.927). Same data, same
   augmentation, same architecture family, same recursion depth — the single
   variable is the sampling channel (ODE pushforward vs explicit-variance head).
   The "measured compounding cap" of the attempt-5 record is therefore
   SUBSTANTIALLY CHANNEL-DEPENDENT, not purely informational: the honest phase-1
   wording ("consistent with an informational limit") survives only for the
   remaining ~10%; the other ~17 points belonged to the NLL-head's sampling noise
   degrading the coarse manifold octave-over-octave.
2. **The extrapolated octave is no longer broken.** P5's phase-1 signature (arm A
   var_slope 0.61 vs 1.12 end-to-end, 45% deficit) reads 0.985/0.937 (−12/−16%)
   here; head-conditional arm A is at 0.98 of real. NOTE: arm A (scale-blind)
   BEATS arm B (dial-conditioned) at the extrapolated octave in both levels — the
   OOD FiLM coordinate hurts rather than helps in this channel (the dial question
   for Stage D just changed shape).
3. **Tails are now the binding deficit, and they compound:** head-conditional oct-2
   kurtosis −12% (5.87/5.88 vs 6.69) grows to −26/−34% end-to-end; oct-3 kurtosis
   −19/−32% head-conditional.
4. **DESCRIPTIVE peaks check** (step-3 machinery on these fields): the peak
   function is biased UPWARD AT ALL THRESHOLDS (+9.1σ at ν=1 → +7.4σ at ν=3 arm A;
   +8.9 → +5.4 arm B; gen 142 vs real 123 peaks at ν=3) — a DIFFERENT tilt than
   the NLL-head generator's sign-flip (+14σ excess low / −12σ missing high). A
   near-dispersion-calibrated generator still fails the higher-order audit: the
   audit-paper's usefulness sentence gets a second, independent exhibit.

## Honest limits

Descriptive throughout (no bars tonight, per prereg); single seed per arm; the
real-side reference is the 64-field production stack (its own N=64 kurtosis wobble
documented in the Gate-A readout applies); the arm-B checkpoint-rule firing is an
oscillation by eye but the rule as written fires — reconvene owns the reading.
