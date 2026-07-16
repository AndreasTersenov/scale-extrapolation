# 2026-07-16 — Stage A readout: GATE A **PASS** (instrument calibrated on exact truth)

Job 16490857 (COMPLETED; first submission 16490749 died at t=0 on the POSIX-sh
`source` gotcha — JOBS.md). Prereg: log/2026-07-16-stageA-prereg.md (rule applied
verbatim, adjudicated mechanically from results_p2/{sandbox_truth,gateA_instrument}.json).

## VERDICT (rule as written, primary N=256)

| oct | metric | truth (SE) | instrument (SE) | rel err | bar = max(floor, 3·SE_rel) | verdict |
|---|---|---|---|---|---|---|
| 1 | var_slope | 1.2587 (0.0098) | 1.2525 (0.0093) | 0.5% | 5.0% | PASS |
| 1 | kurtosis  | 7.833 (0.196)   | 7.659 (0.248)   | 2.2% | 12.1% | PASS |
| 2 | var_slope | 1.1051 (0.0108) | 1.0985 (0.0115) | 0.6% | 5.0% | PASS |
| 2 | kurtosis  | 5.585 (0.156)   | 5.522 (0.219)   | 1.1% | 14.4% | PASS |
| 3 | var_slope | 0.9612 (0.0124) | 0.9571 (0.0132) | 0.4% | 5.6% | PASS |
| 3 | kurtosis  | 4.001 (0.123)   | 3.922 (0.257)   | 2.0% | 21.3% | PASS |
| 4 | var_slope | 0.8085 (0.0141) | 0.8085 (0.0162) | 0.0% | 8.0% | PASS |
| 4 | kurtosis  | 2.678 (0.111)   | 2.812 (0.178)   | 5.0% | 23.4% | PASS |

**GATE A: PASS at every octave, both metrics.** Prediction verdicts: standing P-A 90%
— hit; executor 92% — hit. Stage B and Arm C1 are unlocked per the orders.

## Deliverables

- Exact ensembles: 256 parents × 64 conditional redraws (level-4 Gaussian coarse
  fixed), 128², lognormal (alpha=2.0, sigma_g=0.6), seeds in sandbox/recipe.py;
  arrays at $SCRATCH/scale-extrap-p2/ (ens/parents/cstars/train .npy).
- TRUE conditional statistics with SEs: results_p2/sandbox_truth.json
  (var_slope 1.259 → 0.809, kurtosis 7.83 → 2.68 across octaves 1→4).
- Figure: results_p2/gateA.png — parent + 3 exact conditional redraws (same coarse,
  different detail) and truth-vs-instrument curves inside the tolerance band.

## DESCRIPTIVE (labeled per rider grant 3; no adjudication weight)

At N=64 — the phase-1c production sample size — the instrument's kurtosis at octave 3
reads 3.16 vs truth 4.00 (21% low; its own bootstrap SE ±0.26 covers ~1/3 of that
gap). var_slope stays ≤1.9% at all octaves even at N=64. Implication for reading the
phase-1c record: production kurtosis point-values at deeper octaves carry finite-N
wobble of order 10–20%; the phase-1c kurtosis FAILURES (z≈5, deficits of 50–60%) are
far too large for this to matter, but near-bar kurtosis PASSES (e.g. 4b′'s
"kurtosis at real at oct 3") should be read with this wobble in mind. Filed for the
morning reconvene; no re-adjudication proposed (bars at N=64 already reflect the
larger SEs).

## Updated belief

The frozen instruments measure what they claim to measure, at production sample
sizes, on a field whose truth is exact by construction — the phase-1c verdict stack
rests on a calibrated ruler. The sandbox + truth JSON is now the permanent
calibration bed for Stage-B estimator validation and the C1 bars.
