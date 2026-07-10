# 2026-07-09 — Measurement M1 on the control ladder

## Hypothesis
On real fields, the conditional wavelet statistics are NOT octave-invariant: a GRF shows
no drift (null), while N-body maps drift significantly between adjacent octaves (P9a), and
that drift is low-dimensional (P9b). If so, a weight-tied generator is exactly wrong by
the measured per-octave drift, and the "running couplings" that must repair it are few.

## Setup
- Estimator: `scaledrift` (gates green; see 2026-07-09-gates-and-estimator.md). Headline
  metric = excess conditional-W1 drift (cross-octave W1 of p_j(w|coarse-bin) minus the
  within-octave finite-sample floor), bootstrap over PARENT maps.
- Data: `scripts/measure.py` on the ladder, 18 parent kappa maps per field (1424×176),
  cut into 11 × 128² tiles each (198 tiles/field), Haar wavelet, `periodization`, octaves
  1–5 (1=finest), 8 conditional quantile bins, 150 bootstraps, W1 capped at 40k samples.
- Ran as a CPU SLURM job (`scripts/measure.slurm`, def-lplevass, rc32122, 10m23s,
  COMPLETED) — the login-node cgroup (6 cores/16 GB shared across ~10 sessions) throttled
  and killed local runs. Per-field JSON checkpointing added so a timeout keeps finished
  fields. Fields: GRF_HF (null) → lognormal (control) → gowerstreet (N-body) →
  hf_pm_1024 (cross-check, different sim physics; column `maps`).

## Expectation
GRF_HF drift ≈ 0; lognormal + N-body drift ≫ noise, growing with octave separation;
1–3 smooth running-coupling functions of scale; N-body the strongest.

## Result
| field | adjacent excess-drift z (1→2,2→3,3→4,4→5) | cum. drift z @sep4 | eff-dim(80%) |
|---|---|---|---|
| GRF_HF (null) | 0.6, 0.1, −0.2, −0.1 | 0.5 | (null) |
| lognormal | 2.1, 4.9, 2.8, 1.0 | 6.2 | 2 |
| gowerstreet | 1.7, **5.8**, **7.3**, **5.2** | **9.5** | 2 |
| hf_pm_1024 | **8.2**, **3.6**, **4.2**, 2.0 | 9.8 | 2 |

- **GRF null PASSES**: max|z| = 0.15 in clean octaves (2→3,3→4), 0.61 over all; every
  running coupling flat in j (var_slope ≈ 0.01–0.05, ρ ≈ 0.33, kurtosis ~constant).
- **P9a TRUE**: N-body adjacent drift is 3–7σ and 48–59% of the distance; cumulative
  drift vs separation reaches z ≈ 11 (gowerstreet sep3) / 16 (hf_pm sep3). K-M1b NOT
  triggered → the P5/P6 project lives.
- **P9b TRUE**: 2 PCA components explain ≥85% of the cross-octave variation of the
  conditional-moment profiles, for every drifting field.
- **Running couplings** (gowerstreet, fine→coarse): conditional-variance slope
  1.31→0.42, marginal excess kurtosis 12.8→1.4, cross-octave ρ 0.67→0.53 — smooth,
  monotonic, tight bars. hf_pm has the SAME qualitative shape → drift is field-general.

## Deviations from PLAN notes (not redesign)
- Wavelet = Haar not db4 (symmetry gate; logged in gates entry). Cross-check field =
  hf_pm_1024 rather than JHU turbulence (available, different sim physics — satisfies
  "implementer's pick, different physics").
- Real GRF_HF is not a pure power law, so a small Gaussian spectral drift was allowed for
  under PLAN; it did not appear above noise (null clean), so no analytic correction was
  needed. The synthetic power-law GRF null is the executable gate and is green.
- Marginal excess kurtosis is inflated (~0.9) even for GRF_HF by pooling sky tiles of
  differing amplitude; it is FLAT in octave for GRF so contributes no drift. The
  conditional-variance slope is the clean, contamination-free non-Gaussianity coupling.

## Updated belief
P5 confirmed on the fields: conditional wavelet statistics drift strongly and
low-dimensionally with scale on N-body maps and vanish on a GRF. Naive weight-tying is
insufficient; the repair is a small (≈2-D) scale-conditioning. Proceed to the toy phase
with the three empirical running couplings as the conditioning design spec.
