# 2026-07-09 — Validation gates + estimator (tests-first)

## Hypothesis
The four CLAUDE.md validation gates can be written as executable pytest tests, and an
estimator of the scale-drift of conditional wavelet statistics (Measurement M1) can be
built to pass them — in particular the GRF null must pass before any real-field work.

## Setup
- Env: venv `~/venvs/scale-extrap` on scipy-stack + arrow modules (`source env.sh`).
  Stop hook sources `env.sh` so pytest resolves in its fresh shell.
- Package `scaledrift`: `wavelet` (DWT octave (detail, coarse) pairs, round-trip),
  `fields` (power-law GRF + lognormal), `drift` (binned conditional W1, within-octave
  finite-sample floor, map-bootstrap → **excess** drift + CI), `moments` (conditional
  mean/var/skew profiles, marginal PDF, cross-octave |w| coupling, running-coupling PCA).
- **Headline metric.** Per octave j: pool the 3 orientation sub-bands of the detail
  coeffs, per-octave-standardize detail `w` and coarse `c`; conditional PDF `p_j(w|c-bin)`
  over coarse quantile bins. Drift(j,k) = mean-over-bins W1(p_j, p_k), sizes matched to M.
  **excess = measured − floor**, floor = W1 between two disjoint size-M subsamples of the
  finer octave (finite-sample baseline). Bootstrap resamples MAPS.

## Expectation
Round-trip at machine precision; GRF excess drift consistent with 0 (|z|<3); lognormal
drift highly significant; bootstrap SE ∝ 1/√N_maps; metric invariant under flips/rot.

## Result — all 4 gates green (13 tests, 42 s, well under the 300 s hook)
1. **Round-trip** (haar/db4/sym4): max err 1.8e-15 (haar), <1e-11 (db4), <1e-9 (sym4).
2. **GRF null**: adjacent-octave excess |z| ≤ 2.2 across seeds/octaves (128²–256²);
   test asserts |z|<3. Positive control: lognormal 2→3 z≈6–27. Null PASSES.
3. **Consistency**: mean measured-SE ratio N→2N ≈ 1.4–1.5 (target √2), asserted [1.2,1.7].
4. **Symmetry**: flips + 90/180/270° rotations shift excess by <0.5σ (asserted <3σ).

## Two decisions worth recording (deviations from naive first cut, not from PLAN)
- **Wavelet = Haar, not db4.** db4 is asymmetric → its coefficient stats do not commute
  with flips/rotations on finite samples; the symmetry gate failed at ~2.7σ. Haar is
  exactly symmetric (antisymmetric highpass) → symmetry gate passes at <0.5σ. db4/sym4
  kept as robustness alternatives (PLAN free-periphery: "start db4 or Haar").
- **Floor is a fixed bias correction, not bootstrapped.** First cut recomputed the
  within-octave split-half floor inside each bootstrap resample; duplicated maps then
  matched identical coefficients across the split → artificially low floor → excess biased
  high, point estimate fell outside its own CI. Fix: compute the floor once on the full
  (duplicate-free) data (averaged over 8 partitions); bootstrap only `measured`. Also 3×
  faster.

## Updated belief
Pipeline is trustworthy: null holds, estimator has power, errors scale, metric is
isotropic. Cleared to measure real fields (GRF_HF → lognormal → gowerstreet), GRF first.
