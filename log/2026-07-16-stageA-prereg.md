# 2026-07-16 — Stage A pre-declarations (sandbox build + Gate A adjudication mechanics)

Overnight executor (Fable), per NIGHT-ORDERS.md. Committed BEFORE ensemble generation
and before any Gate-A number exists. Gate-A PASS/FAIL rule is the ORDERS' rule,
applied mechanically; this file only pins the quantities it is applied to.

## Hypothesis / setup

Stage A builds the exact-truth instrument-calibration bed: fully synthetic lognormal
(truth by construction — the orders' PREFERRED path, no shared-data fallback needed),
exact conditional Gaussian redraws given the level-4 Haar coarse via Hoffman–Ribak
(validated tests-first: dense conditional mean at machine tolerance, dense conditional
covariance statistically, constraint reproduction machine-exact at 128²/level 4 —
tests_p2/test_sandbox_conditional.py, committed 6d4ef4e).

## Frozen recipe (selected by a descriptive pre-freeze pilot, 64 fields/cell)

alpha=2.0, sigma_g=0.6, 128², L = exp(0.6·G − 0.18) − 1, unit-variance Gaussian layer.
Pilot table (var_slope, kurtosis per octave 1→4):

| alpha | sig | oct1 | oct2 | oct3 | oct4 |
|---|---|---|---|---|---|
| 2.0 | 0.4 | 0.79, 2.5 | 0.70, 2.1 | 0.61, 1.7 | 0.53, 1.3 |
| **2.0** | **0.6** | **1.25, 7.6** | **1.10, 5.4** | **0.95, 4.0** | **0.81, 2.6** |
| 2.0 | 0.8 | 1.77, 22.0 | 1.55, 13.4 | 1.33, 9.1 | 1.11, 5.0 |
| 2.5 | 0.4–0.8 | (flatter drift; rejected — weaker scale-dependence) | | | |

Rationale: gowerstreet-like kurtosis range (2.6–7.6 vs real 3.9–6.7), clear drift
(1.25→0.81), coordinates inside the phase-1 COORD_NORM [1.5, 13]. Seeds frozen in
`sandbox/recipe.py` (parents 20260716, redraws 20260717, train tiles 20260718).

## Pre-declared subtlety (the log map does not commute with coarse-graining)

Conditional ensembles fix the GAUSSIAN coarse c_G at level 4; the Haar coarse of the
lognormal field itself varies slightly across redraws. Therefore ALL truth values are
ESTIMAND-level: the population functional the frozen instrument estimates (pooled
(w,c); 10 equal-count coarse bins; var_slope = OLS slope of binned Var(w) vs bin mean;
marginal excess kurtosis of pooled standardized w), evaluated on the pooled exact
ensemble (parents ~ prior, redraws exact-conditional ⇒ pooled = exact unconditional
law). Binning is part of the estimand's definition — same n_bins in truth is not
circular. Truth implementation is independent of scaledrift (sandbox/haar.py,
sandbox/truth_stats.py; pywt cross-check ran green under env.sh, auto-skips in the
JAX env). Known estimand property (documented in-test): pooled 3-band kurtosis on a
GRF is a small POSITIVE constant (orientation-band variance mixture), not 0; shared
identically by instrument and truth.

## Gate A adjudication mechanics (pre-declared)

- **Truth:** estimand on all 256×64 pooled exact fields; SE by batch means over 16
  disjoint parent blocks (parents i.i.d.).
- **Instrument (PRIMARY):** the frozen production path
  `scripts/measure_generated.couplings` (pooled scalars, n_bins=10, bootstrap over
  fields, n_boot=200, seed 0) applied to the **256 parent lognormal fields**
  (independent draws — the production-style field-stack input). Descriptive
  secondary, no adjudication weight: same on the first 64 parents (phase-1c sample
  size).
- **SE_rel definition:** SE_rel = sqrt(SE_inst² + SE_truth²) / |truth| (SE of the
  difference, relative to truth).
- **PASS (per the orders, mechanical):** at every octave j ∈ {1,2,3,4}:
  |vs_inst − vs_truth|/|vs_truth| ≤ max(0.05, 3·SE_rel_vs) AND
  |k_inst − k_truth|/|k_truth| ≤ max(0.10, 3·SE_rel_k). FAIL or mixed → STOP the
  gated pipeline (instrument bug), then diagnose within the rider budget.

## Expectations (registered before the numbers)

- Standing P-A: 90% (instruments recover exact truth).
- Executor's number: **92%** pass. Residual risk is not the instrument's math but
  finite-N estimand bias at N=256 vs N=16384 pools (quantile-bin edges and pooled
  standardization are data-dependent; var_slope is a ratio-of-binned-variances
  statistic whose small-sample bias has never been measured — that is exactly what
  Gate A exists to measure). If it fails, expected locus: octave 4 (fewest
  coefficients per field: 3×8²·N pooled samples).

## Job

One CPU SLURM job (def-lplevass, 4 cores, 24 GB, ≤1 h): step 1 generation
(arrays → $SCRATCH/scale-extrap-p2/), step 2 truth (results_p2/sandbox_truth.json),
step 3 instrument (results_p2/gateA_instrument.json). Runner
scripts_p2/gen_sandbox_ensembles.py (incremental/resumable, heartbeats). Gate-A
adjudication happens in-session from the two JSONs after harvest.
