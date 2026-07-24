# RESULTS — Stage-0, Measurement M1

**Scale-drift of conditional wavelet statistics, measured directly on the fields.**
For a reader who knows `PLAN.md` but not the code. Numbers are the final production run
(`results/scores/measurement.json`); figures are in `results/`.

---

## Verdict (one line each)

| Item | Verdict | Basis |
|---|---|---|
| **GRF null (K-M1a)** | **PASS** | real GRF drift consistent with zero (max \|z\|=0.15 in clean octaves); synthetic power-law GRF null is a green gate test |
| **P9a — significant N-body drift** | **TRUE** | gowerstreet adjacent-octave drift 3–7σ and 48–59% of the distance; cumulative drift vs scale z≈9–11 |
| **P9b — drift is low-dimensional** | **TRUE** | 2 PCA components explain ≥85% of the cross-octave variation, every drifting field |
| **K-M1b (project kill gate)** | **NOT triggered** | drift IS significant → naive weight-tying is insufficient → **the P5/P6 project lives** |
| Validation gates (4/4) | **GREEN** | round-trip, GRF null, √N consistency, flip/rotation symmetry |

**Bottom line.** The conditional wavelet statistics are strongly scale-dependent on N-body
maps and flat on a GRF, exactly as P5 predicts. The scale-dependence is captured by ~2
smooth "running couplings." A weight-tied (octave-invariant) generator is wrong by the
measured per-octave drift; P6's repair can be a small scale-conditioning, not a full
per-scale model. Proceed to the toy phase.

---

## What was measured

For each field we DWT every 128² tile (Haar, `periodization`, octaves j=1 finest … 5
coarsest) and, per octave, form the **conditional PDF** `p_j(w | c-bin)` of the
orientation-pooled, per-octave-standardized detail coefficient `w` given the coarse
(approximation) field `c`, binned into 8 coarse quantiles. The headline **drift** between
two octaves is the mean-over-bins Wasserstein-1 distance between their conditional PDFs.

We report **excess drift = measured − floor**, where the floor is the W1 between two
disjoint same-size subsamples of a single octave — the finite-sample W1 that two draws of
the *same* distribution incur. Excess ≈ 0 under scale-invariance, > 0 under real drift.
Error bars bootstrap over **parent maps** (the independent unit). z = excess / bootstrap-SE.

Ladder: **GRF_HF** (null) → **lognormal** (analytic control) → **gowerstreet** (N-body,
the measurement) → **hf_pm_1024** (cross-check, different sim physics). 18 parent kappa
maps/field × 11 tiles = 198 tiles; 150 bootstraps.

## Validation gates (backpressure — all green)

1. **DWT round-trip** at machine precision: max recon error 1.8e-15 (Haar; db4/sym4 ≤1e-9).
2. **GRF null, executable**: a synthetic power-law GRF built in-test shows excess drift
   consistent with zero (\|z\|≤2.2 < 3 across seeds/octaves), while a lognormal positive
   control is detected at z≈5–27 — the estimator is both unbiased on the null and sensitive.
3. **Estimator consistency**: doubling the maps shrinks the bootstrap SE by 1.41 ≈ √2.
4. **Symmetry**: flips and 90/180/270° rotations move the drift by <0.5σ (Haar is exactly
   symmetric, so the transform commutes with the dihedral group).

---

## GRF null verdict — PASS

Adjacent-octave excess drift (real `GRF_HF`), z-scores: **1→2: 0.6, 2→3: 0.1, 3→4: −0.2,
4→5: −0.1**. Every running coupling is flat in scale (conditional-variance slope
0.01–0.05, cross-octave ρ ≈ 0.33, marginal kurtosis ~constant). The pipeline reports no
drift where none should exist. (Real GRF_HF is not a pure power law, so PLAN allowed for a
small Gaussian spectral drift; none appears above noise, so no analytic correction was
needed. The 1→2 pair sits at the tile Nyquist scale and is the noisiest.)

## P9a — N-body drift is significant — TRUE

`gowerstreet` adjacent-octave drift (excess, z, % of the measured distance):

| pair | excess | z | % of distance |
|---|---|---|---|
| 1→2 | 0.007 | 1.7 | 22% |
| 2→3 | 0.026 | **5.8** | 51% |
| 3→4 | 0.037 | **7.3** | 59% |
| 4→5 | 0.047 | **5.2** | 48% |

Three of four adjacent pairs clear the pre-registered bar (>3σ **and** >10%); the finest
pair (1→2, tile Nyquist) is marginal. **Cumulative** drift from the finest octave grows
monotonically: excess 0.045 (z=6.7) / 0.103 (**z=11.4**) / 0.143 (z=9.5) at separations
2/3/4 — see `results/figures/stage0/drift_vs_separation.png` and `results/figures/stage0/drift_adjacent.png`. GRF_HF sits
flat on zero in the same figure. **K-M1b is not triggered.**

## P9b — the drift is low-dimensional — TRUE

PCA of the per-octave conditional-moment profiles (variance + skewness vs coarse bin),
across octaves: the first **2** components explain **≥85%** of the cross-octave variation
for every drifting field (lognormal 87%, gowerstreet 85%, hf_pm 94%; eff-dim@80% = 2 each).
The scale-drift lives in a ~2-D subspace → "few running couplings," as bet.

---

## The empirical running couplings (toy-phase design spec)

Three interpretable, smooth, monotonic functions of octave j (1=finest → 5=coarsest);
these are what a scale-conditioning must carry. Values below are `gowerstreet`
(± bootstrap SE ~0.03–0.06); `results/figures/stage0/running_couplings.png` shows all fields with bars.

| coupling | meaning | j=1 | j=2 | j=3 | j=4 | j=5 |
|---|---|---|---|---|---|---|
| `var_slope` | slope of Var(detail \| coarse) — conditional-variance modulation | 1.31 | 1.15 | 0.93 | 0.66 | 0.42 |
| `kurtosis` | marginal excess kurtosis of detail coeffs | 12.8 | 8.9 | 5.4 | 2.9 | 1.4 |
| `ρ` (pairs) | cross-octave \|w\| coupling (1-2 … 4-5) | 0.67 | 0.64 | 0.59 | 0.53 | — |

Read-off for P6: all three **rise toward finer scales**. The generative task trains at the
128² pixel scale and extrapolates to *finer* octaves, where these couplings are largest —
so a weight-tied model, applying training-scale couplings at the finer generated octaves,
systematically **under-represents small-scale non-Gaussianity**. The required correction is
the extrapolation of these curves, and PCA says it is ~2-dimensional. `var_slope` is the
cleanest, contamination-free coupling; `kurtosis` is partly inflated by pooling sky tiles
of differing amplitude (flat in j for the GRF, so it adds no spurious drift).
`results/figures/readouts/conditional_variance_profiles.png` shows the mechanism directly: for GRF_HF the
per-octave `Var(detail|coarse)` curves collapse (octave-invariant); for gowerstreet they
**fan out** — the finest octave has the steepest rise (~5×), decreasing with scale.

## Cross-check — the drift shape is field-general

`hf_pm_1024` (a different N-body/PM simulation) drifts strongly (adjacent z up to 8.2;
cumulative z up to 16) with the **same** qualitative running couplings (var_slope
0.78→0.36, kurtosis 6.8→1.3 decreasing; eff-dim=2). The lognormal control also drifts
(2→3 z=4.9) with the same monotone shape at smaller amplitude. So the phenomenon is not
specific to one simulation: the conditional non-Gaussianity of the wavelet field runs with
scale, strongest at small scales, everywhere non-Gaussian — and vanishes for a Gaussian
field.

---

## Caveats

- 128² tiles from 1424×176 strips give clean octaves 1–5; octave 5 (few coefficients) and
  octave 1 (tile Nyquist) are the noisiest and drive the widest bars.
- Marginal `kurtosis` mixes intrinsic non-Gaussianity with inter-tile amplitude variation;
  use `var_slope` for the clean coupling. Both are reported.
- Statistics: 18 parents/field. Verdicts are far from their thresholds (z of 5–16 vs a 3σ
  bar; eff-dim 2 vs a 3 bar), so they are robust to the sample size; the toy-phase spec
  would benefit from more maps for tighter coupling curves.

## Files & reproduction

- `results/scores/measurement.json` — every number (drift, couplings, PCA, cross-octave ρ, PDFs).
- `results/npz/profiles.npz` — conditional-moment profiles per field/octave.
- `results/figures/stage0/{drift_adjacent,drift_vs_separation,running_couplings}.png + results/figures/readouts/conditional_variance_profiles.png`.
- Reproduce: `sbatch scripts/measure.slurm` (CPU, def-lplevass) → `python scripts/summarize.py`.
  Gates: `source env.sh && pytest tests/`.
