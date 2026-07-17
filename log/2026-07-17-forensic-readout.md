# 2026-07-17 — NLL-noise forensic readout: branch **F-OVERSHOOT** (both arms) —
# the probe confound fired, and it decomposes the 4a response anyway

Job 16581052 COMPLETED (CPU; benign CUDA-plugin warning on a CPU node). Prereg
log/2026-07-17-prereg-forensic-nllnoise.md; data results_p2/forensic_nllnoise.{npz,json}.
DESCRIPTIVE throughout (no project bar).

## Numbers (frozen scorer; mean-path = frozen 4a ckpts, e^g·z removed)

| arm | oct | var_slope (SE) | kurtosis | detail_std / real |
|---|---|---|---|---|
| A | 2 | **1.536** (0.025) | 32.5 | 0.38 |
| A | 1 | 1.665 (0.031) | 54.0 | 0.28 |
| A | 3/4 | 1.033 / 0.547 | 9.0 / 2.1 | 0.63 / 0.97 |
| B | 2 | **1.676** (0.044) | 78.3 | 0.39 |
| B | 1 | 2.131 (0.039) | 268.4 | 0.38 |
| B | 3/4 | 1.104 / 0.652 | 15.6 / 3.0 | 0.51 / 0.86 |

References: real oct-2 1.020 (kurt 6.7); 4a WITH noise 0.746/0.734 (kurt ~3.0);
C1 plain-CFM 0.921/0.927.

## Branch adjudication (prereg verbatim)

Oct-2 mean-path var_slope 1.536/1.676 > 1.10 → **F-OVERSHOOT both arms** (my
weights: 5; modal F-COLLAPSE 40 — a miss, logged; the pre-registered
interpretation applies: probe confounded, treat descriptively). The pre-declared
amplitude-starvation confound is CONFIRMED quantitatively: detail_std ratios
0.28–0.39 ≈ sqrt(1 − σ-share) for σ-share 86–93%, exactly as computed in the
prereg.

## What the overshoot settles (the R5 question), stated carefully

The 4a generator's end-to-end response decomposes into two miscalibrated
components measured separately for the first time:

- the **μ-cascade alone OVER-modulates** — var_slope 1.5–1.7 with explosive
  kurtosis (32–268): the conditional-mean channel concentrates all its
  (undersized) variance in the bright regions;
- the **σ-channel's white noise UNDER-modulates** — adding it in production
  dragged the pooled response DOWN to 0.746 and the kurtosis down to ~3
  (heavy-tailed μ diluted in a near-flat Gaussian bath).

An information-exhaustion account cannot produce this pattern: a cascade using
the SAME conditioning as production over-shoots the real modulation, so the
information was available at every octave — the with-noise deficit was the
sampler's mixture arithmetic (modulated μ + insufficiently-modulated noise), not
a ceiling on extractable information. This confirms R4's channel-dependence
re-scoping in MECHANISM, complementing C1's single-variable evidence (remove the
head entirely → −10%), though via the overshoot branch rather than the naive
"closes toward −10%" reading — the deficit does not close because mean-path
generation replaces one miscalibration with its mirror image.

Bonus retro-explanation: phase-1c's persistent "kurtosis too tame" (3.0 vs 6.7)
is now legible as the same dilution — the μ-channel's tails were always there
(kurt 32+), drowned by the σ-bath.

## Honest limits

Descriptive; single generation per arm (deterministic sampler — no seed variance
by construction); the mean-path cascade is off-manifold in its own way (1/3
amplitude, progressively smoothed coarse), so its numbers characterize the frozen
checkpoints' channels, not a usable generator; language revisions remain the
reconvene's call (R4/R5).

## Prediction verdicts

Executor branch weights: F-OVERSHOOT 5 fired; modal F-COLLAPSE 40 missed (I
expected amplitude starvation to suppress the modulation signal; instead the
starved-but-perfectly-correlated μ details EXAGGERATE it — the estimand
standardizes pooled w, so concentration, not amplitude, decides). Second
overshoot-direction miss of the campaign for me on this estimand family
(cf. the deterministic-sine shape control at z=−8); noted for calibration.
