# §3 — The audit protocol (instruments before adjudications)

FIRST PARAGRAPH (binding placement): positioning vs Schanz — "extending the
conditional-diversity check of Schanz et al. (width-only, single conditioning,
qualitative tails) into a calibrated, scale-resolved, pre-registered audit
with downstream propagation." Rouhiainen cited as the summary-stats-only
contrast. <!-- cite: Schanz 2310.06929, Rouhiainen 2311.05217; gate0 §4 -->

## 3.1 The exact-truth sandbox

Lognormal field: Gaussianize → conditional GRF sampling given the coarse
projection is exact linear algebra → exponentiate; true conditional ensembles
in closed form. Every instrument calibrated against exact truth before
adjudicating any model: Gate-A pass, all octaves both metrics (var_slope
≤0.6%, kurtosis ≤5.0%). <!-- src: gateA_instrument.json; RESULTS-phase2.md -->
Fig: gateA.png.

## 3.2 The three tiers (R27 wording, verbatim here and everywhere)

1. Power/amplitude level.
2. Conditional marginals: variance modulation (var_slope), tail weight
   (excess kurtosis, q999).
3. Joint/morphological structure: peak-count excess at fixed marginal
   calibration. Literal placement (environment-conditioned rates, peak
   clustering) is UNTESTED pending the placement experiment's instruments
   (registered future work, §6).
Plus the held-out-basis check: starlet-ℓ1 with independent published code,
never used in any design decision.

## 3.3 The protocol's teeth, demonstrated without a good generator (the
inverted pilot) [WRITE]

Self-consistency detects the extrapolation failure without ground truth
(z=7.9 extrapolated vs z=0.7 where the generator is good); held-out
scattering rejects at 98–100% of channels (median |z|≈10); the deployable
curve-referenced variant fails its own calibration — deployable checks need
population-calibrated bands (honest protocol lesson, kept).
<!-- src: RESHAPE §2 step 2; fig pilot_validation.png -->

## 3.4 The downstream demonstration (why tiers matter)

With P(k)-level checks passing (≤7%), the peak function is tilted 6–14σ,
sign-flipping, threshold-dependent — the worst case for peak-based inference,
and both mechanisms were flagged by the audit before a peak was counted.
<!-- src: RESHAPE §2 step 3; fig downstream_peaks.png -->
[WRITE: one paragraph; MultiscaleFlow validation-level contrast lands here or
in §1 — not both.]

## 3.5 Audit of the auditors [WRITE, short]

Instrument-level findings (starlet package gen2/noise-plane issues) forward-
pointer to the reproducibility appendix: the protocol includes auditing its
own validators. <!-- src: starlet_l1_instrument.json -->
