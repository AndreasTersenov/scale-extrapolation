# 2026-07-17 — RECONVENE RULING: forensic adjudicated (F-OVERSHOOT); language revisions executed; C3 APPROVED with conditions

## R8 — the forensic settles R5: the "informational limit" is dead as a general claim

Verified from raw data (forensic_nllnoise.json): mean-path-only regeneration from the
FROZEN 4a checkpoints over-modulates (oct-2 var_slope 1.536/1.676 vs real 1.020) with
hyper-non-Gaussian tails (kurtosis 32.5/78.3 vs real 6.7) at exactly the pre-declared
amplitude confound (detail_std ratio 0.38/0.39 = √(1−σ-share)). Adopted attribution:
**the 4a production response (0.746, kurt≈3.0) was a mixture artifact — an
over-modulated, heavy-tailed μ-cascade diluted by a ~2/3-variance white Gaussian
bath.** An information-exhaustion account cannot over-shoot real modulation from the
same conditioning; the information was present, the sampler's mixture arithmetic
destroyed it. Two retro-explanations adopted: phase-1c's tame kurtosis (μ-channel
tails were always there, drowned); and the channel-dependence of the compounding cap
(R4's provisional wording confirmed at forensic grade).

Scorecard: executor modal F-COLLAPSE 40 MISS, 5% F-OVERSHOOT fired — the second
overshoot-direction miss on this estimand family; their calibration note (the
standardized pooled estimand rewards concentration, not amplitude) is adopted as
estimand lore. Reconvene process note: R5 was authorized without a reconvene-side
branch weight registration — from now on, authorized forensics get reconvene numbers
too.

## R9 — language revisions EXECUTED (evidence in hand, per R5's sequencing)

Applied today, by the reconvene, as appended dated revisions (history preserved):
- RESHAPE-MEMO.md: revision section — the compounding cap re-scoped to the
  variance-head sampler's mixture arithmetic; the attempt-5 negative scoped to its
  substrate; the taxonomy now counts THREE measured mechanisms (mean-memorization
  collapse, cured; variance-head mixture dilution, the false cap; Gaussian-base ODE
  tail deficit, the open frontier) plus two downstream signatures (sign-flip vs
  up-tilt).
- METHOD.md: update paragraph (NLL head retired; collapse cure confirmed in the
  original channel; forensic attribution).
- Gate-0 claim-2 scope: binding sentences unchanged as MEASUREMENTS; the scope
  clause "of the variance-head sampler; the information required for calibrated
  modulation is measurably present in the conditioning (forensic 2026-07-17)"
  attaches wherever claim-2 is quoted. The structured-not-additive flagship (vi)
  survives untouched — it characterizes that sampler's output drift, now with a
  mechanism underneath it.

## R10 — C3 prereg APPROVED, with two conditions and one pre-note

The patched-energy-score choice over almost-fair CRPS is well-argued (joint-structure
teeth; B1's r*≈1 licenses 8×8 patches; β=1 = multivariate CRPS) and the bars are
correctly shaped (kurtosis primary; dispersion must-not-regress as a BAR; branch
weights sum and are sane). Approved as committed (b42edd9), subject to:

1. **Condition — extreme-tail descriptive readout:** alongside kurtosis, report the
   99.9th-percentile absolute-coefficient ratio vs same-convention truth (descriptive,
   never adjudicating). This is the cheap instrument for the executor's own hedge
   that β=1 under-weights extreme tails; if TAILS-FAIL fires we will know at which
   depth of the tail.
2. **Condition — sampler-shape sanity gate (tests-first, with the others):** the
   direct noise-conditioned sampler must pass a two-moment + skewness recovery toy
   BEFORE training (the phase-1c lesson: never trust a new sampling channel's
   calibration by construction).
3. **Pre-note (scope guard):** "objective+sampler package" is accepted as THE single
   variable; consequence pre-registered now — a CAL outcome does NOT attribute the
   win to the objective alone, and any attribution claim would need a later
   decomposition arm. No such arm is authorized tonight.

Standing reconvene numbers (already registered in R6): P-C3a-equivalent 55%,
gowerstreet-halves 50%. Executor's hedged 45/32/70/40 lines noted; the spread
between us is itself informative and will be scored.

GO: the session may write the validation gates and submit per its prereg. STOP at
the readout as always.
