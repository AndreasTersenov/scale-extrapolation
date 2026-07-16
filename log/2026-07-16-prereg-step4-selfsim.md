# 2026-07-16 — PRE-REGISTRATION: Step 4 — the self-similar control

Per the reconciliation ruling (step 4, taken up): bound the scope of the tested
negative. Question: is the measured compounding cap a property of DRIFTING fields in
this data regime, or of the coarse-to-fine architecture itself? Committed before tile
synthesis and submission.

## Design finding first (measured 2026-07-16, before this prereg)

Lognormal fields are NOT a self-similar control: their measured var_slope drifts
1.54 → 0.53 across octaves 1–4 — nearly parallel to gowerstreet (1.30 → 0.66). (Bonus
datum for the memo: the between-scale drift is generic to multiplicative fields.)

## The control (pre-registered)

**A synthesized, exactly scale-invariant cascade in the generator's own model class:**
starting from a smoothed-Gaussian 8×8 coarse at octave 4, recursively apply, at every
octave, the SAME conditional law in standardized coordinates:
detail_j = (a + b·ĉ_j)·z, z ~ N(0,I) i.i.d. per coefficient, ĉ = standardized coarse,
(a, b) = (0.9, 0.25) fixed across octaves (chosen for var_slope in gowerstreet's
mid-range), amplitude std_j applied as a fixed geometric ladder (×1.5 per octave).
By construction: no drift (couplings constant per octave in standardized coordinates)
AND zero model mismatch (the law is exactly the Gaussian-NLL head's class, with a
LINEAR pointwise σ(ĉ)). 386 tiles of 128² (322 train / 64 held-out), deterministic
seed. Measured couplings of the synthesized stack are written to the coords file
under field name "selfsim" and REPORTED in the readout (expectation: ~flat in j;
whatever they measure is the reference).

## Run

Frozen generator config verbatim (NLL head, D4 augmentation, FiLM, 20k steps, seed 0),
`run_two_arms.py --field selfsim`, train octaves 2–4, generate from real octave-4
coarse, extrapolated octave 1. One MIG job, `scripts/arms_selfsim.slurm`,
**config_hash 3c03bded83**. Synthesized-stack couplings, measured (train split):
var_slope 0.559 / 0.548 / 0.517 / 0.486 and kurtosis 1.02 / 0.97 / 0.87 / 0.82 at
octaves 1–4 — flat as designed (vs gowerstreet's factor-2 drift); appended to
`running_couplings.json` under "selfsim".

## Adjudication (pre-registered)

- **S-similar (the scope question):** end-to-end var_slope at the EXTRAPOLATED octave
  within 1σ combined of the real synthesized stack's value. PASS ⇒ the compounding cap
  does not bite when the conditional law is scale-invariant and in-class ⇒ the cap is
  attributable to drift/off-manifold texture of realistic fields (bounds the tested
  negative's scope). FAIL ⇒ the cap is architectural (compounding bites even in the
  most favorable case) ⇒ the negative claim WIDENS.
- Trained octaves reported (expected within 1σ — in-class law, no mismatch).
- Kurtosis reported (conditional is exactly Gaussian ⇒ pooled kurtosis from
  σ-modulation only; the real synthesized stack is its own reference).

## RESULT (2026-07-16, job 16401485, hash 3c03bded83 verified)

Extrapolated octave: real 0.558±0.002, arm A 0.544±0.002, arm B 0.542±0.002 —
**residual error 2.5%** (vs 39–65% on gowerstreet); trained octaves essentially exact
(oct-2 arm A z≈0.3); amplitude 0.801/0.808 vs 0.804; kurtosis within a few %.
**Bar adjudication, honest:** the literal pre-registered 1σ bar FAILS (z≈5) because
the synthetic stack's bootstrap SEs are ±0.002 — the bar demanded 0.4% accuracy. A
bar-calibration miss (mine; same genre as the 4a onset-rule lesson: calibrate bars to
the reference's precision). **The scope question is answered decisively regardless of
the bar's letter:** compounding shrinks by an order of magnitude-plus when the
conditional law is scale-invariant and in-class ⇒ the measured cap is attributable to
the DRIFT / off-manifold texture of realistic fields, not to the coarse-to-fine
architecture — with a small measured floor (~2.5%, z≈5) of finite-sample compounding
even in the ideal case. P-S-similar 60%: substantively the anticipated direction;
formally MISS under the literal bar. Figure: `results/selfsim_control.png` — the
campaign's clearest single image (drifting field: curves sag; invariant field: three
curves coincide, including the extrapolated band).

## Predictions (confidences)

- P-S-similar PASS: **60%.** For: no drift, no mismatch, pointwise linear σ (the toy
  gates showed exact recovery of this law at unit scale); the white-fluctuation
  spatial-correlation issue does not exist here (the true law IS white given coarse).
  Against: the collapse law still applies (322 tiles, though augmentation is frozen
  in); residual compounding from finite-sample head error may still accumulate across
  3 octaves.
- If PASS at oct 1 but trained octaves off: report as partial (in-class calibration
  issue), no adjudication beyond the stated bar.
