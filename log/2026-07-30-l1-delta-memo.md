# 2026-07-30 — L1 DELTA MEMO: the adopted spec's training arm is empty —
# STOP for reconvene review (R38's own rule: any delta → STOP)

R38 pre-cleared the L1 prereg to run IF it instantiates the draft + A2/A3
with no other deltas. While building the machinery I found a structural fact
that makes the faithful instantiation impossible to run in good faith, so
per the rule this is a STOP, not a run.

## The finding (verifiable in one grep)

Octave 1 is in NO training-octave set anywhere in the campaign: c1t sandbox
and c1t gowerstreet train {2,3,4}; stage-D trains {3,4}. The training loop
(arms_p2/c1t/train.py) builds pools ONLY for train_octaves and draws base
samples ONLY inside per-octave steps — an octave-1 base draw NEVER occurs
during training. Octave-1 details are produced exclusively at SAMPLING time
by weight-tied extrapolation (this is the founding bet working as designed).

**Consequence:** the adopted L1 — "retrain the committed arm-A recipe with
the base colored at octave 1 only" — has a training phase that is
IN-DISTRIBUTION IDENTICAL to the committed c1t training (the changed base
component never enters the loss). The sandbox canary would re-validate the
unchanged recipe; the ≥2 gowerstreet seeds would measure fresh-seed variance
(priced at ±3.4 on T_coef by the n=5 ensemble; unmeasured for peaks) on
models that are new draws of the SAME family. ~2.5 H100-h of training would
test nothing about the lever, and the readout would carry a seed confound
the lever itself does not require.

## Proposal: L1′ — the same lever, instantiated where it actually acts

Inference-only, the F/F2 precedent (both prior cures shipped exactly this
way): keep the COMMITTED checkpoints and swap the octave-1 sampling base to
the frozen colored-t.

- Substrate: trained leg = gowerstreet A@16000 (F2 sampling unchanged,
  colored base at j=1 only), THREE fresh generation streams (matching the
  A3 3-stream reference; stream sd priced at 0.72%); edge continuity leg =
  stage-D A@9000, one stream; sandbox must-not-regress canary = sandbox
  A@7500, one stream, standing marginal bars (kill criterion = marginal
  catastrophe, >50% dispersion error at any scored octave) — run FIRST,
  gowerstreet legs launch only if it passes, mirroring the adopted spec's
  canary-first order.
- **A2 verbatim** (scored BEFORE the peak readout is opened, pre-statement
  order): TRANSFER iff C_gen(L1′) ≥ 0.7551 at ≥3σ on its own SEs; splits
  L1-NULL into NULL-WITH-TRANSFER (M-GRAIN wounded) vs NULL-WITHOUT-TRANSFER
  (L2 loss-side becomes live).
- **A3 verbatim**: 3-stream reference — ν2.5 mean +15.290% (stream sd
  0.723%); ν3.0 mean from the same committed streams: (+11.0696 +14.1076
  +12.2447)/3 = **+12.474%** <!-- src: stage0_p3_verdict.json /trained/F2_gowA
  + stage1_p3_replicate_scores.json, R12 --> ; per-stream bootstrap SE ~3.1%
  the CI basis; Δ as in N3 with the shared-real conservatism note.
- Branch weights UNCHANGED (rec/exec): CURED 10/10, IMPROVED 35/40, NULL
  35/30 (A2 splits), REGRESSED 10/10, gates+FLIPPED 10/10.
- Identity gate: with an all-white base the sampler must reproduce the
  committed maps (the replay-gate criterion, twice validated); seeded
  streams recorded.
- Budget: ≲0.2 H100-h (vs ~2.5 for the empty-training version).
- Seed lesson compliance: the lever has NO training component, so training-
  seed variance does not apply; the replicated dimension is the generation
  stream (3 streams, matching the reference's own replication).

## Second decision the reconvene must make: the filter source

Both are fitted, calibrated (post-quantile-map fixed point, ≤2.1% dev),
frozen and committed (results_p2/l1_filter_{gowerstreet,sandbox}.npz):

- **oct1-measured**: strongest correction; but octave 1 is the EXTRAPOLATED
  octave on every leg — using its measured spectrum injects target-octave
  information and weakens the deployment/extrapolation claim.
- **oct2rescaled** (recommended): octave-2 ring shape mapped to octave-1's
  grid in k/N units — deployment-pure, licensed by the MEASURED
  near-scale-invariance of the real coloring (C = 0.786/0.774/0.785 at
  octaves 1/2/3). Proposal: oct2rescaled ADJUDICATES; oct1-measured runs as
  a labeled descriptive ablation stream (cheap, inference-only), never
  entering the branch table.

## Status of the rest of the order set (all executed)

- Minkowski judge: built tests-first, validated on synthetic GRFs only,
  committed FROZEN at b0c36f4 (V2 estimator corrected pre-freeze by the
  synthetic validation — 4/8-connectivity average). Applied to nothing.
- L3 taxonomy curve: done (stage1_p3_l3_taxonomy.json/.png, 20 stacks).
  Reading: dose-responsive association on the REAL field (Gaussian-base c1:
  whiteness z=+5.7 → excess +24.6%; t-base ≈+3.6 → +14%; over-colored
  stage-D arm B z=−12.8/−17.2 → deficits) but NOT universal (sandbox n32:
  whiter z=+4.5 yet deepest deficit −15.7%) — count bias reads as
  whiteness-driven surplus + a field-dependent baseline deficit. C is a
  real-field predictor, not a universal one; paper-grade with that scoping.
- Colored-base machinery: t1–t4 green (JAX gate); two pre-use test-design
  corrections documented in-test (frequency-space D4 maps; detail-space
  coloring synthetics — the instrument-ledger pattern).

**STOP.** Nothing runs until the reconvene rules on (a) L1′ vs the empty
retraining form, (b) the adjudicating filter source. Budget spent this
order set: 0 GPU-h.
