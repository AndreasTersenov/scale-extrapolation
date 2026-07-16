# RESULTS — Phase 2 overnight run (2026-07-16, NIGHT-ORDERS)

<!-- MORNING SUMMARY inserted at top on completion -->

## A — the lognormal sandbox & Gate A (exact conditional truth; instrument calibration)

Prereg log/2026-07-16-stageA-prereg.md; readout log/2026-07-16-stageA-readout.md;
figure results_p2/gateA.png; job 16490857 (first submission died on the POSIX-sh
`source` gotcha, JOBS.md).

- Built: fully synthetic lognormal sandbox (alpha=2.0, sigma_g=0.6, 128²; recipe +
  seeds in sandbox/recipe.py), 256 parents × 64 EXACT conditional redraws given the
  level-4 Haar Gaussian coarse (Hoffman–Ribak; validated tests-first against dense
  linear algebra at machine/statistical tolerance — tests_p2/test_sandbox_conditional.py).
- TRUE conditional statistics (raw convention): var_slope 1.259→0.809, kurtosis
  7.83→2.68 across octaves 1→4 (results_p2/sandbox_truth.json; batch-means SEs).
  Same-convention truth for generator scoring (per-tile normalized):
  var_slope 1.229→0.755, kurtosis 7.02→2.10 (sandbox_truth_normconv.json).
- **GATE A: PASS at every octave, both metrics** (var_slope rel err ≤0.6%, kurtosis
  ≤5.0%, bars max(5%/10%, 3·SE_rel)). P-A standing 90% and executor 92% — both hit.
- Descriptive: at N=64 (phase-1c production sample size) the instrument's oct-3
  kurtosis wobbles 21% low — phase-1c kurtosis FAILURES (50–60% deficits) are
  unaffected, but near-bar kurtosis PASSES should be read with this wobble in mind.
- **Measured convention effect (pre-readout amendment,
  log/2026-07-16-c1-amendment-normconv.md):** per-tile normalization (the wfm
  pipeline's world) shifts the estimand −3…−6% (var_slope) and −12…−19% (kurtosis)
  vs raw-field truth — comparable to adjudication bars; C1 was adjudicated against
  same-convention truth. Phase-1c itself was like-for-like (unaffected); any
  cross-convention comparison (stage-0 raw dial vs generator-side numbers) must
  mind this.

## B1 — dependence range & shape (model-free)

Prereg log/2026-07-16-stageB-prereg.md (+ pre-reading amendment
log/2026-07-16-stageB-amendment-shape-null.md); job 16491648; figure
results_p2/stageB.png; data results_p2/stageB1_{curves,shape}.json. Estimators
validated tests-first against EXACT Gaussian conditional variance
(tests_p2/test_depmeasure.py, test_shape.py — ridge and kNN within 10% of
closed-form truth at every radius).

- **Locality (P-B1a): FIRES — saturation within ~1 coarse pixel.** Gowerstreet
  V(r)/V(0) (mean channel, ridge): one ~9% drop from center-only to r=1, then FLAT
  to r=12 at every octave (oct-2 profile: 1.000, 0.911, 0.906, 0.904, 0.903, 0.894,
  0.902). By the pre-registered mechanical rule r*=0 (the 3·SE(r_max) threshold is
  as large as the whole drop — a bar-sharpness note, same class as bar-miss #5);
  by eye r*≈1. Either way r* ≤ 6 at octaves 1 AND 2 → **P-B1a fires** (standing
  70%, executor 85% — both hit). Sandbox control: same shape (18% drop to r=1,
  flat; r*=1 by the rule — its SEs are 6× smaller).
- **Methods finding:** kNN on annulus MEANS is flat at 1.00 on both fields — the
  near-context information is DIRECTIONAL (gradient-like raw-pixel combinations),
  not radial-mean; and the amplitude channel (|w| target) shows only a weak, slowly
  accumulating trend on gowerstreet (~4% by r=12, not clearly saturated — long-range
  variance correlations exist but are small).
- **Shape/anisotropy (P-B1b): does NOT fire.** The isotropic sandbox control fired
  at z=−3.09, catching a design error in my prereg (the "exchangeability ⇒ zero
  null" claim was wrong — the orientation classifier SELECTS locally-anisotropic
  noise; amendment committed before reading gowerstreet). Diagnostics (4 runs,
  JOBS.md): the negative baseline is systematic (split-half −0.19%/−0.29%;
  gate-off shrinks it — selection mechanism confirmed). Amended verdict
  (difference-in-differences vs the sandbox baseline): gowerstreet oct-2 amplitude
  channel +0.51% ± 0.56% → **z=0.91, NULL** at a measured ~0.5%-of-V sensitivity
  (w channel: −0.16%, z=−1.2). Standing 50% / executor 35% — the NULL outcome
  favors the executor's lean. Filament-transported conditional structure is not
  detectable at matched area on 128² tiles with this machinery; a matched-spectrum
  isotropic control (phase-randomized gowerstreet) is the drafted follow-up.

## B2 — crops inventory

<!-- B2_PENDING -->

## C1 — vanilla CFM + augmentation (the un-run arm), sandbox leg

Prereg log/2026-07-16-prereg-c1-sandbox.md (branch weights pre-registered);
amendment (normconv truth) pre-readout; readout
log/2026-07-16-c1-sandbox-readout.md; verdict table
results_p2/c1_verdict_sandbox.json; figure results_p2/c1_sandbox.png; job 16491750,
config_hash ab9c175f59.

- **Branch: B-C1-TAILS (both arms).** No collapse signature — the oct-2
  head-conditional var_slope curve RISES to 1.02–1.07 (truth 1.070) and holds
  through 20k steps. Dispersion within bars at octaves 2,3,4 BOTH levels
  (head-conditional 4.9–6.8%; end-to-end 4.5–8.7%; bars 10–11.7%). Kurtosis FAILS
  at oct 2 (18.4–21.3% low vs 15–16.1% bars, both arms both levels; arm B also
  end-to-end oct 3).
- **Headline 1: the collapse cure works in the naked ODE-pushforward channel.**
  Plain conditional FM + 8× D4 augmentation, no variance head, reaches and HOLDS
  truth-level dispersion (phase-1 un-augmented: peak 0.97@2k → 0.75@10k). On a
  calibrated bed, phase-1's dispersion pathology is fully accounted for by
  finite-data mean memorization.
- **Headline 2: compounding is NOT generic to drifting fields.** The sandbox
  drifts (1.23→0.75) yet recursion from octave 4 costs ≤1.4% in var_slope at every
  trained octave (phase-1 gowerstreet comparator: −27% at oct 2). The
  "off-manifold texture" component of the informational-limit account is now the
  only surviving candidate for gowerstreet's compounding; the triggered
  gowerstreet leg is the direct discriminator.
- What still fails: tails (oct-2 kurtosis −19/−21%; extrapolated-octave
  DESCRIPTIVE: var_slope −7…−11%, kurtosis −22…−31% — the phase-1 pattern in
  miniature, far milder).
- Prediction verdicts: P-C1a standing 60 HIT / executor 55 HIT; P-C1b standing 45
  MISS / executor 30 leaned right; P-C1c standing 45 HIT / executor 25 MISS (I
  over-weighted compounding); branch: executor modal COLL 35 wrong, TAILS 25 fired.

## C1 — gowerstreet leg (descriptive; triggered by the sandbox dispersion pass)

Job 16491989, config_hash 6f9cdca680; readout
log/2026-07-16-c1-gowerstreet-readout.md; tables
results_p2/c1_descriptive_gowerstreet.json; maps results_p2/c1_maps.png; peaks
results_p2/c1_peaks_gowerstreet_descriptive.json. Registered expectation: my modal
"compounding reappears ≥15%" (65%) MISSED — the 20% "mild" branch fired.

- **Oct-2 end-to-end compounding: −27% (phase-1 NLL-head 4a) → −9/−10% (plain
  CFM), single variable = the sampling channel.** The attempt-5 "measured
  compounding cap" is substantially CHANNEL-DEPENDENT: ~17 of the 27 points
  belonged to the NLL head's sampling noise degrading the coarse manifold, only
  ~10% remains for the informational component on gowerstreet (and ≤1.4% on the
  in-class sandbox).
- **Extrapolated octave no longer broken:** end-to-end var_slope ratios 0.88/0.84
  (phase-1 P5: 0.55); head-conditional arm A at 0.98 of real. Arm A (scale-blind)
  BEATS arm B (dial) at the extrapolated octave — the OOD FiLM coordinate hurts in
  this channel; the Stage-D dial question changes shape.
- **Tails are the binding deficit and they compound:** oct-2 kurtosis −12%
  head-conditional → −26/−34% end-to-end.
- **DESCRIPTIVE peaks:** biased UP at ALL thresholds (+5.4…+9.2σ, no sign flip;
  vs the NLL-head generator's +14/−12 flip) — a near-dispersion-calibrated
  generator still fails the higher-order audit; second independent exhibit for
  the audit paper.
- Arm B's ckpt curve technically fires the running-peak rule on a RECOVERING 4k→6k
  transient (0.995→0.885→1.016) — the oscillation-vs-collapse rule weakness again;
  descriptive, filed to the bar-design ledger.

## Tests & infrastructure

- Chore 0 (154c30b): Stop-hook gate now covers BOTH stacks (tests/ under env.sh;
  tests_wfm/ + tests_p2/ under wl-challenge-env; pyproject lists all three trees —
  loud failure over silent skip; timeout 300→900 s; full gate green 14+32+…).
- tests_p2/ grew to 21 tests tonight (sandbox conditional sampler, B1 estimators
  incl. exact-GRF gates, shape-test nulls/controls) — all green under both stacks
  (pywt cross-checks auto-skip in the JAX env, run green under env.sh).
- SLURM: 5 jobs, all CPU except two MIG legs; cumulative GPU ≈ 0.4 H100-h of the
  3 H100-h cap. One t=0 infra failure (POSIX-sh `source`), documented.
