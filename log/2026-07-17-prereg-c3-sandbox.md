# 2026-07-17 — PREREG: Arm C3, sandbox leg (energy-score-trained detail sampler)
# STATUS: awaiting reconvene review (R6) — NOT submitted; no code written yet
# (tests-first at implementation, after approval)

Authorized track: ruling R6 (log/2026-07-17-reconvene-morning-harvest.md).
SINGLE VARIABLE vs the C1 control = **the training objective** (and the sampler it
entails): conditional flow matching + ODE pushforward → **patched energy score +
direct noise-conditioned sampling**. Everything else frozen at the C1
configuration: same UNet backbone (32/64/128, FiLM), same weight-tying across
octaves, same per-octave std standardization, same data
(data_cache/tiles_sandbox.npz — 322 train + 64 heldout, seeds frozen), same 8× D4
augmentation, same arms A/B, same recursion protocol (gen_from 4), same frozen
scorer and same-convention exact truth (sandbox_truth_normconv.json), same
checkpoint grid {1,2,4,6,8,12,16,20}k. NO NLL head (retired, R3). No crops (B2).

## Objective choice: PATCHED ENERGY SCORE (Pacchiardi line), not AIFS almost-fair CRPS

One-paragraph justification (informed by the Gate-0 reading): our binding deficit
is the conditional MARGINAL's tail weight (kurtosis −18/−21% sandbox, −26/−34%
end-to-end gowerstreet), but the audit's downstream teeth — the peak function
(+5…+9σ on C1) — are JOINT-structure quantities; the patched energy score
(ES(P,y) = E‖X−y‖^β − ½E‖X−X′‖^β summed over local patches; strictly proper for
β∈(0,2)) trains BOTH the per-coefficient marginal and the within-patch dependence,
whereas AIFS's almost-fair CRPS is a per-scalar (pointwise) score whose joint
structure comes only implicitly from shared noise, and its "almost-fair"
finite-ensemble correction is entangled with an autoregressive-rollout training
schedule we would have to port and re-tune — more machinery, weaker
single-variable discipline, and no handle on exactly the joint statistics our
audit flags. Patching is additionally licensed by a measurement we own: B1 found
conditional predictability saturates at r*≈1 coarse pixel, so small local patches
carry essentially all the conditional structure there is. Per-patch with β=1 the
energy score IS the multivariate CRPS generalization, so the CRPS spirit
(calibrated spread from one observation per conditioning — the probabilistic-
weather answer to literally our data regime) is retained.

## Model & training (pinned; implementation details = free periphery, documented at build)

- Generator: ConditionalUNet backbone as in C1, inputs = Gaussian noise z
  (3 channels, detail shape) in place of x_t, t-embedding fed a constant (reuse of
  the frozen architecture with minimal surgery; the surgery is PART of the
  objective package and will be diffed in the implementation commit); output =
  standardized detail sample directly. One forward = one sample.
- Loss: patched energy score, β=1, patch 8×8, stride 4, all three detail channels
  jointly per patch (dim 192); octave-4 maps (8×8) = one patch. m=8 model samples
  per conditioning per step, fair (unbiased) m-sample ES estimator.
- Optimization: Adam lr 1e-3 (C1's), batch 32, 20k steps, seed 0. Cost estimate:
  m=8 forwards per element ⇒ ~8× C1's step cost ⇒ ~1.5–2 h MIG for both arms
  (within budget; ~0.5 H100-h of the 25 H100-h phase cap spent so far).

## Validation gates (tests-first in tests_p2/, BEFORE any training; to be written
## after reconvene approval)

1. ES estimator correctness: patch-size-1/β=1 reduces to per-pixel CRPS — check
   the m-sample estimator against the closed-form Gaussian CRPS (machine/
   statistical tolerance).
2. Propriety-in-practice: a tiny sampler trained on a known 1-D conditional
   recovers mean AND spread (the flat-σ and modulated-σ toys from the phase-1c
   suite, adapted).
3. TAIL gate (the arm's claimed capability): trained on a Student-t (df=5)
   conditional toy, the sampler's kurtosis within a pre-set tolerance — with the
   honest phase-1c caveat stated in-test (pointwise toys can pass while fields
   fail; the discrimination is the full-scale run).
4. Plumbing/determinism/D4-augmentation interaction as in C1's suites.

## Bars (self-adjudicated from same-convention exact truth; KURTOSIS PRIMARY per R6)

SE_rel = sqrt(SE_gen² + SE_truth²)/|truth|, as in C1. Two levels (head-conditional
given real held-out coarse; end-to-end recursion from octave 4), octaves {2,3,4}
binding, octave 1 descriptive — all identical to the C1 prereg.

- **PRIMARY (the arm's question): kurtosis** |k−truth|/|truth| ≤ max(15%, 3·SE_rel)
  at octaves 2,3,4 simultaneously, both arms — per level.
- **Dispersion (must not regress vs C1):** |vs−truth|/|truth| ≤ max(10%, 3·SE_rel),
  both levels — C1 passed these; C3 holding them is a bar, not a bonus.
- Collapse signature on the oct-2 head-conditional var_slope ckpt curve
  (running-peak, drop ≥0.10) as in C1; the kurtosis-vs-steps curve is reported
  DESCRIPTIVELY (new object — no rule on first sight; bar-design ledger lesson).
- Degenerate: end-to-end detail amplitude >25% off the normalized real stack at
  any trained octave, or NaN.
- **Gowerstreet-leg trigger** (identical recipe, descriptive readout): kurtosis
  PRIMARY passes at head-conditional level for both arms AND dispersion bars pass
  both levels both arms. Standing reconvene prediction for that leg (registered in
  R6): P(gowerstreet e2e kurtosis deficit halves vs C1) 50% — executor: 40%.

## Branches with weights (adjudication precedence top→bottom)

| branch | definition (mechanical) | executor weight |
|---|---|---|
| C3-DEG | amplitude/NaN degenerate | 7 |
| C3-DISP-REGRESS | any dispersion bar fails (C1 passed them all) | 18 |
| C3-TAILS-FAIL | dispersion holds; kurtosis fails head-conditional | 25 |
| C3-TAILS-HC-ONLY | kurtosis passes head-conditional, fails end-to-end | 15 |
| C3-CAL | all bars pass (kurtosis + dispersion, both levels) | 30 |
| other/mixed | anything else | 5 |

Prediction lines: **P-C3a (kurtosis at truth, head-conditional, oct 2–4, both
arms): executor 45%** vs standing 55%. **P-C3b (dispersion holds at C1 level):
70%.** **P-C3c (kurtosis calibrated end-to-end too): 32%.** Reasoning: the
energy score directly optimizes calibrated conditional spread including patch
marginals — the mature answer to one-observation-per-condition (Gate-0-verified);
main risks are scoring-rule optimization stability at 322-tile scale, the
m=8-sample gradient variance, and the possibility that tail weight needs
β<1-style emphasis the plain β=1 score under-weights. Dispersion should be the
easy part (ES punishes both under- and over-dispersion); if C3-DISP-REGRESS
fires, the objective swap is dead on arrival regardless of tails — hence its
sizable weight.

## Deliverables (on approval + run)

arms_p2/c3/ (loss, trainer wrapper, sampler), tests_p2/test_energy_score.py,
results_p2/arms_c3_sandbox.npz + c3_conditional_sandbox.json +
c3_verdict_sandbox.json (same scorer scripts, C3 paths), readout log + figure
(ckpt curves incl. the kurtosis curve; bars visual), JOBS.md entries with config
hash at submission. STOP at the sandbox readout for reconvene.
