# 2026-07-16 — NOTE: the solution space for one-sample-per-condition (brainstorm w/ Andreas)

The root problem, named: each coarse map appears ONCE in training; conditional
spread exists only via generalization across similar environments; training
dynamics erode it (the collapse law). Candidate solution families for any FUTURE
generator phase (Option-3 prospectus; the freeze stands — pre-registered arms only):

1. **Constrained-realization conditional ensembles (gold standard; needs own sims →
   3D phase).** Fix large-scale IC modes, randomize small-scale modes, re-run: true
   samples of p(fine|coarse) by construction. Even a few hundred coarses × ~5
   details each gives measured conditional spread for training AND validation truth.
   A training set DESIGNED for conditional learning.
2. **Locality / receptive-field dial (cheapest, theory-backed — WC-RG short-range
   theorem).** Local coarse ENVIRONMENTS repeat hundreds of times even when maps are
   unique; a receptive field larger than the physical interaction range is pure
   memorization capacity. Arm: conditioning-range sweep vs collapse onset +
   end-to-end. MIG-minutes.
3. **Overlapping crops** (stride-shifted tiling of parent maps): ~10–50× correlated
   law-preserving pairs; stacks with the 8× symmetry augmentation.
4. **Proper scoring rules (CRPS / energy score) instead of NLL/regression** — the
   probabilistic-weather answer to literally this problem (one observation per
   condition, calibrated ensembles required; GenCast lineage — fetch-verify).
   Contained objective swap; already inside the Gate-0 kill-test reading scope.
5. **Unconditional prior + inference-time conditioning (posterior sampling).**
   Coarse-graining is a known exact LINEAR operator → detail-given-coarse is a
   linear inverse problem. Train an unconditional model of fine tiles (every tile =
   i.i.d. marginal sample — the pairing scarcity dissolves); condition at sampling
   via DPS-style guidance or SMC (guarantees available; guidance bias is measurable
   by our existing instruments). Prior memorization remains (same cures apply).
6. Honorables: texture-critic adversarial finetuning (blandness fix; mode-seeking
   vs calibration tension — needs the audit as a leash); early-stop at dispersion
   peak + capacity control (small, stackable). Target-side noise: rejected
   (re-invents diffusion, biases the law).

Ranking for a future phase: un-run vanilla-CFM+augment arm (see companion note) →
locality dial → energy score → prior+posterior-sampling → constrained ensembles.
Credit: Andreas's brainstorm (multiple-fine-per-coarse; smart augmentations) —
items 1 and 3 are his directly; 2 fell out of pressing on WHY generalization works
at all.

## Addendum (same day) — Andreas's rulings + reconvene web investigation (search-verified)

Andreas: (1) lognormal needs no generative model per se → re-scoped as TEST-BED (see
below); (2) locality must respect the physics (filament-shaped dependence?) →
measurement-first adopted; overlap idea approved; (5) adversarial path REJECTED.

Investigation results (search-verified today; deep verification still owed to Gate 0):

**(3) Proper-scoring-rule training — VERIFIED, mature:**
- AIFS-CRPS, ECMWF (arXiv:2412.15832; npj AI 2026): operational-class ensemble model
  trained DIRECTLY on "almost fair" CRPS; stochastic, arbitrary ensemble size at
  inference; beats the physics-based IFS ensemble on most variables/lead times.
- Pacchiardi et al., JMLR 25(45) (arXiv:2112.08217): adversarial-free training of
  generative networks by scoring-rule minimization (energy score, prequential);
  follow-up introduces PATCHED energy scores (local patches + stride) — which also
  echoes the locality theme of item 2.
→ The objective-swap arm is de-risked: recipes exist in a field with exactly our
one-observation-per-condition constraint.

**(4) Unconditional prior + posterior-sampling conditioning — pattern EXISTS
(including in Andreas's own lineage); the specific combination appears open:**
- Remy, Lanusse, Jeffrey, Liu, STARCK et al. (arXiv:2201.05561, A&A): score-based
  simulation prior + posterior sampling for the (linear) mass-mapping inverse
  problem. Extensions: DES-Y3 diffusion-prior mass mapping (2511.14667), 3D
  diffusion priors (2606.00803).
- Cosmological SR by diffusion: Stochastic SR with DDMs (2310.06929, OJA;
  "filter-boosted" loss) and 3D conditional diffusion SR emulator (2311.05217,
  ML4PS): both are CONDITIONAL nets on paired low/high — trained-range
  interpolation needing high-res data, same one-sample-per-condition exposure,
  and (to verify in Gate 0) no conditional-calibration audit — i.e. PRIME EXTERNAL
  AUDIT TARGETS for our protocol.
- Exactness machinery: Twisted Diffusion Sampler (arXiv:2306.17775, NeurIPS 2023) —
  asymptotically exact conditional sampling from an UNCONDITIONAL diffusion via
  twisted SMC (particles ↔ accuracy dial). MCGDiff (linear-Gaussian case): from
  memory, NOT yet verified — flag.
→ The apparent gap (Gate-0 to confirm): unconditional fine-field prior + EXACT
wavelet coarse-projection constraint + drift-aware extrapolation BEYOND the trained
range + conditional-calibration audit. The components all exist; the combination
and the audit do not appear to.

**(1) re-scoped — the lognormal sandbox:** for lognormal fields, TRUE conditional
ensembles are obtainable in closed form (Gaussianize → conditional GRF sampling
given the coarse projection is exact linear algebra → exponentiate). A non-Gaussian
field with EXACT p(fine|coarse) truth, for free — the calibration test-bed for any
conditional-learning machinery and for the audit instruments themselves. (We already
measured lognormal drift 1.54→0.53 — in-class, drifting, ideal.)

**(2) refined — physics-shaped locality:** before touching architecture, MEASURE the
conditional dependence range and shape: mask coarse context beyond radius r (and
outside oriented, filament-aligned regions); measure the conditional response
change. If anisotropy confirmed → oriented/steerable kernels or structured
conditioning masks; else plain receptive-field cap. Measurement-before-architecture,
as with everything else here.
