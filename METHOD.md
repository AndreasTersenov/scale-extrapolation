# METHOD — what this project does, and why it's built this way

Plain-language but precise; written 2026-07-16 at Andreas's request. Audience: a
collaborator, committee member, or future-us. Repo results referenced where they
ground a claim. Scope words apply throughout: one field type per experiment,
128² tiles, one-octave validated extrapolation, measured couplings.

## 1. The goal in one paragraph

High-resolution simulations are expensive; low-resolution ones are cheap. We ask
whether a generative model trained only on coarse-resolution fields can generate the
next finer level of structure — beyond its training resolution — with statistics
trustworthy enough for scientific use, and (equally important) whether we can VALIDATE
such beyond-data generation without ground truth at the target scale. Phase 1
falsified the generator half on 2D weak-lensing-like fields and produced its failure
anatomy; the validation half works and is the current deliverable (see RESHAPE-MEMO.md).

## 2. The ladder: scale-by-scale generation

A field is decomposed into an image-pyramid ladder: a coarse map plus, at each scale
("octave" = one factor-2 refinement), the detail needed to reach the next resolution.
Generation is recursive: given the coarse map, sample this octave's detail; assemble;
repeat one octave finer. The field's probability factorizes exactly as
p(coarse) × Π_j p(detail_j | coarse_j).

The physics anchor: the conditional law p(detail | coarse) is NOT the same at every
scale — it drifts (gravitational clustering is progressively more non-Gaussian at
small scales). Our founding, generator-independent measurement (stage-0, RESULTS.md):
that drift is real (3–7σ/octave), LOW-DIMENSIONAL (two smooth components capture
≥85%), similar across simulation types, and exactly zero on Gaussian control fields.
Those two components — the "running couplings" — become a conditioning dial.

## 3. The generator: conditional flow matching + an explicit variance head

- **Backbone:** one small conditional U-Net, weight-tied across octaves (the same
  network generates every rung), conditioned via FiLM on (a) the coarse map,
  (b) the two running couplings — the dial that says where on the ladder it sits.
- **Engine:** conditional flow matching (CFM) — the diffusion-family method that
  learns a velocity field transporting noise to data; sampling integrates an ODE
  (~80 steps). Simple objective, deterministic sampler.
- **The forced amendment (the collapse law, RESULTS-phase1c.md):** vanilla CFM's
  deterministic-ODE pushforward is where conditional VARIANCE lives, and finite-data
  training starves it — the model memorizes the conditional mean and the dispersion
  collapses (confirmed in three variance channels; removed causally by 8× exact
  symmetry augmentation). The frozen generator is therefore a hybrid: the CFM path
  carries the conditional mean/structure; an explicit Gaussian-NLL head predicts a
  per-coefficient log-variance, sampled explicitly; D4 symmetry augmentation
  (322→2576 tiles, conditional law exactly preserved) keeps the head alive.
- Arms: A = scale-blind, B = coupling-conditioned; the pair isolates
  scale-conditioning as the single differing variable.

## 4. Why wavelets (and not raw pixels)

A pixel-space cascade (generate 16², then 16²→32², …) is viable — cascaded diffusion
does exactly this. The wavelet (Haar) factorization buys three specific things:

1. **Exact cross-scale consistency.** A pixel-space stage outputs the whole fine
   image, which CONTAINS the coarse information again — the model can corrupt what it
   was conditioned on, and consistency must be learned/enforced. In wavelet space the
   model generates only the orthogonal complement (the detail coefficients); the fine
   map is assembled by an exact transform, so coarse-graining the output returns the
   conditioning identically, BY CONSTRUCTION. This is the literal meaning of
   "generation that commutes with coarse-graining."
2. **A better-shaped learning target.** Detail coefficients are near-zero-mean,
   localized, heavy-tailed, and statistically similar across octaves — which is what
   makes weight-tying sane and makes the diagnostics (conditional-variance modulation
   = var_slope, coupling curves) directly measurable. Pixel targets are dominated by
   dynamic range the model already knows.
3. **One language for generation, measurement, and physics.** Coarse-graining is the
   RG operation; the couplings, the drift measurement, and the wavelet-ℓ1/scattering
   validation statistics all live in the same kind of basis (the Mallat/WC-RG school's).

What wavelets do NOT do: solve the hard problems. Mean-memorization collapse,
octave-to-octave compounding, and tame tails are facts about finite-data conditional
generation in any coordinates; a pixel cascade hits them too (that is the
exposure-bias literature). The basis makes the task well-posed and auditable — it
does not make it easy.

## 5. The basis is periphery; the factorization is core (incl. the starlet question)

The load-bearing requirement on the transform is: **exact, scale-indexed,
critically-sampled (decimated) factorization** — each octave adds exactly the new
degrees of freedom, and reconstruction is exact. Haar is the simplest instrument with
those properties (orthogonal, local, machine-precision round-trip — a standing test
gate). Any transform with the same properties could substitute; that choice is
implementation periphery.

**Starlet (isotropic undecimated à-trous, the WL-native transform)** is the natural
candidate to raise — it is better matched to convergence fields' isotropic, blobby
structures, and it is the basis of the field's standard statistics. But it trades
away exactly the property the generator leans on: it is UNDECIMATED and redundant
(J+1 full-resolution planes for J scales; non-orthogonal). Consequences if used as
the generative coordinates: (a) no dimension-expanding ladder — every plane has full
resolution, so "generate the next octave's new pixels" must be reformulated;
(b) the planes are not free coordinates (analysis constraints couple them), so
independently sampled planes need not be the transform of ANY field — the exact
factorization p(coarse)Πp(detail|coarse) is lost; (c) the commute-with-coarse-graining
identity degrades to a learned/approximate property — the pixel-space weakness again.

**Where starlet genuinely belongs in this project: the measurement and validation
layer.** Isotropy matters for hand-crafted statistics (peaks are round; Haar's three
directional channels split them awkwardly), and the field's WL instruments
(starlet-ℓ1, peak statistics, the LDT theory anchors) live there. Scoring generated
fields with starlet-based statistics also strengthens the held-out-statistics
principle (validate in a basis NOT used for generation). If a future phase wants a
more isotropic GENERATIVE basis, the candidates preserving (near-)critical sampling
are e.g. a Laplacian pyramid (mildly redundant, consistency approximately enforceable)
— a research change requiring its own gates, and no measured failure so far points at
the basis: the kurtosis and compounding deficits are conditional-modeling problems,
not basis problems.

## 6. Validation: how you trust generation beyond your data

The architecture (piloted, RESHAPE-MEMO.md §2): (i) slide-the-edge — hold out the
finest octave you HAVE, extrapolate into it, score against the hidden truth;
(ii) self-consistency — the generated field's own coupling curve must continue the
measured one (no ground truth needed; detects at z≈8 where it should, passes where
the generator is good); (iii) held-out statistics — score with instruments never used
in design (scattering covariance; starlet-ℓ1 belongs here); (iv) theory anchors (LDT)
where valid; (v) downstream-bias demonstration — the peak function biased +14σ/−12σ
while power-level checks pass, i.e. the audit catches what standard checks ship.

## 7. Honest boundaries (read before quoting anything)

Validated domain: ONE octave of extrapolation, under measured couplings, on 2D fields,
322-tile training sets, Haar basis, one seed per config unless stated. "RG-inspired,"
not RG. The bounded law from the self-similar control (selfsim_control.png):
hierarchical conditional generation is statistically faithful where the between-scale
law is scale-invariant (~2.5% floor), degrades with the measured drift (39–65% here),
and the audit identifies the regime. Novelty claims are gated on the literature
kill-tests (SPEC-novelty-collapse.md) until they return.

## 8. The rival school's basis choices (Mallat/Allys), and the math behind them

That school picks the wavelet by the OPERATION — the same pattern as our §5:
- **Statistics (scattering transform, WPH):** oriented COMPLEX redundant filters
  (Morlet / bump-steerable). Analyticity makes the modulus a smooth local envelope
  (demodulation); orientation captures anisotropy; redundancy is free because
  statistics never reconstruct. Phase harmonics (|z|e^{ikφ}) align phases so
  cross-scale correlations — the non-Gaussianity carriers — do not vanish.
- **Generation by statistic-matching (microcanonical models):** same filters; the
  model is "fields whose scattering/WPH statistics match the observed one," sampled
  by gradient descent — the n=1 school.
- **Conditional factorization across scales (WC-RG — Marchand, Ozawa, Biroli,
  Mallat):** ORTHOGONAL DECIMATED wavelets — the same structural choice we made, for
  the same reason (exact p(detail|coarse) factorization with free coordinates).

Theorem-grade pillars (name+year from memory — fetch-verify before any becomes
load-bearing): Mallat 2012 group-invariant scattering (translation invariance +
Lipschitz stability to diffeomorphisms — the rigorous "good texture statistic");
Bruna–Mallat microcanonical concentration (when statistic-matching defines a sane
ensemble); WC-RG's decoupling result — conditional wavelet distributions have
short-range, well-conditioned interactions even at criticality, i.e. RG makes each
rung an easy learning problem even when the global field is hard. That last theorem
is the mathematical license for the entire per-octave conditional strategy, theirs
and ours; the difference is that they model each rung with interpretable
maximum-entropy potentials trained with data at all scales, while we LEARN the rung
conditional (CFM) weight-tied and bet on extrapolating its measured drift.

**Gate-0 note (2026-07-16):** the novelty kill-test cleared both core findings in
NARROWED form — binding claim boundaries in log/2026-07-16-novelty-collapse.md
(adjudicated: log/2026-07-16-reconvene-gate0.md). "Collapse law" is a paper-internal
name for a synthesized instance of a pre-known phenomenon; our additions are the
channel-invariance, the exact decomposition, and the causal augmentation test. The
structured-not-additive conditioning drift is the sharpest surviving novelty.
Conditional-calibration validation has prior art (Schanz et al. 2310.06929, width-only)
that we systematize, not invent.
