# §2 — The setting and the generator-independent measurements

## 2.1 The object [WRITE from explainer §2 — the source is codename-free]

- Exact orthogonal-wavelet factorization p(x) = p(c_J) ∏ p(d_j | c_j);
  criticality/homogeneity/RG rationale (explainer §3, incl. the starlet
  contrast: redundant transforms are measurement bases, not generation
  coordinates).
- One weight-tied conditional-flow-matching network; weight-tying defined
  verbatim from explainer §10 ("one network, one parameter set, every rung —
  ... the entire license for extrapolation").
- The final model's two deliberate modifications forward-referenced to §4
  (heavy-tailed base; pre-registered checkpoint selection) — introduced as
  cures, not choices. <!-- cite: CTFM 2505.00632 in the CFM paragraph -->

## 2.2 The drift measurement (bedrock result 1)

- GRF null PASSES: adjacent-octave excess drift z = 0.6/0.1/−0.2 (clean
  octaves max |z| = 0.15 per RESULTS.md verdict table); synthetic power-law
  null green as an executable test. <!-- src: RESULTS.md K-M1a -->
- N-body drift significant: adjacent-octave 3–7σ, 48–59% of the
  inter-distribution distance; cumulative z≈9–11.
  <!-- src: RESULTS.md P9a -->
- Low-dimensional: 2 PCA components ≥85% of cross-octave variation, every
  drifting field; components have physical identities (modulation slope,
  tail weight). <!-- src: RESULTS.md P9b -->
- The lognormal drifts like N-body (1.54→0.53) — drift is generic to
  multiplicative fields. <!-- src: log/2026-07-16-prereg-step4-selfsim.md -->
- Fig: stage-0 fan-out (FIGURES.md F2).

## 2.3 Conditional locality (bedrock result 2)

r* ≈ 1 coarse pixel: gowerstreet detail predictability saturates at the
nearest coarse ring (9% drop at r=1, flat to r=12; amplitude channel adds
~4% by r=12); by the pre-registered mechanical rule r*=0, by eye r*≈1 —
stated with the rule caveat. LINEAR-scope caveat + the C2 discriminator
noted as future work (SPEC constraint). WC-RG short-range theorem cited as
the theory anchor; §5 confirms at deployment level.
<!-- src: RESULTS-phase2.md P-B1a; fig maps_locality.png -->

## 2.4 The data accounting (bedrock result 3)

N_eff ≈ 13.7 ≈ parent count at EVERY crop stride (12-parent fallback run;
crops buy ~nothing): the effective sample size of a tiled training set is the
number of independent parents. This is the currency claim the rest of the
paper spends (augmentation cure §4.1; rung-4 data demand §4.3; the placement
experiment's design). <!-- src: RESULTS-phase2.md B2 -->

## 2.5 What these measurements already imply [WRITE]

A scale-blind model is measurably wrong (drift is real); the correction is
low-dimensional; the conditional is short-range; data is parent-limited.
The founding hypothesis (an explicit scale dial is needed) dies in §5 — set
it up honestly here.
