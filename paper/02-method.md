# §2 — The method (the paper's deliverable; every element earned in §4)

## 2.1 The object [WRITE from explainer §2 — codename-free source]

- Exact orthogonal-wavelet factorization p(x) = p(c_J) ∏ p(d_j | c_j);
  criticality/homogeneity/RG rationale (explainer §3, incl. the starlet
  contrast: redundant transforms are measurement bases, not generation
  coordinates).
- One weight-tied conditional-flow-matching network; weight-tying defined
  verbatim from explainer §10 ("one network, one parameter set, every rung —
  ... the entire license for extrapolation"). Octave 1 is generated, never
  trained, on every substrate in this paper — weight-tied extrapolation is
  the method's normal operating mode, not a special case (§4.5).
  <!-- src: c1t_selection_*.json meta train_octaves; R39 verification -->
- <!-- cite: CTFM 2505.00632 in the CFM paragraph -->

## 2.2 The measured setting the method is built on (generator-independent)

Condensed from the bedrock campaign; each result one paragraph + pointer:
- Structured scale drift, 2 PCA components ≥85%, physical identities;
  GRF null clean (z ≤ 0.6); the lognormal drifts like N-body.
  <!-- src: RESULTS.md K-M1a, P9a, P9b -->
- Conditional locality r*≈1 (LINEAR-scope caveat; the C2 discriminator as
  future work; WC-RG short-range theorem as anchor; §3 confirms at
  deployment level). <!-- src: RESULTS-phase2.md P-B1a -->
- N_eff ≈ parents (13.7 ≈ parent count at every crop stride) — the currency
  claim the design spends. <!-- src: RESULTS-phase2.md B2 -->
Figures: F1 ladder opener, F2 drift fan-out, F3 locality.

## 2.3 The final configuration, element by element (each earned, §4 pointers)

| element | what | earned in |
|---|---|---|
| base | unit-variance Student-t(ν=5) | §4.4 (tail starvation; base erasure) |
| training | CFM, D4-augmented, octaves per recipe | §4.2 (collapse cure) |
| selection | caged, pre-registered validation pick at a TRAINED octave — **part of the method** (R43 finding-3 wording; the peak claim attaches to the selected configuration) | §4.4, §5.3 |
| sampling | F2 group-averaged: d = g⁻¹·model(g·c), g ~ Uniform(D4) per field per octave — exactly D4-equivariant in law | §4.5 (lattice-parity cure) |
| oct-1 base calibration | the measured-transfer deconvolution (§2.4) | §4.5, §3 |

## 2.4 The deployment calibration: measured-transfer deconvolution
## [WRITE from the L1→L1″ arc, method-voice]

The recipe, four steps, requiring ONLY the trained checkpoint and generated
maps (no target-octave data): (i) generate a white-base stream; (ii) measure
the per-ring transfer T(k) = S_out(k)/S_white on the extrapolated octave's
detail planes; (iii) deconvolve the target ring shape (a TRAINED octave's
shape rescaled in k/N — deployment-pure; licensed by the measured
near-scale-invariance C = 0.786/0.774/0.785 at octaves 1/2/3
<!-- src: stage1_p3_probes.json coloring -->); (iv) color the base by the
calibrated filter through the t(5) quantile map (copula construction —
exactly D4-invariant by ring-lookup). Predictive record: 5/5 pre-registered
band landings (§3.4); the oracle variant lands ON its prediction at 0.04σ.
<!-- src: l1pp_pt.json; l1pp_verdict.json -->

## 2.5 The declared domain (staged caveat text, R43; Stage-3 prereg §c verbatim)

"The generator models per-tile-standardized flat-sky patches. Three
declared-domain restrictions follow. (i) SYMMETRY: the sampler is exactly
equivariant in law under the discrete group D4 (group-averaged sampling;
the base construction is ring-exact under the grid action); full continuous
isotropy is inherited only to the extent the training data carries it, and
statistics sensitive to sub-group anisotropy should be validated per
application. (ii) STATIONARITY: per-tile standardization removes
patch-scale mean/variance modulation; the validated statistics are those
invariant under per-patch standardization on scales up to the tile size
(128 px); super-tile correlations are outside the domain. (iii) RESOLUTION:
peak-count statistics are calibrated at σ_s ≥ 0.5 px (FWHM ≈ 1.2 px) and
above, covering standard weak-lensing smoothing practice; native-resolution
counts carry a +14% excess of pixel-scale, spectrum-independent origin
(located mechanism, §4.5) and are declared outside the validated domain.
The octave-1 base is calibrated per deployment by the measured-transfer
deconvolution (§2.4), a procedure requiring only generated maps and the
trained checkpoint." <!-- src: log/2026-07-31-prereg-stage3.md §c -->
