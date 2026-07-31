# §1 — Introduction (METHOD-FIRST per R43)

## 1.1 The trust problem [WRITE]

- Surveys consume simulated maps; the constraining information is
  non-Gaussian and small-scale; high-resolution simulation is the cost
  bottleneck. (Explainer §1 prose is the source — codename-free by
  construction.)
- The emerging emulator/SR literature validates at the power-spectrum /
  pixel-PDF level. Verbatim exhibit (respectful): MultiscaleFlow's own
  validation sentence — "samples and test data agree well in terms of the
  power spectrum and pixel probability distribution function"
  <!-- cite: MultiscaleFlow 2306.04689; per reconvene-gate0 §4 -->
  — exactly the level our downstream demonstration indicts (§4.1).
- The Schanz correction governs: conditional-diversity checking EXISTS in
  prior art (width-only, single conditioning, qualitative tails); we extend
  it to a calibrated, scale-resolved, pre-registered audit with downstream
  propagation. <!-- cite: Schanz 2310.06929; forbidden: "SR papers validate
  only summary statistics" -->

## 1.2 What this paper delivers [WRITE — the method-first re-statement]

A generator you can use, with the validation that says exactly where: the
method (§2), its blind validation and robustness tables (§3), the mechanism
evidence behind each design element (§4), and the measured boundaries (§5).
The two inseparable questions (generation beyond training resolution; what
validation even means there — the sample asymmetry, explainer §10) framed
as the reason method and audit ship together.

## 1.3 The three-tier gap and the held-out-tier thesis

Power-level checks → conditional marginals (dispersion, tail weight) →
joint/morphological structure; each tier fails invisibly to the tier below
— demonstrated three times (§4.1, §4.2, §5). R27 wording throughout. NEW
(R43 finding 4): the paper's methodological thesis — validation suites need
a tier the designers never optimized against — is demonstrated ON OUR OWN
GENERATOR: a Minkowski-functional judge, frozen before any cure was
designed and applied exactly once at the blind test, failed the final
configuration where five designed-against tiers passed (§5.2).

## 1.4 Contributions (bulleted, each with its section pointer; R43 claim set)

1. A validated method for single-octave extrapolation of 2-D cosmological
   fields: wavelet-conditional cascade + heavy-tailed base + caged
   pre-registered selection + exactly D4-equivariant sampling (§2),
   passing marginals, held-out-basis starlet-ℓ1, and declared-resolution
   peak counts at a held-out octave under a one-shot blind protocol (§3).
2. A deployment-pure spectral calibration for the extrapolated octave —
   the measured-transfer deconvolution — with a 5/5 pre-registered
   predictive record across seeds, substrates, and the blind protocol
   (§2.4, §3).
3. Three measured failure mechanisms with causal cures (§4): the collapse
   law (paper-internal name, Gate-0 discipline), the mixture/
   instrumentation artifact, and moment-ladder tail starvation.
4. Three measured dissociations locating the native-resolution peak excess
   in pixel-scale phase structure (§4.5): lattice symmetry ✂ peaks,
   base coloring ✂ peaks, octave-1 spectrum ✂ peaks — plus the transfer
   function of weight-tied extrapolation as a measured object.
5. The audit protocol itself, exportable: exact-truth calibration,
   three tiers + a held-out basis + an untouched judge, pre-registration
   with weighted branches, and the boundary declarations (§5) as the
   template for honest emulator validation.
6. Generator-independent measurements the field can reuse (§2.2): 2-component
   drift; conditional locality r*≈1; N_eff ≈ parents.

## 1.5 Scope sentence (verbatim, in the introduction as well as the abstract)

One octave; 2D gravity-only weak-lensing fields; one simulation family
(gowerstreet: 266 training tiles from 30 parent maps, one seed family);
sandbox = exact-truth lognormal (322+64+64 tiles). Never "certified";
"RG-inspired". Peak claim at declared resolution σ_s ≥ 0.5 px, attached to
the SELECTED configuration (R43 finding-3 ordered wording, §5.3).
