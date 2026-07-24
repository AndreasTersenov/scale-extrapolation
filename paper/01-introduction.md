# §1 — Introduction

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
  — exactly the level our downstream demonstration indicts (§3.4).
- The Schanz correction governs: conditional-diversity checking EXISTS in
  prior art (width-only, single conditioning, qualitative tails); we extend
  it to a calibrated, scale-resolved, pre-registered audit with downstream
  propagation. <!-- cite: Schanz 2310.06929; forbidden: "SR papers validate
  only summary statistics" -->

## 1.2 The two inseparable questions [WRITE]

Generation beyond training resolution; and what validation even means there
(the sample asymmetry: training needs thousands of fields, validating summary
statistics needs tens — explainer §10 "Isn't extrapolation speculation?" is
the source).

## 1.3 The three-tier gap (the paper's organizing claim)

Power-level checks → conditional marginals (dispersion, tail weight) →
joint/morphological structure. Each tier fails invisibly to the tier below —
demonstrated three times (§3.4, §4, §5). R27 wording throughout.

## 1.4 Contributions (bulleted, each with its section pointer)

1. A three-tier, exact-truth-calibrated audit protocol for conditional
   generative emulators (§3), with a held-out-basis check and a downstream
   bias demonstration.
2. Three measured failure mechanisms with causal evidence (§4): the collapse
   law (paper-internal name, Gate-0 discipline), the mixture/instrumentation
   artifact, and moment-ladder tail starvation (causal data-size signature at
   two rungs; base-erasure).
3. Generator-independent measurements the field can reuse (§2): 2-component
   drift; conditional locality r*≈1; N_eff ≈ parents.
4. A deployment-blind single-octave extrapolation pass on the cured
   substrate, scale-blind (§5) — with the honest boundary (§5.4, §6).

## 1.5 Scope sentence (verbatim, in the introduction as well as the abstract)

One octave; 2D gravity-only weak-lensing fields; one simulation family
(gowerstreet: 266 training tiles from 30 parent maps, one seed family);
sandbox = exact-truth lognormal (322+64+64 tiles). Never "certified";
"RG-inspired".
