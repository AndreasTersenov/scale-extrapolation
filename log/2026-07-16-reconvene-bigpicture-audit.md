# 2026-07-16 — RECONVENE BIG-PICTURE AUDIT (expert-lens, requested by Andreas after the D1 park)

Standard applied: the same one that parked D1 — would Lanusse or Mishra-Sharma see
through it? Sources: PLAN.md (1c frozen core), RESULTS-toy/1c, the cprime2 ruling +
addenda, the 4a/4b′/4b′-ii record, the positioning reviews (GOLCONDA/ST, Dai–Seljak).

## What survives expert scrutiny

1. **The stage-0 measurement is the bedrock and it is generator-independent.** The
   between-scale drift of N-body fields is real (3–7σ per octave), LOW-DIMENSIONAL
   (2 smooth components ≥85%), field-general (hf_pm replication), and null-clean
   (GRF |z|=0.15). This is a small true fact about the data that nobody disputes
   regardless of what any network does. Measurement-before-architecture is the
   project's best habit and both experts would recognize it.
2. **The validation architecture is the second real asset**: slide-the-edge (hold out
   the finest octave, extrapolate INTO ground truth), held-out statistics never used
   in design (scattering covariance — the rival school's instrument as judge), the
   LDT theory anchor (independent reference in the no-truth regime), self-consistency
   of couplings. Lanusse in particular would approve of the GOLCONDA-alliance +
   LDT-anchor design — it converts the deepest criticism ("you can't validate beyond
   your data") into a research contribution.
3. **The honest positioning already exists in writing** ("if you only need PS+l1 maps
   at sim resolution, use GOLCONDA"; MultiscaleFlow = nearest relative, no tying → no
   extrapolation). No overselling to unwind.

## Where they would press — three uncomfortable findings

**(A) The flagship's best finding has not passed its own pre-registered novelty test.**
Addendum 4 of the cprime2 ruling requires the under-dispersion/collapse-law kill-test
"before any paper claim." It has never run. The law's neighborhood is heavily trodden:
exposure bias / scheduled sampling (our compounding), conditioning augmentation in
cascaded diffusion (we imported the fix from it!), regression-to-the-mean blur in
probabilistic forecasting (GenCast/CRPS lineage), posterior collapse, variance
miscalibration in conditional diffusion/FM. What MIGHT be ours: the monotone-with-
training form, the finite-data mean-memorization mechanism with a CAUSAL data-size
intervention, invariance across three variance channels, on scientific fields. Might.
Until the kill-test runs, internal status = presumed known. SPEC-novelty-collapse.md
written today; assign to a cheap session NOW, in parallel with attempt 5.

**(B) The validated claim is one octave, and today it conditions on measured couplings.**
Slide-the-edge certifies exactly one octave of extrapolation; multi-octave generation
is uncertified extrapolation of the coupling curve itself (the addendum-3 gap:
deployment needs EXTRAPOLATED couplings; phase 1 uses measured ones). Language
discipline, binding from now: "validated one-octave extrapolation under measured
couplings; coupling-curve extrapolation quantified separately" — never "certified
super-resolution," never unqualified "beyond-data generation." The word "certified"
is reserved for D1-style exactness machinery; here we VALIDATE within a declared
domain.

**(C) The use-case economics don't bind on our demo domain.** For 2D convergence maps,
high-resolution sims are not the community's bottleneck — if training pairs exist,
training AT target resolution is usually available. The pitch has teeth where high-res
is genuinely expensive: 3D / hydro / Lyα — which is exactly the Starck-Vilasini/
postdoc direction. Own this in all framing: "method developed and falsified on 2D WL
(cheap, instrumented); the target regime is 3D/hydro." The domain question (stay 2D
for thesis-adjacent demo vs invest toward the binding regime) is an Andreas decision
at the attempt-5 fork, not before.

Also noted, milder: (D) the RG language is an inductive-bias motivation, not a
theorem — weight-tying + scale conditioning is parameter sharing; what makes it
science is the measured 2-D drift justifying it. Write "RG-inspired." (E) The
attempt-ratchet: G-1c said stop after student-t; attempts 4a/4b′/4b′-ii/5 each got
new evidence-driven rulings, all cheap, but an auditor counts five. The attempt-5
finality pre-commitment (log/2026-07-11-reconvene-4bpii.md) is the answer and HOLDS.

## Why the verdict differs from D1's

D1 failed its own performance and usefulness gates and had no user in any branch. D4
has: a real measured phenomenon at its core (survives any generator outcome); a
per-level model that now works (heads at real ceilings post-corruption); exactly ONE
named remaining mechanism with its textbook fix authorized and bounded (attempt 5,
minutes of compute); a pre-named fallback paper that is honest and useful (break +
collapse law + validation architecture — pre-named in cprime2 itself, BEFORE the
failures accumulated); thesis and postdoc fit; and an alliance path. The rational
posture is: proceed on attempt 5, run the overdue kill-test in parallel, enforce the
language rules, and put the domain/paper fork in front of Andreas when attempt 5
reads out.

## The two-branch paper, pre-stated now (so the fork is a choice, not a scramble)

- **Attempt 5 passes the lever bar:** method paper — "RG-inspired conditional
  generation with measured running couplings: one-octave validated extrapolation on
  cosmological fields," with GOLCONDA/ST head-to-heads and the LDT anchor; kurtosis/
  student-t addressed on its own track; aimed explicitly at the 3D/hydro regime as
  future work. NeurIPS-track plausible IF the kill-test leaves us a novel mechanism
  claim; otherwise a strong ML4PS/A&A-methods paper.
- **Attempt 5 fails:** measurement-and-failure-law paper — "the between-scale
  conditional law of cosmological fields: a measured 2-D drift, how learned
  between-scale maps fail (a causally-tested collapse mechanism), and a validation
  architecture for beyond-data claims." Modest, true, useful to everyone building
  Dai–Seljak-style models. Workshop/methods-journal grade unless the kill-test
  upgrades the law's novelty.

Neither branch is embarrassing in front of the two named experts. That is the audit's
bottom line.
