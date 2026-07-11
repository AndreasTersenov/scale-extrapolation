# 2026-07-11 — reconvene ruling: (c') option 2 APPROVED

Evidence accepted: two independent training-time penalties (random-t (c), late-t (c'-1))
failed the frozen bar identically — the deterministic-ODE pushforward variance is the
structural limit, not the loss weighting. Escalation to option 2 (Gaussian-NLL detail
head: model conditional variance explicitly, sample with it) is APPROVED as
pre-registered, with two conditions:
1. **Arm symmetry:** BOTH arms A and B get the new head — the P5/P6 comparison must
   keep scale-conditioning as the only differing variable.
2. **Pre-register the sampling procedure** (how the predicted log-variance enters
   generation) and the same frozen bar: trained-octave var_slope within 1σ at octaves
   2–4 simultaneously, GRF null preserved (the NLL head must NOT break the null —
   Gaussian details on GRF should be learned as such). Then P6/P13 re-adjudication,
   frozen bars unchanged.
If option 2 also fails the bar, the finding "deterministic and variance-headed CFM
both under-disperse" is itself the paper's methods result — report and reconvene.

## ADDENDUM (2026-07-11, foundations review with Andreas) — binds option-2 adjudication

1. **Kurtosis check pre-registered:** the Gaussian-NLL head models mean+variance, not
   tails; conditional kurtosis is one of our running couplings (12.8 at fine octaves).
   Hypothesis permitting a Gaussian head: conditioning on the coarse environment
   Gaussianizes the details (amplitude-modulation lore). MUST be checked, not assumed:
   option-2 success additionally requires conditional kurtosis at trained octaves
   within 2σ of real. Pre-named fallback if it fails: student-t NLL head (df learned or
   swept), same bars.
2. **Held-out-statistics principle (phase-2 frozen-core candidate, noted now):** repair
   design uses {var_slope, kurtosis, cross-octave ρ}; validation must also score
   statistics NEVER used in design (wavelet-L1, peaks — keep them strictly out of any
   tuning loop). Rationale: the tracked-statistics certificate must not commit the
   summary-passes-while-law-is-wrong error (Doeser & Jasche, Andreas's reliability
   thread) against itself.
3. **Coupling-extrapolation gap (phase-2 item):** phase 1 conditions on MEASURED
   target-octave couplings; production requires EXTRAPOLATED couplings. Phase 2 must
   test the one-octave-validated protocol (hold out finest octave; extrapolate the
   2-D coupling curve into it; quantify sensitivity of generated statistics to
   coupling-curve error).
4. **Under-dispersion novelty kill-test (before any paper claim):** sweep the variance
   miscalibration / mean-collapse literature for conditional diffusion/FM; find the
   exact edge of what is new in our law ("monotone worsening with training on
   scientific fields + the structural pushforward mechanism + the fix").

## ADDENDUM 2 (2026-07-11, positioning review vs GOLCONDA + ST school) — phase-2 planning inputs

Three-paradigm map (same wavelet backbone, different source of the law): GOLCONDA
(arXiv:2507.01707, github.com/vilasinits/GOLCONDA) = constraint projection onto
theory-predicted stats (LDT PS+l1+inter-scale; no sims; cosmology-dialable);
ST/microcanonical/WC-RG = matched measured summaries (n=1 capable); D4 = learned
conditional law (needs coarse-res sims; extrapolates the measured 2-D flow).

D4's honest differentiators: (1) extrapolation beyond BOTH theory validity and sim
resolution; (2) the joint law beyond enforced summaries (feeds CNN/field-level
inference); (3) amortized generation (forward pass vs per-map optimization).
Their strongholds (state in paper): GOLCONDA zero-sim + cosmology-parameterized today;
ST n=1 + interpretable; both enforce exactly. "If you only need PS+l1 maps at sim
resolution, use GOLCONDA" goes in the paper.

Phase-2 frozen-core candidates from this review:
- Baselines head-to-head: GOLCONDA (public code) + a microcanonical-ST model, identical
  training information; scored on enforced stats, HELD-OUT stats (peaks, Minkowski,
  scattering covariance — the rival school's instrument as judge), extrapolated-octave
  truth, and cost-per-map.
- **LDT anchor:** validate extrapolated-octave wavelet-l1 against the theoretical LDT
  prediction (2406.10033) where valid — an INDEPENDENT reference in the no-truth
  deployment regime (partial fix to the deepest validation gap). Coordinate with
  Vilasini/Jean-Luc (alliance, not competition; candidate collaboration).
- Cosmology-conditioning (Gower Street grid) noted as the answer to GOLCONDA's
  parameterization advantage — phase-2+ scope decision.
