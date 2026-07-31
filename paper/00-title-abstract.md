# Title options (Andreas picks; METHOD-FIRST direction per R43)

1. **"Validated single-octave extrapolation for cosmological fields: a
   wavelet-conditional generator with a measured-transfer calibration and a
   pre-registered blind audit."** (method-first; the R43 framing)
2. **"A generative method for beyond-resolution cosmological fields, with
   its own audit: blind extrapolation, a deployable spectral calibration,
   and the boundaries we measured."** (method + honesty in the title)
3. **"The moment ladder and its cures: building a validated
   scale-extrapolating generator by audit-guided design."** (mechanism-
   forward alternative; closest to the pre-phase-3 framing)

Recommendation: 1 for astro-methods, 3 for ML (VENUE.md; Andreas decides).

# Abstract skeleton ([WRITE]: connective prose; claims fixed by R43)

Order of sentences (one each):
1. Problem: emulators/SR generators are validated at power-spectrum level;
   the statistics that carry survey information live above that level.
2. METHOD (the paper's deliverable, stated first per R43): an exact
   wavelet-conditional cascade — one weight-tied conditional-flow-matching
   network, heavy-tailed base, pre-registered caged checkpoint selection,
   exactly D4-equivariant group-averaged sampling, and a measured-transfer
   deconvolution that calibrates the finest-octave spectrum at deployment
   with no retraining. "RG-inspired"; the selection protocol is part of the
   method (R43 finding-3 wording). <!-- src: R43 claim set -->
3. VALIDATION (headline): under a pre-registered one-shot blind protocol on
   a fresh training seed, the method passes at the held-out octave:
   marginal bars, the held-out-basis starlet-ℓ1, and peak counts at the
   declared resolution (σ_s ≥ 0.5 px) — excesses −1.1% ± 2.0% and
   −4.0% ± 2.3% at ν = 2.5/3.0.
   <!-- src: stage3_blind_verdict.json E2; branch B3-PARTIAL -->
4. CALIBRATION: the finest-octave coloring lands in its predicted band in
   five of five pre-registered tests across seeds, substrates, and the
   blind protocol — the flow's spectral action is a measured, input-
   independent transfer function, and deconvolving it is standard
   equipment. <!-- src: l1pp_pt.json; stage3_seed1_pt/c, stage3_seed2_pt/c,
        stage3_blind_c.json; l1p_transfer_analysis.json -->
5. MECHANISM: three failure mechanisms found and cured by the audit
   (conditional-variance collapse; an instrumentation artifact; tail
   starvation on a moment ladder), and three measured dissociations that
   locate the surviving native-resolution peak excess in pixel-scale phase
   structure — independent of lattice symmetry, of base coloring, and of
   the full octave-1 power spectrum at fixed weights.
   <!-- src: stage0_p3_verdict.json; l1p_verdict.json; l1pp_verdict.json -->
6. BOUNDARIES (declared, not hidden): native-resolution peak counts (+14%
   excess, located mechanism); Minkowski-functional morphology — an
   untouched held-out judge applied exactly once, which failed the final
   configuration (T = 6.35 vs a 3.5 bar) and thereby demonstrated the
   paper's methodological thesis that held-out validation tiers catch what
   designed-against tiers cannot; and seed-conditionality of the declared
   resolution (fresh seeds carry +4–7% at 0.5 px; printed in the
   robustness table). <!-- src: stage3_blind_verdict.json E3;
        stage3_a_table.json; R43 findings 3-4 -->
7. Scope sentence (verbatim scope words) + the audit protocol and the
   calibration recipe are exportable to any conditional emulator.
