# Title options (Andreas picks; direction per SPEC)

1. **"An audit-guided anatomy of scale-hierarchical generative models for
   cosmological fields: failure mechanisms, causal cures, and validated
   single-octave extrapolation."** (the SPEC's direction, tightened)
2. **"The moment ladder: how conditional generative models of cosmological
   fields fail statistically, and how to know."** (mechanism-first; shortest)
3. **"Trustworthy beyond-data generation for cosmological fields: a
   three-tier audit, three failure mechanisms, and a blind extrapolation
   test."** (audit-first; mirrors the section spine)

Recommendation: 1 for an astro-methods venue, 2 for an ML venue (VENUE.md).

# Abstract skeleton ([WRITE]: connective prose; claims fixed)

Order of sentences (one each):
1. Problem: emulators/SR generators are validated at power-spectrum level;
   the statistics that carry survey information live above that level.
2. Object: exact wavelet-conditional factorization, one weight-tied
   conditional-flow-matching network per ladder; extrapolation = apply the
   rule one octave beyond training. "RG-inspired."
3. Method: pre-registered campaign; exact-truth sandbox calibration
   (instruments to ≲1%); three-tier audit (power → conditional marginals →
   joint/morphological). <!-- src: gateA_instrument.json; R27 wording -->
4. Findings (mechanism): three failure mechanisms invisible to tier-below
   validation — finite-data conditional-variance collapse (causally cured by
   exact-symmetry augmentation), an instrumentation artifact masquerading as
   an information limit (exposed by a forensic decomposition), and tail
   starvation on a moment ladder (causally data-limited at two rungs; cured
   at fixed data by a heavy-tailed base + pre-registered checkpoint
   selection).
5. Result: under a deployment-blind protocol the cured generator passes the
   founding bet — end-to-end dispersion within 4.7%/7.9% and tail weight
   within 3.2%/4.4% at the held-out octave (conditional-level tail check of
   the reference arm passes its 15% floor with a 0.1% margin)
   <!-- src: stageD_verdict.json; R21 margin discipline -->
   — and does so SCALE-BLIND: the coarse conditioning itself carries the
   scale information. <!-- src: R22 -->
6. Boundary: the starlet-ℓ1 passes out-of-basis (+4.0% at the edge) while a
   +13.1%±3.1% / +14.6%±3.1% joint/morphological peak-count excess remains —
   even the field's most constraining marginal statistic cannot see the
   morphological excess. One octave validated; the second octave's cost is
   measured. <!-- src: starlet_l1_edge.json; audit_peak_ci.json; R27 -->
7. Scope sentence (verbatim scope words) + the audit protocol is exportable.
