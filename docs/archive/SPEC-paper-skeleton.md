# SPEC — the paper skeleton (writing phase; authorized by R25, 2026-07-18)

**For the D4 session (Fable or strong model — writing is judgment-heavy). Read
first: log/2026-07-18-reconvene-stageD.md (R21–R25), RESHAPE-MEMO.md incl. the
07-17 revision, log/2026-07-16-novelty-collapse.md (binding claim wordings),
log/2026-07-16-reconvene-gate0.md (citation list). Deliverable: paper/ directory
with a full skeleton draft (LaTeX or md → Andreas's choice of venue format
later): section files, figure plan with existing-artifact pointers, claim-by-claim
citation placement, and a TODO ledger separating [WRITE] from [HARDEN-FIG] tasks.**

## Working title direction (draft 2–3 options, Andreas picks)

Around: "An audit-guided anatomy of scale-hierarchical generative models for
cosmological fields: failure mechanisms, their cures, and validated single-octave
extrapolation." Honest, mechanism-first, extrapolation as the payoff not the promise.

## Section skeleton (map each to existing artifacts — nothing is written from memory)

1. **Introduction** — the trust problem for generative emulators; the three-tier
   validation gap (power → marginal → joint); positioning: systematizing Schanz's
   conditional-width check; WC-RG lineage for the factorization; the
   one-octave-validated scope stated in the abstract already.
2. **The measurements** (generator-independent bedrock): 2-D smooth drift
   (stage-0); conditional locality r*≈1 with the LINEAR-scope caveat and the C2
   discriminator noted as future work; N_eff ≈ parents. Figures: stage-0 fan-out,
   maps_locality, B2 table.
3. **The moment ladder** (mechanism): collapse law at rung 2 (channel-invariance,
   decomposition, causal augmentation cure — Gate-0 wording VERBATIM); rung 4
   tail decay (bake-off trajectories, 8× hold, base-erasure P3). Figures:
   signature_4a, taildyn, c1_sandbox.
4. **The audit protocol**: instruments + Gate-A truth calibration; the inverted
   pilot; the three tiers with their exhibits (NLL-head sign-flip, C1 up-tilt,
   C1-t placement residual). Figures: gateA, pilot_validation, downstream_peaks,
   c1t_maps_peaks.
5. **The audit-guided arc**: G-1c → forensic (mixture artifact; F-OVERSHOOT with
   the estimand caveat) → C1 (head retired) → tail diagnosis → C1-t (base +
   caged selection; attribution: both necessary). Figures: nll forensics panel,
   c1t verdict, maps_channels.
6. **Stage D**: deployment-protocol slide-the-edge; D-PASS-BOTH at bare floors
   (A-hc 0.1% margin STATED); the dial finding (scale-blind works; dial flips
   joint sign; drift real but locally carried); two-octave cost curve. Figure:
   stageD.png (4-panel, near paper-ready).
7. **Discussion**: frontiers (joint placement; multi-octave; 3D/hydro where
   economics bind — Option-3 prospectus paragraph); the data-investment thesis
   (constrained conditional ensembles — parents are the currency, causal at two
   rungs); limitations verbatim from scope words; the under-confidence scorecard
   note as a methods observation.
8. **Reproducibility appendix**: preregs/rulings ledger, config hashes, the
   scorecard table (hits AND misses, both roles).

## Binding constraints (non-negotiable in every draft)

Gate-0 bottom-line wordings + 07-17 scope clause verbatim where claims 1/2 appear;
"collapse law"/"moment ladder" introduced as paper-internal names; never
"certified", never any-scale; arm A = reference generator (B for the dial-effect
section); R12 numbers-by-copy (every number in the draft carries an artifact
pointer comment); the mandatory citation list from reconvene-gate0 placed
claim-by-claim; Schanz positioned as prior art in §4's first paragraph.

## Venue framing (draft the trade-off note, Andreas decides)

A&A/OJA methods-length vs NeurIPS-track: draft §1 both ways (one page each);
the Gate-0 novelty margins (channel-invariance, causal cures, structured-drift,
three-tier audit, scale-blind extrapolation) support either; the physics
demonstrations favor the astro venue. No recommendation baked into the skeleton.

## Process

[WRITE] tasks per section; [HARDEN-FIG] = starlet-ℓ1 into the held-out suite +
nanmedian + SC bands + any figure regeneration (cheap jobs, prereg-exempt as
pure re-rendering of committed numbers). One session, STOP with the skeleton
committed for Andreas's read — no submission-ready prose polishing before his
structural sign-off.

## Addendum (2026-07-24) — updates since this spec was written

1. **The starlet-ℓ1 readout is in** (log/2026-07-18-starlet-l1-readout.md +
   reconvene ruling): §4 gains the held-out-basis exhibit (position-blind ℓ1
   passes where the peak audit fails — the tier claim demonstrated with the
   field's own instrument) and the taxonomy figure (starlet_l1.png); §6 gains
   the edge-leg row (+4.0%). The instrument-findings note (upstream package
   bugs) belongs in the reproducibility appendix as an example of auditing the
   validators.
2. **PROJECT-EXPLAINER.md exists** — use it as the narrative source for §1–§3
   prose (it is codename-free by construction) and mine its §10 (the skeptic
   questions) for the discussion section: the train/validate sample asymmetry,
   the fractality clarification, and the principled-vs-recipe audit of the
   fixes are referee-proofing material.
3. **The placement experiment (SPEC-placement-test.md) is now the frontier
   section's forward pointer** — cite it as registered future work; if its
   readout lands before submission, it becomes a results subsection either way
   (both branches are mechanisms).
4. **Venue note:** the scale-blind result + moment ladder strengthen the
   ML-methods framing; the starlet/peak demonstrations strengthen the astro
   framing. The trade-off memo the spec requests should weigh both WITH the
   Phase-0 audit's independent view.
