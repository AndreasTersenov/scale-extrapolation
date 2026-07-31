# 2026-07-31 — RECONVENE R43: Stage-3 final adjudication — R-SEED-FRAGILE
# and B3-PARTIAL CONFIRMED; PHASE 3 CLOSES; the paper unfreezes
# METHOD-FIRST with the binding claim set below

Ruled on: the STAGE-3 READOUT appended to log/2026-07-31-prereg-stage3.md
(c7b3be9) + artifacts. Same-day git log; queue empty.

## Verification

1. **R12:** every adjudicating number re-checked verbatim — seed table
   (seed0 −1.12/−0.26% PASS @pick 16000; seed1 +3.63/+6.54% @3500; seed2
   +4.48/+6.64% @5500, both failing exactly [declared_res_peaks]; C
   in-band 3/3); blind verdict (branch B3-PARTIAL, failed_entries [E3];
   T_MF 6.3513 vs 3.5, not at-bar; native 5.3042; E1 all four rows pass
   with margins; E2 −1.14%/−4.00% CIs incl. 0, panels not sign-consistent;
   E4 pass; P-T C 0.7765±0.0046 in [0.7678, 0.8445]; determinism 0.0).
   ALL MATCH.
2. **A4 independently re-derived:** recomputed the null statistics from
   the stored per-split values in stage3_mf_null.json — native 1.43/0.47/
   2.65, smoothed 1.48/0.55/2.87, EXACT match to the prereg quotes. The
   conditional pre-clearance was consumed legitimately.
3. **A5/A6 compliance:** scoring order structural in both scorers;
   post-dry-run diff additive-only (scorer files); dry-run outputs
   quarantined and excluded from every adjudicating table. Suites re-run
   at adjudication: tests/ 44 pass; tests_wfm + tests_p2 pass (exit 0).
4. Branch mechanics re-derived for both legs — **R-SEED-FRAGILE and
   B3-PARTIAL (failed entry E3 only) are the unique mechanical outcomes.**

## Final scorecard (the phase's calibration record, both columns)

- (a): both modals (R-ROBUST 50/55) MISS; FRAGILE fired at rec 40 > exec
  35. all-3-declared-res line (55/65): both LOSE, rec less wrong.
  all-3-C-in-band (65/70): both HIT. picks-differ (50/50): FIRES.
- (b): both modals (B3-PASS 40/45) MISS; PARTIAL fired at rec 35 > exec
  30. E3 (62/70): FAILED — both on the wrong side, rec less wrong, and
  the miss is the GOOD kind: the tier was left untouched precisely so it
  could surprise. E1 (85) HIT; E2 (65/70) HIT; E4 (90) HIT; native-echo
  (75/75) HIT. **P-T: 5 consecutive validations across seeds, substrates,
  and the blind protocol — the transfer-model calibration is the phase's
  most reliable prediction engine.**
- Phase pattern, stated for the appendix: reality selected a non-modal
  branch at four of the phase's five decision points; the reconvene's
  systematically heavier pessimistic tails were closer at three of them,
  the executor's measured-dose-response reasoning won the fourth (L1″).

## Adopted findings (verdict-grade)

1. **The founding bet holds blind, third independent training:** a
   {3,4}-trained fresh seed passes edge marginals, starlet, and the
   declared-resolution peak rule at the held-out octave, one shot,
   pre-registered. This is the paper's headline table.
2. **The deconvolution calibration recipe is seed-robust and
   blind-validated** (C in its predicted band 5/5, incl. the two-octave
   deployment-pure target on the blind seed). It ships as the method's
   standard equipment.
3. **The declared-resolution peak claim is SEED-CONDITIONAL at trained
   scales:** it holds for the shipped configuration and at the blind edge,
   but fresh seeds carry +4–7% at 0.5 px — their caged picks landed early
   (@3500/@5500), continuous with the schedule story. ORDERED WORDING for
   the paper: the peak claim attaches to the SELECTED configuration (the
   selection protocol is part of the method), with the fresh-seed spread
   printed in the robustness table. Drafted follow-up (NOT run,
   post-paper): a late-window or matched-step selection rule as the
   candidate cure — it must not enter this paper's claims.
4. **The Minkowski morphology tier fails at both resolutions (6.35
   declared / 5.30 native) and enters the paper as the declared next
   boundary**, continuous with the located pixel-scale phase-texture
   mechanism. The untouched-judge design caught what five designed-against
   tiers could not — the campaign's strongest argument for held-out
   validation tiers in generator audits, and the paper's methodological
   thesis demonstrated on its own generator.
5. Dry-run bonus (appendix, descriptive): the fresh blind seed IMPROVED
   both E2 and T_MF over the committed stage-D configuration — seed
   variance cuts both ways, and the one-shot's pass is not a lucky draw
   of an unusually good seed relative to the committed lineage.

## PHASE 3 CLOSES

Experimental program of the phase: COMPLETE at ≈1.8 of 10 H100-h. The
paper skeleton UNFREEZES, reframed METHOD-FIRST. Binding claim set:

- VALIDATED: one-octave extrapolation on marginals, starlet-ℓ1, parity,
  spacing (pooled convention), and declared-resolution peaks (σ_s ≥ 0.5
  px, selected-configuration wording per finding 3) — trained scales and
  blind edge.
- SHIPPED TOOLS: the F2 group-averaged sampler; the measured-transfer
  deconvolution calibration (5/5 predictive record).
- MECHANISM SECTION: the moment ladder; the three dissociations
  (parity ✂ peaks, coloring ✂ peaks, spectrum ✂ peaks); the transfer
  function of weight-tied extrapolation; the L1→L1″ arc.
- DECLARED BOUNDARIES: native-resolution peak counts (located mechanism);
  Minkowski morphology (the next boundary); seed-conditionality of the
  declared resolution; the isotropy/stationarity/resolution caveat
  (staged text, prereg §c).
- APPENDIX: the reproducibility ledger (R1–R43), the calibration
  scorecard including every modal miss on both sides, the instrument
  lessons, the quarantined dry-run.

Venue decision (astro-first vs ML-methods, both intros drafted in
paper/VENUE.md) RETURNS TO ANDREAS with the phase closed. The next
executor session is a WRITING session per SPEC-paper-skeleton.md +
this ruling's claim set; no further experiments are authorized without a
new reconvene order.
