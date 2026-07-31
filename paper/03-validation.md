# §3 — Validation (the headline section per R43: the blind one-shot table
# + the robustness table with the fresh-seed spread printed)

## 3.1 The audit in one paragraph [WRITE; full protocol in §4.1]

Exact-truth-calibrated instruments; three tiers (R27 wording verbatim);
held-out basis (starlet-ℓ1, independent code, never a design input);
untouched judge (Minkowski functionals, frozen before any cure was
designed, applied exactly once, §5.2); pre-registration with weighted
branches and mechanical rules throughout; one-shot discipline for the
headline (dry-run quarantined; no statistical reruns).

## 3.2 THE HEADLINE TABLE — the one-shot blind edge re-run (final
## configuration, fresh seed, pre-registered bars, scored once)

Protocol: train octaves {3,4} on the real field, fresh seed, caged
selection at octave 3 (a trained octave); octave 2 = the held-out edge;
oct-1 deconvolution target = the OCT-3 ring shape rescaled two octaves
(the oct-2 shape would leak the held-out octave — blindness constraint
stated in the prereg); no edge-octave statistic of any kind consulted
before adjudication. <!-- src: log/2026-07-31-prereg-stage3.md (b) -->

| entry | result | numbers |
|---|---|---|
| edge marginals (hc + e2e) | PASS (pre-registered bars) | hc var_slope rel 14.5% (bar 22.0%), hc kurt 25.5% (bar 59.0%); e2e var_slope 9.5% (bar 19.3%), e2e kurt 31.1% (bar 50.5%) <!-- src: stage3_blind_verdict.json E1 --> |
| peaks at declared resolution (σ=0.5 px) | PASS | ν2.5 −1.14%±2.03 ci[−5.0,+3.0]; ν3.0 −4.00%±2.34 ci[−8.4,+0.8]; panels not sign-consistent <!-- src: idem E2 --> |
| starlet-ℓ1 (held-out basis) | PASS | all scored scales <!-- src: starlet_l1_stage3_blind.json --> |
| Minkowski judge (untouched tier, first application) | **FAIL** | T = 6.35 vs bar 3.5 (declared res); native 5.30 <!-- src: idem E3; §5.2 --> |
| calibration P-T | LANDS | C 0.7765±0.0046 in [0.7678, 0.8445] <!-- src: stage3_blind_c.json --> |

MANDATORY DISCLOSURES (attach wherever this table appears):
- Bar-ledger #9 both ways: the pre-registered marginal bars are
  max(bare floor, 3·SE) with a 32-field reference; on the BARE floors
  (10%/15%) three of four marginal rows would not pass (hc 14.5%/25.5%,
  e2e kurt 31.1%). The original Stage-D run (§3.3) passed the bare floors;
  the blind re-run's marginal margins are weaker — the seed-spread story
  (§3.4) printed here, not hidden.
- Branch: B3-PARTIAL (failed entry E3 only) under the pre-registered
  mechanical table; supporting rows: parity_T 2.10, nn 2.85±0.20 (pooled
  convention), native peaks +30.5/+14.8/+11.4% at ν=1/2.5/3 (descriptive,
  #10). <!-- src: stage3_blind_verdict.json -->

## 3.3 The founding blind pass (the original Stage-D run, third-party
## context for 3.2) [carried from the pre-phase-3 skeleton, condensed]

D-PASS-BOTH on the bare floors: A e2e var_slope 4.7%, kurtosis 3.2%
(hc kurtosis inside its floor by 0.1% — margin discipline verbatim);
dial-beats-scale-blind FALSE — extrapolation works SCALE-BLIND, r*≈1
confirmed at deployment level. Real edge refs var_slope 0.973, kurtosis
6.040. <!-- src: stageD_verdict.json verbatim; R21/R22 -->
Held-out basis at the original edge: starlet-ℓ1 +5.4/+4.0/+3.0/+1.8%
(2–16 px) vs the 10% floor, survey-noise convention robust.
<!-- src: starlet_l1_edge.json -->
Two independent trainings later (the blind re-run 3.2 and the dry-run
appendix row), the founding bet has now passed on three separate seeds.

## 3.4 THE ROBUSTNESS TABLE (n = 3 training seeds, full recipe re-run
## per seed incl. its own calibration; fresh-seed spread PRINTED per R43)

| row | pick | C (own P-T band) | marginals | peaks@0.5px ν2.5/ν3.0 | native peaks (desc.) | parity | starlet | nn (pooled) | row |
|---|---|---|---|---|---|---|---|---|---|
| seed 0 (SHIPPED*) | 16000 | 0.7414 IN | PASS | −1.12%/−0.26% PASS | +13.8/+13.7% | 1.70 | PASS | 2.58±0.44 | PASS |
| seed 1 | 3500 | 0.7628 IN | PASS | +3.63%/+6.54% FAIL | +20.3/+22.3% | 2.47 | PASS | 3.23±0.56 | FAIL [declared-res peaks] |
| seed 2 | 5500 | 0.7558 IN | PASS | +4.48%/+6.64% FAIL | +20.8/+22.3% | 1.25 | PASS | 2.56±0.15 | FAIL [declared-res peaks] |

<!-- src: stage3_a_table.json, verbatim -->
*survivorship labeled: seed 0 is the campaign's pick; seeds 1–2 are the
unbiased draws. ORDERED WORDING (R43 finding 3): the peak claim attaches to
the SELECTED configuration — the selection protocol is part of the method —
with this spread printed. The fresh-seed failures are ONE entry each; their
caged picks landed early (@3500/@5500 vs @16000), continuous with the
schedule story (§4.4); candidate cure (late-window / matched-step
selection) is drafted follow-up and NOT a claim of this paper.
Seed-robust across all rows: the calibration (C in-band 3/3), marginals,
starlet, parity, pooled spacing.

## 3.5 The calibration's predictive record (5/5) [WRITE, short]

Pre-registered band landings: L1″ adjudicating pooled (0.7414 in
[0.7192, 0.7919]); L1″ oracle (0.7815 vs predicted 0.7819 — 0.04σ, ON the
real value 0.7864); seed 1 and seed 2 (own bands); the blind run
(0.7765 in [0.7678, 0.8445], two-octave target). Each prediction committed
before its run. <!-- src: l1pp_pt_prediction.json, l1pp_pt.json,
stage3_seed1_c.json is embedded in stage3_a_table.json rows,
stage3_blind_c.json -->

Figures: the one-shot table renders as the headline table (no figure);
F16 robustness table panel [HARDEN-FIG]; F12 stageD.png for 3.3.
