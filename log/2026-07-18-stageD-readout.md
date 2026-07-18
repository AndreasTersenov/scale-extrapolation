# 2026-07-18 — STAGE D READOUT: **D-PASS-BOTH — P-D, the original bet, PASSES**
# (and the dial adds nothing the coarse field didn't already carry).
# R18 replication: 24/24 — C1T-CAL REPLICATED on 64 fresh fields.

Prereg log/2026-07-18-prereg-stageD.md (a0b6e4c, pre-submission). Jobs: Stage D
GPU 16669982 (21:54); replication 16669983 FAILED at t=0 (env split: the frozen
scorer's pywt cannot share a process with JAX — runner split into sample/score
phases, fix committed) → 16671189 COMPLETED (4:35) + local score phase.
Artifacts: arms_stageD.npz, c1t_selection_stageD.json, stageD_verdict.json,
c1t_repl64{_hc,}.json, c1t_repl64_gen.npz; figure results_p2/stageD.png.
All numbers verbatim (R12).

## Task 2 — STAGE D: the held-out edge (octave 2), gowerstreet

Selected checkpoints (cage at octave 3): A @9000, B @7500. Verdict table
(vs the real TEST fields' own statistics):

| arm | level | var_slope (rel) | kurtosis (rel) |
|---|---|---|---|
| A (scale-blind) | head-conditional | 0.915 (6.0%) | 5.139 (**14.9%**) |
| A | end-to-end | 1.019 (4.7%) | 5.846 (**3.2%**) |
| B (curve dial) | head-conditional | 0.999 (2.6%) | 6.182 (**2.4%**) |
| B | end-to-end | 1.049 (7.9%) | 6.304 (**4.4%**) |

Real edge references: var_slope 0.973, kurtosis 6.040. **All 8 checks PASS →
branch D-PASS-BOTH; P-D (arm B under the deployment protocol) PASSES.**

Two bar-honesty statements, with the same prominence as the pass:
1. The formal kurtosis bars inflated to 56–59% (the 32-field real reference's
   kurtosis SE is large; 3·SE_rel swamps the floor) — nearly vacuous as bars.
   The result therefore stands on the BARE floors: all 8 checks also pass at
   15%/10%, with arm A's head-conditional kurtosis at 14.9% — inside by 0.1%.
   Stated plainly: the e2e numbers (3.2%/4.4%) are unambiguous; A's
   head-conditional margin is razor-thin.
2. AMBIGUOUS=FAIL was never invoked — every check is a clean pass under both
   readings except A-hc-kurt, which passes both but with no margin.

**The dial question (R20's re-scoped Stage-D core): the dial adds NOTHING to
marginal calibration at the edge.** Pinned dial-beats-scale-blind metric: FALSE
(|e2e kurt deficit| B 4.4% vs A 3.2%). Both panels' lines (40/45) MISSED. The
scale-blind arm extrapolated the held-out octave from the coarse field alone —
B1's r*≈1 ("the conditioning carries the scale information") confirmed at the
deployment level, on the question the project was born asking. Nuance the
marginal bars cannot see: the dial CHANGES joint structure — peak-audit excess at
the edge flips sign under B (ν=2.5: −9%, ν=3: −17%) vs A's persistent up-tilt
(+13%, +15%); neither closes the audit. The dial's input error (curve vs
measured: vs +2.9%, kurt +18% at the edge) was absorbed without harm — the arm
is robust to its own extrapolation error at this level.

Descriptive: octave 1 (two-octave extrapolation) degrades to −44/−54% kurt
(worse than the fully-trained C1-t's −29/−40 — each octave of extrapolation
costs); octave 3 e2e kurt runs HIGH (+13/+17%); octave 4 −19/−28%.

## Task 1 — R18 replication: 24/24, C1T-CAL confirmed

64 fresh fields (seed 20260720), frozen picks A@7500/B@2500, identical bars with
~√2 tighter SEs (most kurtosis bars now 15–17%, at or near the floors). ALL 24
checks pass. Named caveats: arm B hc oct-4 kurtosis error is 41.0% riding a
56.0% SE-widened bar (the one bar-carried entry — B's octave-4 tails are
over-heavy on fresh fields, 2.963 vs truth 2.102); arm B e2e oct-2 at 13.3% vs
the 15.0% floor (inside, thin); arm A is clean everywhere (0.0–12.4%). The
"calibrated at the binding octaves" claim now rests on the required replication;
the octave-4 thinness concentrated in arm B, direction OVER.

## Scorecards (the calibration story is now itself a finding)

- Stage D branches: executor modal D-FAIL-BOTH 55 — **large MISS** (D-PASS-BOTH
  18 fired). Fourth consecutive under-confidence on a constructive outcome,
  against R20's explicit warning; the failure-era priors are measurably stale.
- Component lines: dispersion-within-bars 55/55 — both hit. Tails-within-bars:
  reconvene 35 / executor 30 — both MISS low. Dial-beats: 40/45 — both MISS
  (it didn't).
- P-D: PLAN standing 40 vs executor 27 — the PLAN's original number was the
  better-calibrated one.
- Replication: executor modal full-pass 45 — HIT; the oct-4-only-failure 35
  didn't fire (the thinness showed as wide-bar passes, not failures).

## What this means (one paragraph, for the paper decision — Andreas's call)

Per PLAN §4 branch-complete: PASS → the method paper. Per R20.3 it is one paper
either way; Stage D's chapter is now: on a substrate calibrated by the
audit-guided arc, single-octave scale extrapolation WORKS under the deployment
protocol — and works scale-blind: the coarse conditioning itself carries the
scale information the dial hypothesis attributed to an explicit coordinate. The
dial is not the mechanism; the information locality measured in B1 is. The
remaining frontiers are joint/peak structure (neither arm closes the audit; the
dial changes its sign) and multi-octave extrapolation (octave 1 degrades). STOP.
