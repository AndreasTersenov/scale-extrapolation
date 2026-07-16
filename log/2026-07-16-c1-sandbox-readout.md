# 2026-07-16 — C1 sandbox readout: branch **B-C1-TAILS** (both arms) — dispersion
# ALIVE in the pure ODE channel, recursion dispersion-calibrated, tails fail at oct 2

Job 16491750 COMPLETED, config_hash **ab9c175f59**. Prereg
log/2026-07-16-prereg-c1-sandbox.md + pre-readout amendment (normconv truth,
log/2026-07-16-c1-amendment-normconv.md). Adjudication mechanical
(scripts_p2/score_c1_endtoend.py; full table in results_p2/c1_verdict_sandbox.json).
Figure: results_p2/c1_sandbox.png.

## VERDICTS (rules as pre-registered)

- **Branch: B-C1-TAILS, both arms** (dispersion passes BOTH levels at octaves 2,3,4;
  kurtosis fails). Adjudication path: no degenerate (amplitude ≤ few % of the real
  stack), no collapse signature (running-peak rule: curves NEVER drop 0.10 — they
  RISE to a plateau 1.01–1.07 and hold through 20k), head-conditional dispersion
  4.9–6.8% (bars 10–11.7%), end-to-end dispersion 4.5–8.7% (bars 10–10.8%),
  kurtosis FAILS at oct 2 both arms both levels (18.4–21.3% vs 15–16.1% bars) and
  arm B end-to-end oct 3 (21.6%).
- **Gowerstreet-leg trigger: PASS** (dispersion bar, both arms, both levels) —
  leg 2 submitted per the prereg (job 16491989), readout DESCRIPTIVE.
- **Prediction verdicts:** P-C1a (dispersion alive) — standing 60% HIT, executor
  55% HIT. P-C1b (kurtosis at truth) — standing 45% MISS, executor 30%
  (correctly leaned fail). P-C1c (recursion calibrated, dispersion reading) —
  standing 45% HIT, executor 25% **MISS** (I over-weighted compounding: the
  sandbox recursion costs ≤1.4% in var_slope end-to-end vs head-conditional).
  Branch weights: executor modal was COLL 35 — WRONG; TAILS (25) fired.

## The two headline numbers

1. **The collapse cure works in the naked ODE-pushforward channel.** Plain
   conditional FM + 8× D4 augmentation, NO variance head: oct-2 head-conditional
   var_slope trains UP to ~1.02–1.07 against exact truth 1.070 and stays there for
   20k steps (phase-1 un-augmented baseline: peak ~0.97 at 2k then decay to 0.75).
   On a calibrated bed, the dispersion problem of phase 1 is fully accounted for by
   finite-data mean memorization — no head, penalty, or churn needed once the data
   pathology is removed.
2. **Compounding is NOT generic to drifting fields.** The sandbox drifts
   (var_slope 1.23→0.75 normconv) yet end-to-end recursion from octave 4 tracks the
   head-conditional response to ≤1.4% at every trained octave (gowerstreet phase-1
   comparator: −27% at oct 2). Whatever drives the gowerstreet compounding, it is
   not scale-drift per se — the informational-limit story's "off-manifold texture"
   component is now the only surviving candidate on the table, and leg 2 (same
   recipe, real field) is the direct discriminator.

## What failed, precisely

Tails: generated kurtosis at oct 2 is 3.9–4.0 vs truth 4.92 (−19/−21%), i.e. the
ODE pushforward + augmentation produces the variance MODULATION but not the full
tail weight of p(detail|coarse); oct-3/4 kurtosis is within bars (8.7%/5.7% arm B
head-conditional) except arm B's end-to-end oct 3 (−21.6%). DESCRIPTIVE oct 1
(extrapolated): var_slope 1.09–1.15 vs truth 1.23 (−7 to −11%), kurtosis 4.8–5.5
vs 7.02 (−22 to −31%) — the tail deficit grows toward the extrapolated octave, the
phase-1 pattern in miniature but far milder (phase-1 gowerstreet oct-1 var_slope
deficits were 26–52%).

## Cost note

Training+generation+sweep: single MIG job, ~13 min wall. Cumulative GPU tonight
≈ 0.2 H100-h of the 3 H100-h cap.
