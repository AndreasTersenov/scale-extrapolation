# 2026-07-10 — reconvene review of RESULTS-toy (phase-1 verdicts)

Reviewed by the main (reconvene) session; Andreas's standing framework applies.

## Rulings

1. **P-null, P4, P5 verdicts ACCEPTED.** P5 (85% pre-reg) is the paper's load-bearing
   claim and is confirmed robustly (z = 5.8–11.3, 26–52%, mechanism as predicted).
2. **K-T2 does NOT fire.** K-T2 attributes P6 failure to the 2-D conditioning; the
   evidence exonerates it (FiLM: directionally correct, monotone response). The failure
   isolates to a GENERATOR property: L2 flow matching under-disperses conditional
   variance, monotonically worse with training (attempt-3 experiment). P6 and P13 are
   scored **blocked-pending-retest**, not refuted. The frozen P6 bar (>=70% repair) is
   unchanged and will be re-adjudicated after the generator fix.
3. **The under-dispersion finding is promoted to a first-class result** for RESULTS-toy
   and the eventual paper ("conditional FM mean-collapses dispersion statistics under
   heavy fitting") — keep the attempt-2 vs attempt-3 var_slope-vs-loss numbers
   prominent.

## Approved next program: the variance-faithful generator (pre-register each arm)

Ordered, cheapest first, one variable at a time:
(a) **Stochastic sampling of the EXISTING checkpoints** — no retraining. For
    Gaussian-path CFM the score is recoverable from the learned velocity
    (score(x,t) from v(x,t) via the path's drift/variance identities); sample with an
    SDE/churn variant and measure var_slope dispersion recovery on trained octaves
    first (that is the clean test: can the sampler reproduce TRAINED-octave var_slope
    near real?). Verify the identity numerically on the GRF control before trusting it.
(b) If (a) insufficient: **early-stopping / checkpoint sweep** (the 10k-vs-25k evidence
    already points this way) — pick dispersion-faithful checkpoints by trained-octave
    var_slope, not by loss.
(c) If needed: **dispersion-regularized or non-L2 objective arm** (heavier build;
    pre-register the specific objective before implementing).
Then: re-run arms A/B scoring (pipeline unchanged) → fair P6/P13 adjudication →
if P6 passes, rung-(v) transfer re-run.

Success criterion for (a)/(b), pre-registered here: trained-octave var_slope within
1 sigma of real for the chosen sampler/checkpoint; only then is the octave-1 repair
measurement meaningful.

## Notes
- P13's amplitude-transfer success (40% -> 2.5% via arm B on hf_pm) is worth a line in
  the paper even though var_slope transfer stays blocked.
- Budget posture unchanged; all of phase 1 so far is well inside the 15 H100-day cap.
