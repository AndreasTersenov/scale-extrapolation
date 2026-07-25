# 2026-07-25 — PROGRAM-CLOSE READOUT (R35 orders 7a/7b): the 32× joint-window
# test-side confirmation + the 3-seed N1 estimand mini-ensemble

Pre-statement, committed BEFORE the jobs' results are read (the M5/PENDING
pattern; R12 throughout).

## (a) MANDATORY — 32× joint-window confirmation (zero training)

From ckpt_pcure_n32 at the frozen joint criterion's pick (17500, val T_coef
2.88 — the campaign's first sub-bar joint reading): raw-mode test e2e + hc on
the shared 32 test tiles. **Confirmation rule (pre-stated): CONFIRMED iff all
standing marginal bars pass (hc + e2e, octaves 2–4, the c1t convention) AND
test-side T_coef oct-1 < 3. Per ledger #11 (codified in R35): the ambiguity
band is ±0.15 around the 3.0 defect bar and ±10% of bar on marginal entries —
an inside-band landing is reported AT-THE-BAR and disambiguates by
replication on the repl64 stream (one further pass, pre-authorized here) —
not rounded either way.** Meaning of each outcome, pre-stated: CONFIRMED →
data widens the windows AND the joint criterion can harvest them at 32×; the
schedule story's third act ends constructively ("the cure is joint selection
at high data"). NOT-CONFIRMED (named failing entries) → the val-side window
was selection-noise; the joint window remains unwitnessed at any tested
scale; the paper's frontier wording stays at "val-side evidence only".

RESULT: PENDING (verbatim from results_p2/pcure_confirm_verdict.json).

## (b) OPTIONAL, taken — the 3-seed N1 mini-ensemble (≤1 GPU-hour)

Executor's call, exercised: N1 retrained at seeds 2/3/4 (same data, config,
cage; MIG, ~21 min each). Descriptive ONLY: e = raw T_coef oct-1 at each
seed's own caged pick; reported as the n=5 seed ensemble {seed0 committed
15.3, seed1 6.86, seeds 2–4 PENDING} with mean, sd, range — the priced seed
variance R35's seed lesson calls for in the limitations section. No branch,
no adjudication; the caged-pick STEP per seed is reported alongside (the
placement-dominated mechanism predicts the scatter tracks pick placement).

RESULT: PENDING (verbatim from results_p2/pcure_seed_ensemble.json).

## Program-close statement

PENDING both readouts; on (a)'s readout the experimental program of this
phase CLOSES per R35; all further work is WP1 writing pending Andreas's
structural read.
