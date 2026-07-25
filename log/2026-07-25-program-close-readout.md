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

RESULT (job 17440279, 0:59; verbatim from pcure_confirm_verdict.json):
**NOT-CONFIRMED.** Test-side T_coef oct-1 = 3.48 — above the 3.0 bar and
OUTSIDE the ±0.15 ambiguity band (no disambiguation license); all marginal
bars pass, one on the pass side of its band (hc-oct3 kurtosis 17.3% vs the
17.5% bar, disclosed per #11). The pre-stated NOT-CONFIRMED meaning applies
verbatim: the joint window remains unwitnessed at any tested scale; the
paper's frontier wording stays at "val-side evidence only." Honest texture,
stated with it: 3.48 is the LOWEST raw-mode test defect of the entire
campaign (caged picks read 6.9–20.2) and the val→test drift (2.88→3.48) is
ordinary estimator noise — the window's edge is real and near, and the bar
did its job. Process disclosure: this script's first emission printed
"AT-THE-BAR" through a looser trigger (any pass-side at-bar marginal) than
the pre-statement supports; the code was corrected TO the pre-statement
before any verdict was recorded, and both emissions are preserved in the
session log — the pre-statement governs, not the code.

## (b) OPTIONAL, taken — the 3-seed N1 mini-ensemble (≤1 GPU-hour)

Executor's call, exercised: N1 retrained at seeds 2/3/4 (same data, config,
cage; MIG, ~21 min each). Descriptive ONLY: e = raw T_coef oct-1 at each
seed's own caged pick; reported as the n=5 seed ensemble {seed0 committed
15.3, seed1 6.86, seeds 2–4 PENDING} with mean, sd, range — the priced seed
variance R35's seed lesson calls for in the limitations section. No branch,
no adjudication; the caged-pick STEP per seed is reported alongside (the
placement-dominated mechanism predicts the scatter tracks pick placement).

RESULT (jobs 17440280–82, ~21 min each; verbatim from
pcure_seed_ensemble.json): e per seed at its own caged pick — seed 0
(committed) 15.3 @7500 · seed 1 6.86 @5000 · seed 2 13.87 @11000 · seed 3
9.82 @11000 · seed 4 9.88 @7000. **Mean 11.1, sd 3.4, range [6.86, 15.3].**
Two things the n=5 ensemble settles: (i) the committed baseline is the
ensemble MAXIMUM — every cross-seed shrink ratio in arm D rode a 2.3-sd
fluctuation of the reference, the seed-confound downgrade (R35.2) now
priced; (ii) seeds 2 and 3 share the SAME pick (11000) yet differ by 4.05 —
the defect-at-pick variance has BOTH a placement component and a per-seed
model-state component of comparable size, refining the placement-dominated
statement of the D readout: single-seed, single-pick estimands of this
defect carry ±3.4 noise against which no ≤2× data effect can be resolved
(the R35 seed lesson, now with its number).

## Program-close statement

Per R35: with (a) read out, **the experimental program of this phase
CLOSES.** Final state of the record: the founding bet passed blind (Stage D);
the tier-3 residual decomposed into a curable lattice defect (F/F2 — the
symmetrized sampler ships, disambiguated clean) and the surviving nn spacing
anomaly (the named frontier); the moment ladder matured into a schedule with
three mechanistic exhibits; the currency directive is measured (CUR-PARENTS,
matched picks); the joint window is near but unwitnessed (3.48 vs 3.0 — the
honest boundary); and the seed variance of the last estimand is priced (sd
3.4, n=5). All remaining work is WP1 writing — which absorbs the schedule
trilogy, the F/F2 cure arc, the currency directive, the nn frontier, and
ledger #11 — pending Andreas's structural read. STOP.
