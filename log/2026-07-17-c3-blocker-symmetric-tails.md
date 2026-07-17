# 2026-07-17 — C3 BLOCKER: the β=1 patched energy score does not learn SYMMETRIC
# heavy tails (pre-training gate, R10 condition 2) — arm NOT submitted

## What fired

The R10 condition-2 pre-training investigation (full chain:
log/2026-07-17-c3-gate-design.md) ended with a pre-registered decision rule,
written into the log BEFORE the discriminating job finished: if the
production-config Student-t(5) field toy plateaus instead of recovering toward
truth, the arm is blocked and the finding goes to the reconvene. It plateaued —
held-out excess kurtosis **0.49 vs truth 6.0**, flat for the final 1,500 annealed
steps (job 16606069) — so the C3 sandbox leg was NOT submitted. The prereg
(b42edd9, approved R10) is unexecuted, not amended; branch weights stand unused.

## The evidence, compressed (all held-out eval, production net + patch config)

| toy (conditional noise) | shape statistic | truth | recovered | verdict |
|---|---|---|---|---|
| Gaussian, modulated σ(c) | per-bin σ | — | ≤12% everywhere | recovered |
| exp (one-sided), flat σ | skewness | 2.0 | **2.014** | recovered |
| exp (one-sided), flat σ | excess kurtosis | 6.0 | **5.76** | recovered |
| t(5) (symmetric), flat σ | skewness | 0.0 | 0.04 | correct |
| t(5) (symmetric), flat σ | excess kurtosis | 6.0 | **0.49, flat 2.5k→4k** | NOT recovered |

Same architecture, same objective, same budget, same data size (768 fields), same
schedule. 1-D isolation (MLP through the identical loss path, infinite data, 20k
steps, z-dim 1/4/8): skewness 2.0 recovered; t(5) kurtosis plateaus at ~1.1.
The pattern is clean and mechanism-level.

## Why (one paragraph)

The CRPS/energy score pays for distributional error in first-order features of the
predictive law — location, spread, asymmetry — but the score difference between a
symmetric heavy-tailed and a symmetric light-tailed predictive with matched first
two moments is second-order small and carried by rare events, so its training
signal is weak by construction; β=1 is the executor's own prereg hedge, now
measured. Asymmetry (skew) IS first-order — hence exp recovers and t(5) does not.
The failure is not capacity (production net recovers exp kurtosis 5.76 through the
same 192-dim patches), not patch geometry (1-D per-scalar CRPS shows the same
plateau), not z-dimensionality (1/4/8 identical), not data size (infinite-data 1-D
version plateaus too).

## Why this kills the run as configured (not merely risks it)

The arm's PRIMARY pre-registered bar is kurtosis at truth (15%/3σ, octaves 2–4,
both arms). Haar detail marginals are EXACTLY symmetric in the training law (the
8× D4 augmentation enforces w ~ −w; measured sandbox skew ≈ 0) and heavy-tailed
(pooled excess kurtosis 6–7 at the binding octaves). The toy isolates precisely
that increment and shows the objective delivers ~8% of it. The predictable outcome
is C3-TAILS-FAIL (my 25-weight branch) reached by spending a GPU allocation to
confirm a number the toy already measured to first order; R10 condition 2 exists
to prevent exactly this ("never trust a new sampling channel's calibration by
construction").

A quantitative sharpening is running (job 16606223: modulated-σ × t(5) composite —
how much of a POOLED mixture kurtosis the mechanism delivers when modulation is
recoverable and conditional shape is not); its result will be appended here before
this file is committed... APPENDED: see bottom of file.

## What survives, unchanged

- The C3 IMPLEMENTATION (arms_p2/c3/, runner/sweep/scorer, SLURM script) is
  committed, smoke-tested, and gate-validated — ready to run within minutes of a
  reconvene decision on the objective.
- The R10 condition-1 instrument and its truth: tail_q999 validated;
  results_p2/sandbox_truth_q999.json (oct 1–4: 5.974/5.532/5.103/4.591, batch-means
  SEs ≤0.03, job 16606155).
- The two-moment + skewness gate (R10 condition 2 verbatim) PASSES — the
  mechanism's failure is specific to symmetric tail weight.

## Options for the reconvene (executor's read, not a decision)

1. **Threshold-weighted CRPS/ES (Gneiting–Ranjan)**: add a tail-weighted score
   term, w(y) concentrated on |y| ≥ τ — the probabilistic-forecasting literature's
   standard proper-score answer to tail emphasis; keeps the single-variable story
   almost clean (objective family stays "proper score", one new pinned weight
   function). Cheap to prereg-amend; sandbox toys can gate it the same way within
   an hour (the t(5) toy IS the gate).
2. **Transform-space scoring**: score in sign(y)·log(1+|y|/s) space — makes tail
   mass a bulk (first-order) feature; needs care that the back-transformed law
   stays calibrated where the original bars live.
3. **β < 1 energy score**: strictly proper still; whether it materially boosts the
   symmetric-tail signal is measurable by the same toy BEFORE any prereg
   amendment (executor's prior: weak effect — the second-order structure is in the
   score's form, not β).
4. **Retire the C3 objective family** and take the conditional-shape question to a
   different mechanism class (e.g., the deferred C2, or a heavier-tailed base
   distribution for the C1 flow — a t-base ODE pushforward makes tail weight a
   FIRST-order feature of the base, not something the objective must discover).
   Note option 4's second variant touches C1's frozen config: new prereg required.
5. Run C3 anyway as a measured negative (burns the allocation to confirm the toy;
   executor recommends against — the toy's answer is quantitative already).

Executor's ordering: 1 > 3-as-cheap-test > 4(t-base) > 2 > 5. Any of 1–3 can be
gated by the SAME t(5) field toy within ~15 min of CPU SLURM before a prereg
amendment reaches paper.

## Prediction-ledger note

My prereg branch weights implicitly priced this risk at 25 (C3-TAILS-FAIL) with
the β=1 hedge named in P-C3a's reasoning — but I put 45% on P-C3a (kurtosis at
truth) and the reconvene 55%. The pre-training gate has now effectively adjudicated
the hedge BEFORE the run: had we run it, TAILS-FAIL was the strong favorite. Both
panels' numbers were too optimistic on the objective's tail teeth; logged for
calibration (the spread executor-vs-reconvene will not be scored — the arm never
ran — but the direction is on record).

## APPENDED: composite (modulated σ × t(5)) result — job 16606223 COMPLETED (21:30)

Verbatim from jobout/diag_c3_t5mod_16606223.out (copy in results_p2/): truth
reference (data pooled-resid excess kurtosis) train 5.86 / heldout 5.96; recovered
held-out pooled kurt 2.44 @500 → 1.92 @1k → 1.41 @1.5k → 1.17 @2k → 1.10 @2.5k →
1.11 @3k → 1.10 @3.5k → **1.10 @4k** (flat final 1.5k steps).

Reading: **≈18% of the composite pooled kurtosis delivered.** The lift over the
flat-σ t(5) plateau (0.49 → 1.10) is consistent with the recovered σ(c)
modulation's mixture contribution; the conditional-shape increment — C3's entire
reason to exist vs C1 — is absent. Caveat stated plainly: in the SANDBOX estimand
the modulation share of pooled kurtosis is larger than in this toy, so the arm's
realized deficit would be smaller than 82% — but C1 already measured that
modulation is deliverable (its kurtosis deficit was 18–21% WITH dispersion
calibrated); the increment C3 was commissioned to add is precisely the part this
objective does not learn. (The early 2.44 is the untrained net's transient, not
recovery — it decays monotonically as the σ-field organizes.)
