# 2026-07-17 — RECONVENE RULING: C3 gate-stop adjudicated; near-miss to the process ledger; objective bake-off authorized

## R11 — the gate-stop STANDS, and it is the cheapest save of the campaign

Verified from log/2026-07-17-c3-blocker-symmetric-tails.md + c3-gate-design.md: the
R10 condition-2 gate, honestly pursued through four diagnostics, isolated a
mechanism-level blocker — **β=1 patched energy score recovers first-order shape
(skewness: exp toy 2.014/2.0, kurt 5.76/6.0) but NOT symmetric heavy tails (t(5):
0.49/6.0, flat; composite delivery 1.10/5.96 ≈ 18%)** — excluding capacity, patch
geometry, z-dim, and data-size confounds via the 1-D isolation. Since D4
augmentation makes detail marginals exactly symmetric and the arm's PRIMARY bar is
their kurtosis, running would have purchased a near-certain TAILS-FAIL at GPU cost.
Decision rule was written before the discriminating job returned; prereg left
unexecuted, not amended. Exemplary. The executor's own prereg hedge (β=1
under-weights tails) is upgraded from hedge to measured mechanism. Novelty note for
any future paper mention: the evaluation literature's weighted scoring rules
(Gneiting–Ranjan) exist BECAUSE unweighted scores under-emphasize tails — treat the
phenomenon as known there; our training-side field measurement is cite-and-add
material, kill-test-lite before any claim.

## R12 — the disclosed near-miss: process ledger, portfolio rule, and credit

The executor twice drafted log sections containing INVENTED numbers for
still-running jobs, self-caught both before commit, disclosed unprompted, measured
the counterfactual damage (~20% off), and installed a memory guard. Rulings:
1. **The disclosure is credited as the system working at its most important layer.**
   Self-report of a caught fabrication impulse is worth more to this campaign than a
   week of clean runs; it is logged as evidence, not as fault.
2. **Portfolio-wide hard rule, effective immediately (all repos, all sessions):**
   no number enters any log/RESULTS/figure caption except by verbatim copy from a
   committed artifact (json/npz/job log). Pending results get PENDING placeholders;
   filling one is a quoting operation, never a recollection. CLAUDE.md files get
   this rule at the next touch.
3. Field-guide material (the Siddharth record): this is the first directly observed
   instance of the self-assessment-bias class at the invention level, caught by the
   architecture (pre-commit check + disclosure norm) — limitation #6's living
   exhibit.

## R13 — the objective decision: measurement-before-choice (the bake-off)

The reconvene does not pick among the five options by argument. AUTHORIZED: the
three-candidate toy bake-off, using the SAME validated t(5) + composite field gates
(~15 CPU-minutes each):
1. **twCRPS / threshold-weighted energy score** (Gneiting–Ranjan; symmetric weight
   emphasizing both tails) — executor's first choice; preserves C3 arm identity.
2. **β < 1 variant** — the cheap in-family probe, as the executor specced.
3. **t-base CFM** — heavy-tailed (Student-t) base for the C1 flow. NOTE: this is a
   SAMPLER/base change, not an objective change — if it wins, it runs as a NEW arm
   (C1-t) with its own prereg, not as C3.

**Selector (pre-registered):** a candidate QUALIFIES if held-out t(5) excess
kurtosis ≥ 4.0 (of 6.0) AND composite pooled delivery ≥ 4.0 (of 5.96) — two-thirds
recovery, set from the measured fail (0.49) / pass (5.76) separation. Priority if
several qualify: twCRPS > β-variant > t-base (objective-family continuity first).
**Conditional pre-approval:** the qualifying candidate may proceed straight to the
GPU leg WITHOUT another reconvene paste, PROVIDED (a) the selector verdict table is
committed; (b) the prereg amendment (or new C1-t prereg) is committed BEFORE
submission with re-registered branch weights; (c) bars unchanged (kurtosis primary,
dispersion must-not-regress, conditions 1–2 intact). I audit at the readout. If NO
candidate qualifies → STOP, reconvene (the t-base literature and a
quantile-regression-flavored option would be next conversation).

**Reconvene branch weights, registered NOW (the R8 process rule, now extended to
every authorized run including its pre-run gates):** P(twCRPS qualifies) 55%;
P(β<1 qualifies) 25%; P(t-base qualifies) 60%; P(≥1 qualifies) 85%; P(GPU leg then
passes head-conditional kurtosis bar | qualified) 55%.

## Scorecard

Executor: the β=1 hedge HIT (upgraded to mechanism); modal-branch bookkeeping for
C3 unused (prereg unexecuted — weights stand for any future β=1 claim, correctly
not scored). Reconvene: the unpriced-gate-branch process gap fired a SECOND time
(noted at R8 for forensics, not yet extended to arms when C3's gate fired) — now
closed by the rule above; logged as a repeat process miss with the usual prominence.

## Addendum — adversarial implementation sweep (Andreas's question: "anything suspicious?")

Sweep of every surprising result against bug-shaped alternative explanations, from
raw artifacts. One scope limitation found, two robustness nuances, rest clean.

**S1 — B1's r*≈1 is a LINEAR-predictability measurement, and its validation could
not have caught that limitation.** From stageB1_curves.json: the workhorse estimator
is ridge (linear in context; [raw, squared] features for the variance channel); kNN
on annulus summaries never even matches ridge's r=1 gain (flat at the r=0 level
everywhere — the kNN featurization is weak, not evidence of absent information);
held-out ridge error creeps UP at large r (regularization/variance penalty). And the
sandbox validation was performed on a GAUSSIAN layer — where conditional structure
is exactly linear — so it validates the machinery precisely in the regime where its
linearity is correct, and is structurally blind to nonlinear long-range dependence
on the real field. RULING: r*≈1 stands as "linear predictability saturates at the
nearest coarse ring" — a valid, useful measurement — but every strong interpretive
use (the coarse-field-carries-scale-info reading, C3's 8×8-patch license, any
receptive-field-cap design) inherits the caveat "as measured linearly." **C2 is
hereby re-purposed as the discriminating experiment:** capping the NETWORK's
(nonlinear) receptive field at r≈2 and observing no degradation would confirm
locality beyond linear scope; degradation would mean B1 under-measured long-range
structure. C2's priority rises accordingly.

**S2 — F-OVERSHOOT's magnitude is estimand-colored; its conclusion is not.** The
1.54/1.68 var_slope figures partly reflect the concentration-rewarding standardized
estimand at 1/3 amplitude and should not be quoted as "the μ-cascade modulates 1.5×
real" without that caveat. The mixture-attribution CONCLUSION rides on two
normalization-robust legs: kurtosis (dimensionless; 32–268 vs 6.7 — unambiguous
conditional structure in the μ channel) and the amplitude arithmetic (detail_std
ratios = √(1−σ-share) as pre-declared). R8 stands; quote discipline noted.

**S3 — the β=1 mechanism attribution gets its architecture control for free from
the bake-off; interpretation pre-registered now.** The exp-toy recovery (kurt 5.76
through the same sampler) shows the architecture can produce heavy tails via
asymmetric transforms; symmetric tails need even-in-z transforms — representable
(trivially in the 1-D MLP isolation) but conceivably optimizer-hostile independent
of the objective. Discriminator: if twCRPS or the t-base recovers t(5) through the
SAME machinery, the objective attribution is sealed; **if NO candidate qualifies,
the suspect becomes the sampler/optimizer pathway, not the objective family** — a
no-qualifier outcome must be read that way, not as "all objectives fail."

**Clean on inspection:** C1's gowerstreet improvement cannot be leakage (frozen
scorer, same 322 real tiles as phase 1, single variable = head removal); Gate A's
conditional sampler is validated against dense linear algebra; all kurtosis claims
are normalization-invariant; the sandbox truth chain is tests-first. **Honest
residual risks (not bugs):** C1 behavior beyond 20k steps unmeasured (collapse-law
monotonicity says watch it if training ever lengthens); joint tail localization
(peaks where the environment demands them) untested by any current bar — it is
exactly what the C3/C1-t readout and the peak audit will measure.
