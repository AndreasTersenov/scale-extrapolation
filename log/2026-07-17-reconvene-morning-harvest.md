# 2026-07-17 — RECONVENE MORNING HARVEST: overnight run audited; C1 changes the campaign's story

Inputs: RESULTS-phase2.md + all preregs/amendments/readouts (commits f2ea7eb..90f722c),
raw JSONs re-extracted independently (gateA_instrument, c1_descriptive_gowerstreet,
c1_verdict_sandbox, stageB*, sandbox_truth*), figures inspected (gateA, c1_sandbox,
c1_maps, stageB). Queue empty; budget used ~0.45 of 3 H100-h.

## Audit result: CLEAN, and the discretion charter worked as designed

- Decisive numbers verify against raw files (oct-2 e2e 0.921/0.927 vs real 1.020 →
  −9.7/−9.1%, vs NLL-head 0.746; oct-1 0.985/0.937 vs 1.117 → 0.88/0.84; e2e kurtosis
  4.43/4.94 vs 6.69; Gate-A recovery ≤0.6%/≤5.0%).
- Chore-0 hook rewrite audited: exactly as ordered (two stacks, loud failure,
  measured 900s timeout); my independent full-gate run: see addendum.
- Both pre-readout amendments are LEGITIMATE and exemplary: (i) normconv — the
  convention bias was MEASURED (−3..−6% var_slope, −12..−19% kurtosis), the reference
  swapped before any readout, bar formulas untouched, phase-1c like-for-like status
  checked; (ii) shape-null — the executor's own control caught its estimand error,
  the rule was amended before reading gowerstreet numbers, and the original
  prediction was scored as a MISS as worded. This is what the objection channel is for.
- Discretion usage: 4/6 diagnostic runs (zero GPU), bounded B2 fallback after a
  measured infra failure (128GB read), temptations check written, variant N+1
  resisted and drafted instead. The rider is now the template for strong-model
  overnight runs.

## Rulings

**R1 — Gate A PASS stands.** The instrument stack is now calibrated at truth-grade.
Adopted bookkeeping cautions: N=64 oct-3 kurtosis wobble (~21% low) binds any
near-bar kurtosis PASS claims; the raw-vs-normalized convention note binds all
cross-record comparisons. **Convention ruling:** same-convention (normconv) truth is
the standard reference for generator-side scoring from now on; both conventions
reported once per document.

**R2 — Stage B verdicts stand.** B1a: locality confirmed, far stronger than
predicted (r*≈1 coarse pixel; the rule's r*=0 vs by-eye 1 → bar-design ledger #6,
r*-rule to be re-worded for step-shaped curves). B1b: anisotropy does NOT fire under
the amended diff-in-diff rule (z=0.91) — Andreas's filament hypothesis is unsupported
at current power; the drafted phase-randomized control (draft 3) is the right
follow-up if we ever revisit. B2: crops are DEAD as a data lever, and the finding
beneath it is adopted as a first-class result: **the training set's law-level
diversity is ~the parent count (N_eff ≈ 13.7), not the tile count (322).** All
"finite-data" language campaign-wide now means "finite PARENTS"; this mechanically
promotes more-parents strategies (constrained-realization ensembles, more sims) over
any within-parent augmentation for future arms, exactly as PLAN-phase2 §5 anticipated.

**R3 — C1 sandbox branch B-C1-TAILS stands** (dispersion alive in the naked
pushforward, recursion calibrated at ≤1.4% on a drifting field, kurtosis fails).
With this, the collapse cure is confirmed in the ORIGINAL channel: augmentation
alone keeps the pure ODE pushforward at truth-level dispersion. The Gaussian-NLL
head — the phase-1 crutch — is hereby RETIRED from all future arms.

**R4 — C1 gowerstreet, adjudicated against the frozen G-1c project bars:** still a
FAIL as a project verdict (oct-2 e2e dispersion −9.7% vs the ~1σ bar; kurtosis
−26/−34%) — nobody declares the generator working. But the DIAGNOSIS is transformed
and the following re-scopings are provisionally adopted, **pending the NLL-head
forensic (R5):**
- The phase-1 "compounding cap" was substantially CHANNEL-DEPENDENT: ~17 of 27
  points at oct 2 were the NLL head's spatially-white sampling noise degrading the
  coarse manifold; the residual ~10% on the real field (vs ≤1.4% sandbox) is the
  genuinely field-structural component.
- The attempt-5 tested-negative REMAINS VALID for the NLL-head substrate it ran on;
  its generalization is revoked. Candidate revision wording (executor's, endorsed):
  "a measured compounding cap OF THE VARIANCE-HEAD SAMPLER, consistent with an
  informational limit only in that channel."
- Gate-0 claim-2 bindings: measurements (a),(c),(vi) stand as measurements of that
  substrate; scope sentences will be revised after R5 lands.
- The audit-paper thesis GAINS a second exhibit: two generators, two different
  higher-order failure signatures (NLL-head: peak sign-flip; C1: uniform +5..+9σ
  UP-tilt), both invisible to power-level checks.
- The Stage-D question is re-scoped: no longer "does the dial extrapolate" but
  "does the dial add information beyond the coarse field itself at the extrapolated
  scale" (arm A 0.88 beats arm B 0.84 end-to-end at oct 1; consistent with r*≈1 —
  the coarse field locally carries most of the scale information).

**R5 — AUTHORIZED NOW: the NLL-head forensic (executor draft 4).** Descriptive,
frozen 4a checkpoints, noise injection disabled at generation, score e2e var_slope.
This settles the channel-dependence attribution WITHIN phase-1 artifacts before any
memo/paper/Gate-0 language is rewritten. Cheap (MIG-minutes). Language revisions
wait for it — evidence before rewording.

**R6 — Next arm: C3 (tails), executor's draft 1, with reconvene review of the
prereg before submission.** Tails are the binding deficit and they compound
(−18/−21% head-conditional → −26/−34% e2e); the energy-score/CRPS objective is the
Gate-0-verified mature answer. C2 (locality) is DEFERRED — recast per the executor
as a capacity/data-efficiency question, valuable but not binding. Standing reconvene
predictions for C3, registered now: P(kurtosis within 15% of exact truth at oct 2–4,
sandbox) 55%; P(gowerstreet e2e kurtosis deficit halves vs C1) 50%.

**R7 — Scorecard.** Standing (reconvene/plan) numbers: P-A 90 HIT; P-B1a 70 HIT;
P-B1b 50-null CORRECT SIDE; P-C1a 60 HIT; P-C1b 45 MISS; P-C1c 45 HIT; B2
stride-useful 60 MISS. Executor: 8 registered, 4 hits, calibration honest, and its
two self-caught design errors (shape null, normconv) are credited as the system
working. Bar-design ledger #6 (r*-rule) and #7 (running-peak rule, benign) adopted.

## For Andreas (the plain verdict)

The rescue is substantially real: the collapse is cured in the original channel, the
compounding cap was mostly our own variance head's noise, and the generator — still
failing the strict bars — is now failing at the TAILS frontier only, with a clear,
Gate-0-verified next lever (C3). The paper decision should wait for the forensic
(R5) and ideally C3's sandbox leg: the story is visibly migrating from "audit paper
with a dead generator" toward "audit paper with a constructive arc — or better."
