# 2026-07-17 — RECONVENE RULING: bake-off NO-QUALIFIER adjudicated; the tail-decay signature; diagnosis phase authorized

## R14 — verdict stands; the pre-registered no-qualifier reading governs; my miss first

Verified from the six trajectory JSONs. Selector applied mechanically: twCRPS
3.01/3.57, β=0.5 0.51/1.05, t-base 3.56/2.98 — all FAIL the ≥4.0 pair. STOP stands.
Scorecard, reconvene first: **P(≥1 qualifies) 85% — MISS** (the no-qualifier 15%
branch fired); P(twCRPS) 55 miss; P(t-base) 60 miss; P(β<1) 25 correct side. The
executor's registered t-base concern (late Gaussianization) fired — their hedge
record continues excellent. The executor's reconciliation commit (adopting the
pre-registered S3 attribution over its own first framing, history preserved) is the
judge/executor asynchrony pattern working correctly.

Per S3, pre-registered before any result: with three structurally different
objective/sampler pairs failing symmetric tails identically — including the t-base
leg, which is L2-CFM and NOT a proper-score objective, actively REMOVING the tails
its own base supplies — the suspect is the COMMON PATHWAY (training
dynamics/optimization on finite data), not the objective family.

## The finding inside the failure: the tail-decay signature

The trajectories (held-out excess kurtosis vs steps):
- twCRPS: 2.7 → **5.14@2k** → 3.0 (truth 4.15); composite 3.2 → **4.8@2k** → 3.6 (5.96)
- t-base: t5flat 3.6 → **4.11@3k** → 3.6 (truth 4.15 — AT truth mid-training);
  composite 4.3 → **5.13@2k** → 3.0 (5.96)
- β=0.5: monotone decay from 500 steps; never viable.

Four of six runs PASS THROUGH near-truth tail states and then converge away. This
is the phase-1 dispersion-collapse curve one moment-rung up: transiently correct
higher-order conditional structure, eaten by continued training. **Hypothesis,
named: the collapse law is a moment-ladder phenomenon** — finite-data training
starves conditional structure rung by rung (dispersion at rung 2, cured by 8×
augmentation; symmetric tail weight at rung 4, now showing the same signature).
The known signature test transfers: rung-2's cure was established by a DATA-SIZE
causal intervention. Rung 4 gets the same test.

## R15 — AUTHORIZED: the tail-dynamics diagnosis (CPU only, prereg'd, weights incl. gates)

1. **The rung-4 causal test (highest value):** re-run the two best trajectories
   (t-base, twCRPS) at 8× toy data (768 → 6144 fields), same configs, dense
   checkpoints. Pre-registered question: does the tail-decay onset shift ≥2×?
   Reconvene weights: **shift ≥2× at 8× data: 60%**; no shift (mechanism is not
   data-limited → optimizer-flavored): 25%; ambiguous: 15% (= negative, as always).
2. **Checkpoint-viability probe (descriptive):** re-run t-base logging ALL
   statistics (dispersion + skew + kurt + q999) densely: does a checkpoint exist
   where dispersion AND tails are simultaneously near-truth? Reconvene weight:
   **55% yes.** If yes, a pre-registered validation-selected early-stop becomes a
   legitimate candidate lever (selection rule fixed on VALIDATION fields before any
   TEST adjudication; the Goodhart cage specified in the prereg).
3. **t-base mechanism attribution (descriptive):** is the base df fixed or
   effectively learnable, and does the flow actively squash base tails
   (base-vs-flow decomposition over checkpoints)?

Budget: CPU jobs only; the 6-run bake-off pattern reused. STOP at the readout; the
next-arm decision (early-stop lever vs data-scaled arm vs architecture change)
happens with this diagnosis in hand — not before.

## Consequences ledger

The moment-ladder hypothesis, if the causal test confirms, upgrades the paper's
central mechanism from "a collapse law for conditional variance" to "a moment-
resolved law of conditional-structure loss under finite-data training, with a
causal data-size signature at two rungs" — and connects directly to N_eff ≈ parents
(rare-event structure lives in parents; the effective data for rung 4 is smaller
than for rung 2). If the causal test refutes, the optimizer-side investigation
opens with the S3 attribution already in place. Either way the STOP bought
mechanism instead of a burned GPU allocation — twice in one day.
