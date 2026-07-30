# 2026-07-30 — RECONVENE R40: L1′ adjudicated — N-NULL-WITHOUT-TRANSFER
# stands mechanically; its pre-stated MEANING is amended on measured
# mechanism; L2 NOT declared live; L1″ (deconvolved base) AUTHORIZED as a
# quantitative prediction test

Ruled on: the READOUT appended to log/2026-07-30-prereg-l1prime.md
(116152e) + l1p_* artifacts. Same-day git log taken; queue empty.

## Verification

1. **R12:** verdict/A2/transfer artifacts re-checked verbatim — A2 pooled
   C 0.6030±0.0049 (−24.79σ vs baseline 0.7237; threshold 0.7551), streams
   0.6189/0.6056/0.5845, ablation 0.6295; peaks pooled +15.19%±2.93 /
   +13.67%±2.64 vs A3 (+15.290 / +12.474) → Δ 0.02σ / −0.31σ; identity
   gates (canary rel 1.42e-3, main 6.73e-3, corr_min ≥0.9999999) — ALL
   MATCH. Scoring-order enforcement confirmed (l1p_a2.json written before
   peak computation — structural, not procedural).
2. **Branch mechanics re-derived:** both Δ < 2, no at-bar entry, CIs
   entirely positive (no flip), no must-not-regress failure, A2 rule reads
   NO-TRANSFER → **N-NULL-WITHOUT-TRANSFER is the unique mechanical
   outcome.** Suites re-run at adjudication: tests/ 44 pass; tests_wfm +
   tests_p2 pass (exit 0).
3. **Transfer analysis audited:** two-point ring-wise T(k) consistency
   (median ratio 0.981, IQR [0.967, 1.029]); the multiplicative model's
   predicted inversion (0.527 ring-conv) vs measured (0.589) — direction
   and approximate magnitude reproduced; deconvolved base target C = 1.100.
   Labeled descriptive, outside the branch table — correctly.

## Ruling

**1. The branch verdict STANDS; its meaning is AMENDED.** The mechanical
outcome is final: the naive-target colored base changed nothing at the
peak tier (Δ ≈ 0) and its coloring moved AWAY from real — the registered
transfer line (55) lost in an unlisted direction. But the pre-stated
meaning of NULL-WITHOUT-TRANSFER — "the flow does not carry base spectra
into details; the L2 loss-side family becomes live" — is contradicted by
the measured mechanism: the flow carries base spectra at ~unit ring-wise
fidelity through a fixed blue-tilting transfer T(k) (0.66→1.06). The
carried spectrum was aimed at the wrong target: cancellation requires
pre-emphasis AGAINST the transfer (deconvolution), not imitation of the
output target. **L2 is NOT declared live.** The true outcome was a fifth,
unlisted branch: NULL-WITH-INVERTED-TRANSFER (mis-specified target).

**Bar-ledger #14:** a branch may carry a MECHANISM meaning only if its
rule MEASURES that mechanism. A2's rule measured "output moved toward
real," not "spectrum carried" — the two dissociate exactly when the
target is mis-specified. Future preregs state such meanings conditionally.

**2. Scorecard, both roles.** The naive filter target was adopted by the
executor's draft AND by R38/R39 — the deconvolution algebra (output =
T·base, so base must be target/T) was available at design time and neither
role derived it until the data forced it. Shared miss, logged with full
prominence. Credited: the executor's two-point analysis design (running
white + colored streams is precisely what identifies T), the structural
A2-before-peaks enforcement, and the honest flag that the branch meaning
needed amendment rather than silently declaring L2 live. L1′ is hereby
re-framed in the record as the SYSTEM-IDENTIFICATION run: 0.06 H100-h
bought the transfer function of weight-tied extrapolation on this field.

**3. Paper designation (frozen skeleton, recorded for the reframe):** "the
whiteness defect IS the transfer function of weight-tied extrapolation on
this field" + the deconvolution methodology + the L1→L1′→L1″ arc are
mechanism-section material regardless of L1″'s outcome — the
audit-guided-design exhibit in its sharpest form.

**4. L1″ AUTHORIZED** (inference-only, committed checkpoints, F/F2
precedent) as a QUANTITATIVE PREDICTION TEST of the multiplicative model:

- Base filter = S_target/T from the COMMITTED two-point measurement.
  Deployment split as in R39: the ADJUDICATING streams deconvolve the
  oct2-rescaled target (both S_target and T are deployment-available — T
  needs only generated maps and the known base); ONE labeled oracle
  ablation stream deconvolves the measured oct-1 real spectrum. Same
  canary-first order, kill criterion, identity gate, 3 adjudicating
  streams, A3 reference, #11 bands, one licensed disambiguation.
- **Registered premise (scored first, the new A2):** P-T = the
  multiplicative model's point prediction for C_gen(L1″), with a ±3σ band
  from the T measurement's uncertainty, computed and committed in the
  prereg BEFORE the run. Honest scope stated now: input-independence is
  measured on base colorings {0.74, 1.00} and applied at 1.10 — a modest
  extrapolation of the premise, priced in the line. Reconvene:
  **P(P-T lands) = 70.** A P-T failure is itself a mechanism finding
  (input-dependence beyond the measured range) → the NULL split keys on
  P-T, not on the old transfer threshold.
- Peak branch weights (reconvene): **CURED 12 / IMPROVED 38 / NULL 32 /
  REGRESSED 8 / gates+FLIPPED 10.** Executor adds its column in the
  prereg. Watched lines: canary oct2 kurtosis (the L1′ context note — a
  redder base may push it further; still context unless catastrophe);
  nn_T on the adjudicating streams (descriptive; the 4.045→3.33 movement
  on the WRONG-direction coloring is noted and NOT interpreted).
- Pre-delegation as in R39: if the committed L1″ prereg instantiates this
  order with no other deltas, it is pre-cleared to run; any delta → STOP.
  **STOP at the readout.** Budget ≲0.1 H100-h; order-set spend so far 0.06.
