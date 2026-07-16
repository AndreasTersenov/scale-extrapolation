# 2026-07-16 — RECONVENE RULING: reconciling the two big-picture audits (executor's + reconvene's)

Andreas asked the same expert-lens question of both the executor session and the
reconvene, in parallel. Both audits are on record (executor's: relayed in chat,
to be committed by the session with its reshape memo; reconvene's:
log/2026-07-16-reconvene-bigpicture-audit.md). This ruling reconciles them so the
sessions do not proceed on divergent plans (the anti-race-condition rule).

## Convergent findings (adopted without further debate)

Both audits independently: (1) the two literature kill-tests are MANDATORY and first;
(2) the audit/validation machinery — not the generator — is the asset; (3) language
discipline ("any-scale" framing is dead permanently; P5's value is the quantification,
not the surprise); (4) no downstream task, no utility.

## Scorecard (reconvene's misses, logged with the usual prominence)

The executor's audit strictly dominates mine on constructive moves. I missed:
(a) the downstream-bias demo (parameter-level bias invisible to P(k), caught in
advance by our diagnostics) — the single best usefulness converter either of us
produced; (b) the inverted validation pilot (run it on the FAILING frozen generator —
the protocol demonstrates best against a known failure; my plan had it gated on P6
passing, which was backwards); (c) the self-similar control (a cascade where the
scale-flow IS smooth — discriminates method-failure from data-regime-limit);
(d) Seitzer et al. β-NLL as the sharpest adjacent citation for the σ-head starvation.
All four are adopted. SPEC-novelty-collapse.md updated accordingly.

## Adjudicated divergence 1: the "informational limit" is MEASURED, not PROVEN

The executor states phase 1c "proved" the compounding deficit is not an engineering
gap. Not granted. What the record supports: (i) end-to-end response ≈ drifted-input
response (attenuation matching, 4b′-ii); (ii) the drift is structured, not additive;
(iii) naive over-trust (s=0) beat honest calibration (s=0.1) for pushed-forward
statistics. Those are measurements about THIS generator's outputs. "No generator in
this data regime can carry sufficient conditional information one octave down" is an
interpretation — and the discriminating experiment is exactly attempt 5
(self-conditioning): if training the heads on generated coarse fixes compounding,
the limit was train/test mismatch (engineering); if it fails, the informational-limit
reading gains the status of a tested negative — MUCH stronger in the reshaped paper
than an assumption. The self-similar control then bounds its scope from the other
side. Interpretation discipline: until attempt 5 reads out, the phrase is
"a measured compounding cap consistent with an informational limit."

## Adjudicated divergence 2: attempt 5 PROCEEDS — re-purposed, not rescinded

The executor's reshape memo omits attempt 5 (generator treated as dead). Overruled,
with changed framing: attempt 5 is no longer the last rescue of the generator; it is
the DISCRIMINATOR the reshaped paper needs for its central negative claim, and it
costs minutes. Bars unchanged (they now define the discrimination readout). My
P-attempt5-lever drops 55% → **40%** on the executor's evidence (structured drift +
honesty-hurts inversion both point against train-side fixes); the prediction is
re-registered here at the new value before the run. Finality clause unchanged:
no attempt 6 in any branch.

## The reconciled program (phase-1d candidate; Andreas signs off on the memo)

- **Gate 0 (cheap session, parallel, blocking all paper claims):** the kill-tests —
  SPEC-novelty-collapse.md, now covering BOTH claims (collapse law; informational
  compounding limit incl. the honesty-hurts inversion) with the Seitzer/β-NLL and
  exposure-bias lineages added.
- **Step 1 (minutes):** attempt 5 as discriminator, bars as ruled in 4bpii.
- **Step 2 (MIG-minutes):** the validation pilot, INVERTED — run slide-the-edge,
  self-consistency, and held-out statistics (incl. scattering covariance) on the
  frozen failing generator; the protocol's job is to catch the failure we know is
  there, and its success at catching is the demonstrable product.
- **Step 3 (days — the usefulness centerpiece):** downstream-bias demo — propagate
  frozen-generator fields into peak counts and/or a small SBI posterior; show the
  parameter-level bias that P(k)-level checks miss; show our diagnostics flag it in
  advance. This is Readout-C's lesson from D1 applied here, and it speaks the
  language of Andreas's reliability spine.
- **Step 4 (optional, executor's call on cost):** the self-similar control.
- **Then:** the executor drafts the reshape memo (pivot options, costs, ordered) —
  target deliverable per both audits: "a failure taxonomy and pre-registered audit
  protocol for scale-hierarchical generative emulators, demonstrated on weak
  lensing," upgraded by whatever survives Gate 0 and Step 1. Andreas decides.

## Bias notes, both directions (standing)

The executor flagged its own incentive to call the wreckage valuable; noted. The
mirror bias is mine: the reconvene authorized five attempts and has an incentive to
see the program vindicated as "discoveries." Both are priced the same way as always:
every claim above rides on pre-registered, adversarially-gated measurements or it
does not ride at all.
