# 2026-07-11 — RECONVENE RULING: attempt 4a adjudicated

Inputs: RESULTS-phase1c.md (4a section), log/2026-07-11-prereg-4a-augment.md,
results/signature_4a.png, results/arms_aug_score.json, jobs 15744601 (hash 1e61bd812a
verified). Audit: suite re-run green (14/14) in repo env; diff b423c7c..HEAD clean — no
hook/settings changes, no existing-test edits; new tests_wfm/test_augment.py adds
validation (exact orbit, invariant amplitude, preserved per-bin conditional variance).

## Ruling 1 — the onset rule: manifest-intent reading adopted. 4a = CONFIRMED (strongest form).

The literal rule ("first checkpoint ≥0.10 below the curve's maximum") fires at the 2k
warm-up dip, which PRECEDES the curve's own peak (6k). A collapse onset cannot precede
the growth it is the decay of — the literal reading contradicts the quantity the rule
operationalizes. The rule was drafted against monotone-from-early-peak curves (true of
the baseline, where both readings coincide at 4k) and is defective for non-monotone
curves. Adopted reading: first drop ≥0.10 below the RUNNING maximum → never fires →
onset_aug censored >20k ≥ 5× onset_base (4k) → **CONFIRMED**, the prereg's own
strongest-form censoring branch.

This is NOT an "ambiguity = negative" case: that doctrine prices ambiguous EVIDENCE
against the hypothesis. Here the evidence is unambiguous (no collapse within the
horizon; σ-share 86–93% vs baseline 81→17%); the defect is in the rule's drafting.
**Scorecard: bar-design miss #4** (onset rule unspecified for non-monotone curves) —
charged to the reconvene, whose ruling specified the staged test. The executor's
both-readings-reported handling was exactly right.

## Ruling 2 — the collapse law is now causally confirmed; augmentation is adopted.

Removing finite-data memorization headroom (322→2576 tiles, conditional law exactly
preserved) removes the collapse — a causal intervention on the hypothesized mechanism,
not another correlate. With the three prior confirmations (L2-CFM pushforward, penalty
target, NLL head), the law — *finite-data memorization of the conditional mean starves
the conditional variance in whatever channel carries it* — is the campaign's
strongest methods finding. Corollary confirmed on GRF/arm-B: the OOD amplitude
blow-up was memorization-coupled too (gone under augmentation; P4 passes at oct 1).

D4 augmentation enters the FROZEN generator config from here on (data-level, law-
preserving, test-gated). Not a tuned hyperparameter; not subject to per-attempt ablation.

## Ruling 3 — original 4b is DEPRIORITIZED, not dead; the round's second stage is re-scoped.

The g1c ruling made 4b (two-stage frozen-mean residual variance) available "if the bars
are still unmet". Bars remain unmet (end-to-end oct-2 var_slope 0.746±0.014 vs
1.020±0.038, ~7σ; kurtosis ~3.0 vs 6.7). But 4a's second finding moved the deficit:
the head's conditional response GIVEN REAL COARSE is ~0.96 of real — original 4b's
target is measured nearly closed. The binding failure is now **recursion compounding**
(each octave conditions on generated, slightly flattened coarse; the flattening
accumulates 0.475→0.746 across three octaves).

**Attempt 4b′ (single variable): conditioning robustness / anti-compounding.**
Named default lever: conditioning augmentation — during training, corrupt the coarse
input (noise level exposed to the model, e.g. via FiLM) so generation is robust to its
own drift; this is the standard, well-precedented fix for exactly this failure in
cascaded diffusion (Ho et al. 2021, Cascaded Diffusion Models; used in Imagen). It
enters as a pre-registered arm like any reconvene suggestion — if the executor judges a
different anti-compounding mechanism better-matched to modulation-flattening drift
(e.g. self-conditioning on generated coarse), the choice is periphery; the SINGLE
VARIABLE is "conditioning robustness", whatever its implementation. No bundling with
anything else.

## Ruling 4 — bars: two tiers, both from measured references (rule 3 compliance).

Pre-4b′ requirement: measure and report in the prereg the **ceiling** — the head's
given-real-coarse implied var_slope at octaves 2, 3, 4 (existing 20k checkpoints;
no new training). Then:

- **Lever bar (adjudicates 4b′):** end-to-end var_slope within 1σ (combined) of the
  measured ceiling at octaves 2, 3, 4 simultaneously. This judges decompounding against
  what decompounding can achieve.
- **Project bar (G-1c dispersion, FROZEN, unchanged):** within 1σ of REAL at octaves
  2, 3, 4 simultaneously. Flagged now: at oct 2 the measured ceiling (~0.96) sits
  ~1.5–1.7σ below real — the frozen bar may be unreachable by decompounding alone.
  If 4b′ passes the lever bar but the project bar fails BECAUSE the ceiling binds,
  that is the revival condition for original 4b (residual head under-response becomes
  the binding deficit again) — pre-named now to prevent relitigating.

Kurtosis: still failing and now MEANINGFUL (with σ alive, generated kurtosis finally
measures the conditional, not the memorized mean). The student-t conditional remains
the pre-named lever for the variance-passes/kurtosis-fails branch — AFTER 4b′, never
bundled with it.

Bounded-OOD variance requirement: descriptively satisfied under augmentation (arm B
blow-up gone; oct-1 detail_std 0.689–0.729 vs real 0.743). Remains binding at every
bar adjudication.

## Predictions (reconvene, pre-registered before 4b′ is designed)

- P-4b′-lever (lever bar met): **55%** — the fix is standard for this failure class,
  but additive-noise corruption may model modulation-flattening drift poorly.
- P-project-dispersion after 4b′ alone: **30%** (ceiling risk at oct 2).
- P-ceiling-binds (lever passes, project fails via ceiling → 4b revival): **25%**.

## Standing rules (unchanged)

Pull and read this ruling before pre-registering. One variable. Grep-verify scripted
edits. Budget note: 4a cost 7.6 MIG-minutes; 4b′ expected similar — no cap concern.
