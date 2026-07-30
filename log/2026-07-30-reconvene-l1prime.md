# 2026-07-30 — RECONVENE R39: L1 delta memo ruled — L1′ (inference-only)
# APPROVED; oct2rescaled filter ADJUDICATES, oct1-measured = labeled
# ablation; R38's retraining spec logged as a reconvene design miss

Ruled on: log/2026-07-30-l1-delta-memo.md (9eedcf1) + commits b0c36f4
(judge freeze), 7ba3130 (L3), ce0df4a (machinery). Same-day git log taken.

## Verification

1. **The delta's load-bearing claim CONFIRMED independently:** committed
   selection metadata reads train_octaves = [2,3,4] (c1t gowerstreet),
   [2,3,4] (sandbox), [3,4] (stage-D) <!-- src: c1t_selection_*.json
   /meta/train_octaves -->; arms_p2/c1t/train.py builds pools and draws
   base samples only inside `for j in train_octaves`. An octave-1 base draw
   never occurs in any training run of the campaign. The adopted L1's
   training arm was EMPTY: the colored component never enters the loss, so
   retraining = a fresh seed of the unchanged family plus ~2.5 H100-h of
   confound.
2. **A3 arithmetic re-derived:** ν3.0 3-stream mean (11.0696 + 14.1076 +
   12.2447)/3 = +12.474% — MATCH. Near-scale-invariance license numbers
   match the committed N1 C_real values (0.7864/0.7737/0.7853).
3. **Order-set execution audited:** Minkowski judge frozen at b0c36f4
   (synthetics-only validation, 5 tests, V2 connectivity correction
   disclosed pre-freeze); L3 committed with figure + 20-stack table;
   machinery t1–t4 green with two pre-use test corrections documented
   in-test. Zero GPU spent. STOP honored.

## Scorecard (process, not predictions)

- **Reconvene miss, logged with full prominence:** R38 adopted a retraining
  instantiation whose training arm was empty, and attached a training-seed
  rider to a lever with no training component. The executor's night draft
  carried the same flaw; the build-time code-read caught what two rounds of
  design-time reasoning missed. Ledger entry: **before authorizing any
  intervention, verify WHERE the intervened component enters the committed
  pipeline (training loss vs inference path) by reading the code, not the
  recipe's description.** The pre-delegation rule ("any delta → STOP")
  performed exactly as designed — credited.
- **Structural insight adopted into the record:** octave 1 is ALWAYS an
  extrapolated octave in this architecture (weight-tied application beyond
  the training set) — on the trained legs too. The oct-1 whiteness defect
  therefore lives precisely where weight-tied extrapolation meets the real
  field, while the same extrapolation on sandbox produces no whiteness —
  despite the real coloring being measured near-scale-invariant. Why the
  network under-carries a nearly self-similar coloring on one field and not
  the other is now a sharp open question attached to M-COMPOSITION; it
  refines (does not overturn) R37's field-not-scale reading.

## Ruling

(a) **L1′ APPROVED as the Stage-2 instantiation** — inference-only colored-t
base at octave 1 on the COMMITTED checkpoints, the F/F2 precedent, exactly
as the memo specifies: sandbox canary first (standing marginal bars, kill
criterion) → gowerstreet trained leg 3 streams + edge continuity leg 1
stream; identity gate = all-white base reproduces committed maps under the
twice-validated replay criterion; A2 verbatim and scored BEFORE the peak
readout is opened; A3 verbatim (ν2.5 +15.290% sd 0.723; ν3.0 +12.474%);
branch weights carried unchanged (rec/exec: CURED 10/10, IMPROVED 35/40,
NULL 35/30 with the A2 split, REGRESSED 10/10, gates+FLIPPED 10/10); the
replicated dimension is the generation stream, correctly — the seed rider
does not apply to a lever with no training component.

(b) **Filter source: oct2rescaled ADJUDICATES.** Octave 1 is the
extrapolated octave on every leg; a filter measured from real octave-1
details injects target-octave information and would weaken the paper's
deployment claim to exactly the extent the cure worked. The rescaling is
licensed by the measured near-scale-invariance of the real coloring.
**oct1-measured runs as ONE labeled descriptive ablation stream** — never
entering the branch table — because the gap between deployment-pure and
oracle filter is itself paper material: it prices what perfect knowledge of
the target octave's spectrum buys.

(c) **L3 ACCEPTED as committed descriptive evidence with its own scoping:**
on the real field the whiteness→excess association is dose-responsive
(Gaussian-base +5.7σ → +24.6%; t-base ≈+3.6σ → +14%; over-colored arm B →
deficits); universality fails on sandbox (whiter n32 yet deepest deficit).
Adopted reading: count bias = whiteness-driven surplus + field-dependent
baseline deficit; C is a real-field predictor. This scopes M-GRAIN's
causal claim to the deployment field — which is the case the paper claims.

(d) **Judge freeze ACCEPTED.** The Minkowski scorer touches no generation
until Stage 3, including all L1′ outputs.

## Order

Run L1′ per (a)+(b): pre-statement with PENDING placeholders before any
result is read; canary → streams → A2 → peaks; ablation stream labeled;
artifacts → results_p2/l1p_*; R12 throughout; budget ≲0.4 H100-h of the
order set's remaining ~4. **STOP at the readout for reconvene
adjudication.**
