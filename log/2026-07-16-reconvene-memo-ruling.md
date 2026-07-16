# 2026-07-16 — RECONVENE RULING: reshape memo audited and ENDORSED; decision to Andreas

Inputs: RESHAPE-MEMO.md; step readouts + preregs (steps 1–4); raw JSONs re-extracted
independently (arms_selfsim_score.json, downstream_peaks.json, pilot_validation.json);
figures inspected (selfsim_control.png, downstream_peaks.png, pilot_validation.png);
attempt-5 record verified at e96ce89 (2026-07-11 19:08).

## Audit result

- **Numbers verify.** Self-similar control: extrapolated-octave residual 2.4% (arm A)
  / 2.9% (arm B) vs 39–65% on gowerstreet; trained octaves exact; amplitude 0.4/0.5%;
  kurtosis clean. Downstream demo: peak-count bias z = +14.1/+10.2 (ν=1) flipping to
  −6.6/−12.0 (ν=3) while P4 amplitude checks pass ≤7% — the memo's sentences match
  the raw files. Pilot: detection + specificity structure as reported.
- **Pre-registration discipline held** at every step: step-2 amendment committed
  PRE-readout (frozen U-Net cannot ingest 4×4 pairs → second edge re-scoped);
  step-3 criteria, direction, and falsifier fixed before computation; step-4
  pre-registered with its design probe (lognormal drift 1.54→0.53 — itself a bankable
  observation: the drift is generic to multiplicative fields).
- **Suites:** tests/ re-run green under the repo env (14/14). tests_wfm/ re-running in
  background at ruling time (compute-heavy on login cores); executor's pre-readout
  gates reported green. **Infrastructure flag: the Stop hook gates `pytest tests/`
  ONLY — tests_wfm/ is not covered by the backpressure gate.** Fix required before
  paper work: extend the hook (wl-challenge-env interpreter) or add a second gate.
  Filed as a gate-coverage gap, not an accusation — wfm gates were run manually and
  reported, but manual is what the hook exists to replace.

## Scorecard — the judge's misses first, as always

1. **Stale-view rulings (mine, the campaign's clearest judge-side process failure):**
   both 2026-07-16 rulings were written without checking for the attempt-5 readout
   committed five days earlier — a direct violation of handbook re-anchoring rule #4
   ("check disk immediately before every ruling"; my re-anchor read cprime2 and
   PLAN.md but not the fresh git log). The executor's reconciliation is CORRECT in
   every particular and is adopted: the 55→40 re-registration is VOID (a prediction
   cannot be re-registered after the outcome exists); the binding pre-run predictions
   stand (reconvene 55% — MISS; executor B-distribution modal B4 35%, outcome B5 5% —
   also a miss, differently). Handbook amendment: re-anchoring before a ruling means
   git log + newest log/ entries IN THE REPO BEING RULED ON, that day, no exceptions.
2. **Bar-calibration miss #5 (shared):** the step-4 literal 1σ bar against ±0.002
   bootstrap SEs (z≈5 at a 2.5% residual). The substantive readout is unambiguous and
   is adopted on the memo's reading; the bar class (absolute-σ bars against
   arbitrarily small SEs) joins the bar-design ledger — bars on relative residuals
   with pre-declared floors from now on.

## Ruling on the memo

**ENDORSED as written, including its recommendation.** Specifically:

- **Option 1 (the audit paper) with Option 2 folded in** is also the reconvene's
  recommendation. The step-4 control upgraded the paper's spine from a negative claim
  to a bounded law: *hierarchical conditional generation is statistically faithful
  where the between-scale law is scale-invariant (measured floor ~2.5%), degrades
  with the measured drift (39–65% here), and the audit protocol identifies the regime
  without ground truth.* That sentence is the paper.
- **Gate 0 remains blocking**: no novelty claim is written before the kill-tests
  (SPEC-novelty-collapse.md) return. The memo's fallback sentence structure ("we
  observe X, consistent with [refs], and add [causal test / cross-channel invariance
  / field context]") is pre-approved for the ADJACENT-KNOWN outcome.
- **Option 3 (3D/hydro generator program) is deferred, not killed** — revival
  requires a new phase plan, an owner for the data problem, and Andreas's sign-off;
  Option 1 is its prospectus. Option 4 (GOLCONDA/LDT alliance) slots into Option 1's
  discussion section.
- **Conditions before writing starts:** (a) Gate-0 readout in; (b) estimator
  hardening as listed (nanmedian, population-calibrated self-consistency bands);
  (c) the Stop-hook coverage fix above; (d) scope words verbatim from memo §4 in
  every draft.

**Decision requested from Andreas (the memo's ask, seconded):** confirm Option 1 +
fold-in-2, or direct otherwise. On confirmation the executor gets the paper-skeleton
paste (structure = memo §1–2 tables + the bounded law + the audit protocol as
methods; Gate-0-dependent slots marked).
