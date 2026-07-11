# 2026-07-11 — RECONVENE RULING: 4b′ adjudicated — lever bar failed as registered; mechanism half-tested; 4b′-ii authorized

Inputs: RESULTS-phase1c.md (4b′ section), log/2026-07-11-prereg-4bprime-condrobust.md,
results/readout_4bp.png, arms_4bp jobs 15753842/15753843 (hashes verified). Audit:
suite green (14/14 + new tests_wfm/test_cond_corrupt.py validation gates); diff
0125bbd..HEAD clean — no hook/settings changes, no existing-test edits; prereg
committed before submission; pull-before-preregister complied with (two-tier bars and
measured ceiling implemented as ruled).

## Ruling 1 — LEVER BAR FAILED. Registered verdicts stand.

End-to-end oct-2 var_slope vs the 4a ceiling: z ≈ 4.7–7.6 across arms and s_max.
Project bar also failed. Scorecard: reconvene P-4b′-lever 55% → MISS; executor
P-lever 60% → MISS; executor P-s0.3>s0.1 → HIT. No relitigation: under the bars as
written, 4b′ as run did not work.

## Ruling 2 — but the experiment tested HALF the mechanism, and said so cleanly.

The decomposition (readout_4bp.png) is unambiguous: corruption training lifted the
heads' own given-real-coarse ceilings essentially to real (oct 2: 0.956→0.996 vs real
1.020; oct 3: 0.683→0.748 vs 0.801) and moved kurtosis to real at oct 3 — a free
regularization gain — while end-to-end did not move, because generation was run at
s_gen = 0, the "trust the conditioning fully" mode, so the trained robustness is
never engaged. The cascaded-diffusion recipe this lever imports applies non-zero
conditioning noise AT INFERENCE; the pre-registered s_gen = 0 choice (made to avoid
a hand-tuned knob) removed the mechanism's operating mode. That is a design gap in
the arm, not a refutation of the lever. Charge: shared — the ruling that authorized
4b′ (mine) did not specify the inference side either. Logged as a joint miss.

## Ruling 3 — 4b′-ii AUTHORIZED: engage the robustness at inference. Same variable, zero training.

Reuses the existing s=0.3 checkpoints; one generation job (~10 MIG-min). This is the
SAME single variable (conditioning robustness) with its operating mode enabled — not
a new lever, not bundling.

Anti-tuning discipline (binding): s_gen must come from a MEASURED reference, not a
sweep-and-pick. Procedure: measure the actual conditioning drift — the per-octave
discrepancy statistic between generated coarse and real coarse that the corruption
model can express (e.g. the relative residual level implied by the end-to-end vs
given-real-coarse gap, or an equivalent measured mismatch statistic; exact estimator
is executor periphery, but it must be computed BEFORE any generation run and written
into the prereg). Primary adjudication at s_gen = s_matched ONLY. A pre-registered
sensitivity pair {s_matched/2, 2·s_matched} is reported descriptively — it never
enters adjudication. If s_matched falls outside the trained range (> s_max = 0.3),
report and stop; do not extrapolate the conditioning.

Bars unchanged (two-tier): lever bar = end-to-end within 1σ combined of the
CORRUPTION-TRAINED model's own ceiling (now ≈ real — so the lever and project
dispersion bars nearly coincide at oct 2; adjudicate both, report both). Bounded-OOD
variance requirement binds: injected conditioning noise must not inflate amplitude
(oct-1 detail_std within the established band). Kurtosis: report at all octaves; the
student-t branch stays pre-named for variance-passes/kurtosis-fails, AFTER this
readout, never bundled.

## Predictions (reconvene, before 4b′-ii)

- P-4b′ii-lever: **50%.** For: the mechanism is now complete and the literature
  applies it exactly here; the heads are calibrated. Against: the drift is a
  systematic modulation deficit, not additive noise — matched Gaussian corruption may
  be the wrong noise model for it.
- P-project-dispersion (all three octaves vs real): **35%** (ceilings now ≈ real, so
  it rides on the lever).
- P-kurtosis-bar at oct 2 after 4b′-ii alone: **15%** (3.90 vs 6.69 is a tail-shape
  gap; expect the student-t branch to be next regardless).

## Standing

Pull and read this ruling before pre-registering 4b′-ii. One variable. Grep-verify.
STOP at the readout.
