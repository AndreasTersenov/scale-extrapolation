# 2026-07-28 — PREREG: Phase-3 Stage 0 — the full audit battery under the
# FINAL SHIPPED configuration (committed BEFORE any run or scoring;
# reconvene lines to be added at review — STOP stands until then)

Authorization: PLAN-phase3.md Stage 0 (Andreas's reopening sign-off in its
header); house rules per R36 (#1–#11, seed lesson, ambiguity=negative,
numbers-by-copy). Zero training.

## The question

The deployment-edge peak excess (arm A +13.13%±3.10 at ν=2.5, +14.62%±3.08
at ν=3.0 <!-- src: audit_peak_ci.json, stage_d leg -->) was measured on the
PRE-symmetrization sampler, and the lattice defect it predates was itself a
peak producer. Does the final shipped configuration — heavy-tailed base +
caged selection at the committed Stage-D picks + the D4 group-averaged
sampler (F2, ships per R34) — close, shrink, or leave the excess?

Known texture cutting BOTH ways, stated now: the Stage-D arm-A edge maps
showed only weak level-1 parity (T = 2.4, A′ M1) but significant coarser
block-level bias (levels 2/3: T = 3.7/4.1) — and count excess can also ride
symmetric graininess that symmetrization does not touch. Genuinely open.

## What runs (one MIG job, after review clears this prereg)

F2 group-averaged e2e generations from the committed Stage-D checkpoints
(arm A @9000 ADJUDICATING; arm B @7500 descriptive) on the frozen Stage-D
test tiles (tiles_pnull[298:330]) — the deployment-edge leg. The
trained-scales leg scores the COMMITTED F2 gowerstreet generations
(f2_test_gen.npz F2_gowA_e2e @16000, already on disk) — no new generation.
Identity-gate convention as in F2 (zero-offset equivalence already
established for the machinery; the runtime gate re-asserted on the Stage-D
substrate). Seeded group assignments recorded.

## The battery (scored after the job; all conventions frozen from the record)

1. **Peaks (PRIMARY):** counts at ν ∈ {1, 2.5, 3} vs the real test tiles,
   field-bootstrap CIs (n_boot 5000) + the per-parent panel (blocks
   [(0,10),(10,21),(21,32)], verified); ν=1 never headlined (#10); the
   adjudicating quantities are the ν=2.5 and ν=3.0 excesses at the edge,
   arm A.
2. **Spacing (descriptive, WATCHED):** nn profile, K_real = 174, frozen
   edges; expectation registered below.
3. **Starlet-ℓ1 (descriptive):** the frozen scorer's edge and trained legs
   on the F2 maps vs the standing 10% floors (the shipped configuration
   must keep the SL1 pass — a regression here is gate-grade news).
4. **Marginal suite (descriptive context):** e2e couplings vs the real test
   references at octaves 2–4 + edge, standing bars as context; the F2
   sandbox verdicts stand as the committed marginal evidence (no bar
   changes in Stage 0 per the PLAN).

## Branches (mechanical; arm A edge peaks adjudicate; ambiguity = negative)

Let (e25, e30) = the new excesses with SEs (s25, s30); references
(r25, r30) = (+13.13%, +14.62%) with SEs (3.10%, 3.08%). Reduction
significance: Δν = (rν − eν)/√(sν² + 3.1ν²... the committed SEs) — computed
per threshold as (rν − eν)/hypot(sν, sν_ref).

| branch | rule (BOTH thresholds unless stated) | executor weight | meaning |
|---|---|---|---|
| S0-CLOSED | 95% CI of eν includes 0 at both ν; per-parent panel shows no sign-consistent excess | 22 | the excess was the lattice artifact; the peak tier CLOSES for free → PLAN Stage 3 (skip 1–2) |
| S0-SHRUNK | not CLOSED; both eν < rν/2 AND both Δν ≥ 2 | 43 | symmetrization removed the lattice component; a genuine residual remains → Stage 1 profiles it with a measured head start |
| S0-UNCHANGED | both eν ≥ rν/2 (or reductions insignificant, Δν < 2, at both) | 25 | the excess never was lattice-driven; symmetric graininess or genuine joint miscalibration → Stage 1 unchanged, Stage 2's colored base is the live lever |
| mixed/at-bar | thresholds disagree on category, OR any eν within 0.5·sν of a category boundary (#11 band) | (in gates) | the WORSE category governs for sequencing; disambiguation pre-authorized: ONE regeneration with a fresh sampler PRNG (doubling the gen-side sample), then re-apply the rules once |
| gates | infra (one resubmission); identity-gate failure; catastrophic marginal regression (>50% dispersion error at any scored octave) | 10 | infra/repair; a marginal catastrophe would be an F2-substrate interaction unseen at trained scales — STOP, report |

Weights sum: 22 + 43 + 25 + 10 = 100. Reconvene lines to be added at review.

**Registered expectations (descriptive lines, scored at the readout):**
- nn spacing at the edge, F2 maps: P(T ≥ 3, i.e. the residual persists) = 85
  (it survived four cures; nn is D4-invariant).
- starlet-ℓ1 edge leg still passes all scored scales: P = 90.
- Arm B (descriptive) shows larger deviations than A wherever they differ:
  P = 75 (its record: every off-binding pathology).

## Sequencing consequence (pre-stated, from the PLAN)

S0-CLOSED → Stage 3 directly. S0-SHRUNK/S0-UNCHANGED → Stage 1 (mechanism
profiling) with this readout's numbers as its measured references. Either
way the marginal/starlet/spacing rows become the method paper's
final-configuration audit table rows.

## Discipline

Zero training; one MIG job + CPU scoring; artifacts →
results_p2/stage0_p3_*. R12 verbatim throughout; tests-first N/A (all
instruments frozen and validated in the committed record; any NEW estimator
would trigger the tests-first rule). **STOP now for reconvene review of this
prereg; STOP again at the readout.**
