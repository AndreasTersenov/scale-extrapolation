# 2026-07-25 — PREREG (DRAFT FOR RECONVENE REVIEW — R31 order 2; NO RUNS SUBMITTED)
# The lattice-defect cure experiment: correction, joint selection, and the data dial

Re-poses Phase B of log/2026-07-24-prereg-placement.md (R30-approved, GATE-S
fired before any Phase-B run) in the light of Phase A′
(log/2026-07-25-parity-localization-readout.md): the tier-3 sandbox residual
is a lattice-parity defect that LIVES IN THE DETAIL COEFFICIENTS as transient
nonzero channel means, is harvested by the marginal-optimal early stop
(A@7500 = the defect-curve maximum), and trains away at 1× by 17.5k steps.
The reconvene's registered lean (architectural/synthesis-grid, 55%) is scored
against this at review. Executor weights registered below; reconvene
re-registers at review; freezes on approval.

## The re-posed question

Not "is placement the next data rung" but: **the moment ladder makes
different statistics healthy at different training times (tails early,
lattice symmetry late); single-checkpoint marginal selection cannot satisfy
both at 1×. Which cure restores joint health without spending the tail
rescue — and does data widen the joint-viable window?** Every branch teaches:
a correction result, a selection result, or a data-scaling result, each
mechanism-grade for the paper and the PM program.

**Primary estimand (from A′, mechanically upstream of peak parity):**
T_coef = the octave-1 coefficient D4-statistic T (max |z| over the 9
components, per-field profiles vs the 32 held-out truth tiles, phase-A/A′
frozen conventions; "clean" = T_coef < 3, the standing null bar). Secondary,
always reported: octave-2 T_coef, level-1 peak-parity T and odd-odd share,
the marginal verdict suite (kurtosis/var_slope bars as in the C1-t prereg —
must-not-regress guards), and nn (descriptive; its mild real-field signal is
a parity-independent open residual, stated).

**Measured references (A′, verbatim):** committed picks A@7500 T_coef = 15.3,
B@2500 = 10.8; A's curve minimum 2.1 @20k; truth-null bar 3.0.

## Arms (ordered cheap-first; F and J are inference-only from committed checkpoints)

**F — inference-time DC correction (zero training, hours).** Estimate the
per-channel, per-octave mean offset of the generated details on VALIDATION
generations at the committed selected checkpoint (arm A; B descriptive);
subtract the constant at sampling; adjudicate on TEST. The correction is a
symmetry-restoring calibration (truth channel means are zero in expectation);
fitted on val only (cage discipline), one constant per channel/octave, no
re-picks.
- F-CLEAN: corrected T_coef < 3 AND parity T < 3 AND all marginal bars still
  pass (kurtosis/var_slope, hc + e2e, octaves 2–4 + edge conventions of the
  substrate). Executor 45. Meaning: the deployed defect is curable for free;
  the paper's generator ships with the correction and the audit gains a
  "calibrate the symmetry, not just the moments" exhibit.
- F-PARTIAL: T_coef drops ≥2× but ≥3, or parity residual remains. 30.
  Meaning: the defect is not a pure DC offset — higher spatial structure in
  the bias; localization continues (the M2 corr components say where).
- F-BREAKS: any marginal bar regresses below its floor. 15. Meaning: the
  offset is entangled with the tail calibration — correction trades one
  disease for another; the joint window (J/D) is the only route.
- Gate (degenerate estimation: val offset estimates unstable across K-fold
  val splits by >2× their SE): 10. Null meaning: correction is
  under-determined at 32 val fields — refit on repl64's val convention.

**J — joint selection (zero training, hours).** Re-run the selection sweep
on the committed dense checkpoints with the pre-registered joint criterion:
argmin max(rel_vs/0.10, rel_kurt/0.15, T_coef_val/6.0) on val-32 (the 6.0
normalizer prices T_coef = 6 as equal to a bar-critical marginal; fixed here,
before any scoring); adjudicate the joint pick on test.
- J-WINDOW: a checkpoint exists whose test verdict passes marginal bars AND
  T_coef < 3. Executor 35. Meaning: 1× has a joint-viable window the marginal
  cage missed; selection criterion, not data, was the binding constraint.
- J-TRADEOFF: no checkpoint satisfies both (the M5 curve suggests tails decay
  before the defect does). Executor 50. Meaning: at 1× the ladder's timing
  makes single-checkpoint selection insufficient — F (if clean) or D is the
  cure; a load-bearing paper sentence either way.
- Gate (val/test pick instability as in C1-t's noted selection noise): 15.
- Precedence: J adjudicates INDEPENDENTLY of F (different cure class).

**D — the data dial (training; the R30-approved arms, estimand upgraded).**
N1 (322 fields, seed 1 — the fresh-seed rider unchanged), N8 (2576), N32
(10304), C-ENS (322 × 8 exact conditional redraws, matched to N8) — data
recipes exactly as in the R30-approved prereg (make_placement_data.py,
committed unused). Identical training config and marginal-caged selection.
Estimands per arm, at the marginal-caged pick: T_coef (primary); plus the
joint-window indicator (J's criterion re-run on that arm's curve).
- D-WIDEN: T_coef at the caged pick < 3 at N8 (and stays clean at N32), i.e.
  the tail-flat window of 8× (taildyn) overlaps the parity-clean region —
  data dissolves the trade. Executor 40. TRANSFER CLAUSE (R27, verbatim):
  this licenses "the joint window is data-limited in this model class,"
  never "more parents cure the gowerstreet excess"; the real-field echo
  stays load-bearing.
- D-SHRINK: T_coef at the pick improves ≥2× vs N1's 15.3 but ≥3. 25.
  Meaning: data helps; combined with F/J for full cure.
- D-FLAT: <2× improvement at N8 AND N32. 15. Meaning: the defect's harvest
  is selection-timing-locked, data-independent — H-selection alone; F/J are
  the designated cures; the reconvene's architectural lean partially
  vindicated at the selection interface.
- Currency (conditional on D-WIDEN/D-SHRINK): CUR-PARENTS e(C-ENS) worse
  ≥1.25× than N8: 40 · CUR-ENSEMBLE better ≤0.8×: 25 · CUR-EQUIV: 35 —
  meanings as in the R30 prereg (PM data design).
- Gates (infra/degenerate marginals, one resubmission): 10. N1-seed-1 rider:
  marginal bars scored descriptively (seed robustness), non-adjudicating.

**Real-field echo (descriptive, load-bearing):** F applied to the committed
gowerstreet arm-A generations (offset re-estimated on its val split);
coefficient and parity statistics before/after; parent-blocked conventions
(#10). The mild nn spacing residual is re-scored after correction: does the
real-field position-pure signal survive the symmetry fix? (Its answer feeds
the paper's frontier section either way.)

## Discipline

Order: F, J (inference-only; can run immediately on approval) → readout →
D (data build + 4 MIG training jobs, the R30 pattern) → readout. Bars from
the measured references above with the standing floors; final-state
statistics (#8); reference noise budgeted (#9); parent/field exchangeability
conventions (#10); numbers by verbatim copy (R12); tests-first for the F/J
machinery (offset estimator, joint criterion) before any scoring; STOP at
each readout; the reconvene adjudicates. NO RUNS until this freezes on
reconvene approval.
