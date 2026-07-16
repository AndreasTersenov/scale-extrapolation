# NIGHT ORDERS — phase-2 overnight run: Stage A → Stage B → Arm C1 (2026-07-16)

Reconvene-authored. This file pre-delegates gate adjudication for ONE overnight run
under branch-complete rules. Read PLAN-phase2.md FIRST (it governs; this file only
instantiates its gates with numbers). Andreas has selected rescue-first; Gate-0 is
CLEARED (log/2026-07-16-reconvene-gate0.md). One session, cheap model, sequential.
**Ambiguity at any gate = STOP, commit, write the morning summary, end.** Morning
harvest is a reconvene; nothing beyond these orders is authorized.

## Chore 0 (before any science) — the test-gate fix

Extend the Stop hook to BOTH test trees under their correct envs (tests/ via env.sh
stack; tests_wfm/ + new tests_p2/ via ~/wl-challenge-env python; pyproject
`testpaths` updated). Verify by running the full gate once (must be green:
14 + 32 + whatever tests_p2 holds). This was a standing pre-writing condition and it
protects tonight's work. Commit separately.

## Stage A — the lognormal sandbox (new code under sandbox/, tests in tests_p2/)

1. Prefer FULLY SYNTHETIC lognormal: choose a power-law-ish P(k) and shift; generate
   the Gaussian layer yourself → truth by construction (no fitting ambiguity). Using
   the shared lognormal data is the fallback, documented if taken.
2. Conditional truth: exact conditional GRF sampling of the Gaussian layer given the
   Haar coarse projection (linear algebra), then exponentiate. If exact conditioning
   under the log map is messier than expected, the pre-authorized fallback is
   mode-redraw ensembles (fix coarse Gaussian modes, redraw fine) — which is exact.
   Tests-first: a closed-form validation test of the conditional sampler on a small
   grid (analytic conditional mean/cov vs sampled, machine-tolerance) BEFORE
   generating ensembles.
3. Deliver: ≥256 coarse fields × ≥64 conditional details each, at the phase-1
   geometry (128², octaves 1–4); TRUE conditional statistics (var_slope, kurtosis
   per octave) with bootstrap SEs; committed as data recipes + seeds (arrays to
   $SCRATCH, summaries in-repo).

**GATE A (self-adjudicated):** apply the frozen scaledrift instruments to the true
ensembles. PASS = at every octave, instrument var_slope within max(5%, 3·SE_rel)
relative of truth AND kurtosis within max(10%, 3·SE_rel). FAIL or mixed → STOP
(instrument bug; nothing downstream is interpretable). Record per-octave table.

## Stage B (runs only if Gate A passed)

**B1 — dependence range & shape.** Estimator (your periphery: k-NN or regularized
regression on patches) validated FIRST on the sandbox, where conditional
predictability is computable. Then on gowerstreet: predictability-saturation curve
(conditional detail variance at a point vs coarse-context radius r), per octave;
then the SHAPE test — filament-aligned oriented masks vs isotropic disks at MATCHED
AREA. Deliver r* per octave + anisotropy verdict with bootstrap bars. Pre-register
expected shapes with weights BEFORE computing on gowerstreet (P-B1a 70% locality /
P-B1b 50% anisotropy are the standing reconvene numbers; add yours).

**B2 — crops inventory.** Correlation-adjusted effective pair count for stride-
shifted overlapping crops (stride 32 and 64) on gowerstreet parents, on top of the
8× symmetry group. Table only, no training.

No gate here — B is measurement; report and continue.

## Arm C1 (runs only if Gate A passed) — vanilla CFM + augmentation, the un-run arm

The point: the full-distribution model (NO Gaussian-NLL head, NO variance crutch;
plain conditional FM, the phase-1 architecture otherwise) WITH the collapse cure it
never got. Two legs, sandbox first.

- **Data regime must mirror phase 1:** 322 sandbox tiles (not more) + 8× symmetry
  augmentation (+ crops ONLY if B2 shows it's law-preserving at chosen stride —
  else without; single variable discipline: if in doubt, WITHOUT crops).
- **Prereg (commit BEFORE training):** checkpoint curve {1,2,4,6,8,12,16,20}k as in
  4a; gate-branch weights for the outcomes (alive-calibrated / alive-but-tails-fail
  / collapses-again / degenerate), your numbers alongside the standing reconvene
  numbers (P-C1a 60% dispersion alive, P-C1b 45% kurtosis at truth, P-C1c 45%
  recursion calibrated).
- **Sandbox-leg bars (self-adjudicated, from measured truth):** at final checkpoint,
  all trained octaves simultaneously: dispersion |gen−truth|/truth ≤ max(10%,
  3·SE_rel); kurtosis ≤ max(15%, 3·SE_rel); end-to-end recursion from octave 4:
  same bounds at every octave. (Phase-1 failures were 25–65% — these bars
  discriminate cleanly.) Report the implied-var_slope training curve as in
  signature_4a.
- **Gowerstreet leg (runs only if the sandbox dispersion bar passes):** identical
  recipe on the real 322 tiles + augmentation; score with the FROZEN scorer;
  readout is DESCRIPTIVE (tables + the standard figures) — adjudication happens at
  the morning reconvene against the G-1c project bars. No further legs, no
  variants, no tuning passes regardless of outcome.

## Discipline (standing, all night)

SLURM for anything past ~1 core-minute (MIG h100_20gb, ≤2:59, account
rrg-lplevass; CPU jobs → def-lplevass). Pre-submission job logging in JOBS.md
(ID, config hash, expected outcome). Heartbeat prints in long scripts. Tests-first
for every new estimator (tests_p2/). Grep-verify scripted edits. Commit+push after
every unit. No edits to PLAN-phase2.md, no phase-1 file changes (additive layout:
sandbox/, depmeasure/, arms_p2/c1/). Budget cap tonight: 3 H100-hours (expected
usage well under 1). If the queue stalls >2h on any job, park and proceed to
whatever doesn't need it; note in the summary.

## Morning deliverable

RESULTS-phase2.md with sections A / B1 / B2 / C1 (tables + figure pointers), topped
by a MORNING SUMMARY block: gates passed/failed, prediction verdicts vs the
pre-registered weights, the three numbers that matter most, and open questions for
the reconvene. Commit, push, STOP. Do not design C2/C3/C4. Do not touch the paper.
