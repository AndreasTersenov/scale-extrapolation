# CLAUDE.md — scale-extrapolation

Stage-0 data study for the RG-consistent generation project (D4). Read `PLAN.md` first,
always. Wider context (only if needed):
`~/claude-notes/brainstorms/2026-07-09-dl-project-directions.md` (§D4) and
`...novelty-sweep-RESULTS.md` (§D4).

## Hard rules

- `PLAN.md` "FROZEN CORE" (measurements, ladder, predictions, gates) is immutable in
  this session. Objections go to `log/` and `RESULTS.md`, not into silent redesign.
  Implementation details are yours.
- The GRF null gate comes FIRST. No interpretation of real-field drift before the null
  passes.
- Every distance/estimator gets bootstrap error bars; a drift claim without error bars
  doesn't count.

## Environment & conventions

- Rorqual. Data: `/project/rrg-lplevass/shared/wl_chall_data/` (`GRF_HF`, `lognormal`,
  `gowerstreet*`) — exists on compute nodes; may be absent on login nodes (known gotcha,
  see `~/experiments/CLAUDE.md`). Big intermediates → `$SCRATCH` via `~/links/scratch`;
  plots/summaries → in-repo `results/`.
- SLURM account `rrg-lplevass`; `rorqual-jobs` skill for queue strategy. Most of this is
  CPU-friendly; GPU fine for batched wavelet transforms (`~/software/wl_stats_torch`).
- Log format: `log/YYYY-MM-DD-<slug>.md`, hypothesis → setup → expectation → result →
  updated belief. Commit early and often; local-only repo (no remote yet).

## Git discipline (decided 2026-07-09)

- **Commit after every meaningful unit**: a validated estimator, a completed measurement
  grid, a log entry, a RESULTS.md section. WIP commits are fine; uncommitted work at
  session end is not.
- **Push after committing** if a remote is configured (`git remote -v`) — currently
  local-only; Andreas is setting up SSH auth + private GitHub remotes. Never force-push;
  never rewrite pushed history.
- Worktrees: NOT used in stage-0 (one sequential agent per repo). They become the tool in
  the toy phase for parallel variant exploration (note: local-only repos need
  `worktree.baseRef: "head"` since there is no origin/HEAD yet).

## Backpressure (non-negotiable)

- **Tests-first**: before implementing any estimator, write its validation test in
  `tests/`. A Stop hook (`.claude/settings.json`) runs pytest and blocks session
  completion while tests fail — this is deliberate; fix or xfail-with-justification.
- A number plotted or written into RESULTS.md whose validation test is not green does
  not exist.
- Validation gates for THIS repo:
  1. DWT round-trip: reconstruction error at machine precision.
  2. GRF null as an executable test: on a synthetic power-law GRF generated in-test,
     measured drift consistent with zero within bootstrap CI.
  3. Estimator consistency: doubling the number of maps shrinks bootstrap error ~sqrt(2).
  4. Symmetry: drift metrics invariant under flips/90-degree rotations within noise.

## Compact instructions

When compacting, preserve: modified file paths, test commands and their latest status,
the measurement/grid currently running, SLURM job IDs, and any deviation-from-PLAN notes.

## Long jobs

Prefer Bash run_in_background or the Monitor tool to babysit SLURM jobs within a session;
/loop for periodic in-session polling. Consult the rorqual-jobs skill before submitting.

## Scripted-edit rule (added 2026-07-10 after three silent no-op edits)

Every scripted find/replace (sed/python -c/etc.) must be followed by a grep verifying
the change landed (and, for outputs, a check that results are not byte-identical to the
prior run when a change was intended). Silent no-ops corrupt attribution.
