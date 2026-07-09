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
