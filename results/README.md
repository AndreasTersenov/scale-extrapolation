# results/ layout (reorganized 2026-07-24)

Phase-0/1 outputs. Phase-2 outputs live in `results_p2/` (flat, unchanged).

- `figures/maps/` — map-level galleries (`gallery_*`, `maps_*`, `nll_sigma_maps`)
- `figures/explainers/` — roadmap, phase1_story, intuition_1..3
- `figures/stage0/` — stage-0 drift measurement (drift_*, running_couplings,
  pilot_validation)
- `figures/readouts/` — arm readouts, gates, verdicts, diagnostics
  (readout_*, g1c_verdict, nllhead_gate, nll_diagnosis, signature_4a,
  smatched_4bpii, downstream_peaks, selfsim_control, toy_underdispersion,
  conditional_variance_profiles)
- `scores/` — all *.json (score/verdict/measurement files)
- `npz/` — all *.npz (tracked: arms_aug, nll_diagnosis, profiles,
  signature_4a; the rest are gitignored regenerable bundles)
- `joblogs/` — SLURM job logs (gitignored)

Historical `log/*.md` entries written before 2026-07-24 reference the old flat
paths (`results/<name>`); map them here by basename. Scripts and the living
docs (README, RESULTS*.md) were updated to the new paths in the same commit.
