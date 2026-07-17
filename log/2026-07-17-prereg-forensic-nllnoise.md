# 2026-07-17 — PREREG: NLL-head noise forensic (R5; executor draft 4, descriptive)

Authorized by log/2026-07-17-reconvene-morning-harvest.md R5. Committed BEFORE any
generation. DESCRIPTIVE: no training, no project bar; settles the
channel-dependence attribution (R4's provisional re-scoping) within the frozen
phase-1 artifacts before any memo/paper/Gate-0 language is rewritten.

## Protocol (pinned)

- FROZEN checkpoints `data_cache/ckpt_aug/arm{A,B}_gowerstreet.pkl` (4a, hash
  1e61bd812a lineage), untouched.
- Regenerate the standard end-to-end stack: 64 held-out gowerstreet tiles
  (tiles_pnull, last 64, normalize_tiles), recursion from octave 4, identical to
  run_two_arms EXCEPT the sampler: **detail = mu only** — the single forward at
  (x_t=0, t=0) with the e^{g}·z noise term REMOVED (sample_nll minus its noise).
  std_by_j from the ckpt (includes the extrapolated octave-1 value the 4a run
  used); arm A unconditioned, arm B coords/COORD_NORM as frozen.
- Score with the FROZEN scorer (measure_generated.couplings, n_boot=200) →
  results_p2/forensic_nllnoise.json + npz of the stacks.
- Reference points (frozen record): real oct-2 var_slope 1.020±0.041; 4a
  WITH-noise e2e 0.746/0.734; C1 plain-CFM e2e 0.921/0.927.

## Pre-declared confound (read before the branches)

In the 4a checkpoints the σ-channel carries 86–93% of conditional detail variance
(signature_4a record), so mean-path details have only ~26–37% of the full
amplitude: each recursion step assembles a coarse that is progressively SMOOTHED,
a different off-manifold direction than the with-noise case. The probe therefore
tests "is the white noise the damaging ingredient?" only if smoothing is benign;
per-octave detail_std of the mean-path stacks is measured alongside as the direct
amplitude-starvation gauge.

## Branches (oct-2 end-to-end var_slope, frozen scorer, BOTH arms unless split)

| branch | definition | weight | interpretation (pre-declared) |
|---|---|---|---|
| F-CLOSE | ≥ 0.87 both arms (deficit ≤ ~15%, closes toward C1's −10%) | 30 | white-noise attribution CONFIRMED within phase-1 artifacts; R4 re-scoping adopted |
| F-MID | 0.78–0.87 | 20 | partial closure; noise is A driver, smoothing also costs |
| F-COLLAPSE | ≤ 0.78 (at/below the with-noise level) | 40 | probe CONFOUNDED by amplitude starvation (report detail_std); attribution rests on the C1 single-variable comparison alone |
| F-OVERSHOOT | > 1.10 (real + ~2SE) | 5 | smooth-coarse modulation overshoot; treat as confounded, report |
| OTHER | arms straddle boundaries / anomalies | 5 | report per arm |

Secondary descriptives from the same run (no bars): per-octave detail_std ratio
(mean-path vs real), kurtosis, octave-1 values.

## Job

One CPU SLURM job (def-lplevass, 4c/16G/0:30): generation under wl-challenge-env
(single forward per octave — no ODE), scoring under the env.sh stack.
scripts_p2/forensic_nllnoise.py + scripts_p2/score_forensic.py. Job ID in JOBS.md
at submission.
