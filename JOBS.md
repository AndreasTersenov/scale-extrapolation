# JOBS — in-flight SLURM jobs & harvest instructions

Last updated 2026-07-10. Phase-1 toy (break & repair). Envs: `source env.sh` (scaledrift,
CPU, has pywt) for measurement; `~/wl-challenge-env/bin/python` (JAX) for training/
generation. GPU via SLURM only.

## Ladder status — phase 1 COMPLETE (see RESULTS-toy.md)
- Rung (i) single-octave overfit — **GREEN** (`7045cea`).
- Rung (ii) two-octave recursion — **GREEN** (`9343214`).
- Rung (iii) GRF end-to-end null — **GREEN / P-NULL PASS** (job 15617056).
- Rung (iv) gowerstreet P4/P5/P6 — **CONCLUDED**: P4 PASS, **P5 HOLDS (robust break)**,
  P6 NOT demonstrated. 3 configs run (jobs 15627183 add, 15628956 FiLM, 15629332 big).
- Rung (v) transfer P13 — **CONCLUDED**: not demonstrated (same cap; arm B transfers an
  amplitude fix). `RESULTS-toy.md` written.

## IN FLIGHT
**None.** All jobs harvested.

## VARIANCE-FAITHFUL PROGRAM (reconvene-approved; unblocks P6/P13)
Ordered (a)→(b)→(c), each pre-registered. Success bar = trained-octave var_slope within 1σ
of real; frozen P6/P13 bars unchanged.
- **(a) SDE churn on existing checkpoints — DONE, insufficient.** Score identity
  s=(t·v−x)/(1−t) verified (`tests_wfm/test_score_identity.py`); churn saturates ~9–10σ short
  on the mean-collapsed 10k checkpoint. `log/2026-07-10-prereg-varfaithful-a-sde.md`.
- **(b) checkpoint sweep — DONE, insufficient alone.** var_slope PEAKS at ~2k steps and
  collapses with training (the dispersion-collapse curve). Peak 2k: oct2 3σ, oct4 within 1σ.
  `log/2026-07-10-prereg-varfaithful-b-ckptsweep.md`.
- **(a)+(b) — near-faithful, gate not clean.** 2k+churn4: oct2 1.9σ/oct3 0.6σ but oct4
  overshoots (global churn is uniform, deficit is octave-dependent). **BUT octave-1 P6 repair
  = 90%** here → the repair works once dispersion is restored.
- **(c) dispersion-regularized objective — HANDED OFF (next step).**
  `log/2026-07-10-prereg-varfaithful-c-objective.md`.

## To run step (c)
```bash
# 1. implement in wfm/cfm.py: cfm_loss_dispersion = cfm_loss + λ·Σ_bin (sd_pred(bin) − sd_data(bin))^2
#    with x1_hat = x_t + (1-t)*v (Tweedie); add --lambda-disp to scripts/run_two_arms.py.
# 2. train (MIG) sweeping λ∈{0.1,0.3,1.0}, keep early stopping (~2-4k steps):
sbatch scripts/train_gowerstreet_film.slurm       # after adding --lambda-disp to the script
# 3. verify trained-octave var_slope within 1σ of real (deterministic ODE, no churn), then:
source env.sh && python scripts/measure_generated.py --npz results/arms_film.npz  # P6/P13
```
Prior: (a)+(b) already showed 90% octave-1 repair, so P6 is likely to PASS once (c) makes the
generator per-octave faithful. Do NOT over-train (var_slope peaks ~2k) and do NOT hand-tune
per-octave churn (that fits the answer).

## DONE (env facts)
GPU: MIG `h100_20gb` sees `CudaDevice(id=0)`; ~217 s (8k steps) / ~270 s (10k) / ~430 s
(25k, 48/96/192). Benign ptxas 12.6.77 clamping warning. CPU generation (transfer) via
affinity-pinned `~/wl-challenge-env` python.

## NEXT (after rung iii green)
Rung (iv): reuse `run_pnull.py` with `--field gowerstreet` (add gowerstreet's real stage-0
2-D running-coupling coordinate as arm B's `cond_by_octave`, replacing the placeholder
`scale_coord`), then `measure_generated.py` → P4/P5/P6. Arm A should BREAK at the
extrapolated octave (P5: var_slope/kurtosis far from real); arm B should repair ≥70% (P6).
This is the load-bearing run and wants a full H100 (`--gpus-per-node=1`, ≤2:59).
