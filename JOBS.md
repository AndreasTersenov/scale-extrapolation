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
- **(c) dispersion-regularized objective (Tweedie-mean-std matching) — DONE, INSUFFICIENT.**
  Implemented (`cfm_loss_dispersion`, `--lambda-disp`, tests green); swept λ∈{0.1,0.3,1.0}
  (job 15648042). Bar NOT met (oct2,3 still 6–9σ low). Diagnosis: the candidate matches the
  std of the Tweedie MEAN E[x1|x_t], which is structurally below the data std (total
  variance) and t-dependent → mis-specified. `log/2026-07-10-prereg-varfaithful-c-objective.md`.
- **(c') corrected objective — PRE-REGISTERED, HANDED OFF (next step).** Target the SAMPLED
  conditional variance, not the mean's: either a late-t / t-consistent dispersion penalty, or
  a Gaussian-NLL detail head (log-variance). Same frozen bar.

## To run step (c') (next)
```bash
# Implement ONE in wfm/cfm.py (see the c-objective log for both):
#  (1) late-t penalty: weight the sd penalty toward t~1, OR penalize the model's implied
#      E[Var(x1|x_t)] (residual x1 - x1_hat) vs the data conditional variance; or
#  (2) Gaussian-NLL detail head: 2nd output channel-group = log-variance, NLL detail loss.
# Then sweep its weight (MIG, ~2-4k steps, early stop), verify trained-octave var_slope
# within 1σ of real at oct 2,3,4 SIMULTANEOUSLY (deterministic ODE, no churn) + GRF null,
# THEN re-run arms A/B (full H100, <=2:59) and re-adjudicate P6/P13 (frozen bars).
```
Strong prior from (a)+(b): 90% octave-1 repair once dispersion is restored. Do NOT over-train
(var_slope peaks ~2k) and do NOT hand-tune per-octave churn.

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
