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

## IN FLIGHT (phase 1c, step 1 — Gaussian-NLL detail head)
Pre-registered `log/2026-07-11-prereg-1c-nllhead.md`.
- **15738957** `scripts/pnull_nll.slurm` (MIG) → `results/pnull_nll.npz`,
  config_hash dc676a1f01. The GRF-null condition — **still QUEUED** (post-drain
  backlog). Harvest (verdict-independent; G-1c already failed on dispersion): verify
  the runtime hash in the job log, then `source env.sh;
  python scripts/measure_generated.py --npz results/pnull_nll.npz` — both arms'
  octave-1 var_slope should be consistent with real GRF (record in RESULTS-phase1c.md).
- **15738958 HARVESTED**: **G-1c FAILED → STOP at the gate (reconvene).** Dispersion
  bar 5–8σ short at oct 2–3 both arms; kurtosis ≈5σ; student-t fallback not triggered.
  See `RESULTS-phase1c.md` (verdict + mechanism: the mean memorizes finite data and
  starves ANY variance channel — third confirmation of the collapse law; exp head
  OOD-unstable at the extrapolated octave). No further generator variants without a
  new ruling.

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
- **(c') option 1 (t-consistent late-t penalty) — DONE, INSUFFICIENT** (job 15671262, tests
  green, `--disp-t-lo 0.6`, λ∈{0.3,1,3}). Bar NOT met (oct2,3 still 6–10σ low). Twice-confirmed:
  a training-time penalty on the deterministic model can't fix the ODE-pushforward
  under-dispersion.
- **(c') option 2 — HANDED TO RECONVENE for a design decision.** Fix must change the generative
  process (stochastic, learned conditional noise). (2a) hybrid learned-σ SDE = FM augmentation
  (free periphery, recommended); (2b) pure Gaussian detail head = a FROZEN-CORE change (replaces
  FM) → reconvene, not silent redesign. Details in `log/2026-07-10-prereg-varfaithful-c-objective.md`.

## Next (after the 2a/2b reconvene decision)
```bash
# 2a (recommended, within FM): add a per-pixel log-sigma head to ConditionalUNet; train by
# Gaussian NLL on r = detail-(x_t+(1-t)v): NLL=0.5*(r^2/e^{2g}+2g); sample via sample_sde with
# per-location noise scaled by e^{g}. Sweep, verify trained-octave var_slope within 1σ (det. ODE
# + learned noise) at oct 2,3,4 + GRF null, THEN re-run arms A/B (full H100, <=2:59) -> P6/P13.
```
Prior from (a)+(b): 90% octave-1 repair once dispersion is restored. Do NOT over-train (peak
~2k) or hand-tune per-octave churn.

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
