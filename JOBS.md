# JOBS — in-flight SLURM jobs & harvest instructions

Last updated 2026-07-10. Phase-1 toy (break & repair). Envs: `source env.sh` (scaledrift,
CPU, has pywt) for measurement; `~/wl-challenge-env/bin/python` (JAX) for training/
generation. GPU via SLURM only.

## Ladder status
- Rung (i) single-octave overfit — **GREEN**, committed (`7045cea`).
- Rung (ii) two-octave recursion — **GREEN**, committed (`9343214`).
- Rung (iii) GRF end-to-end null (P-null) — **GREEN** (job 15617056 harvested; P-NULL PASS,
  both arms extrapolate GRF; verdict in `log/2026-07-10-job-pnull-gpu.md`).
- Rung (iv) arms A/B on gowerstreet (P4/P5/P6) — **NEXT** (cleared; see below).
- Rung (v) transfer P13 + `RESULTS-toy.md` — not started.

## IN FLIGHT
**None.** Job 15617056 (rung iii P-null) completed and was harvested — P-NULL PASS.
GPU env confirmed working: `scripts/train_gpu.slurm` on a MIG `h100_20gb` slice sees
`CudaDevice(id=0)` and finishes an 8000-step GRF run in ~217 s.

## NEXT (after rung iii green)
Rung (iv): reuse `run_pnull.py` with `--field gowerstreet` (add gowerstreet's real stage-0
2-D running-coupling coordinate as arm B's `cond_by_octave`, replacing the placeholder
`scale_coord`), then `measure_generated.py` → P4/P5/P6. Arm A should BREAK at the
extrapolated octave (P5: var_slope/kurtosis far from real); arm B should repair ≥70% (P6).
This is the load-bearing run and wants a full H100 (`--gpus-per-node=1`, ≤2:59).
