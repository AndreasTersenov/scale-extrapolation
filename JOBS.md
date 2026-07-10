# JOBS — in-flight SLURM jobs & harvest instructions

Last updated 2026-07-10. Phase-1 toy (break & repair). Envs: `source env.sh` (scaledrift,
CPU, has pywt) for measurement; `~/wl-challenge-env/bin/python` (JAX) for training/
generation. GPU via SLURM only.

## Ladder status
- Rung (i) single-octave overfit — **GREEN**, committed (`7045cea`).
- Rung (ii) two-octave recursion — **GREEN**, committed (`9343214`).
- Rung (iii) GRF end-to-end null (P-null) — **IN FLIGHT** (job below). Scaffolding +
  full pipeline committed and CPU-validated (`7828525`, and the P-null runner commit).
- Rung (iv) arms A/B on gowerstreet (P4/P5/P6) — not started (needs rung iii green).
- Rung (v) transfer P13 + `RESULTS-toy.md` — not started.

## IN FLIGHT

### Job 15617056 — rung (iii) P-null GRF training  (config_hash `7618ef0a81`)
- Submitted 2026-07-10 via `sbatch scripts/train_gpu.slurm` (MIG h100_20gb, ≤59 min).
- Pre-submission log + expected outcome: `log/2026-07-10-job-pnull-gpu.md`.
- Output: `results/pnull_generated.npz` (gen_A, gen_B, real), job log
  `results/train_15617056.log`.

**Check status**
```bash
squeue -j 15617056 --format="%.12i %.9T %.10M %R"
sacct -j 15617056 --format=State,Elapsed,ExitCode -n
tail -20 results/train_15617056.log     # first line prints `jax devices:` — must show a GPU
```

**Harvest (when COMPLETED and results/pnull_generated.npz exists)**
```bash
source env.sh                                   # scaledrift env (pywt)
python scripts/measure_generated.py --npz results/pnull_generated.npz
```
Then read the printed table + `P-NULL:` line and `results/pnull_generated_score.json`.
- **PASS** (both arms' var_slope at octave 1 consistent with real GRF, |Δ|<0.1; detail_std
  within a few %; kurtosis near real): rung (iii) is GREEN. Update
  `log/2026-07-10-job-pnull-gpu.md` Result, mark task, commit `pnull_generated_score.json`,
  proceed to rung (iv).
- **DEVIATES**: G-null gate fails → the generator/pipeline is buggy (NOT physics). Debug
  before any gowerstreet verdict. Likely suspects: too-few steps (raise `--steps`), the
  std extrapolation, or the sampler step count.

**Failure modes to check first**
- `jax devices:` shows only CPU → the MIG/cuda module didn't expose the GPU; add/adjust
  `module load cuda/…` in `scripts/train_gpu.slurm` and resubmit (CPU would likely TIMEOUT).
- Job TIMEOUT at 59 min → it ran on CPU, or 8000 steps is too slow; confirm GPU, or lower
  steps / raise `--time` (keep ≤ 2:59 to stay on the fast b1 pool).

## NEXT (after rung iii green)
Rung (iv): reuse `run_pnull.py` with `--field gowerstreet` (add gowerstreet's real stage-0
2-D running-coupling coordinate as arm B's `cond_by_octave`, replacing the placeholder
`scale_coord`), then `measure_generated.py` → P4/P5/P6. Arm A should BREAK at the
extrapolated octave (P5: var_slope/kurtosis far from real); arm B should repair ≥70% (P6).
This is the load-bearing run and wants a full H100 (`--gpus-per-node=1`, ≤2:59).
