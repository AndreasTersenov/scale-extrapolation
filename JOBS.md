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

### Job 15627183 — rung (iv) gowerstreet break & repair  (config_hash `71a9fd7e6d`)
- `sbatch scripts/train_gowerstreet.slurm` (MIG h100_20gb). Pre-registration:
  `log/2026-07-10-prereg-rung-iv-gowerstreet.md`. Output `results/arms_generated.npz`
  (gen_A, gen_B, real) + checkpoints `data_cache/ckpt/arm{A,B}_gowerstreet.pkl`; job log
  `results/arms_15627183.log`.

**Harvest (when results/arms_generated.npz exists)**
```bash
source env.sh
python scripts/measure_generated.py --npz results/arms_generated.npz   # P4/P5/P6
```
- Convergence sanity FIRST: arm A/B var_slope at TRAINED octaves 2,3,4 should approach real
  (stage-0 ~0.66–1.15). If they sit near ~0.15 (as the 400-step CPU dry-run did), the run is
  under-trained — raise `--steps` and rerun (a convergence fix, predictions unchanged).
- Then read octave-1 verdicts: P5 BREAK (arm A z>3 & >10%), P6 repair ≥70%, P4 amplitude
  <10%. Update the prereg log Result, commit `arms_generated_score.json`, then rung (v).

### Rung (v) transfer (P13) — READY, run AFTER rung iv green
Needs the arm-B checkpoint from job 15627183. Then (CPU ok, no training):
```bash
JAX_PLATFORMS=cpu ~/wl-challenge-env/bin/python scripts/run_transfer.py   # gowerstreet ckpts -> hf_pm
source env.sh && python scripts/measure_generated.py --npz results/transfer_generated.npz
```
P13: arm B repairs >40% of arm A's octave-1 var_slope error on hf_pm (hf_pm tiles + coords
already prepared in data_cache).

## DONE
Job 15617056 (rung iii P-null) — P-NULL PASS. GPU env confirmed: MIG `h100_20gb` sees
`CudaDevice(id=0)`, ~217 s for an 8000-step run. (Benign ptxas 12.6.77 clamping warning.)

## NEXT (after rung iii green)
Rung (iv): reuse `run_pnull.py` with `--field gowerstreet` (add gowerstreet's real stage-0
2-D running-coupling coordinate as arm B's `cond_by_octave`, replacing the placeholder
`scale_coord`), then `measure_generated.py` → P4/P5/P6. Arm A should BREAK at the
extrapolated octave (P5: var_slope/kurtosis far from real); arm B should repair ≥70% (P6).
This is the load-bearing run and wants a full H100 (`--gpus-per-node=1`, ≤2:59).
