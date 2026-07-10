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

## RECONVENE RECOMMENDATION (the one open scientific issue)
P5 (the non-Gaussian break under extrapolation) is confirmed. P6/P13 (the repair) are
blocked by a diagnosed generator property: **the L2 flow-matching objective under-disperses
conditional variance, and it worsens with training** (attempt 3, loss 0.6→0.08, made
var_slope WORSE and collapsed the detail amplitude). The FiLM scale-coordinate acts in the
RIGHT direction (repair −65% → +16%), so the conditioning is not the blocker. Next lever is
the **generator** (variance-preserving / stochastic sampler, or a non-L2 / dispersion-aware
objective, or early stopping), NOT the conditioning or architecture scale — see the logged
objection in `log/2026-07-10-rung-iv-film.md`. The whole pipeline (coordinates, both arms,
scoring, transfer) is in place to re-run once the generator is fixed.

## To resume the P6 attempt after a generator fix
```bash
# edit wfm/cfm.py sampler (add SDE/Langevin noise) or the objective, then:
sbatch scripts/train_gowerstreet_film.slurm     # arm B FiLM, moderate steps (best var_slope so far)
source env.sh && python scripts/measure_generated.py --npz results/arms_film.npz
```
Note: moderate training (~10k steps) gave BETTER var_slope than 25k — do NOT over-train.

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
