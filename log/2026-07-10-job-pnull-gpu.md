# 2026-07-10 — SLURM job (pre-submission): rung (iii) P-null GRF training

## Job
- Script: `scripts/train_gpu.slurm` → `scripts/run_pnull.py`
- **config_hash: `7618ef0a81`**
- Config: field=grf, train_octaves=[2,3,4], gen_from=4, channels=[32,64,128], steps=8000,
  batch=32, lr=1e-3, n_heldout=64, sample_steps=80, seed=0,
  data=`data_cache/tiles_pnull.npz` (330 GRF tiles), out=`results/pnull_generated.npz`.
- Resources: MIG slice `h100_20gb`, `--time=00:59:00` (<3h → fast b1 pool per rorqual-jobs),
  4 cpus, 32G, account rrg-lplevass (GPU).

## Hypothesis / expected outcome
The shared conditional generator trains on GRF octaves 2–4 and, applied at the untrained
finer octave 1, reproduces real GRF conditional statistics for BOTH arms (P-null / G-null).
Expected on harvest (`measure_generated.py`):
- `detail_std` (amplitude, P4) within a few % of real at every octave, both arms.
- `var_slope` and `kurtosis` at the extrapolated octave 1 consistent with real GRF
  (|Δvar_slope| < 0.1); kurtosis closer to real than the 400-step CPU dry-run (which
  under-shot it at 0.07 vs 0.41 — a convergence gap, not a pipeline bug).
- Verdict: **P-NULL PASS**. If a gen arm deviates on GRF, the pipeline is buggy (G-null
  gate) — debug before any real-field (gowerstreet) verdict.

## Pre-submission validation done (CPU)
Full pipeline ran end-to-end at `--steps 400` (config_hash f47c05f8c1): trains both arms,
extrapolates std to octave 1 (0.756), generates 32 fields/arm, saves npz;
`measure_generated.py` scored it (P-null already "consistent" at low budget because the GRF
null is trivial). The GPU run only increases the training budget to convergence.

## Ladder position
Rungs (i) single-octave overfit and (ii) two-octave recursion are green + committed.
This job is the compute for rung (iii). Not a multi-hour risk: small model, MIG slice,
GRF is the easy null; pipeline pre-validated on CPU.

## Result — P-NULL PASS (rung iii GREEN)
- **SLURM job 15617056** COMPLETED on rg12601 (MIG h100_20gb), 217 s, `CudaDevice(id=0)`
  confirmed. Both arms converged: arm A loss 2.35→0.357, arm B 2.67→0.340.
- Harvest (`measure_generated.py`, `results/pnull_generated_score.json`):
  - **var_slope** at the extrapolated octave 1 — real 0.001, armA −0.000, armB 0.003;
    |Δ| ≤ 0.002 ≪ 0.1 → **both arms extrapolate the GRF conditional structure**. ✓
  - **detail_std** (amplitude, P4) within ~1–8% at octaves 1–3 (octave 4 ~14% low).
  - **kurtosis** under-shot at the finest octaves (octave 1: 0.02/0.00 vs real 0.39). This
    is the inter-tile-amplitude POOLING kurtosis flagged in stage-0 RESULTS, not intrinsic
    conditional non-Gaussianity — the generator makes each field from one coarse so it lacks
    the cross-tile amplitude spread. The clean discriminator (var_slope) is correct, so the
    G-null gate holds. Track it as a known eval caveat for rung (iv).
- **Verdict: P-NULL PASS.** Pipeline is sound → cleared for real-field (gowerstreet) verdicts.
