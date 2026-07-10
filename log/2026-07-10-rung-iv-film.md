# 2026-07-10 — Rung (iv) attempt 2 (FiLM conditioning): honest P6 tuning

Attempt 1 (additive conditioning) confirmed **P5 (break)** but **P6 (repair) failed**: arm B
ignored the scale coordinate (arm B ≈ arm A at every trained octave) and, out-of-range at
octave 1, only perturbed. PLAN K-T2 prescribes trying the conditioning MECHANISM before
reconvening; this is that attempt. Predictions are UNCHANGED (P6 ≥ 70% repair) — only the
free-periphery "how the coordinate enters" changes (add → FiLM).

## Job (pre-submission)
- `scripts/train_gowerstreet_film.slurm` → `run_two_arms.py --cond-mode film`.
- **config_hash `ee62f09bb1`**: as attempt 1 (field=gowerstreet, octaves 2–4, gen_from 4,
  channels 32/64/128, steps 10000, batch 32, lr 1e-3, n_heldout 64, sample_steps 80,
  seed 0) but cond_mode=film, out=results/arms_film.npz, ckpt_dir=data_cache/ckpt_film.
- MIG h100_20gb, ≤59 min, rrg-lplevass. Arm A is identical to attempt 1 (cond_dim 0, same
  seed) → same P5 baseline; only arm B changes.

## Expected outcome
FiLM lets the coordinate modulate features multiplicatively — the coarse field cannot
substitute for it, so the network must learn a coordinate→conditional map. If that map is
smooth and monotonic (P9b), arm B should raise octave-1 var_slope toward real (1.12) and
**repair ≥ 70%** of arm A's error. Diagnostic to check first: does arm B now DIFFER from
arm A at trained octaves? If arm B still ≈ arm A, FiLM also failed to engage the coordinate.

## Decision rule
- repair ≥ 70% (and trained octaves not degraded > ~1σ) → **P6 HOLDS**, rung (iv) GREEN.
- 30–70% → partial; report, consider one more mechanism (concat) if cheap.
- < 30% after this honest attempt → **K-T2**: the 2-D conditioning hypothesis is
  insufficient with these mechanisms; STOP and reconvene on mechanism (not architecture
  scale). P5 (the break) still stands as the phase's confirmed result.

## Result
(job id + repair verdict filled after completion)
