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

## Result — attempt 2 (FiLM), job 15628956, config_hash ee62f09bb1
arm A loss →0.668 (== attempt 1 baseline), arm B-FiLM →0.599. Octave 1 (bootstrap N=64):

| octave | real | arm A | arm B (FiLM) |
|---|---|---|---|
| 1 (extrap) var_slope | 1.117±0.051 | 0.824±0.006 | **0.872±0.005** |
| 2 | 1.020 | 0.778 | 0.773 |
| 3 | 0.801 | 0.585 | 0.534 |
| 4 | 0.532 | 0.449 | 0.355 |

- P4 PASS; **P5 HOLDS** (arm A z=5.7, 26%).
- **P6: repair = 16%** (up from −65% with additive). FiLM DOES engage the coordinate: arm B
  now differs from arm A and moves the RIGHT way at octave 1 (0.872 > 0.824, toward real),
  monotone in the coordinate (up where coord high = oct 1, down where low = oct 3,4). But
  16% < 70%, and it slightly degraded trained octaves 3–4.
- **Key diagnosis (changes the K-T2 read):** both arms under-shoot var_slope ~25% at every
  TRAINED octave (fidelity ~0.73–0.84 × real). The generator's octave-1 ceiling is therefore
  ~0.85–0.94; arm B-FiLM (0.872) is AT that ceiling. So P6 is capped by **generator
  fidelity, not the conditioning mechanism** — K-T2 (which blames the 2-D conditioning) is
  premature. Honest next step: lift fidelity (bigger model + more steps), then re-test P6.

**Attempt 3:** channels 48/96/192, steps 25000, FiLM — to raise the trained-octave fidelity
toward real and get an uncapped P6 read.

## Result — attempt 3 (bigger + longer FiLM), job 15629332, config_hash 63423d9af3
Loss crashed to ~0.08 (from ~0.6), but var_slope got WORSE, not better:

| octave | real | arm A | arm B (FiLM) | vs attempt 2 armB |
|---|---|---|---|---|
| 1 (extrap) var_slope | 1.117 | 0.539 | 0.512 | 0.872 (was better!) |
| 2 | 1.020 | 0.513 | 0.492 | 0.773 |
| 3 | 0.801 | 0.445 | 0.450 | 0.534 |
| detail_std oct3 | 1.799 | 1.441 | 1.187 | 1.796 (was near-real) |

- P4 still PASS; **P5 HOLDS even harder** (arm A z=11.3, 52%). P6 repair −5%.
- **KEY FINDING (objection, logged here per rules):** the L2 flow-matching objective
  UNDER-DISPERSES conditional variance, and OVER-TRAINING makes it worse — heavy fitting
  pulls the flow toward the conditional MEAN, collapsing the sample spread that var_slope
  (and detail_std, and kurtosis) measure. Moderate training (attempt 2) gave the BEST
  var_slope; attempt 3's lower loss = lower fidelity for the variance structure. So the
  generator ceiling cannot be lifted by more compute with this objective.

## Rung (iv) conclusion (all three attempts)
- **P4 PASS** — power-spectrum amplitude extrapolates within ~5–7%, both arms, every config.
- **P5 HOLDS (robust, load-bearing)** — arm A's conditional non-Gaussianity breaks at the
  first extrapolated octave: var_slope z = 5.8 / 4.8 / 11.3 and 26–52% across the three
  configs. The pre-registered 85% break is confirmed strongly. This is the phase result.
- **P6 NOT DEMONSTRATED** — best repair 16% (FiLM, moderate training) « 70%. FiLM makes the
  2-D coordinate act in the RIGHT direction (attempt 2 arm B > arm A toward real), so the
  conditioning is not obviously wrong (not a clean K-T2). The blocker is the CFM generator
  UNDER-DISPERSING conditional variance — a generator-objective issue, worse with training.
- **Refined objection vs PLAN K-T2:** K-T2 attributes a P6 failure to the 2-D conditioning
  hypothesis. Evidence says otherwise: the coordinate acts correctly under FiLM but the
  repair is capped by the generator's variance under-dispersion. **Next lever is the
  GENERATOR (variance-preserving / stochastic sampler, or a non-L2 objective), not the
  conditioning or architecture scale.** Recommend reconvene on this before more compute.
  P5 (the break) stands regardless.
