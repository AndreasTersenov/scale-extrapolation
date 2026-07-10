# 2026-07-10 — PRE-REGISTRATION: variance-faithful generator, step (b) checkpoint sweep

Step (a) (SDE churn on the 10k checkpoint) was insufficient: trained-octave var_slope
saturated ~9–10σ short of real. Evidence that var_slope is NON-monotone in training
(400 steps → 0.15; 10k → 0.77; 25k → 0.51 at octave 2) says the generator MEAN-COLLAPSES
with over-fitting, so there is an intermediate "dispersion-optimal" checkpoint. Step (b)
finds it. Pre-registered before running.

## Method
Train arm A + arm B (FiLM) on gowerstreet octaves 2–4 ONCE, saving checkpoints at steps
{1k, 2k, 3k, 4k, 6k, 8k, 12k}. For each checkpoint, generate from held-out real coarse and
measure TRAINED-octave (2,3,4) var_slope (deterministic ODE; optionally + a modest churn).
Select the checkpoint by trained-octave var_slope closeness to real — NOT by loss.

## Success criterion (frozen, unchanged from the reconvene)
Trained-octave var_slope within 1σ of real (gowerstreet oct2/3/4 = 0.995 / 0.779 / 0.514)
for the selected checkpoint (± churn ≤ 8). Only then re-adjudicate octave-1 P6 with the
frozen bars (arm A break z>3 & >10%; arm B repair ≥70%).

## Decision
- If a checkpoint (± churn) meets the bar → adopt it; re-adjudicate P6/P13.
- If the PEAK var_slope over all checkpoints is still < real − 1σ at octaves 2,3 → step (b)
  insufficient too: the generator cannot represent the conditional dispersion at any
  training horizon with this architecture/objective → proceed to step (c)
  (dispersion-regularized / non-L2 objective), pre-registered separately.

## Notes
Report the var_slope-vs-steps curve (the "dispersion collapse" curve) — it is itself a
first-class result (conditional-FM mean-collapse). Budget still well inside 15 H100-days.

## Result — the dispersion-collapse curve (job 15640510)
arm A trained-octave var_slope vs training steps (real oct2/3/4 = 1.02 / 0.801 / 0.532):

| step | loss | oct2 | oct3 | oct4 |
|---|---|---|---|---|
| 2000 | 1.30 | **0.847 (3σ)** | **0.684 (2σ)** | **0.512 (0σ)** |
| 4000 | 1.11 | 0.751 (5σ) | 0.642 (3σ) | 0.490 (1σ) |
| 6000 | 0.32 | 0.751 (5σ) | 0.603 (4σ) | 0.460 (1σ) |
| 8000 | 0.34 | 0.748 (5σ) | 0.587 (4σ) | 0.449 (2σ) |
| 12000 | 0.15 | 0.757 (5σ) | 0.557 (5σ) | 0.433 (2σ) |

- **First-class result: var_slope PEAKS early (~2k steps) and COLLAPSES monotonically with
  training** — the exact inverse of loss (best at 12k). Selecting by loss picks the WORST
  dispersion. This is the "conditional-FM mean-collapse" curve for the paper.
- Peak (2k): octave 4 within 1σ, octave 3 at 2σ, octave 2 at 3σ (0.847 vs 1.02) — vastly
  better than the 10k ckpt_film (15σ at oct2), but octaves 2–3 still exceed the 1σ bar.
## Result — (a)+(b) combined (job 15642098), 2k peak checkpoint + churn
arm A trained-octave var_slope (σ to real) and octave-1 P5/P6:

| churn | oct2 | oct3 | oct4 | oct1 armA (z) | oct1 armB | P6 repair |
|---|---|---|---|---|---|---|
| 0 | 0.87 (5.8σ) | 0.71 (3.2σ) | 0.55 (0.5σ) | 0.86 (z9.9) | 0.88 | 10% |
| 4 | 0.97 (1.9σ) | 0.82 (0.6σ) | 0.65 (3.3σ) | 0.87 (z9.5) | **1.14** | **90%** |
| 8 | 1.10 (2.7σ) | 0.89 (3.1σ) | 0.70 (4.5σ) | 0.97 (z5.4) | 1.33 | −46% |

- **Step (b) insufficient per the pre-registered rule** (peak, churn 0: oct2 5.8σ, oct3
  3.2σ — both < real−1σ). And (a)+(b): **global churn cannot make fidelity uniform** — it
  adds ~constant dispersion, but the deficit is octave-dependent, so churn 4 fixes oct2/oct3
  while OVER-correcting oct4 (3.3σ high). No single churn puts all trained octaves within
  1σ. The frozen fidelity gate is NOT cleanly met → proceed to step (c).
- **Strong positive signal for P6:** at churn 4, arm B octave-1 var_slope = 1.14 ≈ real 1.12
  (**repair 90%**, ≥70% bar) while arm A stays broken (z=9.5). So the running-coupling repair
  WORKS once dispersion is restored — it just rides on global churn rather than a natively
  faithful generator. This is what step (c) must deliver cleanly.
- Do NOT tune per-octave churn to hit the target — that fits the answer. Step (c) is a
  principled per-octave-faithful generator (dispersion-regularized objective).
