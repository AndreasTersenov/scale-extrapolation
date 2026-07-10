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
- **Next: (a)+(b) combined** — churn on the 2k peak checkpoint (step (a) added ~+0.09 to
  oct2). Expected ~0.94 at oct2 (~1.5σ). Test directly: train arm A+B (FiLM) to 2000, save
  checkpoints, churn-sweep, measure trained octaves (success bar) AND octave-1 (P6).
