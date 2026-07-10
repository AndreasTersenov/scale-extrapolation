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

## Result
(filled after runs)
