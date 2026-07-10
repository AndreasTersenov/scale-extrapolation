# 2026-07-10 — PRE-REGISTRATION: variance-faithful generator, step (c) objective

Steps (a) SDE churn and (b) checkpoint sweep each helped but neither gives per-octave
variance fidelity: churn adds ~uniform dispersion while the deficit is octave-dependent
(2k+churn4: oct2 1.9σ under, oct4 3.3σ over). A principled per-octave-faithful generator is
needed. Pre-registered before implementing (heavier build, per the reconvene).

## Diagnosis recap (the result this fixes)
L2 conditional flow matching mean-collapses conditional variance, monotonically worse with
training (var_slope: 400→0.15, 2k→0.85, 10k→0.77, 25k→0.51 at oct2; real 1.02). The
deterministic pushforward of the approximate conditional-mean velocity under-disperses.

## Candidate objective (specific, to implement)
Augment the CFM loss with a **conditional-dispersion matching regularizer**, evaluated on
the same minibatch, no extra sampling in the hot loop:
  L = L_cfm + λ · Σ_bin ( sd_pred(bin) − sd_data(bin) )^2
where, per coarse quantile bin at the training octave, sd_data = std of the target detail
and sd_pred = std of the model's ONE-STEP data estimate x1_hat = x_t + (1−t)·v(x_t,t)
(Tweedie mean; already available from the velocity). Rationale: the mean-collapse shows up
as sd_pred(bin) shrinking below sd_data(bin); penalizing that directly opposes the collapse
without a sampling loop. Sweep λ ∈ {0.1, 0.3, 1.0}; keep early stopping from (b).
Fallbacks if insufficient: (i) a two-headed model predicting (velocity, log-variance) with a
Gaussian-NLL detail head; (ii) a variance-exploding path.

## Success criterion (frozen, unchanged)
Trained-octave var_slope within 1σ of real at ALL of oct 2,3,4 SIMULTANEOUSLY, with the
deterministic ODE (no churn crutch). GRF null preserved. THEN re-adjudicate octave-1 P6/P13
with the frozen bars (arm A break z>3 & >10%; arm B repair ≥70%). The (a)+(b) run already
gives a strong prior that P6 will pass (90% repair at churn 4 once dispersion is restored).

## Result — candidate objective INSUFFICIENT (job 15648042); frozen bar NOT met
Implemented (`wfm.cfm.cfm_loss_dispersion`, tests green), swept λ∈{0.1,0.3,1.0} on
gowerstreet, 3000 steps, deterministic ODE. Trained-octave var_slope (real 1.02/0.80/0.53):

| λ | oct2 | oct3 | oct4 | oct1 armA(z) | armB | repair |
|---|---|---|---|---|---|---|
| 0.1 | 0.79 (8.6σ) | 0.64 (5.9σ) | 0.51 (0.5σ) | 0.93 (z6.6) | 0.66 | −149% |
| 0.3 | 0.82 (7.6σ) | 0.64 (6.1σ) | 0.50 (0.9σ) | 1.04 (z2.6) | 0.72 | −426% |
| 1.0 | 0.79 (9.1σ) | 0.60 (7.3σ) | 0.45 (2.3σ) | 0.93 (z7.0) | 0.64 | −151% |

- **Fidelity bar NOT met at any λ** (oct2,3 still 6–9σ low); arm B destabilized (repair < 0).
- **Diagnosis — the pre-registered candidate targets the wrong quantity.** `sd_pred` is the
  std of the Tweedie conditional MEAN E[x1|x_t]; by total variance
  Var(x1)=Var(E[x1|x_t])+E[Var(x1|x_t)], so sd_pred is STRUCTURALLY < sd_data, and it depends
  on t (→0 at t→0, →sd_data at t→1). Averaged over random t, matching sd_pred to sd_data is
  mis-specified — it cannot be satisfied and perturbs training. Not a coding bug; a
  specification bug in the candidate.

## Next: (c') CORRECTED objective — pre-registered, HANDED OFF
The dispersion penalty must target the SAMPLED conditional variance, not the mean's. Two
concrete, pre-registered options (implement one, sweep, adjudicate against the SAME frozen
bar):
1. **Late-t / t-consistent penalty:** evaluate sd_pred only near t≈1 (x1_hat faithful), or
   compare per-bin std of the residual (x1 − E[x1|x_t]) to E[Var(x1|x_t)] — i.e. penalize the
   model's implied conditional variance, computed consistently in t.
2. **Gaussian-NLL detail head (fallback (i) from the original pre-reg):** a second head
   predicts log-variance; train the detail conditional with a proper NLL so variance is
   modelled explicitly, then sample with it. Cleanest, heaviest.
The (a)+(b) evidence (90% octave-1 repair once dispersion is restored) still stands as the
prior that P6 passes once the generator is per-octave faithful. Pipeline unchanged.
