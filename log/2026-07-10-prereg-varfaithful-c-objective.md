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

## Status: HANDED OFF (not run this session)
Heavier build; this session executed the ordered program (a)+(b) with pre-registration and
the fidelity diagnosis. (c) is the next step — see JOBS.md. Implement the regularizer in
`wfm/cfm.py` (new `cfm_loss_dispersion`), add a `--lambda-disp` arm to `run_two_arms.py`,
train (MIG), sweep λ, then re-adjudicate with `measure_generated.py` (pipeline unchanged).
