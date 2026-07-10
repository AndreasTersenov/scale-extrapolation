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

---

## PRE-REGISTRATION — c'-option-1 (concrete form), before submitting
Reconvene endorsed (c'); implement cheapest-first = option 1 (t-consistent penalty).

**Diagnosis of why the original (c) failed:** it matched std(x1_hat) to std(detail) averaging
over t~U(0,1); but x1_hat=E[x1|x_t] is the conditional MEAN, whose variance
Var(E[x1|x_t]) < Var(x1) for all t<1 (total variance), and →0 as t→0. So the target was
mis-specified, worst at small t.

**Option-1 fix (t-consistent):** evaluate the dispersion penalty only in a LATE-t window,
where x_t already carries most of x1 so x1_hat is a faithful estimate and Var(x1_hat)≈Var(x1)
for a good model, while a mean-collapsed velocity still shows a spread deficit. Concretely,
in `cfm_loss_dispersion(..., t_lo)`: keep the CFM loss at t~U(0,1); add a SECOND forward at
t~U(t_lo, 1) with fresh noise, form x1_hat = x_t + (1−t)·v, and penalize per coarse quantile
bin  mean_bin ( std(x1_hat) − std(detail) )². **t_lo = 0.6, n_bins = 8.**

**Sweep:** λ ∈ {0.3, 1.0, 3.0} (late-t penalty is weaker; allow larger λ), gowerstreet FiLM,
early stop 3000 steps (~ the 2–4k dispersion window), deterministic ODE.

**Frozen success bar (unchanged):** trained-octave var_slope within 1σ of real at oct 2,3,4
SIMULTANEOUSLY, deterministic ODE, no churn; GRF null preserved.

**Escalation trigger (pre-authorized):** if NO λ meets the bar, escalate to option 2
(Gaussian-NLL detail head: a second output group predicts log-variance; train the detail
conditional with NLL and sample stochastically with the learned variance) — pre-register its
concrete form before submitting.

**On success:** re-run arms A/B at the winning λ (full H100, ≤2:59) and re-adjudicate P6/P13
with the unchanged frozen bars.

### Result — c'-option-1 INSUFFICIENT (job 15671262); frozen bar NOT met
Swept λ∈{0.3,1.0,3.0}, t_lo=0.6, 3000 steps, deterministic ODE (real oct2/3/4 = 1.02/0.80/0.53):

| λ | oct2 | oct3 | oct4 |
|---|---|---|---|
| 0.3 | 0.80 (8.4σ) | 0.64 (5.9σ) | 0.48 (1.4σ) |
| 1.0 | 0.79 (9.1σ) | 0.63 (6.5σ) | 0.49 (1.2σ) |
| 3.0 | 0.76 (10.1σ) | 0.60 (7.6σ) | 0.48 (1.5σ) |

Bar NOT met at any λ (oct2,3 still 6–10σ low; no better than random-t (c)); arm B
destabilized (repair < 0). **Twice-confirmed conclusion:** a training-time penalty on the
deterministic model — whether it shares the CFM t (c) or uses a late-t window (c'-1) — cannot
fix the generated under-dispersion, because the deterministic-ODE PUSHFORWARD variance is what
under-shoots and it is not a function of the penalized training-data quantities. The fix must
change the GENERATIVE PROCESS (stochastic sampling with a learned, conditional noise scale).

### Escalation → option 2 (Gaussian-NLL / learned-variance) — needs a design decision
Two concrete forms, with a FROZEN-CORE caveat:
- **(2a) Hybrid, WITHIN flow matching (recommended):** keep the velocity head; ADD a per-pixel
  log-σ head g(x_t,t,coarse,cond) trained by Gaussian NLL on the residual
  r = detail − (x_t+(1−t)v):  NLL = ½[ r²/e^{2g} + 2g ]  (so e^{g} ≈ conditional std). Sample
  with the score-SDE (`sample_sde`) but scale the injected noise per-location by e^{g}/mean(e^{g})
  — a LEARNED, coarse/scale-dependent churn (fixes the octave-dependent deficit that global
  churn could not). This is an FM augmentation → free periphery.
- **(2b) Pure Gaussian detail head:** model detail|coarse ~ N(μ,σ²) directly, NLL, sample μ+σε.
  Captures var_slope exactly and would test P6 cleanly, BUT it REPLACES flow matching for the
  detail — a change to the FROZEN-CORE "conditional flow-matching generator". **This is a
  frozen-core redesign → per the hard rules it must go to reconvene, not silent redesign.**

**Status: HANDED to reconvene for the design decision** (2a vs 2b), then implement/pre-register/
submit. Everything else (coords, both arms, scoring, the score-SDE sampler, the NLL machinery
sketched above) is in place. Prior unchanged: (a)+(b) showed 90% octave-1 repair once dispersion
is restored, so P6 is expected to pass once the generator disperses natively.
