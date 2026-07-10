# 2026-07-10 — PRE-REGISTRATION: variance-faithful generator, step (a) SDE sampling

Per the reconvene ruling (`log/2026-07-10-reconvene-phase1.md`): K-T2 did NOT fire; the P6
blocker is generator variance under-dispersion. Program step (a): stochastic/SDE sampling
of the EXISTING checkpoints, NO retraining. Pre-registered before running.

## Score-from-velocity identity (OT/linear path)
Path x_t = (1−t)·x0 + t·x1, x0~N(0,I). Learned velocity v(x,t)=E[x1−x0|x_t]. Then
- E[x1|x_t=x] = x + (1−t)·v,  and
- **score s(x,t) = ∇log p_t(x) = (t·v(x,t) − x)/(1−t).**
Derivation: v=(E[x1|x]−x)/(1−t); Tweedie on the Gaussian kernel N(x; t·x1, (1−t)^2 I)
gives s = (t·E[x1|x]−x)/(1−t)^2 = (t·v − x)/(1−t). Limit checks: t→0 ⇒ s=−x (score of
N(0,I)). ✓

## Marginal-preserving churn SDE (generative, t: 0→1)
dx = [v + ε(t)·s] dt + sqrt(2 ε(t)) dW,  any ε(t)≥0 preserves the marginals p_t
(Fokker–Planck: the score+diffusion terms cancel). Choose ε(t)=ε0·(1−t) so both the
drift correction ε·s = ε0·(t·v − x) and the noise sqrt(2 ε0 (1−t)) stay finite at t→1.
Euler–Maruyama step:
    x += [v + ε0·(t·v − x)]·dt + sqrt(2·ε0·(1−t)·dt)·z,  z~N(0,I).
ε0=0 recovers the deterministic probability-flow ODE (current sampler).

## Verification (GRF control) — before trusting the sampler
1. **Identity, analytic:** on a Gaussian x1 (the GRF case — detail coeffs are Gaussian),
   compute the exact v* and s* and check s* == (t·v* − x)/(1−t) to machine precision, and
   against a finite-difference ∇log p_t. Unit test `tests_wfm/test_score_identity.py`.
2. **Null preservation:** SDE-sampling a GRF-trained generator must keep var_slope ≈ 0
   (churn must NOT inject spurious dispersion) — else any "recovery" on gowerstreet is an
   artifact. Run at the same ε0 used for gowerstreet.

## Success criterion (frozen, from the reconvene)
For the chosen ε0: **trained-octave (2,3,4) var_slope within 1σ of real** on gowerstreet,
AND GRF null preserved. Only then is the octave-1 (P6) repair measurement meaningful.

## Plan
Sweep ε0 ∈ {0, 0.25, 0.5, 1, 2, 4} on the ckpt_film gowerstreet checkpoints (generation
only, CPU); measure trained-octave var_slope vs real; pick the ε0 (if any) meeting the bar;
confirm GRF null at that ε0; then re-adjudicate P6/P13 with the frozen bars (arm A z>3&>10%,
arm B repair ≥70%). If NO ε0 meets the bar → step (a) insufficient → proceed to step (b)
(checkpoint/dispersion sweep). The frozen P6/P13 bars are unchanged.

## Result — step (a) INSUFFICIENT (routes to step b)
Identity verified analytically (GRF/Gaussian control): `tests_wfm/test_score_identity.py`
green (matches exact score + finite difference to 1e-10 / 1e-4). SDE sampler implemented
(`wfm.cfm.sample_sde`, churn wired through `generate_recursive`).

Churn sweep on the ckpt_film gowerstreet checkpoints (trained-octave var_slope; real
oct2/3/4 = 0.995 / 0.779 / 0.514):

| churn | arm A oct2 / oct3 / oct4 | Δ to real |
|---|---|---|
| 0 (ODE) | 0.77 / 0.57 / 0.46 | 15σ / 13σ / 3σ |
| 1 | 0.80 / 0.58 / 0.46 | 12σ / 12σ / 3σ |
| 2 | 0.82 / 0.59 / 0.47 | 11σ / 11σ / 2σ |
| 4 | 0.84 / 0.60 / 0.47 | 10σ / 11σ / 2σ |
| 8 | 0.86 / 0.60 / 0.47 | 9σ / 10σ / 2σ |

- Churn moves var_slope the right way but **SATURATES ~0.86 / 0.60**, far short of real
  (0.995 / 0.779) — 9–10σ at octaves 2,3. **The success bar (within 1σ) is NOT met** →
  step (a) alone is insufficient.
- Note: arm B octave-1 var_slope crosses real (0.87→1.13 over churn 0→8), but that is
  generic dispersion injection, not faithful trained-octave structure — which is exactly
  why the bar is placed on TRAINED octaves. The GRF-generator null-preservation check
  (pre-registered) is not gating here since (a) fails the primary bar regardless; it would
  only matter to certify a PASS.
- Consistent with the attempt-2 (10k, var_slope 0.77) vs attempt-3 (25k, 0.51) evidence,
  the ckpt_film generator is already mean-collapsed; churn cannot fully undo it. → **step
  (b): checkpoint/early-stopping sweep, selecting by trained-octave var_slope, not loss.**
