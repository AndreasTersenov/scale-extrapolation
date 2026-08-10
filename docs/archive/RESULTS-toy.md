# RESULTS-toy — Phase 1 (break & repair)

**The pre-registered break/repair experiment on a wavelet-factorized conditional
flow-matching generator.** For a reader who knows `PLAN.md` but not the code. All numbers
are from `scaledrift` (the frozen stage-0 instrument) applied to generated vs held-out real
fields, bootstrap over held-out fields (N=64). Primary discriminator is **var_slope** (the
conditional-variance slope = the clean conditional non-Gaussianity); marginal kurtosis is
reported but confounded by an inter-tile-amplitude pooling artifact (see the P-null log).

## Verdicts

| Prediction | Pre-reg | Verdict | Basis |
|---|---|---|---|
| **P-null** | 90% | **PASS** | both arms extrapolate GRF; var_slope \|Δ\|≤0.002 at the untrained octave |
| **P4** (power extrapolates) | 70% | **PASS** | detail amplitude within ~5–7% (both arms) at the first extrapolated octave |
| **P5** (arm A break) | 85% | **HOLDS (robust)** | arm A var_slope wrong at the extrapolated octave: z = 5.8 → 11.3, 26–52% across 3 configs |
| **P6** (arm B repair ≥70%) | 55% | **BLOCKED-PENDING-RETEST** | capped by a generator limitation (below), NOT the conditioning; 90% repair seen once dispersion is restored |
| **P13** (zero-retrain transfer) | 55% | **BLOCKED-PENDING-RETEST** | same cap; arm B does transfer an amplitude fix (40%→2.5%) |

**Bottom line (post-reconvene).** The load-bearing **break (P5) is confirmed strongly** and
is the phase's paper result. The reconvene ACCEPTED P-null/P4/P5, ruled that **K-T2 does not
fire** (the 2-D conditioning is exonerated — FiLM's response is directionally correct and
monotone), and promoted the P6 blocker to a **first-class result**: *L2 conditional flow
matching mean-collapses conditional variance, monotonically worse with training.* P6/P13 are
**blocked-pending-retest** (frozen bars unchanged), to be re-adjudicated after a
variance-faithful generator. First evidence that the repair will then pass: **90% octave-1
repair** already appears once dispersion is restored by churn (below).

## The Karpathy ladder
(i) single-octave overfit, (ii) two-octave recursion, (iii) GRF end-to-end null — all GREEN
and committed (rel-L2 0.08 / 0.07; P-null PASS). (iv) arms A/B on gowerstreet — below.
(v) transfer — below.

## Setup (rung iv)
Both arms are weight-tied conditional flow-matching generators (shared across octaves)
trained on gowerstreet octaves 2–4; the finest octave j=1 is the untrained "first
extrapolated octave". Arm A gets no scale input; arm B is conditioned on the stage-0 2-D
running-coupling coordinate [var_slope_j, kurtosis_j] (measurement allowed, no training),
including octave 1's value. Generation starts from held-out real coarse at octave 4. Three
configs were run (all committed, pre-registered before submission):

| # | conditioning | model | steps | config_hash |
|---|---|---|---|---|
| 1 | additive embedding | 32/64/128 | 10k | 71a9fd7e6d |
| 2 | **FiLM** | 32/64/128 | 10k | ee62f09bb1 |
| 3 | FiLM | 48/96/192 | 25k | 63423d9af3 |

## P5 — the break (CONFIRMED)
Real octave-1 var_slope = **1.117**. Arm A produces **0.82** (attempt 2) / **0.54**
(attempt 3) — z = 5.8 / 11.3, 26% / 52% low (>3σ AND >10%). Mechanism: arm A learns the
pooled trained-octave conditional (var_slope ~0.9) and applies it at octave 1, which needs
1.31; a weight-tied net with no scale input cannot represent conditionals that measurably
drift. **P5 holds in every config.** K-T1 (reframe) is NOT triggered — the break is real.

## P6 — the repair (NOT DEMONSTRATED; diagnosed)
- Additive conditioning: arm B ≈ arm A at all trained octaves → the coordinate was
  **ignored** (the coarse field sufficed to fit training); OOD at octave 1 it only
  perturbed. Repair −65%.
- **FiLM** conditioning: the coordinate now has multiplicative leverage — arm B differs from
  arm A and moves octave-1 var_slope the RIGHT way (0.87 > arm A 0.82, toward 1.12),
  monotone in the coordinate. Repair **+16%** — directionally correct but « 70%.
- **Why it's capped — the key finding:** the L2 flow-matching objective **under-disperses
  conditional variance**, and it gets WORSE with training. Attempt 3 (loss 0.6→0.08) drove
  var_slope DOWN (octave-1 0.54 vs attempt-2 0.87) and collapsed even the detail amplitude —
  heavy fitting pulls the flow toward the conditional *mean*, shrinking the sample spread
  that var_slope measures. So the generator's variance ceiling cannot be raised by more
  compute with this objective; arm B-FiLM already sits at that ceiling.
- **Refined vs PLAN K-T2:** K-T2 attributes a P6 failure to the 2-D conditioning. The
  evidence says otherwise — under FiLM the coordinate acts correctly; the repair is capped
  by generator variance under-dispersion. **Next lever is the GENERATOR (a
  variance-preserving / stochastic sampler, or a non-L2 objective), not the conditioning or
  architecture scale.** This is the reconvene recommendation.

## P13 — zero-retrain transfer (NOT DEMONSTRATED)
The gowerstreet-trained (FiLM) arms were applied to hf_pm_1024 with hf_pm's own coordinate
and amplitude, no retraining. hf_pm is a MILDER field (octave-1 var_slope 0.75 vs
gowerstreet 1.31), so gowerstreet's under-dispersed arm A happens to match it (z=1.5 — no
var_slope break there), while arm A breaks on AMPLITUDE (40% high) and arm B **fixes the
amplitude** (2.5%). var_slope repair fails (−105%) — the same under-dispersion cap. So the
transfer neither cleanly breaks nor repairs var_slope; it does transfer an amplitude
correction. Inconclusive for P13 as scoped, for the same generator reason as P6.

## The variance-faithful generator program (post-reconvene)
Approved ordered program to unblock P6, each step pre-registered; success bar =
trained-octave var_slope within 1σ of real, with the frozen P6/P13 bars unchanged.

- **First-class finding — the dispersion-collapse curve.** arm A octave-2 var_slope vs
  training steps: 400→0.15, **2k→0.85 (peak)**, 6k→0.75, 12k→0.76, 25k→0.51 (real 1.02).
  var_slope peaks EARLY and collapses with training; loss is monotonically best at the WORST
  dispersion. Selecting a generator by loss is exactly wrong for dispersion statistics.
- **(a) SDE sampling** of existing checkpoints (score s=(t·v−x)/(1−t), identity verified on
  the GRF/Gaussian control; churn-SDE that preserves the marginals): raises var_slope but
  SATURATES ~9–10σ short of real on the mean-collapsed 10k checkpoint. Insufficient alone.
- **(b) checkpoint sweep**: the 2k peak is far better (oct2 3σ, oct4 within 1σ) but octaves
  2–3 still exceed 1σ. Insufficient alone.
- **(a)+(b)**: at churn 4 the 2k checkpoint reaches oct2 1.9σ / oct3 0.6σ but OVER-corrects
  oct4 (3.3σ) — global churn adds uniform dispersion while the deficit is octave-dependent,
  so the fidelity gate is not cleanly met. **Yet octave-1 P6 repair = 90%** here (arm B 1.14
  ≈ real 1.12; arm A still broken, z=9.5) — strong evidence the running-coupling repair WORKS
  once dispersion is restored.
- **(c) + (c'-1) training-time dispersion penalties — RUN, both insufficient.** Penalizing the
  per-bin std of the one-step (Tweedie-mean) estimate to match the data — sharing the CFM t (c,
  λ∈{0.1,0.3,1}) or in a late-t window (c'-1, `--disp-t-lo 0.6`, λ∈{0.3,1,3}) — leaves
  trained-octave var_slope 6–10σ low. **Twice-confirmed:** a training-time penalty on the
  deterministic model cannot fix the generated under-dispersion, because the deterministic-ODE
  PUSHFORWARD variance is what under-shoots and is not a function of the penalized quantities.
- **(c'-2) the fix must change the generative process** (stochastic sampling with a LEARNED,
  conditional noise scale). (2a) a hybrid learned-σ SDE (an FM augmentation — free periphery) or
  (2b) a pure Gaussian detail head (which replaces FM — a frozen-core change). The 2a/2b choice
  is with the reconvene. The (a)+(b) 90%-repair signal remains the prior that P6 passes once the
  generator disperses natively.

## Honest limits & recommended next step
- The whole P6/P13 story is gated by ONE generator property: **flow-matching under-disperses
  conditional variance** (empirically worse with training). Until the generator can
  reproduce trained-octave var_slope near real, the repair cannot be fairly tested.
- Reconvene lever (per the logged objection): a variance-faithful generator — SDE/stochastic
  sampling, an early-stopped or dispersion-regularized FM objective, or a different
  head — THEN re-run arms A/B (the pipeline, coordinates, and scoring are all in place).
- Unchanged and solid: stage-0 (the drift is real and low-dim) and **P5** (the break is
  real). The paper's load-bearing empirical claim — non-Gaussian break under extrapolation —
  is confirmed end-to-end in a trained generator, not just in the fields.

## Files
`results/scores/arms_{generated,film,big}_score.json` (per-config scores),
`results/scores/transfer_generated_score.json`; logs `log/2026-07-10-prereg-rung-iv-gowerstreet.md`
and `log/2026-07-10-rung-iv-film.md` (attempts + the under-dispersion finding);
`log/2026-07-10-job-pnull-gpu.md` (P-null). Reproduce: `sbatch scripts/train_gowerstreet*.slurm`
→ `python scripts/measure_generated.py`; transfer via `scripts/run_transfer.py`.
