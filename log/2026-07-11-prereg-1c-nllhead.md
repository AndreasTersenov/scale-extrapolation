# 2026-07-11 — PRE-REGISTRATION: Phase-1c Step 1, Gaussian-NLL detail head (option 2)

Per PLAN.md (frozen core, 2026-07-11) and the reconvene ruling + addendum 1
(`log/2026-07-11-reconvene-cprime2.md`). Written and committed BEFORE job submission.

## Hypothesis

The under-dispersion is a property of the deterministic-ODE pushforward, not of the
model's information content (twice-confirmed: training penalties don't fix it; churn
does but is uniform). Modeling the conditional variance EXPLICITLY — a per-coefficient
log-sigma head trained with a proper Gaussian NLL, sampled with its variance — makes the
generator variance-faithful at trained octaves.

## Design (free periphery, fixed here before training)

- **Model:** `ConditionalUNet(variance_head=True)` — a second zeros-initialised 1×1 conv
  head `g` (log-sigma, 3 detail channels) off the shared trunk; output `[v, g]`.
  Zeros-init ⇒ sigma starts at 1 = the standardized-detail scale. BOTH arms get the head
  (arm symmetry, binding condition 1).
- **Loss** (`wfm.cfm.cfm_loss_nll`): standard CFM L2 on `v` at random (x0, t), PLUS the
  Gaussian NLL of the full conditional at the flow's center anchor (x_t=0, t=0):
  `NLL = mean(0.5·((detail−mu)²·e^{−2g} + 2g))`, `mu = v(0,0|coarse,cond)`,
  `g = g(0,0|coarse,cond)` clipped to [−5, 3]. At the CFM optimum
  `v(x,0) = E[x1] − x`, so mu is the conditional mean and `e^{2g}` regresses
  `Var(detail|coarse,cond)` — exactly the object var_slope measures. The NLL trains the
  mean jointly (validation gate showed CFM-only leaves `v(0,0)` shrunk ~15% at extreme
  coarse via t>0 contamination). Loss weight 1.0, no schedule.
- **SAMPLING PROCEDURE (binding condition 2, pre-registered):**
  `detail = mu + e^{g}·z`, `z~N(0,I)`, with `(mu, g)` from one forward at
  (x_t=0, t=0 | coarse, cond) — the deterministic mean-path endpoint plus explicit
  learned variance. NO churn, NO stochastic ODE, no double-counting (the mean path is
  deterministic given coarse, so ALL conditional variance is carried by `e^{2g}`).
  Recursion coarse-to-fine unchanged; per-octave detail_std unchanged
  (`wfm.cfm.sample_nll`, `generate_recursive(nll=True)`).
- **Validation gates (green before this prereg, commit de6c032):**
  `tests_wfm/test_nll_head.py` — conditional mean+sigma recovery on a known law,
  per-coarse-bin sampler variance faithfulness (≤12%), flat-sigma null (no spurious
  slope on constant-variance data), determinism/API. Full tests_wfm + tests suites green.

## Jobs (logged pre-submission; cluster is in maintenance drain — jobs will queue)

1. **GRF null** — `scripts/pnull_nll.slurm` (MIG h100_20gb, ≤0:59):
   `run_pnull.py --nll-head`, steps 8000, channels 32/64/128, batch 32, n-heldout 64,
   out `results/pnull_nll.npz`. **config_hash dc676a1f01**.
2. **gowerstreet arms A/B** — `scripts/arms_nll.slurm` (full H100, ≤0:59):
   `run_two_arms.py --nll-head --cond-mode film`, steps 10000, channels 32/64/128,
   batch 32, n-heldout 64, ckpt-steps 2000 4000 6000 8000 (PLAN's ~2k-granularity
   guard), out `results/arms_nll.npz`, ckpt-dir `data_cache/ckpt_nll`.
   **config_hash 4866f6e236**.

Scoring: `measure_generated.py --npz results/{pnull_nll,arms_nll}.npz` (CPU, scaledrift
env), bootstrap N=200 over 64 held-out fields — the frozen stage-0 instrument, untouched.

## Adjudication (frozen bars, applied verbatim)

- **Dispersion bar:** for EACH arm, |var_slope − real| ≤ 1·SE (combined bootstrap SE) at
  octaves 2, 3 AND 4 simultaneously, on `arms_nll.npz` (final 10k-step checkpoint;
  the 2k-granularity checkpoints are the mechanism diagnostic, expectation: no
  dispersion collapse with training, unlike L2-CFM).
- **Kurtosis check (addendum 1):** conditional kurtosis within 2·SE of real at trained
  octaves (both arms). If variance passes and kurtosis fails → the single pre-named
  escalation: student-t NLL head (df learned or swept), same bars, then STOP regardless.
- **GRF null:** on `pnull_nll.npz`, both arms' extrapolated-octave (j=1) var_slope
  consistent with real GRF within the same z criterion as the original P-null pass
  (|Δ| within bootstrap error; original run had |Δ|≤0.002).
- **G-1c:** if the head (incl. student-t fallback if triggered) fails the dispersion
  bar → STOP, report, reconvene (the under-dispersion law graduates to the standalone
  methods finding). No further generator variants without a new ruling.

## Predictions (Claude, pre-registered)

- P-NLL-var (dispersion bar passes at 2,3,4 both arms): **65%**. Main risk: the
  mean-path + white-noise conditional misses spatially-correlated residual structure
  that pools into the binned variance profile (the unit toy says the per-bin variance
  is matched, but 128² real tiles ≠ 16² toy).
- P-kurt (kurtosis within 2σ at trained octaves): **55%** (real fine-octave kurtosis is
  heavy, ~12.8 marginal; the amplitude-modulation hypothesis — sigma varying with
  coarse — generates excess pooled kurtosis from a Gaussian conditional; whether it's
  ENOUGH is genuinely uncertain). With the student-t rescue: combined ~80%.
- GRF null preserved: **90%** (the head learns flat sigma on GRF — the unit null gate
  passed; residual risk is small-sample z fluctuation).
- Trained-octave detail amplitude stays within a few % (no degradation): **85%**
  (mu² + e^{2g} matches the second moment by NLL optimality).

## Submitted

2026-07-11: job **15738957** (pnull_nll, MIG) and **15738958** (arms_nll, full H100).
Both PENDING on ReqNodeNotAvail — the cluster is in a maintenance drain (68/72 b1 nodes
draining); left queued per policy, no resubmission churn.

## RESULT (2026-07-11, harvested same day): G-1c FAILED — STOP

Job 15738958 completed (293 s, hash 4866f6e236 verified). Dispersion bar FAILED
(oct 2: 7–8σ low both arms; oct 3: 5–7σ) and kurtosis FAILED (≈5σ); student-t fallback
not triggered (it presupposes variance passing). Full verdict, tables, and the
mechanism decomposition in `RESULTS-phase1c.md` + `results/{g1c_verdict,nll_diagnosis,
nll_sigma_maps}.png`. Headline diagnosis: the conditional MEAN memorized the finite
training set and starved the variance head (e^{2g}→~0, flat); the implied var_slope
decays with training exactly like the L2 dispersion-collapse curve — third independent
confirmation of the law, now in an explicit-variance channel. Arm B's exp(g) is
OOD-unstable at the extrapolated octave (+71% amplitude, negative modulation).
GRF-null job 15738957 harvested (hash dc676a1f01 verified): null NOT cleanly preserved —
arm A z≈3.2 spurious var_slope at j=1; extrapolated-octave amplitude +14% (A) / +43% (B,
whose placeholder coordinate is also OOD at j=1 — corroborates the exp-head instability
on a second field). The 90% null-preserved prediction failed.

## Expected timings

pnull ~6–10 min on MIG (8k steps, NLL adds a second forward ≈1.5×; sampling is ~160×
cheaper than the 80-step ODE). arms ~10–15 min on a full H100 (10k steps, 2 arms).
Both far under the 0:59 caps.
