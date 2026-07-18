# 2026-07-18 — PREREG: starlet ℓ1-norm held-out readout (SPEC-starlet-l1)

Executing on Andreas's explicit instruction of 2026-07-18 ("execute it exactly").
The SPEC's "runs after the paper skeleton" ordering is superseded by that
instruction. Committed BEFORE any leg statistic is read; the only numbers
touched so far are instrument-verification numbers, recorded in the committed
artifact results_p2/starlet_l1_instrument.json (R12 source for this file).

## Instrument verification (done pre-prereg; gate branch resolved)

- Package suite: "45 passed, 1 skipped in 3.87s" (starlet_l1_instrument.json).
- Forward transform is a clean first-generation starlet: sum-of-scales
  round-trip error 4.4e-16 (float64) — the identity our statistics rely on.
- **Instrument finding 1 (flagged to Andreas, unused path):**
  `verify_installation.py` FAILS its Starlet check because it calls
  `reconstruct(gen2=True)`: gen2 reconstruction error 1.41 on a unit-variance
  GRF — the gen2 filter does not match the gen1 forward transform. Never used
  here (ℓ1 needs the forward transform only); the package's own tests avoid it.
- **Instrument finding 2 (flagged to Andreas; convention chosen around it):**
  `get_noise_levels` uses zero-padded *linear* convolution ('same' mode) while
  the forward transform is periodic-consistent (rot90-commutes at 1e-16). On
  unmasked periodic sim tiles the shipped SNR plane therefore carries border
  artifacts: spatial cv of the noise plane 0.107/0.067/0.057/0.065 (detail
  scales, scalar σ), and shipped SNR fails rot90 commutation with per-scale max
  errors 8.8/6.7/5.2/4.9 — it would fail our D4 gate. Convention (below) uses
  the *interior plateau* of the package's own propagation instead; the plateau
  values for σ=1 (0.891/0.201/0.0855/0.0412) reproduce the analytic B3-starlet
  σ·||h_s||₂ table, and plateau-normalized ℓ1 is rot90-exact (4.5e-15).
- Our gates (tests_p2/test_starlet_l1.py, all green under the Stop hook):
  gen1 sum identity; GRF half-vs-half consistency within 3σ bootstrap;
  batch-vs-loop identity; D4 (rot90/rot180/fliplr/flipud) exactness;
  bin-coverage identity (binned total == Σ|SNR|, nothing outside bins).

## Conventions (declared now, before scoring; executor periphery)

- `WLStatistics(n_scales=5)` float64 CPU: 4 detail scales of support ~2/4/8/16
  px ↔ octaves 1/2/3/4; coarse plane excluded from all curves.
- **Noise normalization (primary):** per-leg global scalar σ = pixel std of
  that leg's REAL stack (declared constant, identical for real and generated;
  pure convention on noiseless sims). Per-scale SNR denominator = σ × interior
  plateau of the package's impulse-response propagation (see finding 2).
- Bins: n_bins=31 per scale; per-scale range = [min, max] of SNR over the
  COMBINED map sets of the leg (real ∪ gen_A ∪ gen_B), ends padded by 1e-6 of
  the span (float32-threshold guard); identical bins for every set in a leg;
  nothing falls outside (per-map binned total == Σ|SNR|, tested).
- **Scored scalar per scale:** T = mean over maps of the per-map total ℓ1
  (sum over bins), bootstrap SE over maps (n_boot=400, fixed seeds).
  **Pass rule:** |T_gen − T_real| ≤ max(1·hypot(SE_gen, SE_real), 0.10·T_real)
  — the SPEC's "1σ-with-10%-floor". Descriptive per scale: the full 31-bin
  curve with bootstrap band, and the tail share ℓ1(|SNR|≥3)/total.
- **Scored scales:** legs 1–2: {4, 8, 16 px} (octaves 2–4, the binding set).
  Leg 3 (edge): same three scored, headline = 4 px (the held-out octave 2).
  2 px (octave 1, never bound anywhere) always descriptive. Verdicts adjudicate
  on **arm A** (reference arm per reconvene 9efdac3); arm B same tables,
  descriptive.
- **Secondary convention (descriptive, legs 2–3 only):** survey-like noise —
  additive white Gaussian, σ_n = 2.0 × σ_real (declared Euclid-like
  noise-to-signal ratio at ~arcmin pixels), fixed seed 20260718, independent
  realizations, identical procedure for real and generated; SNR denominator
  then uses σ_n. Reported as arm-A total-ℓ1 rel. differences only, no verdicts.

## Legs (all from FROZEN committed npz stacks; no new generation)

1. **Sandbox truth-referenced:** arms_c1t_sandbox.npz — 32 truth-process
   fields vs 32 e2e samples per arm. (The 64-field replication gen npz is
   committed but its truth tiles are not; 32v32 is the committed pairing.)
2. **Gowerstreet trained octaves:** arms_c1t_gowerstreet.npz — 32 real test
   fields vs 32 C1-t samples per arm.
3. **Stage-D edge:** arms_stageD.npz — same real test fields vs the
   deployment-protocol extrapolated maps (train {3,4}, octave 2 held out).
4. **Taxonomy panel (descriptive, no verdicts):** same statistic across the
   generator generations on gowerstreet — 4a NLL-head with-noise
   (results/arms_aug.npz, frozen into the repo with this commit; produced
   2026-07-11 by scripts/arms_aug.slurm, nll_head=true, 64 held-out fields),
   4a mu-only forensic skeleton (forensic_nllnoise.npz), C1
   (arms_c1_gowerstreet.npz), C1-t (arms_c1t_gowerstreet.npz). Shared σ from
   the 64-field real stack; shared ranges over the union of all panel sets;
   arm A curves. Note: C1-t conditions on the test-32 subset of the 64
   held-out fields; the panel is descriptive so the mixed conditioning is
   stated, not scored.

## Pre-registered expectations (reconvene lines from SPEC; executor added now)

| line | question | reconvene | executor |
|---|---|---|---|
| P-SL1-trained | arm A within 1σ-w-10%-floor at all scored scales, leg 2 | 65 | **80** |
| P-SL1-edge | same at the Stage-D edge (all three scored; headline 4px) | 55 | **70** |
| P-SL1-blind | ℓ1 passes legs 2 AND 3 (arm A) while the committed peak audit fails — position-blind statistic below the placement tier | 70 | **75** |
| gate: install/env failure | — | 10 | **2** (resolved pre-prereg; residual scoring surprises only) |
| gate: convention ambiguity forces second primary | — | 5 | **5** |

Executor reasoning (priced with failure-era priors retired per 9efdac3):
total ℓ1 is a mean-|w| statistic, lower-moment than the kurtosis bars C1-t
already passed at these octaves (leg-2 binding-octave deficits −4.0/−8.0%;
edge e2e 3.2/4.4%); amplitude-marginal match should carry ℓ1. The known
peak excess (+13/+15% at high ν) is an arrangement phenomenon invisible to an
amplitude histogram — hence blind at 75, capped by the sandbox-B tail-hot
pattern possibly showing in leg-1 tails and by the 1σ leg of the bar being
tight (SE-dominated false-fail risk at ~1σ is the main hazard for the 80).

P-SL1-blind operationalization (pinned): fires iff arm A passes all scored
scales on leg 2 AND leg 3 under the pass rule, given the already-committed
peak-audit failure (c1t_peaks_gowerstreet.json; stageD_verdict.json peaks,
arm A +13/+15% at ν=2.5/3). If ℓ1 *fails* where peaks fail, ℓ1 is stronger
than the tier model predicts → §4 tier ordering gets revised (finding either
way, per SPEC).

## Execution

CPU-only, login-light (seconds of batched float64 transforms), no SLURM job.
Scorer scripts_p2/score_starlet_l1.py → results_p2/starlet_l1_{sandbox,
gowerstreet,edge,taxonomy}.json; one figure results_p2/starlet_l1.png; readout
log with verdict-vs-expectation table. All readout numbers by verbatim copy
from the four JSONs (R12). STOP at the readout.
