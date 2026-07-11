# 2026-07-11 — PRE-REGISTRATION: Attempt 4a — D4 augmentation as the diagnosis test

Per the reconvene ruling `log/2026-07-11-reconvene-g1c.md`: one further round, staged;
4a is symmetry augmentation ALONE and doubles as the falsification test of the G-1c
diagnosis ("finite-data memorization of the conditional mean eats the conditional
variance"). Written and committed before submission.

## Change under test (the ONLY change vs the G-1c generator config)

D4 field-level training augmentation (`wfm.dataset.d4_augment`, `--augment`): the
training tiles are expanded to all 8 flip/rotation orientations BEFORE wavelet
decomposition (322 → 2576 tiles), so the conditional law of each block is exactly the
data's law under that orientation (validation gates green: exact orbit, invariant
per-octave amplitude, exactly preserved per-bin conditional variance —
`tests_wfm/test_augment.py`). Held-out tiles are split off BEFORE augmentation (no
leakage). Model, loss, sampler, FiLM, batch, lr, seed: unchanged from hash 4866f6e236.

The 20k-step horizon and denser checkpoints ({1,2,4,6,8,12,16,20}k) are part of the
measurement instrument — required to resolve a ≥4× onset shift — not a generator change.

## Pre-registered signature (the diagnosis test)

S(s) = implied var_slope of the generator at octave 2, arm A, at checkpoint s, computed
by the exact decomposition of `scripts/diagnose_nll.py` (Var(mu|bin) + E[e^{2g}|bin]
over 10 coarse-quantile bins, 64 held-out fields — no sampling noise).

**onset(curve) := the first checkpoint where S has fallen ≥ 0.10 (absolute) below the
curve's maximum.** Baseline (G-1c run, `results/nll_diagnosis.npz`): S = [0.967@2k,
0.793@4k, 0.782@6k, 0.773@8k, 0.747@10k] → **onset_base = 4k**.

Branches (report and STOP at this readout in ALL branches, per the ruling):
- **CONFIRMED** (diagnosis supported): onset_aug ≥ 16k (≥4×). 4b becomes available if
  the bars are still unmet — separate pre-registration, not this session's readout.
- **REFUTED**: onset_aug ≤ 8k (≤2× — the curve did not materially move) → **HARD
  STOP**, no 4b; reconvene reshapes around break + law + validation architecture.
- **INTERMEDIATE**: 8k < onset_aug < 16k → report; the reconvene judges.
Censoring: if S never falls 0.10 below its max by 20k, onset_aug > 20k ≥ 4× → CONFIRMED
(strongest form: collapse suppressed within the horizon).

Secondary, descriptive only (not adjudicated at this readout): octave-3 curve; the
σ-term's share and modulation (does e^{2g} stay alive?); final-checkpoint distances to
the frozen bars; arm B curves. The new bounded-OOD variance requirement binds at bar
adjudication (4b-stage mechanism, "mechanism periphery, requirement core") — not part
of 4a.

## Job

`scripts/arms_aug.slurm` (MIG h100_20gb, ≤1:30, b1 pool): run_two_arms.py with
--nll-head --augment, steps 20000, ckpt-steps 1000 2000 4000 6000 8000 12000 16000,
out `results/arms_aug.npz`, ckpt-dir `data_cache/ckpt_aug`.
**config_hash 1e61bd812a** (absolute paths pinned in the SLURM script).
Expected wall: ~35–45 min (2 arms × 20k NLL steps at MIG ~2/7 compute).

## RESULT (2026-07-11, harvested same day): readout delivered — rule-dependent, STOPPED

Job 15744601 completed (453 s, hash 1e61bd812a verified). **The collapse is gone within
the horizon**: augmented oct-2 implied var_slope flat at 0.95–1.01 from 4k to 20k
(real 1.02; baseline had sunk to 0.75 by 10k); σ-share 86–93% vs baseline 81→17%.
**Rule caveat, reported honestly:** the literal onset rule fires at 2k on a warm-up dip
that PRECEDES the curve's peak (6k) → literal reading REFUTED; the running-peak intent
reading gives onset censored >20k → CONFIRMED (strongest form). Both computed in
`scripts/signature_4a.py`; reconvene adjudicates. Descriptive extras: end-to-end frozen
scorer still 0.746±0.014 at oct 2 (~7σ) — the deficit moved from the head (fixed) to
RECURSION COMPOUNDING (generated coarse flattens octave-by-octave); arm B's OOD
amplitude instability disappeared under augmentation; P5 intact; kurtosis still short.
Stopped at the readout per the ruling.

## Predictions

P-4a-shift (signature CONFIRMED): **65%** (the ruling's number; mine agrees — the
decomposition mechanism is clean but augmentation only delays memorization of a still-
finite pool; 8× data is worth ~3 doublings, and the onset need not scale linearly with
data). If confirmed, the residual risk moves to the LEVEL of the σ-term (delaying
collapse ≠ restoring modulation): P(dispersion bar met at 20k by 4a alone) ~25%.
