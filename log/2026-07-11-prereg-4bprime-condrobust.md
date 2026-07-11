# 2026-07-11 — PRE-REGISTRATION: Attempt 4b′ — conditioning robustness (anti-compounding)

Per the reconvene ruling `log/2026-07-11-reconvene-4a.md` (4a CONFIRMED strongest form;
augmentation FROZEN into the generator config; original 4b superseded — the binding
deficit is recursion compounding). Written and committed before submission.

## Single variable

**Conditioning robustness.** Implementation = the ruling's named default (accepted): the
cascaded-diffusion conditioning-augmentation recipe (Ho et al. 2021; Imagen). During
training the coarse input is corrupted per example, coarse′ = coarse + s·std(coarse)·ε
with s ~ U(0, s_max), and s is EXPOSED to the model as one extra conditioning dimension
(both arms symmetrically; arm A's cond becomes [s] — carrying no scale information, so
the P5 contrast is preserved). At generation s = 0, no sampling-time corruption knob.
I considered the self-conditioning alternative (train on generated coarse) and rejected
it for this round: it couples training to sampling (heavier machinery, a moving target)
and the standard recipe's robustness is not specific to additive drift — it penalizes
over-trust in conditioning texture generally, which is the diagnosed failure mode.
Gates green first (`tests_wfm/test_cond_corrupt.py`): plumbing + clean-eval (s=0)
non-inferiority on the known-truth toy. Everything else identical to the 4a frozen
config (NLL head, D4 augmentation, FiLM, 20k steps, seed 0).

**Sweep (pre-registered):** s_max ∈ {0.1, 0.3}. Bar met if EITHER value meets it; both
reported. No other values without a new prereg.

## Fixed reference: the measured ceiling (ruling 4 pre-requirement; no new training)

Given-real-coarse implied var_slope of the existing 20k augmented checkpoints
(`scripts/measure_ceiling.py`, bootstrap-over-64-held-out-fields SE,
`results/ceiling_4a.json`):

| arm | oct 2 | oct 3 | oct 4 |
|---|---|---|---|
| A | 0.956 ± 0.035 | 0.683 ± 0.021 | 0.474 ± 0.020 |
| B | 1.016 ± 0.039 | 0.699 ± 0.022 | 0.505 ± 0.020 |

Context (end-to-end 4a run vs this ceiling): the compounding gap is concentrated at
oct 2 (A: 0.746 vs 0.956; B: 0.734 vs 1.016); octaves 3–4 are already at ceiling.
Flag beyond the ruling's oct-2 note: at oct 3 the ceiling itself (0.68–0.70) sits
~3σ below real (0.801±0.037) — the pre-named ceiling-binds branch may fire at oct 3
even if decompounding is perfect.

## Bars (two tiers, adjudicated separately)

- **Lever bar (adjudicates 4b′):** per arm, end-to-end var_slope (frozen scorer,
  bootstrap over held-out fields) within 1·σ_combined of that arm's measured ceiling
  above, at octaves 2, 3 AND 4 simultaneously. σ_combined = hypot(SE_end2end,
  SE_ceiling).
- **Project bar (FROZEN, G-1c):** within 1σ of REAL at octaves 2, 3, 4 simultaneously
  — reported separately. If the lever bar passes and the project bar fails BECAUSE the
  ceiling binds (gen consistent with ceiling, ceiling inconsistent with real), that is
  the pre-named revival condition for original 4b — not relitigated here.
- **Kurtosis (2σ at trained octaves):** reported; the student-t conditional remains the
  pre-named post-4b′ lever for the variance-passes/kurtosis-fails branch, never bundled.
- **Bounded-OOD variance response:** gowerstreet extrapolated-octave amplitude reported
  at this readout (4a reference: 0.729/0.689 vs real 0.743); the GRF-extrapolated-octave
  demonstration accompanies any PROJECT-bar adjudication (separate pnull run, not part
  of this readout's jobs).

## Jobs (MIG h100_20gb, ≤1:30 each, absolute paths pinned)

1. `scripts/arms_4bp_s01.slurm`: --cond-corrupt 0.1 → `results/arms_4bp_s0.1.npz`,
   ckpt-dir `data_cache/ckpt_4bp_s0.1`. **config_hash 4f5bbe7b0f**.
2. `scripts/arms_4bp_s03.slurm`: --cond-corrupt 0.3 → `results/arms_4bp_s0.3.npz`,
   ckpt-dir `data_cache/ckpt_4bp_s0.3`. **config_hash 9f56a059ad**.

Expected ~8–10 min each (4a ran 453 s; no mid-training checkpoints here). Harvest:
`measure_generated.py` on each npz; plus post-hoc ceiling of the NEW checkpoints
(descriptive: did corruption training lower the head's own given-real-coarse response?
— the attenuation risk).

## RESULT (2026-07-11, harvested same day): LEVER BAR FAILED — STOPPED at the readout

Jobs 15753842/15753843 completed (491/507 s, hashes verified). Oct-2 end-to-end vs the
fixed ceiling: z≈6.7/4.7 (arm A, s=0.1/0.3), z≈7.6 (arm B, both) → lever FAILED both
sweep values, both arms (oct 3–4 pass, already at ceiling). Project bar also failed;
ceiling-binds branch not reached. KEY decomposition: corruption training RAISED the
heads' own ceilings (A oct-2 0.956→0.996 ≈ real; oct-3 0.683→0.748) and moved kurtosis
toward real (A s=0.3 oct-3 at real), but end-to-end did not move — the trained
robustness is never engaged because generation samples at s_gen=0. The literature's
operating mode (inference-time matched conditioning noise) was excluded by this
prereg's own s_gen=0 choice; re-sampling the EXISTING s=0.3 checkpoints with a
pre-registered s_gen>0 is the zero-training follow-up candidate — reconvene's call.
Full record: RESULTS-phase1c.md §4b′, results/readout_4bp.png.

## Predictions (Claude; reconvene's alongside)

- P-lever (either s_max passes the lever bar): **60%** (reconvene: 55%). Precedented
  mechanism; only oct 2 must move (~0.21), oct 3–4 already at ceiling. Risks: additive
  noise may mis-model modulation-flattening drift; attenuation may drag the s=0
  conditional response down (clean-eval gate bounds this at toy scale only).
- P-s_max=0.3 outperforms 0.1 at oct 2: **55%** (drift is small but structured; more
  robustness range should help until attenuation bites).
- P-project-dispersion (both arms, all three octaves, vs real): **20%** (reconvene:
  30%) — I weight the oct-3 ceiling shortfall more heavily now that it is measured.
- P-ceiling-binds (lever passes, project fails via the ceiling): **35%** (reconvene:
  25%) — raised for the same reason.
