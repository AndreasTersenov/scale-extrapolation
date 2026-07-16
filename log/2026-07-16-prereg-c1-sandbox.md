# 2026-07-16 — PREREG: Arm C1, sandbox leg (vanilla CFM + augmentation — the un-run arm)

Committed BEFORE training (NIGHT-ORDERS Arm C1). Gate A passed (calibrated
instruments + exact truth in results_p2/sandbox_truth.json). SINGLE VARIABLE vs the
frozen 4a configuration: **the NLL head is REMOVED** (plain conditional FM, ODE
sampler); augmentation stays frozen-in; same architecture (32/64/128 FiLM),
same training protocol (20k steps, batch 32, lr 1e-3, seed 0), same arms A/B,
same 64-heldout split. NO crops (B2 un-adjudicated; single-variable discipline).

## Data (mirrors the phase-1 regime per the orders)

`data_cache/tiles_sandbox.npz['sandbox']` = 386 tiles: 322 training (Stage-A stream,
seed 20260718) + 64 held-out (fresh stream, seed 20260719) — matching the selfsim
precedent (386 = 322+64; noted: the 4a gowerstreet run itself trained on 266 = 330−64,
the "322" of the record is the step-1c-era count; the orders say 322, we give 322).
Coords for arm B: `data_cache/running_couplings_sandbox.json` = the TRUTH estimand
values per octave (the exact-law dial), octaves 1–4.

## Command lines (pinned; hashes recorded at submission in JOBS.md)

- Leg 1 (this prereg): `scripts_p2/arms_c1_sandbox.slurm` →
  run_two_arms.py --field sandbox --train-octaves 2 3 4 --gen-from 4 --channels 32
  64 128 --cond-mode film --steps 20000 --batch 32 --n-heldout 64 --sample-steps 80
  --augment --ckpt-steps 1000 2000 4000 6000 8000 12000 16000 (NO --nll-head)
  → results_p2/arms_c1_sandbox.npz, ckpts data_cache/ckpt_c1_sandbox/;
  then scripts_p2/c1_conditional_sweep.py (head-conditional curve + final octaves).
- Leg 2 (gowerstreet; runs ONLY if the sandbox dispersion bar passes, per the
  orders): identical recipe on tiles_pnull.npz gowerstreet + running_couplings.json,
  out results_p2/arms_c1_gowerstreet.npz — readout DESCRIPTIVE (adjudication at the
  morning reconvene against the G-1c project bars).

## Bars (self-adjudicated, from measured truth; SE_rel = sqrt(SE_gen²+SE_truth²)/|truth|)

Truth (from results_p2/sandbox_truth.json): var_slope [oct2,3,4] = [1.105, 0.961,
0.809]; kurtosis = [5.585, 4.001, 2.677]. Two measurement levels, both arms:

1. **Head-conditional** (generated detail given REAL held-out coarse, estimand on
   pooled (w_gen, c_real), bootstrap-over-fields SE): at octaves 2,3,4
   SIMULTANEOUSLY — dispersion |vs−truth|/|truth| ≤ max(10%, 3·SE_rel); kurtosis
   |k−truth|/|truth| ≤ max(15%, 3·SE_rel).
2. **End-to-end recursion from octave 4** (frozen scorer on the generated stacks):
   same bounds at octaves 2,3,4. **Orders-ambiguity resolution (pre-declared):**
   "same bounds at every octave" is adjudicated on the TRAINED octaves {2,3,4};
   octave 1 (the extrapolated octave) is reported DESCRIPTIVELY — extrapolation is
   Stage D's question, and no phase-1 comparator bar exists for it on the sandbox.

**"Sandbox dispersion bar passes" (the leg-2 trigger)** := level-1 AND level-2
var_slope components pass at octaves 2,3,4 for BOTH arms.

**Checkpoint curve** (collapse signature, the adopted running-peak rule): oct-2
head-conditional var_slope at {1,2,4,6,8,12,16,20}k; collapse fires iff the curve
drops ≥0.10 below its RUNNING peak by 20k.

## Branches with weights (pre-registered; adjudication precedence top→bottom)

| branch | definition (mechanical) | executor weight |
|---|---|---|
| B-C1-DEG degenerate | end-to-end detail amplitude >25% off at any trained octave, or NaN | 5 |
| B-C1-COLL collapses-again | collapse signature fires on the ckpt curve, OR final head-conditional var_slope fails LOW at any trained octave | 35 |
| B-C1-REC alive-but-recursion-fails | head-conditional dispersion passes (2,3,4); end-to-end dispersion fails | 20 |
| B-C1-TAILS alive-but-tails-fail | dispersion passes both levels; kurtosis fails anywhere | 25 |
| B-C1-CAL alive-calibrated | all bars pass (dispersion + kurtosis, both levels) | 10 |
| other/mixed | anything else | 5 |

Standing reconvene numbers: P-C1a (trained-octave dispersion alive on sandbox) 60%,
P-C1b (kurtosis at truth) 45%, P-C1c (recursion calibrated) 45%.
**Executor's numbers: P-C1a 55%, P-C1b 30%, P-C1c 25%.** Reasoning in brief: the
ODE-pushforward channel never demonstrated full dispersion in phase 1 even at its
2k peak (~3σ low at oct 2 un-augmented); augmentation removes the memorization
DECAY (4a, in the σ-channel) but has never been shown to raise the pushforward's
LEVEL; kurtosis via a deterministic ODE from Gaussian noise was persistently tame;
the sandbox drifts like gowerstreet, so compounding should reappear. C1 is exactly
the experiment that separates "collapse law was the whole problem" (then dispersion
lives) from "the ODE pushforward is structurally under-dispersed" (then it fails
low without a collapse signature).

## Deliverables

results_p2/arms_c1_sandbox.npz, c1_conditional_sandbox.json, c1_endtoend_sandbox.json
(CPU scoring step, frozen scorer), readout log + figure (ckpt curve + bars visual),
JOBS.md entries with config hash at submission.
