# 2026-07-18 — PREREG: Arm C1-t, sandbox leg (t(5)-base CFM + validation-selected
# early stop). Committed BEFORE submission; R12 rule throughout.

Authorization: log/2026-07-18-reconvene-taildyn.md R17. SINGLE variable vs the C1
control = the base distribution + the pre-registered checkpoint-selection rule
(selection is part of the arm by design — R17 core). Everything else is the C1
configuration verbatim: same UNet (32/64/128, FiLM), weight tying, per-octave std
standardization, same data file and TRAINING split (322 tiles + 8× D4 — identical
to C1's training set), same recursion protocol (gen_from 4, heun-80), same frozen
scorers and same-convention exact truth. ν = 5 FIXED (R17). Runner:
scripts_p2/run_c1t_arms.py (smoke-tested end-to-end on CPU); field gates:
tests_p2/test_c1t_field.py (green; trainer-pipeline identity with C1, recursion
shapes/determinism/std-scaling — the R17 "light sanity check", complementing the
bake-off's toy-scale statistical validation of the t-base ODE).

## The Goodhart cage (pinned)

- Held-out block (last 64 tiles) split ONCE: **VALIDATION = first 32** (selection
  only), **TEST = last 32** (adjudication only; never touched by selection).
- Dense checkpoints every 500 steps (40 per arm, both arms saved to
  data_cache/ckpt_c1t_sandbox).
- **Selection rule (one pick, no re-picks, mechanical):** at each checkpoint,
  sample K=4 octave-2 details per VALIDATION field (heun-80), pool (w_gen,
  c_real), compute the estimand, and score = max(rel_err_var_slope / 0.10,
  rel_err_kurtosis / 0.15) vs the normconv exact truth — the scaled worst case
  against the two binding bars. Selected checkpoint = argmin score; ties → the
  EARLIER step. Selection uses octave 2 only (the binding octave); octaves 3, 4
  ride along at adjudication.
- Adjudication: head-conditional at the SELECTED checkpoint on TEST fields (K=1,
  bootstrap over fields, n_boot 200 — the C1/C3 convention) + end-to-end
  recursion from the selected checkpoint on TEST coarse, frozen production
  scorer. The 20k-checkpoint head-conditional numbers are reported DESCRIPTIVELY
  (the selection-effect, same arm).

## Bars (C3 frame, unchanged per R17)

- **PRIMARY: kurtosis** |k−truth|/|truth| ≤ max(15%, 3·SE_rel), octaves 2–4,
  both arms, per level (head-conditional and end-to-end).
- **Dispersion must-not-regress:** var_slope ≤ max(10%, 3·SE_rel), both levels.
- q999 DESCRIPTIVE (R10 condition 1), never adjudicating.
- Degenerate: e2e detail amplitude >25% off the test real stack, or NaN.
- **Gowerstreet-leg trigger:** kurtosis PRIMARY passes head-conditional both arms
  AND dispersion bars pass both levels both arms → identical-recipe gowerstreet
  leg (its own val/test split of the 64 held-out, same selection rule),
  DESCRIPTIVE readout, reconvene adjudicates.

## Attribution at zero cost (R17 core)

scripts_p2/score_c1_tails.py scores C1's EXISTING sandbox checkpoints (Gaussian
base, same architecture) for oct-2 tails on the SAME validation fields with the
same K=4 protocol — separating base-effect (t vs Gaussian at matched steps) from
selection-effect (early vs late checkpoint). Descriptive; runs as a CPU job.

## Branches with weights (adjudication precedence top→bottom)

| branch | definition (mechanical) | executor |
|---|---|---|
| C1T-DEG | amplitude/NaN degenerate | 3 |
| C1T-DISP-REGRESS | any dispersion bar fails (either level) | 17 |
| C1T-TAILS-FAIL | dispersion holds; kurtosis fails head-conditional | 27 |
| C1T-TAILS-HC-ONLY | kurtosis passes hc, fails end-to-end | 13 |
| C1T-CAL | all bars pass (kurtosis + dispersion, both levels) | 35 |
| other/mixed | anything else | 5 |

Reconvene (R17, registered there): P(sandbox kurtosis bar passes at the selected
checkpoint) 55; P(joint viable window exists at the real-arm regime) 60;
P(gowerstreet e2e kurtosis deficit halves vs C1 | triggered) 50;
P(dispersion regresses) 20.
**Executor lines:** P(kurtosis bar at selected ckpt, head-conditional, both arms)
**45** (selection noise on 32 fields + the real regime is MORE starved than toy
1× — N_eff ≈ parents cuts the effective data; the toy 1× window was real but
narrow); P(joint viable window exists on the validation curve) **55**;
P(gowerstreet deficit halves vs C1 | triggered) **45**; P(dispersion regresses —
selection picks an under-trained dispersion state) **15** (the selection score
penalizes dispersion error at 1.5× the kurtosis weight per unit bar, and toy
dispersion recovered by step 1000 everywhere).
**Gate branches:** infra resubmission needed: **12**; selection-curve pathology
(no finite scores / all-NaN → descriptive STOP, reconvene): **4**.

## Deliverables

results_p2/arms_c1t_sandbox.npz + c1t_selection_sandbox.json +
c1t_verdict_sandbox.json + c1_tails_val.json (attribution); readout log +
figure (selection curves with the pick marked, bars visual, attribution
overlay); JOBS.md entries with config hash. Gowerstreet leg only on trigger,
descriptive. STOP at the readout.
