# 2026-07-18 — PREREG: R18 replication (64 fresh fields) + STAGE D (slide-the-edge,
# the plan's final experiment). Committed BEFORE either submission; R12 throughout.

Authorization: log/2026-07-18-reconvene-c1t.md (R18 required replication; R20
Stage D on C1-t as the passing substrate, one run, PLAN-phase2 §4 cores frozen).

## Task 1 — R18 replication (CPU)

64 FRESH sandbox test tiles, new seed stream 20260720 (disjoint from train
20260718 and held-out 20260719; recipe committed: make_c1t_repl_data.py), scored
at the FROZEN selected checkpoints (A@7500, B@2500, read from the committed
selection json — no re-selection, no retraining) with the identical bars
(kurtosis 15%/3·SE_rel primary, var_slope 10%/3·SE_rel; head-conditional
octaves 2–4 + end-to-end; ~√2 tighter SEs at 64 fields). Runner:
c1t_replication.py (std_by_j recomputed deterministically from the training
tiles — dense ckpts store params only).

**Executor weights:** P(all 24 bars pass — full replication) **45**;
P(only octave-4 entries fail, octaves 2–3 clean — "calibrated at the binding
octaves, thin at oct 4" confirmed) **35**; other **20**. (The sandbox verdict's
own texture said octave 4 rides the widened bars; tighter SEs squeeze exactly
there. Gate branch: infra resubmission **8**.)

## Task 2 — STAGE D (GPU, MIG-minutes, ONE run)

**Frozen cores (PLAN §4):** gowerstreet; C1-t substrate (t(5) base + caged
selection); slide-the-edge — train octaves **{3,4}**, octave 2 is the HELD-OUT
edge (octave 1 stays extrapolated as always); arm-B dial = **coupling-curve
extrapolation** (deployment protocol: no measured target-octave couplings);
both arms; AMBIGUOUS = FAIL.

**Pinned periphery:**
- Dial curve (make_stageD_coords.py, mechanical, chosen on trained-visible
  octaves j∈{5,4,3} ONLY): per component, linear vs log-linear in octave index by
  in-sample RSS → var_slope LINEAR (edge value 1.1836), kurtosis LOG-LINEAR
  (edge value 10.5199); measured j≥3 verbatim; j=2,1 extrapolated. The dial's
  input error vs the (scoring-only) measured couplings is itself a readout
  quantity.
- Selection cage: identical to C1-t (val-32/test-32, dense ckpts every 500, K=4,
  argmin max(rel_vs/0.10, rel_kurt/0.15), ties→earlier, one pick) with the
  selection octave moved to **3** — the finest TRAINED octave (octave 2 must stay
  untouched by every deployment-side choice). Reference:
  gowerstreet_val_ref_oct3.json (real val-32 oct-3 estimand: vs 0.845,
  kurt 4.2153), committed pre-submission.
- Runner: run_c1t_arms.py with --train-octaves 3 4 --sel-octave 3 (parameter
  added; default path unchanged — existing green tests cover it). Octave-2 and
  octave-1 detail amplitudes: log-linear extrapolation from trained octaves (the
  standing P4 protocol, deployment-legal).
- **Adjudication (score_stageD.py):** at the EDGE (octave 2), BOTH levels
  (head-conditional given real coarse = conditional calibration; end-to-end from
  octave 4 = the deployment path), vs the REAL TEST fields' own statistics:
  var_slope ≤ max(10%, 3·SE_rel) AND kurtosis ≤ max(15%, 3·SE_rel). **P-D rides
  arm B** (the dial under deployment extrapolation); arm A is the scale-blind
  control. Suite: estimand bars + q999 + PEAK AUDIT (ν∈{1,2.5,3}, 32-field
  means, descriptive) + octaves 3/4/1 descriptive. starlet-ℓ1 and scattering
  covariance are NOT in the suite (the hardening chores have not landed — stated
  here per the ruling's "if the hardening lands first").

**Branches (mechanical, precedence as listed) and weights:**

| branch | definition | reconvene (R20 lines) | executor |
|---|---|---|---|
| D-PASS-BOTH | A and B pass all 4 edge checks | — | 18 |
| D-PASS-B-ONLY | only B passes | — | 14 |
| D-PASS-A-ONLY | only A passes | — | 8 |
| D-FAIL-BOTH | neither passes | — | 55 |
| gate/infra | resubmission/pathology | — | 5 |

Component lines (comparable to R20's): P(dispersion within bars at the edge,
arm B, both levels): reconvene **55** / executor **55**. P(marginal tails
[kurtosis] within bars there): reconvene **35** / executor **30** (the moment
ladder says the finest octave's tails are the most data-hungry thing the
weight-tying must now supply unseen; C1-t's oct-1 record shows tails degrade
fastest under extrapolation). P(dial arm B beats scale-blind arm A on |e2e edge
kurtosis deficit|): reconvene **40** / executor **45** (phase-1's P6 precedent
says the dial helps exactly here; curve-extrapolation error — vs +2.9%, kurt
+18% at the edge — is the new unknown that caps my number).

P-D itself (arm B passes all 4 edge checks — the original project bet): **PLAN
standing 40; executor 27** (= my TAILS line 30 × dispersion ~90 given the
campaign's dispersion record, both levels).

## Deliverables

Task 1: results_p2/c1t_repl64.json + verdict lines in the readout.
Task 2: arms_stageD.npz + c1t_selection_stageD.json + stageD_verdict.json +
readout log + figure (edge bars, selection curve at oct 3, peak audit, dial
input-error note). JOBS.md entries. STOP at the readout — the paper decision
(R20.3) is Andreas's, taken with this verdict in hand.
