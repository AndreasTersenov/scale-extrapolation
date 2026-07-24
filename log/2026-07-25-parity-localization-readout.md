# 2026-07-25 — PHASE A′ READOUT: the lattice-parity mechanism localized —
# it lives in the DETAIL COEFFICIENTS (nonzero channel means, a D4 violation),
# originates at the finest/extrapolated octave, and predates the t-base.

Authorized by R31 order 1 (log/2026-07-25-reconvene-phaseA.md). Descriptive,
committed artifacts only; zero training; M5 is inference from committed dense
checkpoints (job 17368763, logged pre-submission in JOBS.md). Estimators
validated tests-first (tests/test_parity_localization.py, 6 tests, env.sh
gate — pywt cannot share a process with JAX, the standing env split; corner
weights derived NUMERICALLY from pywt synthesis, convention-proof). All
numbers verbatim from results_p2/parity_localization.json (R12);
checkpoint-curve numbers from results_p2/parity_ckpt_curve.json.

## M1 — output-side peak parity by block level (T = max|z| vs real)

| stack | level 1 (2×2) | level 2 (4×4) | level 3 |
|---|---|---|---|
| c1t sandbox A / B | **6.8** / 4.2 | **4.7** / 1.4 | 3.1 / 0.8 |
| c1t repl64 A / B | **7.5** / 5.6 | **6.3** / 1.1 | 2.2 / 2.6 |
| C1 sandbox A / B | 4.4 / 4.3 | 0.7 / 0.6 | 0.4 / 1.5 |
| c1t gowerstreet A / B | 2.5 / 2.6 | 4.1 / 3.6 | 1.9 / 1.9 |
| C1 gowerstreet A | 0.6 | 2.1 | 2.6 |
| Stage-D A / B | 2.4 / **20.8** | 3.7 / 3.7 | 4.1 / 2.4 |
| 4a NLL-head A | 2.7 | 1.5 | 2.0 |
| 4a μ-only A | **22.4** | 2.6 | 0.6 |

## M2/M3 — the coefficient-level origin (the deliverable's first half)

**The bias is IN THE COEFFICIENTS, not created at synthesis.** The generated
detail channels violate D4 symmetry through NONZERO CHANNEL MEANS — a
statistic that is exactly zero for the truth (and measures so: real-vs-truth
and truth-half nulls clean). C1-t sandbox arm A, octave 1: mean-V z = −15.0,
mean-D z = +11.5, mean-H z = −5.8, plus matching sign-rate shifts and a
corr(H,V) = +8.7 violation (repl64: −17.3/+13.1/−7.2 — replicated on fresh
fields). The corner-argmax instrument (M3) confirms the deterministic link:
high-amplitude 2×2 blocks over-select the (1,1) corner (0.272 vs truth 0.245),
matching the observed odd-odd peak excess.

**Octave gradient = the second half of the deliverable.** The violation is
strongest at octave 1 and decays through the trained octaves (c1t sandbox A:
T = 15.0 / 5.7 / 4.7 / 3.3 at octaves 1/2/3/4; repl64: 17.3 / 9.1 / 3.9 /
3.2). Octave 1 is the EXTRAPOLATED octave of the sandbox e2e path — the
defect is largest exactly where no training data disciplined the rung, and
each generated octave's coefficients bias the parity at its OWN block level
(the level-2 output signal traces to octave-2 coefficients), consistent with
one weight-tied defect expressed at every rung it generates.

## M4 — hybrid transplants (causal, index-paired)

Transplanting ONLY the generated octave-1 detail triple onto otherwise-real
fields reproduces the level-1 parity bias (sandbox A: T = 5.5; repl64: 7.3);
transplanting any other single octave does not (T ≤ 1.0); removing the
generated octave-1 details from full generations removes it (T = 1.7 / 0.9).

## Taxonomy attribution (which design choices carry it)

- **The defect predates the t-base:** C1 (Gaussian base) shows it at octave 1
  only — sandbox A/B M2 T = 5.3 / 9.5 with the same channel-mean signature,
  output parity T = 4.4/4.3 — weaker, and absent from trained octaves
  (oct-2 T = 1.1/1.5). C1-t amplifies it (15–17 at oct 1) and extends it
  into trained octaves (5.7–9.1 at oct 2).
- **It was always in the mean path:** the variance-head era's μ-only skeleton
  has the largest legacy signal (output T = 22.4; oct-1 coefficient T = 39.3)
  — while production 4a scores CLEAN at output level (T = 2.7): the head's
  white-noise bath randomized peak sub-positions and hid the lattice defect,
  the mixture artifact's third disguise.
- **The dial arm under deployment is the worst case:** Stage-D arm B oct-1
  mean-V z = −104.9, output parity T = 20.8 — a further independent
  confirmation of arm-A-reference.
- **Real field:** the same defect class with a different signature and weaker
  output expression (c1t A: oct-1 M2 T = 7.5, mean-H dominant; output level-1
  T = 2.5, level-2 4.1) — field-structure-dependent expression, mechanism
  shared.

## M5 — the checkpoint curve (does the caged selection influence it?)

Job 17368763 (MIG, inference only): e2e generations from committed dense
checkpoints, arms A and B, steps 2500–20000 by 2500, scored for level-1
parity T + octave-1/2 coefficient stats.

RESULT: PENDING (filled verbatim from results_p2/parity_ckpt_curve.json when
the job lands).

## What this means (for the re-posed prereg; the reconvene's 55% lean is on
## "architectural/synthesis-grid rather than data-limited")

The defect is a learned, systematic DC offset in the per-channel conditional
detail output — not a property of the synthesis grid itself (truth
coefficients through the same synthesis are clean), and not created by the
sampler's stochasticity (the μ-only path carries it). Three facts point AWAY
from "purely architectural" and toward "trainable-but-undisciplined":
(i) the octave gradient tracks training-signal strength (worst where
extrapolated, mildest at the best-trained coarse octaves); (ii) the C1→C1-t
difference shows design choices (base + caged early stop) MODULATE it;
(iii) the training data is exactly D4-symmetrized, so nothing in the data
demands the offset — finite optimization leaves it. Whether more parents
shrink it is precisely the re-posed Phase-B question, now with a coefficient-
level estimand (channel-mean z) that is far more powerful than peak parity
(T = 15–17 vs 7) and directly mechanical. The M5 curve adds the
selection-interaction answer. STOP after the re-posed prereg draft, per R31.
