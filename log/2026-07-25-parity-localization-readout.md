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

Job 17368763 (MIG, inference only, COMPLETED 3:16): e2e generations from
committed dense checkpoints, arms A and B, steps 2500–20000 by 2500, scored
for level-1 parity T + octave-1/2 coefficient stats. Verbatim from
results_p2/parity_ckpt_curve.json:

| step | A: coef-oct1 T | A: parity T (odd-odd) | B: coef-oct1 T |
|---|---|---|---|
| 2500 | 10.6 | 2.4 (0.240) | **10.8 (B's committed pick)** |
| 5000 | 13.0 | 3.1 (0.248) | 16.9 |
| 7500 | **15.3 (A's committed pick — the curve MAXIMUM)** | 7.3 (0.303) | 11.6 |
| 10000 | 7.4 | 4.6 (0.254) | 6.0 |
| 12500 | 5.4 | 1.9 (0.255) | 6.5 |
| 15000 | 8.2 | 2.4 (0.264) | 8.5 |
| 17500 | 2.3 | 1.9 (0.260) | 8.3 |
| 20000 | 2.1 | 2.0 (0.261) | 7.0 |

**Three facts.** (i) At 1× data the defect TRAINS AWAY in arm A: by
17.5k–20k steps the coefficient statistic is null-consistent (T = 2.3/2.1) —
the defect is an optimization TRANSIENT, not a fixed property of the
architecture or synthesis grid (its channel-mean signs also wander along the
curve). (ii) **The caged marginal-optimal selection harvested the defect:**
A@7500 is the exact maximum of the defect curve; B@2500 sits at 10.8 — the
early stop that rescues the tails (disease III's cure) lands on the
parity-dirty part of the curve. The selection-interaction question R31 asked
is answered descriptively: yes, and in the harmful direction. (iii) Arm B
never becomes clean in this window (T = 7–8.5 late) — the dial arm holds the
defect, consistent with its Stage-D blow-up.

**The mechanism statement this yields:** the moment ladder from a new angle —
different statistics are healthy at different training times (tails early,
lattice symmetry late), and SINGLE-checkpoint selection on marginals cannot
satisfy both at 1× data. The re-posed Phase B asks whether (a) an
inference-time DC correction removes the defect at the tail-optimal
checkpoint, (b) a joint selection criterion finds a both-clean checkpoint at
1× (window may not exist), and (c) more data widens the joint-viable window
(the taildyn precedent: 8× holds tails flat for thousands of steps — a late,
parity-clean checkpoint may then also be tail-clean).

## What this means (for the re-posed prereg)

The defect is a learned, transient DC offset in the per-channel conditional
detail output — not a property of the synthesis grid (truth coefficients
through the same synthesis are clean), not the sampler's stochasticity (the
μ-only path carries it), and not permanent (it trains away by 17.5k steps at
1× in arm A). The deployed generator carries it because the tail-rescuing
early stop harvests the dirty part of the curve: the proximate cause is the
SELECTION-LADDER TRADE, with the reconvene's registered 55%
"architectural/synthesis-grid" lean set against this evidence for scoring at
the review. The octave gradient (worst at the extrapolated octave), the
C1→C1-t amplification, and the arm-B persistence are all consistent with an
optimization transient whose decay is slowest where the training signal is
weakest. The re-posed prereg (drafted alongside this readout) tests the three
candidate cures — inference-time DC correction, joint selection, and the data
dial (does 8×/32× widen the joint-viable window?) — with the coefficient
channel-mean statistic as primary (T = 15–17 at the committed picks vs 7 for
peak parity: more powerful and mechanically upstream). STOP after the prereg
draft, per R31.
