# 2026-07-11 — PRE-REGISTRATION: Attempt 5 — self-conditioning (FINAL generator attempt of phase 1c)

Per the ruling `log/2026-07-11-reconvene-4bpii.md`. The reshape-on-failure
pre-commitment stands: if the lever bar fails, the generator freezes at "calibrated
heads + measured compounding limit" and phase 1c reshapes around the banked results.
Committed before submission.

## Single variable: the conditioning distribution

Aligned-pair self-conditioning (scheduled-sampling recipe): for octaves 3 and 2 — the
levels that receive drifted conditioning at generation — each training example
conditions, with probability p, on the GENERATED coarse recursed from that same
training tile's real octave-4 start by the FROZEN 4a checkpoints (arm-matched); the
target stays the real detail. Octave 4 keeps real conditioning (matches generation,
which starts from real coarse). Static pool, built once in-job, deterministic seed;
no self-referential refresh (the ruling's feedback risk). **No leakage:** pools are
built from the training split only, before held-out tiles are touched. No s-channel;
augmentation frozen in; everything else identical to the 4a config.
**Sweep: p ∈ {0.5, 1.0}** (mixing = textbook scheduled sampling; pure = exact
generation-time distribution). Bar met if either passes; both reported.

Gates green first (`tests_wfm/test_selfcond.py`): plumbing/alignment + non-inferiority.
**Honest toy null, disclosed:** the toy mechanism test could NOT show an improvement
under drifted conditioning, because a pointwise toy has no neighborhood-texture
fragility to repair. Consequence: this experiment discriminates two readings of the
compounding gap — **behavioral** (feature fragility; self-conditioning recovers
modulation) vs **informational** (drifted coarse does not carry the modulation; the
honest response given drifted input ≈ the current end-to-end, and no conditioning-side
training can beat it).

## Bars (unchanged, two-tier) and references

- **Lever bar:** end-to-end var_slope within 1σ combined of THIS model's own
  given-real-coarse ceiling (measured with bootstrap SEs at harvest, before scoring
  end-to-end), octaves 2, 3, 4 simultaneously, per arm.
- **Project bars (frozen):** dispersion within 1σ of real at octaves 2–4; kurtosis
  within 2σ (reported at all octaves; student-t branch stays pre-named for
  variance-passes/kurtosis-fails, after this readout).
- **Bounded-OOD:** extrapolated-octave (j=1) detail_std within 10% of real (band
  established since 4a: 0.689–0.736 vs real 0.743).

## Gate branches WITH weights (the 4b′-ii lesson; Claude's, reconvene's alongside)

Reference end-to-end (4a run, oct 2): A 0.746±0.014, B 0.734±0.014.
- **B1 — lever AND project dispersion pass: 20%.**
- **B2 — lever passes, project fails (ceiling binds, e.g. oct 3): 15%.**
- **B3 — lever fails but material partial recovery (oct-2 end-to-end improves by
  ≥ 0.05 for either arm): 25%.** → reshape per pre-commitment, with "compounding is
  partly behavioral" noted.
- **B4 — no material movement (|Δoct2| < 0.05 both arms): 35%** — the informational
  reading; the honest conditional given drifted coarse is ~what we already produce.
  Sharp corollary if B4: the self-conditioned model's own ceiling ON GENERATED coarse
  should ≈ the end-to-end value (measured at harvest as the discriminator).
- **B5 — degradation (oct-2 end-to-end drops > 0.05, or own real-coarse ceiling drops
  > 2σ, or bounded-OOD violated): 5%.**
(Reconvene: P-lever 55% ≙ B1+B2; mine is 35% — the toy null and the 4b′-ii
information-limit reading shift my weight to B4. Registered disagreement.)
P-project-dispersion: mine 25% (reconvene 40%). P-kurt-oct2: 10% (reconvene 15%).

## Jobs (MIG h100_20gb, ≤1:30 each, absolute paths pinned)

1. `scripts/arms_a5_p05.slurm`: --selfcond-p 0.5 → `results/arms_a5_p05.npz`,
   ckpt-dir `data_cache/ckpt_a5_p05`. **config_hash ffdeac4d4b**.
2. `scripts/arms_a5_p10.slurm`: --selfcond-p 1.0 → `results/arms_a5_p10.npz`,
   ckpt-dir `data_cache/ckpt_a5_p10`. **config_hash ccc86ccf1b**.

Harvest: own-ceiling (real coarse, with SEs) per model → lever bar; `measure_generated`
per run → end-to-end + project bars + bounded-OOD; the B4 discriminator (own ceiling on
GENERATED-coarse pools); readout figure. STOP at the readout.
