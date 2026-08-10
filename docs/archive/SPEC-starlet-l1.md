# SPEC — starlet ℓ1-norm held-out readout (Andreas's request, 2026-07-18)

**Runs AFTER the paper skeleton lands (one-session rule). Pre-registered held-out
CHARACTERIZATION — not a new gate on any arm (C1-t's bars are adjudicated and
closed); results go into paper §4/§6 whichever way they fall.**

## Instrument

`~/software/wl_stats_torch` (Andreas's package; already cloned):
`WLStatistics(n_scales=...).compute_wavelet_l1_norms(...)` — starlet basis,
batched, float64. Torch stays behind a numpy boundary (load npz → torch → stats →
json; never port). Verify install (`verify_installation.py`) and run the package's
own test suite first; then OUR validation gates (tests_p2, tests-first): (i) GRF
sanity — starlet-ℓ1 of a synthetic GRF ensemble consistent between two independent
halves within bootstrap; (ii) batch-vs-loop identity; (iii) D4-symmetry invariance
within noise. The held-out principle holds by construction: starlet ≠ Haar
(different basis), ℓ1-norm never used in any design or selection loop.

## Conventions (pre-register ONE primary, report the other descriptively)

The ℓ1-norm is SNR-binned → a noise convention is required on noiseless sim maps.
Primary: fixed global σ = std of the REAL test ensemble per resolution (declared
constant, identical for real and generated — pure convention, no tuning). 
Descriptive secondary: Euclid-like survey noise added identically to both (the
thesis-world convention). n_scales and bins: executor periphery, declared before
scoring, same for every map set.

## What gets scored (all from FROZEN committed npz stacks; no new generation)

1. **Sandbox, truth-referenced (the strongest leg):** exact conditional-truth
   ensembles vs C1-t samples — starlet-ℓ1 per scale with truth SEs.
2. **Gowerstreet trained octaves:** real test fields vs C1-t (arm A primary; B
   descriptive).
3. **The Stage-D edge:** real edge statistics vs the deployment-protocol
   extrapolated maps — the headline row.
4. **Taxonomy panel (descriptive):** same statistic across the generator
   generations (4a NLL-head, C1, C1-t) — does the field's constraining statistic
   separate the diseases the audit found?

## Pre-registered expectations (reconvene; add yours before scoring)

- P-SL1-trained (C1-t within 1σ-with-10% floor of real per scale, trained
  octaves): **65%**.
- P-SL1-edge (same at the Stage-D edge): **55%**.
- **P-SL1-blind (the tier question): starlet-ℓ1 does NOT flag the known ~15%
  peak-placement bias** (passes while the position-aware peak audit fails):
  **70%.** If it fires, the paper gains the sentence: "even the field's most
  constraining summary sits below the placement tier — position-blind statistics
  cannot see where the extremes land." If it does NOT fire (ℓ1 catches it), the
  statistic is stronger than the tier model predicts and §4's tier ordering gets
  revised — a finding either way.
- Gate branches: install/env failure 10%; convention ambiguity forcing a second
  primary 5%.

Deliverables: results_p2/starlet_l1_{sandbox,gowerstreet,edge,taxonomy}.json +
one figure (per-scale ℓ1 curves, real vs generated, all four legs); readout log
with verdict-vs-expectation table. R12 numbers-by-copy. STOP at the readout.
