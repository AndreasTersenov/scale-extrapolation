# 2026-07-29 — DRAFTED preregs (NOT RUN; NIGHT-ORDERS-2 NOT-CONFIRMED path)

GATE-N1 read NOT-CONFIRMED (one clear octave, not two), so no training ran
tonight. These drafts encode what the N1 evidence says the next levers are.
They are proposals for the morning reconvene — nothing here is
self-authorizing. Weights are the executor's, registered now so the morning
review can co-sign or amend them BEFORE any run.

## L1 — the ONE-OCTAVE colored base (the narrowed doctrine lever)

**Premise (measured tonight):** the whiteness defect is real, ~5σ, but lives
at octave 1 ONLY (gowerstreet trained C_gen 0.7237±0.0072 vs real
0.7864±0.0100; octaves 2–3 null; sandbox: absent entirely). The D1 spectrum
gives the filter's shape: a broadband ~15% deficit at k < Nyquist closing at
the pixel ring — i.e. the base must ADD low/mid-k weight (equivalently
suppress the near-Nyquist tilt) at octave 1 only.

**Construction:** copula-style colored-t base AT OCTAVE 1 ONLY (white-t
unchanged at octaves ≥2 — sharper single-variable than the N2 multi-octave
spec): color a Gaussian field by the annular filter measured from TRAINING
detail planes at octave 1 (per field, frozen + committed pre-training),
quantile-map marginals to t(ν=5). Tests t1–t4 exactly as NIGHT-ORDERS-2 N2
specifies (spectrum match, t tails, D4-invariance proof via isotropic-filter
argument + orbit statistics, seeded reproducibility). One variable vs the
committed arm-A recipe; F2 sampling unchanged.

**Branches (executor draft weights):** L1-CURED 10 (both trained-leg CIs
include 0, no regression) / L1-IMPROVED 40 (both Δ ≥ 2) / L1-NULL 30 /
L1-REGRESSED 10 / gates+FLIPPED 10. Same must-not-regress set as N3;
sandbox canary first with the kill criterion; ≥2 gowerstreet seeds
(seed lesson). Note the honest hedge: the D1 deficit is BROADBAND in the
detail plane, so a base-coloring cure assumes the flow largely preserves
base spectral structure into the details — measured nowhere yet. If L1
lands NULL, that assumption is the finding.

## L2 — training-side pixel-scale regularization (NOT a base lever; draft only)

If L1 reads NULL, the alternative family is loss-side: an oct-1 spectral
penalty (match annular band powers of generated details to real during
training) — a different doctrine (auxiliary loss vs base structure), needs
its own reconvene debate before any prereg is finalized. Registered as a
direction, not a spec.

## L3 — sandbox oct-3 over-coloring probe (descriptive, cheap, CPU)

Tonight's D3 found sandbox gen OVER-colored at octave 3 (z=−2.9 c1t, −2.1
F2) where peak counts show DEFICITS — the mirror image of gowerstreet. A
half-day descriptive pass profiling C vs octave vs field vs arm across the
committed taxonomy (all existing npz stacks) would turn the two-field
anecdote into a coloring-vs-count-bias curve; if the association holds
across the taxonomy, C becomes a cheap pre-training predictor of the peak
bias sign — paper-grade methodology. No GPU.

## What Stage 3 should look like if a lever lands (for the morning debate)

Unchanged from PLAN-phase3 except: the fresh held-out judge should be chosen
NOW (Minkowski functionals) and its scorer validated tests-first BEFORE any
lever run, so the judge stays untouched by tonight's mechanism knowledge.
