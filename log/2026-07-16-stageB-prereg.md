# 2026-07-16 — Stage B prereg (B1 dependence range & shape; B2 crops inventory)

Committed BEFORE any gowerstreet number is computed (Gate A passed; estimators
validated tests-first in tests_p2/{test_depmeasure,test_shape}.py, commits
a5c6bf3 + 02477b5). No gate here per the orders — B is measurement; report and
continue. Pre-registered rules and predictions below are binding for the readout's
verdict lines.

## B1 protocol (pinned)

**Estimand:** V(r) = held-out residual variance of the best in-class predictor of
the octave-j detail coefficient from same-level coarse values within radius r
(per-band fits, band-averaged; every V(r) upper-bounds the true conditional
variance). Two channels: 'w' (the coefficient — mean channel; ridge on raw pixels =
PRIMARY, kNN on annulus summaries = secondary) and 'absw' (|coefficient| — the
variance/amplitude channel, ridge on [raw, squared] features). Fit even / eval odd
fields; SE by eval-field batch means.

**Data:** gowerstreet = 330 tiles (data_cache/tiles_pnull.npz, non-periodic,
interior margins); sandbox control = 256 parents (periodic). Octave r-grids:
oct1 {0,1,2,3,4,6,8,12}; oct2 {0,1,2,3,4,6,8}; oct3 {0,1,2,3,4}; oct4 {0,1,2}
(oct4 descriptive — thin interior). Positions/field ≤768 (oct1) / ≤1024 (oct2) /
all interior (oct3-4); fixed seeds.

**r\* rule (mechanical):** smallest r on the grid with
V(r) ≤ V(r_max) + 3·SE(V(r_max)), computed on the ridge curve per channel.
**P-B1a fires iff** on gowerstreet, w-channel: r\* ≤ 6 at octaves 1 AND 2
(saturation strictly inside the grid). absw-channel r\* reported alongside; a
w-vs-absw disagreement is a C2-design finding, not a bar failure.

**Shape test (estimand amended from the orders' literal wording — measured reason):**
the ordered disk-vs-oriented comparison is SHAPE-CONFOUNDED (elongation at matched
area loses ~0.2% under exact isotropy — measured during estimator development, kept
as a test). PRIMARY estimand: Δ_align = V_misaligned − V_aligned (same elongated
mask, aspect 4:1, area = disk r=4 exactly; class assignment rotated 90°) — exact
null under isotropy (exchangeability), validated positive control (stochastic
row-amplitude × oriented texture, z=3.8). The ordered disk-vs-aligned number
(Δ_disk) is reported as descriptive with the confound stated. Channels: absw
(PRIMARY — validated with power), w (reported; measured power limitation: mean
channel saturates in the compact core on GRF-like fields). Octave 2 primary,
octave 1 secondary; sandbox = isotropy control (expect NULL).
**P-B1b fires iff** gowerstreet octave 2, absw channel: Δ_align > 3·SE.

## B2 protocol (pinned)

30 gowerstreet parent maps (scaledrift.data.iter_parent_maps, 'gowerstreet-train',
seed 0), crops via scaledrift.data.tile_map at strides {128 (disjoint baseline),
64, 32}. Per-crop summaries at octave 2: var_slope and detail_std (each crop scored
alone). Correlation-adjusted effective count: N_eff = N² / Σ_ij ρ(Δ_ij) with ρ(Δ)
the measured cross-crop correlation at grid offset Δ pooled over parents (parents
independent; ρ set to 0 across parents). Report per stride × per summary; the MIN
over summaries is the inventory number. On top of the exact 8× D4 group. Table
only, no training decision tonight: C1 runs WITHOUT crops per the orders'
single-variable discipline (crops enter only if B2 says law-preserving at stride —
adjudicated by the morning reconvene, not tonight).

## Predictions (registered before computation)

- **P-B1a** (standing 70%) — executor: **85%** (WC-RG short-range prior; the
  phase-1 U-Net's receptive field was never the binding constraint; alpha≈2-like
  spectra put most mean-channel information in the nearest coarse pixels).
- **P-B1b** (standing 50%) — executor: **35%** (aligned-vs-misaligned at matched
  area needs variance transported along structures at mask scale; the measured
  shape-effect floor is ~0.5–1% of V and lensing kappa at 32² coarse may be too
  isotropized; the sandbox control must be NULL — executor 90% on the control null).
- Branch (standing ≤15%): no saturation below map scale at oct 1–2 → C2 dies AND
  the phase-1 memorization account needs revisiting → flagged to reconvene.
- **B2 expectation** (descriptive): stride-64 N_eff/crop-count ≥ 0.5 (worth it),
  stride-32 strongly diminishing (N_eff/count < 0.3) — 60%.

## Job

One CPU SLURM job (def-lplevass, 4c/24G/1:30): scripts_p2/run_stageB.py →
results_p2/stageB1_curves.json, stageB1_shape.json, stageB2_crops.json + figure
results_p2/stageB.png. Sandbox control computed in the same job (periodic path).
