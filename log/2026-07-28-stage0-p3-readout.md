# 2026-07-28 — PHASE-3 STAGE-0 READOUT: the full audit battery under the
# FINAL SHIPPED configuration (prereg 62db5f0; APPROVED + A1 per
# log/2026-07-28-reconvene-stage0-review.md)

Pre-statement, committed BEFORE the job's results are read (the M5/PENDING
pattern; R12 throughout). Job 17621159 (MIG b1, 59-min cap): F2 group-averaged
e2e from the committed Stage-D checkpoints (arm A @9000 ADJUDICATING, arm B
@7500 descriptive; picks asserted at runtime) on the frozen Stage-D test tiles.
Zero training. Machinery committed at 812723d with the branch logic
tests-first (7 cases, incl. A1/at-bar/mixed/panel-cascade — the pcure lesson
applied preemptively).

## Gates (pre-stated in the sampler's docstring)

- G1 identity (binding, the F2 convention): all-identity group assignment at
  A@9000 with the original key PRNGKey(1) == in-process
  generate_recursive_tbase output EXACTLY. Result: **PASS, exact (0.0)**
  (job 17622970 log).
- G2 substrate-chain: the same in-process reference vs the COMMITTED
  arms_stageD.npz gen_A; corrected criterion per the execution note below.
  Result: **PASS** — corr_min 0.9999999843, ratio_mean 1.000003 (maxdev
  1.011e-05), rel max-abs 1.019e-02. The chain (checkpoints, std
  reconstruction, coarse) reproduces the committed run's fields exactly up
  to recursively amplified float noise — the first job's firing was the
  instrument, not the substrate.
- Marginal catastrophe (scorer): arm-A e2e var_slope rel err > 50% at any
  scored octave (2-4). Result: **did not fire** (arm-A max 23.7%, oct3).

### Execution note (2026-07-28, before the resubmission ran): G2 fired on
### job 17621159 — instrument mis-design, disclosed; corrected criterion
### pre-stated here

Job 17621159 FAILED at G2: rel max-abs 9.881e-3 vs the 1e-3 tolerance
(verbatim from the job log). G1 never ran (script order: G2 first). The
1e-3 threshold was an instrument-design error of mine: it was calibrated to
per-op float noise, but the comparison is made AFTER 4 octaves of recursive
generation (80-step ODE integrations, each octave conditioning the next), so
cross-run XLA/TF32 algorithm-selection noise is amplified multiplicatively —
the same genre as the Phase-A NaN-bin fixes (instrument corrected on
validation evidence, before any model quantity is scored; no adjudicating
number existed yet). A ~1e-2 post-recursion diff is consistent BOTH with
amplified float noise AND with a genuine ~1% chain mismatch; the original
gate recorded nothing that separates them. Corrected G2, PRE-STATED before
the resubmission and before its diagnostics were seen (thresholds chosen
blind): per-field Pearson corr(ref, committed) min >= 0.99 (same fields) AND
per-field amplitude-ratio mean within 5e-3 of 1 (same std chain) AND rel
max-abs <= 5e-2 (sanity ceiling); all three asserted, all recorded. A
failure of the corrected criterion is a real substrate-chain mismatch ->
STOP, gates branch. This consumes the prereg's ONE infra resubmission.
Resubmitted job: 17622970.

## Adjudication (mechanical; the prereg branch table + A1; references
## +13.13%±3.10 @ν2.5, +14.62%±3.08 @ν3.0, read at runtime from
## audit_peak_ci.json)

Weights (executor / reconvene): CLOSED 22/12 · SHRUNK 43/38 · UNCHANGED
25/30 · FLIPPED —/10 · gates 10/10.

All verbatim from results_p2/stage0_p3_verdict.json (job 17622970 +
deterministic scoring, frozen instruments):

- Edge peaks, arm A, ν=2.5: **+12.45% ±3.09**, ci95 [+6.55%, +18.86%]
  (ref +13.13% ±3.10; Δ-reduction 0.16σ; not at-bar).
- Edge peaks, arm A, ν=3.0: **+12.94% ±3.00**, ci95 [+7.38%, +19.30%]
  (ref +14.62% ±3.08; Δ-reduction 0.39σ; not at-bar).
- Per-parent panel: ν=2.5 [+8%, +12%, +18%], ν=3.0 [+20%, +15%, +5%] —
  sign-consistent positive at both thresholds (parent-robust excess).
- **BRANCH: S0-UNCHANGED** — mechanical, nothing at the bar, thresholds
  agree, no A1 flip, no disambiguation license. The pre-stated meaning
  applies verbatim: the excess never was lattice-driven; symmetric
  graininess or genuine joint miscalibration → Stage 1 unchanged, Stage 2's
  colored base is the live lever.

The cleanest statement of the result: D4 group-averaging moved the peak
counts by a fraction of an SE in BOTH arms (pre-F2 committed, from
audit_peak_ci.json: A +24.7/+13.1/+14.6% at ν=1/2.5/3.0, B +37.5/−9.4/−17.3%;
F2 today: A +22.9/+12.4/+12.9%, B +38.0/−8.7/−17.7%) — while the SAME F2 maps
show the lattice channel dead (arm A parity_T = 1.00, T_coef_oct1 = 2.18,
both < 3 at the edge). The peak-count excess and the parity/lattice defect
are fully dissociated: the cure worked on its target and the excess is
orthogonal to it. This is the reconvene's review reasoning ("the
parity→edge-peak-excess causal link was never established") landing as
measurement; the reconvene's UNCHANGED 30 was better-calibrated than the
executor's 25.

## Registered descriptive lines (scored here; prereg P / reconvene co-sign)

- nn spacing at the edge, F2 arm A, T >= 3 (persists): P=85/85. Result:
  **nn_T = 2.57 — the line LOSES** (the 15% side happened, for both weight
  columns). Context, stated carefully: this is the FIRST nn reading at the
  stage-D edge (no pre-F2 edge value exists); the named tier-3 frontier
  values (T ≈ 3.5–4.1) were measured on sandbox and gowerstreet TRAINED
  legs and stand untouched — arm B's edge nn_T = 4.76 shows the instrument
  has power here. Calibration appendix: another registered line lost on its
  minority side.
- starlet-ℓ1 edge leg all scored scales pass: P=90/90. Result: **HOLDS** —
  edge leg gen_A PASS, gen_B PASS; trained leg (F2_gowA) PASS
  (starlet_l1_stage0_{edge,trained}.json, frozen scorer, fresh seed streams
  leg_idx 6/7).
- arm B worse than A where they differ: P=75/70. Result: **LOSES on the
  strict quantifier** — B is worse at ν=1.0 (+38.0% vs +22.9%), worse at
  ν=3.0 (|−17.7%| vs |+12.9%|), worse on nn (4.76 vs 2.57), and
  catastrophically worse at oct1 marginals (var_slope 70.7% rel); but at
  ν=2.5 B's magnitude is SMALLER (|−8.7%| vs |+12.4%|, sign-flipped) and
  its oct3/4 marginals are cleaner. "Wherever they differ" fails;
  the reconvene's shaded-down 70 ("the group average may compress
  inter-arm differences") was directionally right for the wrong entry — B's
  pathologies are essentially UNCHANGED by F2 (its sign-flip and ν=1
  explosion predate the symmetrization, see the committed B row above).

## Battery context (descriptive)

- Edge peaks ν=1 (never headlined, #10): A +22.9% ±3.9, B +38.0% ±4.3.
  The split-half null's ν=1 floor is +19.3% ±6.6 (committed) — ν=1 stays
  un-headlined for exactly this reason.
- Marginal suite (e2e octaves 1-4, bars as context, no bar changes per the
  PLAN): arm A — oct2 (the EDGE) passes cleanly (var_slope 8.5%, kurtosis
  5.2% rel; the committed pre-F2 edge e2e was 4.7%/3.2%); oct1/oct4 pass;
  oct3 var_slope 23.7% vs its 22.2% noise-floored bar — inside the #11
  ±10%-of-bar band, reported AT-THE-BAR, not rounded either way (context
  entry; no committed pre-F2 oct3 e2e row exists to compare). Arm B — oct1
  fails hard (var_slope 70.7%, kurtosis 59.2%), oct2-4 pass; oct1 is
  beyond-edge and descriptive.
- parity_T / T_coef_oct1 at the edge, arm A: **1.00 / 2.18** — both below 3.
  The F2 sampler's target defect is dead on the stage-D substrate.
- Trained-scales leg (gowerstreet test tiles, committed maps only):
  F2_gowA +14.5% ±3.1 (ci95 [+8.6, +20.8]) at ν=2.5, +11.1% ±2.8
  (ci95 [+5.8, +16.6]) at ν=3.0; committed pre-F2 gen_A +14.1% ±3.1 /
  +15.3% ±2.9. The REAL-field excess at TRAINED scales is the same size as
  at the edge, and F2 barely moves it — while the committed SANDBOX trained
  legs are significantly NEGATIVE (repl64 −6.79%/−9.37%, sandbox32
  −4.93%/−9.07%; the A1 motivation numbers). Texture, stated as texture
  (different fields, different models): the sign of the count bias tracks
  the FIELD, not the scale. On gowerstreet the "edge excess" is not an
  extrapolation defect — it is a field-level joint-texture mismatch already
  present at trained scales. Stage 1's mechanism question should be posed
  accordingly, and it sharpens the S0-UNCHANGED row's pre-stated meaning:
  the live lever is the base/texture (Stage 2), not the edge.
- starlet-ℓ1 trained leg (F2_gowA): PASS (all scored scales).

## Sequencing consequence (pre-stated, from the prereg)

S0-UNCHANGED → PLAN Stage 1 (mechanism profiling) with this readout's
numbers as its measured references — and per the branch row's pre-stated
meaning, Stage 2's colored heavy-tailed base is the live lever. The
trained-scales texture above (excess present at trained scales on the real
field, sign flipping with the field) belongs in Stage 1's framing. The
marginal/starlet/spacing rows become the method paper's final-configuration
audit table rows, as pre-stated.

## Execution accounting

One MIG job spent twice (17621159 gate-failed at the mis-set G2, 17622970
completed — the one licensed resubmission, disclosed above); scoring CPU
in-session; zero training. GPU cost ≈ 0.2 H100-h against the PLAN's 10 h
budget. Artifacts: stage0_p3_gen.npz, stage0_p3_sample.json,
stage0_p3_verdict.json, starlet_l1_stage0_{edge,trained}.json. Machinery
committed pre-run (812723d, 0ea7a0a); branch logic tests-first (7 cases,
green in the env.sh gate).

**STOP at this readout for reconvene adjudication (R-order: adjudication
before any sequencing).**
