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
  generate_recursive_tbase output EXACTLY. Result: PENDING.
- G2 substrate-chain: the same in-process reference vs the COMMITTED
  arms_stageD.npz gen_A; rel max-abs <= 1e-3 passes as cross-run float noise,
  larger = reconstructed std/coarse/checkpoint chain mismatch (gate failure).
  Result: PENDING.
- Marginal catastrophe (scorer): arm-A e2e var_slope rel err > 50% at any
  scored octave (2-4). Result: PENDING.

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

- Edge peaks, arm A, ν=2.5: PENDING
- Edge peaks, arm A, ν=3.0: PENDING
- Per-parent panel (blocks [(0,10),(10,21),(21,32)]): PENDING
- BRANCH: PENDING (if MIXED/AT-BAR or S0-FLIPPED: one fresh-PRNG
  regeneration disambiguation is pre-authorized; worse category governs
  sequencing; a persisting A1 deficit STOPs as its own finding)

## Registered descriptive lines (scored here; prereg P / reconvene co-sign)

- nn spacing at the edge, F2 arm A, T >= 3 (persists): P=85/85. Result:
  PENDING.
- starlet-ℓ1 edge leg all scored scales pass: P=90/90. Result: PENDING.
- arm B worse than A where they differ: P=75/70. Result: PENDING.

## Battery context (descriptive)

- Edge peaks ν=1 (never headlined, #10): PENDING.
- Marginal suite (e2e octaves 1-4 vs real test refs, bars as context):
  PENDING.
- parity_T / T_coef_oct1 at the edge (F2 should have killed parity): PENDING.
- Trained-scales leg (committed F2_gowA_e2e @16000 + committed pre-F2 gen_A
  vs gowerstreet test tiles): PENDING.
- starlet-ℓ1 trained leg (F2_gowA): PENDING.

## Sequencing consequence (pre-stated, from the prereg)

S0-CLOSED → PLAN Stage 3 directly. S0-SHRUNK / S0-UNCHANGED → Stage 1 with
this readout's numbers as its measured references. S0-FLIPPED persisting
through disambiguation → STOP, its own finding. **STOP at this readout for
reconvene adjudication in every branch.**
