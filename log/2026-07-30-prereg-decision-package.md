# 2026-07-30 — PREREG: R41 decision package (committed BEFORE any probe
# runs; instantiates R41 order 3; zero training, one negligible GPU stream)

## Probe 1 — nn stream-scatter from disk (CPU, committed maps only)

nn_T (frozen instrument, K=174, phase-A edges) per stream on all ten
trained-leg streams: WHITE = committed F2_gowA_e2e + rep1 + rep2; L1′ =
adj1–3 (wrong-direction coloring); L1″ = adj1–3 (deconvolved); ORACLE = the
oct1-measured L1″ stream. **Mechanical read (pre-registered):** within-
config stream sd s = pooled over the white and L1″ triplets
(sqrt of mean variance); difference D = mean(nn_T, white) − mean(nn_T,
L1″-adj); EXCEEDS iff |D| ≥ 2·s·√(2/3) (the 2σ difference-of-means bar).
**Executor line: P(EXCEEDS) = 55** — the 4.045→3.33→2.90 single-stream
sequence suggests real movement, but each config had n=1 until now and the
L1″ C per-stream scatter (0.019) warns that stream variance is not small.
Meaning, pre-stated: EXCEEDS → the spacing frontier responds to coloring
even though counts do not (mechanism finding + a reason the final config
ships the deconvolved base); NOT-EXCEEDS → the nn sequence was stream
noise; the frontier stays where R36 left it.

## Probe 2 — resolution-scoped peak audit (CPU on committed maps + ONE
## negligible GPU stream)

Missing substrate, generated per the deployment recipe (the only GPU;
~0.02 H100-h): the DECONVOLVED EDGE stream — stage-D A@9000, F2 sampling,
base at oct-1 = deconvolved with T measured from the COMMITTED stage-D
white-base maps (stage0_p3_gen gen_A oct-1 spectra — the per-checkpoint
deployment recipe, exactly as a practitioner would) and the oct2rescaled
target (its stage-D information caveat is already on record from the L1′
pre-statement; this leg is decision-support, not adjudication). Key 3701 /
grng 20260816. Runtime replay gate, asserted: white base with key 3101 /
grng 20260729 must reproduce the committed stage0_p3_gen gen_A under the
replay criterion.

Sweep: Gaussian σ_s ∈ {0 (anchor), 0.25, 0.5, 0.75, 1.0} px applied to
both stacks before counting; configs × legs: trained-white (pooled 96),
trained-L1″ (pooled 96), trained-oracle (32), edge-white (committed
stage0 gen_A, 32), edge-deconvolved (32, new); each vs its real; both ν ∈
{2.5, 3.0}; 5000-boot CIs; per-parent panel (blocks [(0,10),(10,21),
(21,32)]) at every σ_s. **Mechanical reads (pre-registered):** (a) zero
crossing = linear interpolation of excess(σ_s) where the sign changes,
per config × leg × ν; (b) "declared-resolution pass at σ_s" = both ν
ci95 include 0 at that σ_s AND the per-parent panel is not sign-consistent
at either ν; the DECLARED RESOLUTION candidate = the smallest grid σ_s
where ALL FIVE config×leg combinations pass. **Executor lines:** P(white
trained-leg zero crossing lies in (0.25, 0.75) px) = 70 (N1: +14.5% → −1.7%
between 0 and 0.5 px); P(a declared-resolution pass exists at σ_s ≤ 0.5 px
for all five combos) = 55 (the edge legs are unmeasured under smoothing;
the N1 1-px overshoot to −8/−10% warns the pass window may be narrow or
config-dependent).

## Then

The one-page scope fork memo (R41 order 4): Option A (declared-resolution
close + Stage 3) vs Option B (training-side redesign), measured costs,
executor recommendation + a slot for the reconvene's, recorded separately.
**STOP — the fork belongs to Andreas.** Artifacts →
results_p2/l1pp_decision_*.
