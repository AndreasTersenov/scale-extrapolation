# 2026-08-05 — PREREG: Phase-3b order set (R47) — band-limit check, the
# oracle-at-oct-1 arm, the phase-texture instrument (committed BEFORE
# anything runs or is read; R12; artifacts → results_p2/oct1fix_*)

Standing refusals carried: no χ-matching, no topology losses; the
texture-transfer corrector is DRAFT-only (R47 order 4). Budget ≤1 H100-h.

## BL — band-limit MF check (CPU, committed maps, runs FIRST)

Construction: band-limit(f) = reconstruct with the oct-1 detail triple
zeroed (frozen DWT machinery); applied IDENTICALLY to gen and real stacks;
declared-resolution smoothing convention preserved (0.5 px on both, after
band-limiting). Legs: trained (l1pp adj pooled 96 vs gow real 32) and
blind (stage3_blind pooled 96 vs stage-D real 32). Frozen judge, frozen
3.5 bar, #11 band 0.25. Also scored, descriptive: the FULL-BAND
trained-leg T_MF (the judge's first reading on the trained leg — the
baseline O must beat, and BL's own reference).
Lines: P(both legs pass at the frozen bar): rec 80, **exec 70** (the
oct-2 single-transplant read T = 10.2 — confounded by misalignment, but a
genuine mid-octave residual is not excluded; ambiguity = negative if
at-bar). Meaning, pre-stated: PASS → topology joins the declared-domain
family with a crisp band statement ("valid with the finest octave
discarded") and no mid-octave residual hides behind oct-1.

## O — the ORACLE-AT-OCT-1 arm (labeled ORACLE; the direct solve where
## fine-scale data exists)

Training: run_c1t_arms.py, gowerstreet, **train-octaves {1,2,3,4}**,
everything else verbatim (arm A, 20k steps, batch 32, caged selection at
octave 2 — a trained octave, unchanged; seed 0, the production default,
stated). ckpt-dir ckpt_oct1fix_oracle; MIG ≤2:45 (~0.65 H100-h — the
oct-1 steps are ~4× heavier). Sampling: final config (F2 group-averaged);
deconvolution recalibrated per the standard per-checkpoint recipe with ONE
stated difference: the oct-1 target is the OCT-1-MEASURED training shape —
legitimate here because octave 1 IS a trained octave in this arm (no
deployment leak; the ORACLE label is exactly this); T from the arm's own
white stream. Streams: white 5201/20260870; final 5211-3/20260871-3;
determinism gate asserted.

Scoring (A5 order: gates → C/P-T context → entries): MF declared
(adjudicating vs 3.5) + native (descriptive); native peaks (pooled 96 vs
gow real, ν 2.5/3.0, vs the committed A3 reference +15.290/+12.474%);
must-not-regress: marginal suite hc+e2e octaves 1–4 (standing bars,
octave 1 scored with the same formula and NOTED as a new trained octave),
starlet trained leg (frozen scorer, leg_idx 15), parity_T < 3, identity/
determinism. The phase-texture instrument (I) scored on the O maps as
texture verification (descriptive).

| branch | rule | rec | exec |
|---|---|---|---|
| O-FIXES | MF declared ≤ 3.5 AND both native excesses < half the committed values; no regression | 45 | 45 |
| O-PARTIAL | exactly one of the two targets; no regression | 25 | 28 |
| O-NULL | neither target (an architecture/capacity finding, NOT a diagnosis failure) | 20 | 17 |
| O-REGRESSED | any must-not-regress failure | 5 | 5 |
| gates | infra (one resubmission per job); determinism | 5 | 5 |

Executor reasoning, registered: trained-octave texture has passed every
audit tier it was scored on; turning oct-1 from extrapolated to
interpolated should carry the texture with it (and #16 argues for the
concentrated branch); PARTIAL slightly above rec because the two targets
can split — MF and native counts need not move together (the three
dissociations cut both ways). Sub-lines: P(MF declared ≤ 3.5) = 60;
P(native excesses both < half) = 55; P(recalibrated C in its own P-T
band — 6th consecutive) = 80.

## I — the oct-1 phase-texture instrument (tests-first, synthetics only,
## before real data)

Two statistics on the oct-1 CONTRIBUTION field u = f − band-limit(f)
(standardized per map; both D4-invariant by construction):
1. **Fragmentation profile:** at ν_u ∈ {0.5, 1, 1.5}: component count,
   hole count, χ, and small-component fraction (area ≤ 4 px) of the
   excursion set of u — 4-connectivity, the judge's family.
2. **Alignment statistic:** A = corr(u², |∇ band-limit(f)|²) per map
   (does fine texture organize on coarse structure?) — the T1 confound's
   mechanism, made into a measurement.
Validation tests (tests/, env.sh): (a) fragmentation discriminates
synthetic organized-vs-scrambled u (phase-randomized u at fixed spectrum —
the null the corrector would target) at z > 5; (b) A discriminates
aligned vs independent planted u; (c) D4 invariance (machine precision);
(d) SE scaling ~√2. Then scored, descriptive: real vs blind e2e vs
trained l1pp vs (when landed) O maps.

## Sequencing

Prereg commit → BL (CPU) → O training submission → I tests + instrument on
committed maps (while O trains) → O sampling (white → fit → final) →
O scoring → readout appended here → **STOP.**
Results: BL PENDING · O PENDING · I PENDING.
