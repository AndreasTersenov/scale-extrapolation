# 2026-07-31 — PREREG: STAGE 3 (Option A decision record 4bab9ab; committed
# BEFORE anything runs; REQUIRES reconvene review — the (b) one-shot has no
# pre-delegation; R12, ledger #8–#14, ambiguity = negative throughout)

**The FINAL CONFIGURATION (fixed by the decision record):** committed arm-A
checkpoints + F2 group-averaged sampler + deployment-pure deconvolved oct-1
base (per-checkpoint recipe: T measured from the checkpoint's own
white-base F2 maps; target = trained-octave ring shape rescaled in k/N
units; copula colored-t construction, frozen machinery). Peak claim at
declared resolution σ_s ≥ 0.5 px, both legs. Spacing rows: pooled
multi-stream (mean ± stream sd; never single-stream — probe-1 lesson).

---

## (a) Training-seed ensemble — the robustness table

**Runs:** TWO fresh trainings of the production recipe (gowerstreet,
run_c1t_arms.py arm A verbatim: train-octaves {2,3,4}, 20k steps, caged
selection at octave 2, seeds **1 and 2**; committed seed 0 = the shipped
row). n = 3 rows; seed 0 is the campaign's pick and is labeled as such
(fresh seeds 1–2 are the unbiased draws; survivorship stated in the
table). ~0.6 H100-h (2 MIG trainings) + sampling ≈ 0.1.

**Per seed:** (i) caged pick (own validation curve); (ii) ONE white-base F2
stream → measure T(k) (deployment recipe); (iii) deconvolve the
oct2rescaled target by the seed's own T, calibrate (frozen fit machinery,
tolerance 3%), compute the seed's P-T band exactly as L1″ (multiplicative
model + MC over its own two-point-unavailable T — here sigma_lnT from the
L1″ committed spread, stated as a transferred prior); (iv) THREE
final-config streams (keys/grngs in the table below). Replay identity gate
per job (white stream must reproduce nothing here — fresh weights — so the
gate is the PLUMBING equivalence test, already green, plus seedwise
determinism: same key → same maps, asserted).

**The robustness table rows (per seed, frozen instruments):** marginal
suite hc+e2e octaves 2–4 (standing bars, 3·SE floors); starlet-ℓ1 trained
leg; parity_T; coloring C pooled with the seed's P-T band; smoothed peaks
(σ = 0.5 px, ν ∈ {2.5, 3.0}, pooled 3-stream CIs + per-parent panel —
declared-resolution rule); native peaks descriptive (#10 for ν=1); nn_T
pooled (mean ± stream sd, DESCRIPTIVE — no bar, per the fork decision).

**Branches (rec weights at review; executor):**
| branch | rule | exec |
|---|---|---|
| R-ROBUST | every seed passes: marginals + starlet + parity < 3 + declared-resolution peak rule + C in its P-T band | 55 |
| R-SEED-FRAGILE | ≥1 seed fails ≥1 entry (entries NAMED; meaning conditional per #14 — a single at-bar marginal ≠ a failed method; the table prints what it prints) | 35 |
| gates | infra (one resubmission per job); determinism-gate failure | 10 |

Registered lines: P(all 3 seeds pass the declared-resolution peak rule,
trained leg) = 65; P(all 3 seeds' C lands in their own P-T band) = 70 (the
calibration method's cross-seed test); P(caged picks differ by > 5000
steps across seeds) = 50 (context for the schedule story).

---

## (b) THE ONE-SHOT BLIND EDGE RE-RUN — the headline table
## (RUNS ONLY AFTER RECONVENE REVIEW OF THIS PREREG; EXACTLY ONCE)

**Protocol (Stage D verbatim + final config):** fresh training, gowerstreet,
train-octaves **{3,4}** (octave 2 = the held-out edge), seed **7** (never
used), 20k steps, caged selection at octave 3 (a TRAINED octave, Stage-D
convention), arm A only. Blindness constraints, pre-stated: the
deconvolution target at oct-1 uses the **oct-3 ring shape rescaled two
octaves** in k/N units (octave 2 is held out; the two-octave rescaling is
licensed by the measured near-scale-invariance C = 0.786/0.774/0.785 and
is the deployment-pure choice — the oct2rescaled target would leak the
held-out octave, the flag first raised in the L1′ pre-statement); T from
the run's own white-base maps (post-training, needs no real data). No
edge-octave statistic of ANY kind is consulted before adjudication.

**Adjudicating entries (bars frozen NOW):**
1. Edge marginals (octave 2, hc + e2e, var_slope ≤ max(10%, 3·SE_rel),
   kurtosis ≤ max(15%, 3·SE_rel) — the Stage-D bars verbatim).
2. **Smoothed peaks at σ = 0.5 px** (ν ∈ {2.5, 3.0}, the THREE final-config
   streams pooled — 96 maps — vs the 32 real test tiles, both stacks
   smoothed before counting): declared-resolution rule — both ν ci95
   include 0 AND per-parent panel (per-stream panels averaged at the
   excess level, the corrected convention) not sign-consistent at either ν.
3. **Minkowski judge, FIRST APPLICATION, at the declared resolution**
   (both stacks smoothed σ = 0.5 px before scoring — the judge validates
   the CLAIMED domain): PASS iff T_MF ≤ 3.5. Measured null, frozen here:
   real split-half (20 seeded 16v16 splits, committed stage-D real tiles)
   native T mean 1.43 sd 0.47 max 2.65; smoothed 1.48 / 0.55 / 2.87 — the
   3.5 bar sits 3.7 null-sd above the smoothed null mean. #11 band:
   |T_MF − 3.5| ≤ 0.25 = AT-THE-BAR (reported as such; one-shot → NO
   replication license; the worse reading governs the branch).
   Native-resolution T_MF reported DESCRIPTIVE (expected to inherit the
   native peak excess via V2 at high ν — stated now so it surprises no
   one).
4. Starlet-ℓ1 edge leg (frozen scorer, 10% floors).
5. Supporting descriptive: parity_T, coloring C at oct-1 + the run's P-T
   band, nn_T pooled (mean ± sd), native peaks (#10).

**Branches (executor weights; reconvene adds at review):**
| branch | rule | exec |
|---|---|---|
| B3-PASS | entries 1–4 all pass | 45 |
| B3-PARTIAL | ≥1 of {2,3} fails or lands at-bar while 1 and 4 pass (failed entries NAMED; meanings conditional per #14) | 30 |
| B3-FAIL | entry 1 or 4 fails, or ≥2 adjudicating entries fail → the PLAN's written scope shrink governs the paper | 15 |
| gates | infra (one resubmission per job — INFRA ONLY, no statistical rerun); determinism gate; A1-guard: smoothed-peak ci95 entirely below 0 at both ν = over-smoothing deficit, its own named outcome inside PARTIAL | 10 |

Registered lines: P(entry 2 passes) = 70 (one committed stream passed the
rule at the edge; fresh seed + fresh T adds variance); P(entry 3 passes) =
70 (never applied; smoothed-domain morphology should track the passing
peak/marginal rows, but it is the untouched tier for a reason); P(entry 1
passes) = 85 (Stage-D precedent + n=5 seed evidence that caged picks pass
marginals); P(native T_MF > 3.5, the expected descriptive echo) = 75.

**One-shot discipline:** the sampling/scoring pipeline will be DRY-RUN
end-to-end on the committed stage-D substrate (seed-0 checkpoints, labeled
DRY-RUN, never entering the headline table) before the blind training
launches, so the one shot cannot die to plumbing; the dry-run's outputs are
quarantined in results_p2/stage3_dryrun_* and reported only in the
appendix. All adjudication numbers come from the single blind run.

---

## (c) Isotropy/stationarity declared-domain caveat (method-section draft;
## paper/ remains frozen — text staged here for insertion when WP resumes)

DRAFT: "The generator models per-tile-standardized flat-sky patches. Three
declared-domain restrictions follow. (i) SYMMETRY: the sampler is exactly
equivariant in law under the discrete group D4 (group-averaged sampling;
the base construction is ring-exact under the grid action); full continuous
isotropy is inherited only to the extent the training data carries it, and
statistics sensitive to sub-group anisotropy should be validated per
application. (ii) STATIONARITY: per-tile standardization removes patch-
scale mean/variance modulation; the validated statistics are those
invariant under per-patch standardization on scales up to the tile size
(128 px); super-tile correlations are outside the domain. (iii)
RESOLUTION: peak-count statistics are calibrated at σ_s ≥ 0.5 px (FWHM
≈ 1.2 px) and above, covering standard weak-lensing smoothing practice;
native-resolution counts carry a +14% excess of pixel-scale,
spectrum-independent origin (located mechanism, §Mechanism) and are
declared outside the validated domain. The octave-1 base is calibrated
per deployment by the measured-transfer deconvolution (§Method), a
procedure requiring only generated maps and the trained checkpoint."

---

## Streams, seeds, keys (recorded now; all disjoint from every committed stream)

| leg | training seed | white/T stream | final streams | notes |
|---|---|---|---|---|
| (a) seed 0 | committed | committed F2_gowA (T committed) | l1pp adj1–3 (COMMITTED — the shipped row reuses the L1″ streams) | no new GPU |
| (a) seed 1 | 1 | key 4101 / grng 20260820 | 4111–4113 / 20260821–23 | |
| (a) seed 2 | 2 | key 4201 / grng 20260824 | 4211–4213 / 20260825–27 | |
| (b) blind | 7 | key 4701 / grng 20260830 | 4711–4713 / 20260831–33 | oct3-rescaled target |
| (b) dry-run | 0 (stage-D committed) | committed stage0 gen_A | 4901–4903 / 20260841–43 | quarantined |

Budget: (a) ≈ 0.7, (b) ≈ 0.4 (one {3,4} training ≈ 0.3 + streams), dry-run
≈ 0.02, scoring CPU → ≈ 1.1 of the ~2 allocated (phase ≈ 0.56 spent).
Suites green required before every submission; JOBS.md pre-submission
entries; commit+push per unit.

**STOP — this prereg goes to reconvene review before ANYTHING runs; the
(b) one-shot additionally runs only exactly once, after that review.**
