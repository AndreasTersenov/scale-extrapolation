# 2026-08-05 — PREREG: Phase-3b topology diagnosis (R46; committed BEFORE
# any probe computes; reconvene lines in R46, executor lines here)

Target: the E3 defect — T_MF 6.35 at declared resolution, V2/χ at ν≈0
(fragmentation: island excess + hole deficit, deeper underdense trough,
survives 0.5-px smoothing). Refused in advance (R46, carried): χ-matching
post-processing; topology-aware losses before diagnosis. Zero GPU except
the licensed T1 hc generation (≤0.05 H100-h). Artifacts →
results_p2/topo_diag_*.

## T1 — the deployment split (decisive)

**hc-map construction, pre-stated:** for real tile i, take its Haar
pyramid {c4, d4..d1}; at each octave j = 4..1 sample d_j^gen conditioned
on the REAL coarse c_j^real (final-config sampling: F2 group-averaged;
oct-1 base = the leg's committed deconvolved filter; std per the leg's
training convention); reconstruct from {c4^real, d4^gen..d1^gen}. Every
conditioning is real — the construct isolates conditional-detail quality
from recursion compounding. NO committed hc-map stacks exist (hc STATS
only) → the licensed generation: blind config (ckpt_stage3_blind @10500,
stage3_blind_filter, 3 streams, keys 5101-3 / grngs 20260850-2) +
stage-D-lineage context (ckpt_stageD A@9000, l1pp_decision_edge_filter,
1 stream, key 5105 / grng 20260853). Replay-style gate N/A (no committed
hc reference); the determinism gate (same key → same maps) asserted
in-job.

**Scoring:** frozen MF judge, declared resolution (0.5-px both stacks),
vs the frozen 3.5 bar, #11 band 0.25; native descriptive. e2e comparison
= the committed stage3_blind_final pooled stack (its T = 6.35 stands; no
rescore needed, but the scorer re-emits it from the same code path as a
consistency line).

| branch | rec | exec |
|---|---|---|
| T1-CONDITIONED-PASS (hc ≤ 3.5, e2e fails) | 25 | 30 |
| T1-BOTH-FAIL | 55 | 50 |
| T1-MIXED / at-bar (worse governs; ONE fresh-stream disambiguation licensed) | 15 | 15 |
| gates (infra, one resubmission) | 5 | 5 |

Executor reasoning, registered: hc marginals have always been cleaner than
e2e (compounding is real), and the N1 transplants showed single-octave gen
details REDUCE structure counts inside real pyramids — but the
fragmentation texture lives in the details' own fine structure, which real
conditioning does not repair. Slight shift toward CONDITIONED-PASS vs the
reconvene, still BOTH-FAIL modal.

## T2 — octave attribution, χ-scored

Substrate: the blind final1 stream (32 maps, paired with real by
conditioning). Hybrids from the FROZEN transplant instrument: single-octave
(gen map i's octave-j details into real tile i's pyramid, j = 1..4) and
CUMULATIVE (gen details at octaves ≤ j, real above). Scored: judge_T
(declared res) + the χ(0) shift fraction (hybrid vs the full-gen shift on
the same 32-map pairing). Lines: P(a single octave reproduces ≥ half the
full-gen χ(0) shift): rec 40, **exec 35** (the peak transplants were
composition-dominated); NEW executor line: P(the cumulative-through-oct-2
hybrid reproduces ≥ 80% of the shift) = **60** (the fragmentation should
live in the fine octaves if it is a detail property at all).

## T3 — the skewness hypothesis (reconvene hypothesis, house rule 1)

New scorer (tests-first, per R46): per-octave detail-coefficient skewness
(channel-pooled, per-map → stack mean ± tile-bootstrap SE) + map-level
skewness at declared resolution. Line (rec): P(generated details
under-skewed at ≥2 of octaves 2–4 at ≥3σ) = 55. **Executor counter-line,
registered with its argument: 20.** Signed detail-coefficient skewness
vanishes in law for BOTH stacks — the real ensemble by statistical parity
(a flip maps each detail channel to minus itself in law), the generated
ensemble by the exact D4 group-averaged sampler — so the signed-skew probe
should read ≈0/≈0 and the hypothesis, as operationalized on details,
should NULL. The map-level variant is where the χ asymmetry's odd-moment
signature can live: **executor companion line: P(map-level skewness
deficit, gen vs real, ≥3σ at declared resolution) = 55.** If T3 nulls on
details but fires at map level, the mechanism points at ODD CROSS-MOMENTS
(coarse–detail-squared type couplings) rather than detail marginals — a
readout note, not a new probe (scope discipline).

## Sequencing and discipline

Prereg commit → T3 tests-first → one MIG job (hc streams) → CPU scoring
T1/T2/T3 → readout appended here (PENDING slots below) → **STOP for
reconvene adjudication (the lever menu is chosen there).**

Results: T1 PENDING · T2 PENDING · T3 PENDING.
