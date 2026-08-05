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

---

## READOUT (appended; verbatim from topo_diag_verdict.json +
## topo_diag_t2_streams.json; job 18402965, determinism gate exact)

### T1 — mechanical branch: **T1-BOTH-FAIL** (hc 47.27, e2e 6.35, both
### fail-side) — WITH a construct confound, disclosed

The hc leg's number is not what the pre-stated meaning assumed: T_hc =
47.27 (native 67.20; stage-D context 27.31) — an order WORSE than e2e,
with the mechanism visible in T3's map-level row: the hc maps' skewness
COLLAPSES (real +2.32 → hc +0.83, z = −8.5, while e2e reads −2.4).
**Instrument lesson (ledger material): the hc-MAP construct is confounded
by composition misalignment** — each octave's details were sampled against
the REAL coarse but composed onto intermediate coarse fields that differ
from it, scrambling the cross-scale alignment that carries map skewness
and mid-level connectivity. The prereg's claim that the construct
"isolates conditional-detail quality" was WRONG; per-octave hc STATISTICS
(the campaign's convention) remain valid, hc MAPS do not follow from them.
Consequence, stated honestly: the deployment-split question T1 posed is
UNANSWERED by this probe (the branch's pre-stated meaning transfers to T2,
which answers it cleanly); the only deployment-mode fact in evidence is
that e2e — which IS octave-4-conditioned deployment — fails at 6.35.

### T2 — THE DIAGNOSIS: the fragmentation is an octave-1 detail-texture
### property

| substitution | T_MF | χ(0) shift fraction |
|---|---|---|
| single oct-1 | 11.30 | **+1.15** |
| single oct-2 | 10.19 | −0.91 |
| single oct-3 | 8.50 | −0.40 |
| single oct-4 | 6.65 | −0.25 |
| cumulative ≤2 | 16.32 | +0.49 |
| cumulative ≤4 (≡ full gen) | 5.40 | +1.00 (sanity) |

Substituting ONLY the finest-octave generated details into otherwise-REAL
pyramids reproduces MORE than the full fragmentation shift — stream-stable
across all three blind streams (+1.15 / +1.13 / +1.26
<!-- src: topo_diag_t2_streams.json -->). Misalignment alone does NOT
produce the signature: misaligned oct-2/3/4 substitutions push χ(0) the
OPPOSITE way (which also explains cumulative-≤2's cancellation, 0.49).
The mid-level topology defect lives in the oct-1 generated detail texture
itself — the same locus as the cured coloring defect, on the axis the
spectrum lever measurably does not reach (dissociation 3). Lines: single-
octave ≥ half FIRED (rec 40 / exec 35 — the minority side on both
columns, noted for the calibration appendix); executor's cumulative-≤2 ≥
80% line LOSES (0.49 — the oct-2 cancellation is the reason, now
measured).

### T3 — the skewness hypothesis: REFUTED as operationalized

Detail-coefficient skewness: all |z| ≤ 1.3 at every octave for every
generated set — the registered parity argument confirmed (signed detail
skew vanishes in law for both stacks; real ≈ 0 by statistical parity,
generated ≈ 0 by the exact D4 sampler). Reconvene line (55) does not
fire; executor counter-line (20) on the right side. Executor's map-level
companion line ALSO does not fire: e2e map-skew deficit z = −2.4 ( < 3σ;
the 55 line loses — recorded). Descriptive: the −2.4σ map-level deficit
is suggestive texture for the odd-cross-moment reading (coarse–detail²
couplings), stated as a readout note only. The drafted skewed-base
lever's premise is measured ABSENT at the detail-marginal level.

### Synthesis for the lever menu (measured directions; the choice is the
### adjudication's)

The defect: octave-1 generated detail TEXTURE — phase/higher-order, not
spectrum (exhausted, 5/5-validated calibration unaffected), not detail
skewness (refuted), not composition misalignment (wrong sign), expressed
even inside fully-real pyramids. Any lever needs an oct-1 phase/topology
instrument first (component-size / mid-level connectivity of detail
contributions — the F20-style decomposition scoped to oct-1 is the
natural candidate); loss-side interventions would target oct-1 only;
χ-matching and topology losses remain refused pending that instrument.
Budget: one MIG job ≈ 0.02 H100-h; everything else CPU on committed maps.

**STOP — reconvene adjudication; the lever menu is chosen there.**
