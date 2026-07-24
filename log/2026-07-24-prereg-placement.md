# 2026-07-24 — PREREG (DRAFT FOR RECONVENE REVIEW — R29 STOP; NO RUNS SUBMITTED)
# The placement experiment: is joint/morphological structure the moment ladder's next rung?

Supersedes the design section of SPEC-placement-test.md per its own "DESIGN
OPEN until Phase 0" clause, incorporating the takeover audit §3.6 upgrades
(accepted R28) and the R27 rulings. Reconvene registers its weights against
this document at review; upon approval + reference fill-in (Phase A), this
freezes as usual. Executor weights below are registered now.

## The question (R27 wording)

The cured generator's one open audit failure is joint/morphological: a
peak-count excess at fixed marginal calibration on the real field
(+13.1%±3.1% / +14.6%±3.1% at the deployment edge, parent-robust), and — the
audit's sign-flip finding — a significant peak DEFICIT on the exact-truth
sandbox (arm A −9.4%±1.7% at ν=3 on 64 fresh fields, nulls clean)
<!-- src: results_p2/audit_peak_ci.json -->. Literal placement is untested;
this experiment's instruments make it measured. Hypotheses:

- **H-data:** joint structure is the next data-starved rung (currency:
  independent parents). Prediction: placement error shrinks causally with
  parent count, as dispersion (rung 2) and tails (rung 4) did.
- **H-arch:** conditional regression through this sampler cannot express the
  needed joint law at any reachable data size. Prediction: flat in parent
  count. Fallback route if this fires: the unconditional-prior /
  posterior-sampling reframe (solution-space note item 5).
- **H-selection (audit upgrade, first-class):** the checkpoint cage
  optimizes marginal statistics and may select against joint structure.
  Prediction: placement error at the caged checkpoint materially exceeds the
  checkpoint-curve minimum at the same data size.

**THE TRANSFER CLAUSE (R27-4.2, verbatim, binding on every readout sentence):**
PL-DATA licenses "joint structure is data-limited in this model class,"
never "more parents cure the gowerstreet excess"; the real-field echo leg is
load-bearing and its branch meanings say so. (The sandbox and real-field
residuals differ in SIGN; the sandbox adjudicates the mechanism class, the
echo anchors the real-field relevance.)

## Phase A — instruments (tests-first; no model runs; sets the measured references)

Arena: the exact-truth lognormal sandbox (closed-form conditional ensembles;
machinery: gen_sandbox_ensembles.py — 256 parents × 64 exact conditional
redraws exists as the pattern).

Candidate instruments (each computed on 128² fields at the binding octave's
resolution; implementations position-pure by construction):
1. **Environment-conditioned peak rate** — peaks per coarse-quantile bin,
   NORMALIZED by the map's total peak count (the "wrong place vs wrong
   texture" discriminator; rate profile across 8 coarse bins, L2 error vs
   truth ensembles).
2. **Peak nearest-neighbor distance distribution** — at COUNT-MATCHED
   thresholds (per-map adaptive threshold at the truth ensemble's peak-count
   percentile, so the statistic is orthogonal to residual marginal/count
   miscalibration); W1 distance vs truth.
3. **Peak two-point function at small separations** (r ≤ 8 fine pixels),
   rate-normalized; band-averaged error vs truth.
4. **Octave-seam statistic** — the same peak statistics restricted to bands
   straddling vs avoiding coarse-pixel boundaries (separates transition
   artifacts from genuine joint miscalibration — a confound gate, not a
   primary candidate).

Validation gates per instrument (tests in tests_p2/, green BEFORE any model
scoring; Stop-hook enforced):
- **Truth null:** truth-vs-truth ensemble comparison consistent with zero
  within bootstrap (the GRF-null discipline).
- **Surrogate detection:** ≥5σ detection of a phase-randomized surrogate
  (detail phases scrambled, marginal histogram exactly matched by rank
  remapping — joint structure destroyed, marginals kept). An instrument that
  cannot see pure joint destruction is dropped.
- **Convention:** bootstrap over fields (sandbox fields are independent
  parents; ledger #10 satisfied); final-state statistics only (ledger #8);
  reference-side SE budgeted into every bar (ledger #9).

**Primary-instrument rule (mechanical, Goodhart-caged):** the primary is the
validated instrument with the highest surrogate-detection z per unit
truth-null width, chosen from truth-side properties ONLY (no model scoring
enters the choice); the choice commits in the Phase-A readout appended to
this prereg BEFORE any Phase-B scoring. Non-primary validated instruments
are descriptive.

Phase A also scores, descriptively: the COMMITTED 1×-regime artifacts
(arms_c1t_sandbox.npz, c1t_repl64_gen.npz) with all validated instruments —
the measured reference for Phase-B floors — and the real-field echo on
committed gowerstreet/Stage-D npz with PARENT-BLOCKED bootstrap (#10).

**GATE-S (signal gate):** if the committed 1×-regime generations sit within
the primary instrument's truth-null band, the arena cannot discriminate →
STOP, report (meaning: the known peak-count deficit is not expressible in
position-pure coordinates — itself a finding: the sandbox residual would be
morphological-texture, not placement). Expected clear given the z≈4–5
count-level deficits, stated. Executor weight: fires 8.

## Phase B — the causal runs (after reconvene approval + reference fill-in)

Arms (all: C1-t final recipe — t(5) base, identical architecture, 20k steps,
ckpt every 500, the identical caged selection rule on val-32; one seed each;
matched total steps across arms, the rung-2/4 harness convention — the known
limitation "8×/32× see each field fewer times" stated, taildyn precedent:
8× holds flat for thousands of steps):
- **N1:** 322 independent training fields (the 1×/phase-1-mirror regime).
  Doubles as the fresh-seed replication of the C1-t sandbox marginal result
  (rider: its marginal bars scored descriptively; pass = seed-robustness
  evidence for the paper; fail = seed-sensitivity finding, to limitations —
  either way NOT adjudicating this experiment).
- **N8:** 2576 independent fields.
- **N32:** 10304 independent fields (the third point — curve, not line).
- **C-ENS (currency arm, R28 IN):** 322 coarse environments × 8 EXACT
  conditional detail redraws each = 2576 pairs (matched total to N8).

Estimand: e(arm) = primary-instrument error vs exact conditional truth at
the SELECTED checkpoint (cage identical everywhere), with bootstrap SE.
Also recorded per arm: the full checkpoint-curve of the primary (for
PL-SELECTION), all descriptive instruments, marginal statistics.

PENDING reference slots (filled verbatim from Phase-A artifacts before
Phase-B submission, per house rule): truth-null band per instrument;
committed-1× errors; surrogate z's.

## Branches (mechanical; precedence as listed; executor weights; every null's meaning stated)

Let S8 = e(N1)/e(N8), S32 = e(N8)/e(N32); "flat" = CI-consistent with 1;
"floor" = within the primary's truth-null band.

| branch | rule | weight | meaning if it fires |
|---|---|---|---|
| GATE-I | primary fails truth-null or surrogate gate | 4 | instrument redesign; no model claims |
| GATE-S | committed 1× inside truth-null band | 8 | arena lacks placement signal; the sandbox residual is texture-morphological — report, redesign on the real-field echo instead |
| GATE-D | any arm fails marginal sanity catastrophically (dispersion off >50%) | 3 | training pathology; one resubmission, else STOP |
| PL-DATA | S8 ≥ 2 AND (e(N32) < e(N8) with CI separation OR e(N32) at floor) | 33 | joint structure is data-limited in this model class (TRANSFER CLAUSE ATTACHES); constrained/larger ensembles are the designated cure |
| PL-SAT | S8 ≥ 2 AND e(N32) flat vs e(N8) above floor | 14 | data helps then a ceiling binds — H-data and H-arch both partially true; the plateau is the measured ceiling |
| PL-WEAK | 4/3 ≤ S8 < 2 (N32 descriptive); OR S8 < 4/3 but e(N32) ≤ 0.75·e(N1) | 19 | causally data-responsive but weaker than rungs 2/4 — the joint rung's effective-data demand is markedly harsher; currency question becomes decisive |
| PL-FLAT | S8 < 4/3 AND e(N32) > 0.75·e(N1) | 19 | not data-limited at reachable scales; H-arch strengthened; the posterior-sampling reframe is the designated next candidate |

Sum: 100. **Co-firing FLAG (not exclusive):** PL-SELECTION — at any arm, the
checkpoint-curve minimum of the primary ≤ 0.5× its selected-checkpoint value
with ≥3·SE separation. Executor P(fires) = 15. If it fires: the branch
verdict gains the qualifier "selection-confounded"; the per-arm
placement-optimal checkpoints are re-scored DESCRIPTIVELY (no
re-adjudication); the mechanism finding is that the marginal-optimizing cage
trades away joint structure, and a joint selection criterion becomes a named
future arm. Meaning of the flag's null (does not fire): the cage is not the
binding constraint on placement — the residual is model/data, not selection.

**Currency comparison (conditional on the data lever operative — PL-DATA,
PL-SAT, or PL-WEAK):** CUR-PARENTS: e(C-ENS) > 1.25·e(N8) — environment
diversity is the currency (PM design: many independent ICs). Executor 40.
CUR-ENSEMBLE: e(C-ENS) < 0.8·e(N8) — conditional multiplicity is the
currency (PM design: constrained realizations per IC). Executor 25.
CUR-EQUIV: between — either axis converts to joint structure at par;
budget on cost alone. Executor 35. If PL-FLAT/gates fire, C-ENS is
descriptive only (a non-data-limited residual moves under neither axis —
stated now so the null is meaningful).

**Real-field echo (descriptive, load-bearing per R27):** validated
instruments on committed gowerstreet/Stage-D generations, parent-blocked
bootstrap, reported alongside every branch verdict with the transfer clause.

## Cost & discipline

Phase A: CPU only (ensembles + scoring + tests). Phase B: 4 sandbox training
runs (MIG-minutes each, the arms_c1t_sandbox pattern) + CPU scoring. All
artifacts → results_p2/placement_*. One readout log; STOP at the readout;
the reconvene adjudicates. Numbers by verbatim copy (R12) throughout.

## What each overall outcome teaches (the brief's requirement)

Every non-gate branch names a mechanism: data (with the currency resolved),
ceiling (with its height measured), weak-response (with the demand curve),
architecture (with the fallback named), or selection (with the trade
quantified). The gates' nulls are themselves findings (GATE-S especially).
No branch confirms-without-teaching; no branch is uninterpretable.

---

## PHASE-A READOUT (2026-07-24, appended per the prereg; R12 verbatim from
## results_p2/placement_phase_a.json) — **GATE-S FIRES; STOP**

Instrument validation ran first (tests_p2/test_placement_instruments.py, 7
tests green before any model scoring). Three instrument-definition
refinements were made DURING validation, before model scoring, all from
discrete-geometry findings: the (0,1] and implied (1,2] pair classes are
structurally empty (strict 8-neighbour maxima are never closer than 2 — the
(0,1] bin removed); duplicate NN decile edges from discrete grid distances
(edges deduplicated); constant classes guarded in the z computation.

**Frozen conventions:** K = 160 (mean truth count above nu=2.5: 160.09);
NN edges [2, 2.236, 2.828, 3, 3.162, 4, 5, 6.708]; truth reference = 256
parents x redraws 0..7, parent-aggregated (#10).

**Gates (T_null must be <3; T_surrogate >=5 for primary candidates):**

| instrument | T_null | T_surrogate | verdict |
|---|---|---|---|
| env_rate | 2.03 | 4.70 | FAIL (surrogate, by 0.30) |
| nn | 1.67 | 5.0028 | PASS — margin 0.003, razor-thin, stated |
| pk2pt | 1.61 | 4.31 | FAIL (surrogate) |
| parity | 1.02 | 0.82 (n/a) | PASS as confound (truth-null only; the surrogate gate cannot apply — truth AND surrogate parity are uniform by symmetry) |

**PRIMARY = nn** (sole survivor of the mechanical rule; committed before any
model scoring). Near-misses of env_rate/pk2pt disclosed as texture.

**GATE-S: FIRES.** T_nn(committed sandbox 1x, arm A) = 2.62 < 3; repl64 arm A
= 2.16 — consistent at both field counts. Per the pre-registered meaning: the
known sandbox peak-count deficit (audit 2c, z~4-5) is NOT expressible in
position-pure coordinates — the reference arm's sandbox joint residual is
morphological-texture (count), not placement. Null integrity: the REAL
held-out tiles pass both instruments against the truth ensemble (parity
T=2.51, nn T=2.50) — the reference and instruments are clean.

**The finding inside the gate — a NEW positional mechanism, caught by the
confound instrument:** the generated sandbox fields carry a strong PEAK-PARITY
bias: arm A concentrates 30.2% of its top-160 peaks in the (odd,odd) fine-grid
parity class (truth: 25.0%; z=+7.9 at 32 fields, z=+9.9 at 64; arm B
T=6.0/8.2). This is an octave-seam/synthesis artifact — peaks preferentially
placed at a fixed corner of the finest 2x2 Haar synthesis blocks — invisible
to every marginal statistic, to peak counts, and to starlet-l1. On the REAL
field the trained arms show no significant parity bias at 32 fields (A/B
T=2.4-2.6), but the deployment-protocol dial arm explodes: Stage-D arm B
parity T=20.8 (shares [0.09, 0.37, 0.38, 0.16] vs uniform 0.25) alongside
nn T=8.7 — a further, severe off-binding pathology of arm B, again
independently confirming the arm-A-reference ruling.

**Real-field echo (descriptive, transfer clause):** arm A shows a MILD but
genuine position-pure signal on the real field — nn T=3.49 (c1t leg) / 3.90
(Stage-D edge), pattern: slight excess at small NN separations, deficit in
the largest-separation class (z=-3.9); per-parent L2s consistent across all
three parents. Sharpened joint picture: the sandbox residual is count-only;
the real-field residual is count-excess PLUS a mild genuine spacing anomaly.

**Consequence per the prereg: STOP.** Phase B does not submit (the gate's
pre-registered meaning removes its premise: there is no arm-A sandbox
placement error to dial). No training data was built; no jobs submitted.
Reconvene owns the re-pose. Executor recommendation for the reconvene:
Phase B re-posed with the PARITY/seam statistic as the primary placement
estimand — it has robust 1x signal (T 7.9-9.9), passes truth-null, is
position-pure, and carries a crisp candidate mechanism (2x2 synthesis-block
imprinting) that makes the H-data/H-arch question sharp: an architectural
synthesis artifact should NOT shrink with parent count; a learned within-block
detail-correlation error might. The branch table, currency arm, and N1 rider
transfer unchanged; nn becomes the descriptive secondary; the surrogate gate
is re-specified for parity-class instruments (sensitivity demonstrated by
direct 1x detection rather than joint-destruction response). The real-field
echo's mild nn signal stands as the transfer-clause anchor either way.
