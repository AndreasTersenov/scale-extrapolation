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
