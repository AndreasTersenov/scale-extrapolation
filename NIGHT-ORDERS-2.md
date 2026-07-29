# NIGHT ORDERS 2 — phase-3 overnight run: Stage 1 → gate → Stage 2 → audit
# (2026-07-29, reconvene-authored)

Andreas has authorized an autonomous overnight run: this file SUPERSEDES R37
order 2's "no Stage-2 training until the Stage-1 harvest" — the harvest gate
is pre-delegated here under mechanical rules, exactly as NIGHT-ORDERS.md
pre-delegated Gate A. Executor model = Fable; the RIDER at the end governs
your discretion — read it before starting. PLAN-phase3.md governs scope;
this file instantiates its Stages 1–2 with numbers and rules.

Read first (you have the capacity to hold all of it — use it):
PLAN-phase3.md, log/2026-07-28-reconvene-stage0-adjudication.md (R37 — the
field-not-scale reframe is tonight's premise), log/2026-07-28-stage0-p3-readout.md,
log/2026-07-28-prereg-stage0-phase3.md (branch-table conventions), and the
Stage-2 paragraph of PLAN-phase3.md twice.

## The night's shape (three phases, two mechanical decision points)

```
N1 mechanism profiling (descriptive, cheap)
   └─ GATE-N1: is the whiteness premise CONFIRMED?
        ├─ YES → N2: train the colored heavy-tailed base (the single variable)
        │         └─ N3: frozen audit battery → branch table (provisional)
        └─ NO  → deep diagnosis on the discretionary budget; drafted preregs
                 for alternatives; STOP with a mechanism report (full success)
```

Tonight trains AT MOST one variant recipe (the colored base). No other
training lever exists tonight, whatever the evidence suggests — draft
preregs for anything else.

## Phase N1 — mechanism profiling (field-first per R37)

Primary venue: TRAINED scales on gowerstreet (the excess lives there:
F2_gowA +14.48%±3.12 at ν=2.5, +11.07%±2.76 at ν=3.0 <!-- src:
stage0_p3_verdict.json /trained/F2_gowA -->). One edge leg for continuity
(refs +12.45%±3.09 / +12.94%±3.00 <!-- src: idem /edge/A -->). Commit a
short prereg with your expectations+weights per probe BEFORE computing;
reconvene lines for the gate are already registered below.

1. **Detail-coefficient coloring (THE gate probe; new estimator →
   tests-first).** Per octave (1–3), compare the spatial structure of real
   vs generated detail coefficients: annular power spectrum of the detail
   planes, summarized by a coloring index C = P(low-k band)/P(high-k band)
   (bands fixed in the test, chosen on synthetic validation BEFORE touching
   data), bootstrap SEs over tiles. Validation test: C discriminates
   synthetic white vs colored noise at machine-comfortable significance and
   is flip/rotation invariant.
2. **Peak morphology.** Height-resolved excess (ν ∈ {2.5, 3, 3.5, 4}) and
   smoothing response: excess vs Gaussian smoothing at ~1–2 px. If the
   excess dies under 1-px smoothing the surplus is grainy/small-scale; if it
   survives, the surplus peaks are real-shaped. Descriptive.
3. **Octave attribution (hybrid transplant).** Real pyramid, substitute the
   GENERATED details at octave j only, reconstruct, count peaks — the
   per-octave excess contribution. Uses frozen synthesis machinery.
   Descriptive.
4. **Spacing co-location.** nn profile on the same maps; does the trained-leg
   spacing residual (T≈3.5–4.1 committed) share octave/height structure with
   the count excess? Descriptive.
5. **Reference replication (unconditional, cheap).** TWO fresh-PRNG F2
   regenerations of the trained leg (gen-side replicates of the +14.5/+11.1
   reference). Tightens N3's reference and prices generation-stream
   variance. ~0.4 H100-h total.

## GATE-N1 (self-adjudicated, mechanical — the colored-base premise)

**CONFIRMED** iff at ≥2 octaves of {1,2,3}: C_real − C_gen ≥ 3·hypot(SE_real,
SE_gen) with C_real > C_gen (real details more spatially colored than
generated). **Reconvene line, registered now: P(CONFIRMED) = 60.** Add yours
in the N1 prereg.

- CONFIRMED → N2 authorized immediately, no reconvene review overnight.
- NOT CONFIRMED → no training tonight. This is major news, not a failure:
  the doctrine lever's premise is measured absent. Spend the diagnostic
  budget on the mechanism (morphology/transplant follow-ups), draft
  alternative-lever preregs (NOT run), write the report, STOP.
- Ambiguous (exactly 1 octave, or at-bar within 0.5·SE of the 3σ line) =
  NOT CONFIRMED (ambiguity = negative, standing rule).

## Phase N2 — the colored heavy-tailed base (single variable; only via GATE-N1)

**Construction (your periphery — judgment-heavy, make it excellent):** base
noise carrying the MEASURED per-octave spatial autocorrelation of real
detail coefficients with the established heavy-tailed marginals. The naive
order (color white-t by spectral filtering) Gaussianizes the marginals by
central-limit mixing — the known-good shape is copula-style: color a
Gaussian field with the measured isotropic (annular) filter, then map
marginals to the t target by quantile transform. Implementation is yours;
these constraints are not:

- **Tests-first (tests_p2/), before any training:** (t1) target spectrum
  match within tolerance; (t2) marginal tails match the t target (kurtosis /
  quantile ratios) within tolerance; (t3) D4-symmetry compliance — the base
  law must be exactly D4-invariant (isotropic filter + iid seed ⇒ prove it
  in-test by group-orbit statistics), because the F2 sampler's
  character-theory guarantee assumes it; (t4) seeded reproducibility.
- **One variable:** IDENTICAL recipe to the committed arm-A configuration —
  data regime, augmentation, architecture, checkpoint curve, caged
  validation selection, F2 sampling — except the base. The coloring filter
  is measured from TRAINING data only, per octave, frozen and committed
  before training.
- **Sandbox canary first (1 seed):** train on the sandbox recipe; bars from
  measured truth as in NIGHT-ORDERS C1 (dispersion ≤ max(10%, 3·SE_rel),
  kurtosis ≤ max(15%, 3·SE_rel) at all trained octaves, end-to-end).
  KILL criterion during any training: at checkpoint ≥4k, sandbox dispersion
  error > 40% → kill the leg, diagnose on the discretionary budget. The
  gowerstreet legs run only if the sandbox canary passes its bars.
- **Gowerstreet: minimum 2 seeds, 3 if the queue permits.** The seed lesson
  (R35) is binding: single-seed comparisons carry ~2× hidden variance.
  Adjudication in N3 uses the seed-mean; a 1-seed night ends N-PENDING-SEED
  (provisional, not adjudicable).
- Prereg with weights committed BEFORE training (branch table below +
  yours); pre-statement with PENDING placeholders BEFORE reading any
  readout (the M5 pattern); R12 throughout.

## Phase N3 — the frozen audit battery on the colored-base variant

Frozen instruments only (audit_peak_ci peaks, marginal suite bars as
context, starlet-ℓ1 scorer, placement nn, parity/G1 identity re-asserted).
Score trained leg (PRIMARY: seed-mean excess at ν=2.5/3.0 vs the N1-replicated
reference) and the edge leg (descriptive continuity). MUST-NOT-REGRESS set:
marginal suite within standing bars (no >50% catastrophe anywhere), starlet
edge+trained pass, parity_T < 3, G1 exact.

Branch table (mechanical; Δ = (ref − new)/hypot(SE_new, SE_ref), shared-real
conservatism noted N1-style; #11 half-SE at-bar bands; worse category
governs; ONE fresh-PRNG disambiguation licensed):

| branch | rule (both ν unless stated) | reconvene weight |
|---|---|---|
| N-CURED | both trained-leg 95% CIs include 0; no regression | 15 |
| N-IMPROVED | not cured; both Δ ≥ 2; no regression | 40 |
| N-NULL | both Δ < 2 (or mixed → worse); no regression | 30 |
| N-REGRESSED | ANY must-not-regress failure, regardless of peaks | 10 |
| N-FLIPPED | either trained-leg 95% CI entirely below 0 (the A1 guard) | (in gates) |
| gates | infra (one licensed resubmission per job); N-FLIPPED; identity-gate failure | 5 |

Executor weights alongside at the N2 prereg. Edge-leg movement is
descriptive tonight (the field-level mechanism predicts it follows the
trained leg — note agreement/disagreement explicitly). All N3 branch calls
are **PROVISIONAL pending the morning reconvene adjudication**.

## Discipline (standing, all night)

SLURM for anything past ~1 core-minute (MIG h100_20gb ≤2:59, rrg-lplevass;
CPU → def-lplevass). Pre-submission JOBS.md entries (ID, config hash,
expected outcome). Tests-first for every new estimator. Grep-verify scripted
edits. Commit+push after every unit. No edits to PLAN-phase3.md, no paper/
edits, additive layout (stage1_p3_*, colored_base/ or similar). **Budget cap
tonight: 6 H100-h** (N1 ≤1 incl. replicates; N2 ≤4; N3 ≤1; expect ~2–3
total; phase cap 10 with 0.4 spent). MIG pace arithmetic before every
submission (2/7 of a full H100 — one attempt died at a time wall from this).
Queue stalls >2h: park that leg, proceed with what doesn't need it, note it.

## Morning deliverable

NIGHT-REPORT-2.md, topped by a MORNING SUMMARY block: the path taken at each
node (gate verdicts vs registered weights), the three numbers that matter,
budget spent, provisional N3 branch call (or the mechanism report if GATE-N1
failed), discretionary-budget ledger, drafted follow-up preregs, open
questions for the reconvene. Verdict blocks and interpretation blocks
labeled and separate. Commit, push, STOP. Do not touch the paper skeleton,
do not begin Stage 3.

---

## RIDER — the discretion charter (Fable executor; read twice)

**The goal, stated precisely:** tonight's goal is NOT that the colored base
wins. It is that tomorrow's reconvene reads a maximally informative,
maximally trustworthy record of whether the field-texture mechanism is what
R37 says it might be, and whether the doctrine's third application cures it.
A clean NOT-CONFIRMED gate with a sharp mechanism profile is a full success.
A muddy N-IMPROVED is a failure. Point your goal-orientation at epistemic
yield.

**Frozen — no discretion, no exceptions tonight:** GATE-N1 and the N3
branch rules AS WRITTEN (objections → morning report, never mid-run
adaptation); the one-training-lever rule; the 2-seed adjudication minimum;
prereg-before-run and pre-statement-before-reading; bars and frozen
instruments; budget caps; single-threaded execution (no subagent fan-outs /
workflows — standing cost rule); the supersession scope (tonight's autonomy
covers Stages 1–2 ONLY — Stage 3 and the paper remain reconvene-gated).

**Granted — where your judgment is expected, not just permitted:**
1. The colored-base construction and the coloring-index estimator are
   genuinely judgment-heavy — design them well; validation tests still come
   first, and the C-band choice must be fixed on synthetic data before real
   data is touched.
2. **Diagnosis on surprise:** on any anomaly, gate failure, or kill, you
   hold up to 6 descriptive runs / 1 additional H100-h to characterize what
   happened. Descriptive = measures existing artifacts or the failure
   itself; NEVER trains a variant, never feeds back into tonight's verdicts.
   One-line intent + expectation in JOBS.md before each. Park with a
   diagnosis, not a shrug.
3. **Opportunistic descriptive measurements** free on existing outputs,
   labeled DESCRIPTIVE, kept out of verdict tables.
4. **Sequencing and infrastructure** (queue strategy, flaky nodes, env
   breakage): full discretion, documented.
5. **Interpretation:** the morning report gets a genuine analysis section —
   belief updates with numbers, competing mechanisms ranked, drafted (not
   run) preregs for the follow-ups the evidence suggests, including what
   Stage 3 should look like if N3 lands well.
6. **One bug-repair re-run per leg** is licensed IF a disclosed
   implementation bug is found and documented (the G2 precedent: repair
   blind, pre-state the corrected criterion, disclose in place). "The result
   looked wrong" is not a bug.

**Named temptations (your profile's failure modes — refuse them):** pushing
through GATE-N1 on a favorable reading of an ambiguous octave; softening a
must-not-regress bar because the peak number moved; training a second
variant because the fix is obvious (draft the prereg instead); reading
1-seed improvement as adjudicable; a morning summary that outruns the
verdict tables (the campaign's "preprint-ready" scar has a name on it);
spending the diagnostic budget on rescue. If you catch yourself
rationalizing any of these, write that down in the report — it will be read
as competence.
