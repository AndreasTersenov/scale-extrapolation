# PLAN — Phase 1c: variance-faithful generator, fair P6/P13, validation-architecture pilot

**Status entering this phase** (re-anchor from disk, not compacted memory: RESULTS-toy.md,
PLAN-phase1.md, log/2026-07-10-prereg-varfaithful-*.md, log/2026-07-11-reconvene-cprime2.md
INCLUDING both addenda): P5 (the break) is confirmed and accepted. P6/P13 are
blocked-pending-retest: L2 flow matching under-disperses conditional variance (twice-plus-
once-confirmed: random-t penalty, late-t penalty, and churn all fail; dispersion peaks at
~2k steps and collapses with training; the deterministic-ODE pushforward is the structural
limit). Reconvene approved **option 2: an explicit-variance (Gaussian-NLL) detail head**,
with binding conditions from the addenda. The foundations review (2026-07-11) also fixed
the paper's validation architecture; this phase pilots it.

---

## FROZEN CORE — do not modify (pre-registered 2026-07-11; Andreas + reconvene)

### Step 1 — the variance-faithful generator (option 2, as ruled)

Gaussian-NLL detail head: the model predicts (velocity/mean, log-variance) per detail
coefficient; the detail conditional is trained with a proper NLL and SAMPLED with its
variance. Binding conditions (addendum 1):
- **Arm symmetry:** BOTH arms A and B get the head — scale-conditioning stays the only
  differing variable.
- **Success bar (unchanged):** trained-octave var_slope within 1σ of real at octaves
  2, 3, 4 SIMULTANEOUSLY, deterministic sampling of the mean-path plus explicit variance
  (no churn), GRF null preserved (on GRF the head must learn Gaussian details and pass
  the null).
- **Kurtosis check (pre-registered):** conditional kurtosis at trained octaves within 2σ
  of real. Hypothesis permitting the Gaussian head: conditioning on the coarse
  environment Gaussianizes details. If the variance bar passes but kurtosis fails →
  **pre-named fallback: student-t NLL head** (df learned or swept), same bars, one
  escalation only.
- Pre-register the sampling procedure before training. Do not train past the dispersion
  peak without checkpoints (~2k-granularity early).

**Gate G-1c:** if option 2 INCLUDING the student-t fallback fails the dispersion bar,
STOP — the under-dispersion result graduates to the standalone methods finding; report
and reconvene. No further generator variants without a new ruling.

### Step 2 — fair P6/P13 re-adjudication (frozen bars, original definitions)

With the passing generator: re-run arms A/B on gowerstreet exactly as PLAN-phase1
specified. **P6 (55% original): arm B repairs ≥70% of arm A's extrapolated-octave
non-Gaussian drift, measured couplings allowed (the original definition), trained
octaves undegraded.** P13 (55%): the zero-retrain hf_pm transfer, original definition.
- **NEW pre-registered variant P6x (60%):** same run, but arm B conditioned on couplings
  EXTRAPOLATED from a smooth fit to the training octaves only (the production-honest
  mode; closes the measured-vs-extrapolated gap from addendum 1). Bar: repair ≥50%.
- **K-T2 (now live):** if P6 < 30% with a dispersion-faithful generator, the
  low-dimensional-conditioning hypothesis has failed its fair test — STOP, reconvene.

### Step 3 — validation-architecture pilot (only if P6 passes)

The paper's second contribution, piloted at small scale (full protocol is phase 2):
1. **Slide-the-edge:** repeat the train/extrapolate experiment at a SECOND edge position
   (train octaves 3–5 → test into octave 2, alongside the existing 2–4 → 1). Report
   one-octave extrapolation error (arm B, var_slope z and drift metrics) at both edges.
   **P-edge (60%): the two errors agree within a factor 2** — the extrapolation
   operator, not a particular map, is what's validated.
2. **Self-consistency:** measure the running couplings ON generated fields at the test
   octave; **P-selfcons (65%):** they land on the extrapolated coupling curve within its
   fit uncertainty wherever P6 passes.
3. **Held-out statistics:** score the test octave on statistics NEVER used in training
   or repair design — wavelet-L1, peak counts, and the scattering covariance (the rival
   school's instrument; use a standard ST library). **P-heldout (50%, genuinely
   uncertain — the law-vs-summaries test):** arm B within 3σ on at least half of the
   held-out statistics wherever tracked statistics pass.
4. **Error-bar demo (stretch, non-gating):** propagate the coupling-curve fit
   uncertainty (generate at fitted ±1σ couplings) into a band on ONE statistic at the
   test octave — the certificate prototype.

### Out of scope

GOLCONDA / microcanonical-ST head-to-head baselines, the LDT theory anchor
(coordination with Vilasini/Starck), cosmology conditioning, D6b, full error-budget
machinery, any paper writing — all phase 2. The under-dispersion novelty kill-test
(literature sweep) is NOT this session's job — reconvene will assign it separately.

### Predictions summary (Claude, 2026-07-11)

P-NLL-var 65% · P-kurt 55% (student-t rescue → combined ~80%) · P6 70% (the 90%-repair
prototype is the prior) · P6x 60% · P13 55% · P-edge 60% · P-selfcons 65% ·
P-heldout 50%.

### Budget & discipline

Within the standing 15 H100-day phase cap (spend to date is small; MIG slices suffice
for training, full H100 for the load-bearing generation runs, ≤2:59 per the cluster
rules — nodes may be draining; queued jobs start when the drain lifts, do not churn
resubmissions). Every job logged pre-submission. Report and STOP at G-1c, at the
P6/K-T2 verdict, and at phase completion.

---

## FREE PERIPHERY

NLL-head parameterization details, variance floor/clamping, how log-variance enters
sampling, checkpoint cadence, the smooth-fit family for coupling extrapolation
(document the choice and its uncertainty estimate), ST library choice, plot styles.
Reuse scaledrift untouched (any modification needs written justification).

## Deliverable

`RESULTS-phase1c.md`: G-1c outcome with the dispersion/kurtosis tables; P6/P6x/P13
verdicts with the frozen bars applied verbatim; the pilot's three validation panels
(edge-slide, self-consistency, held-out stats) + the error-bar demo if reached;
prediction verdicts; honest limits. Written for the reconvene; assumes this PLAN.
