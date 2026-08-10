# NIGHT ORDERS 3 — the texture campaign: three principled levers, full
# machinery build, parallel arms (2026-08-05, reconvene-authored)

Andreas's authorization, explicit: a large-scope overnight run that
implements and tests the principled lever menu; the executor MAY deploy
subagents / parallel fan-outs and spend tokens accordingly — **this
overrides the standing no-fanout cost rule FOR THIS NIGHT ONLY.** Executor
model = Fable; the RIDER governs discretion. PLAN-phase3.md + R46/R47
govern scope; this file instantiates. Budget cap tonight: **3.5 H100-h**
(phase ≈2 spent of 10; R47's set may consume ~1 of tonight's cap if not
yet run).

Read first: log/2026-08-05-reconvene-topo-adjudication.md (R47 — the
oct-1-texture diagnosis is tonight's premise), NIGHT-ORDERS-2.md's RIDER
(the discretion charter pattern), the R43 claim set, and the lever
rationale in the reconvene chat record mirrored below.

## The three levers (principled = changes assumptions/coordinates/training
## distribution on physical or structural grounds; judged only by
## instruments the training never saw; NO audit statistic in any loss)

1. **AUG — pseudo-octave augmentation:** downsample training fields ×2 so
   pyramid roles shift; the tied network trains ACROSS octave-role
   transitions instead of extrapolating off the edge of them. Uses no
   target-octave information. Targets the measured defect directly
   (role-shift at the first untrained octave).
2. **TIDAL — tidal-frame conditioning:** expose the coarse field's local
   Hessian eigenframe (orientation/anisotropy features) to the conditional
   — or generate in that frame. Physics: anisotropic collapse; structure:
   makes the conditional law more scale-invariant, which is what
   weight-tying assumes. GATED on the instrument (below).
3. **CASC — multiplicative-cascade base (PROBE ONLY tonight):** the
   density field is multifractal; a cascade base carries spatially
   clustered seeds. Tonight: inference-only two-point transfer probe (does
   base PHASE structure survive fixed weights at all? — the L1″ lesson).
   NO cascade training tonight regardless of outcome.

## Phase N0 — harvest / complete the R47 set (prerequisite logic)

If BL / O(oracle {1,2,3,4}) / I(texture instrument) have not run, run them
FIRST (their preregs and weights stand: O = 45/25/20/5/5, BL = 80).
Pre-delegated mechanical adjudication for tonight's sequencing only (the
morning reconvene re-adjudicates): O's branch determines the
interpretation frame — O-FIXES ⇒ the defect is role-shift ⇒ AUG is its
deployable approximation (raise AUG's stakes); O-NULL ⇒ architecture
limit ⇒ AUG likely null too, run it anyway (the pair O/AUG then cleanly
separates data-availability from architecture).

## Phase N1 — machinery (PARALLEL; subagents encouraged; every estimator
## tests-first; single-committer rule below)

a. **Texture instrument** (if pending from R47): oct-1-scoped
   component/hole decomposition + a cross-scale ORIENTATION-ALIGNMENT
   statistic (local orientation of oct-1 texture vs coarse-field
   eigenframe; validated on synthetic oriented/isotropic textures).
b. **GATE-T (mechanical, from the instrument):** TIDAL trains ONLY IF
   generated oct-1 texture shows orientation decoherence vs real at ≥3σ
   (real texture aligned with coarse eigenframe, generated less so).
   Reconvene line: P(GATE-T fires) = 60. NOT-FIRED ⇒ TIDAL descoped
   tonight (budget saved; the null is itself a mechanism finding).
c. **AUG pipeline:** downsampled-copy dataset builder; tests: role-shift
   correctness, tile counts, NO oct-1-of-original-resolution data
   anywhere in training, augmentation composes with the 8× symmetry.
d. **TIDAL machinery** (build regardless of gate; train only if gated
   in): eigenframe features (smoothed Hessian of the conditioning
   coarse), D4-compatibility PROVEN in-test (features must transform
   covariantly so the F2 group-average guarantee survives — this is the
   design-heavy piece; get it right).
e. **CASC sampler + probe:** 2-D multiplicative cascade (log-normal MRW
   family), isotropic, D4-exact in law, seeded; the transfer probe =
   L1′-style two-point comparison (cascade base vs white base through
   committed weights) scored on the texture instrument + MF(dev).
   Reconvene line: P(base phase structure measurably carried) = 30.
f. **JUDGE-2 FREEZE (mandatory before ANY arm is scored):** the Minkowski
   judge is hereby RECLASSIFIED as a development metric (we are
   engineering against it; it can no longer serve as held-out). Build and
   freeze a NEW untouched judge: persistent-homology Betti curves (b0/b1
   vs threshold, declared + native conventions), tests-first, validated
   on synthetic GRFs ONLY, committed FROZEN, applied to NO generation
   until the next blind shot. The audit thesis requires an unburned tier.

## Phase N2 — training arms (SLURM, parallel; canary-first; one licensed
## bug-repair resubmission per job)

- **Arm AUG:** production recipe + augmentation, train {2,3,4}(+pseudo
  roles), seeds 11 & 12. Sandbox canary first (standing bars, kill:
  dispersion >40% at ckpt ≥4k).
- **Arm TIDAL** (iff GATE-T): production recipe + eigenframe conditioning,
  seeds 21 & 22. Its OWN sandbox canary (new conditioning path).
- Controls: committed production model (baseline) + O arm (oracle
  ceiling). No other variants, no exceptions — draft preregs instead.

## Phase N3 — the audit (frozen battery + instrument + MF-as-development;
## same branch rules as O for cross-arm comparability, per arm, vs the
## committed baseline; ambiguity = negative; #11 bands; one fresh-PRNG
## disambiguation license per arm)

| branch (per arm) | rule | rec weights AUG / TIDAL |
|---|---|---|
| X-FIXES | MF(dev) declared ≤ 3.5 AND native ν2.5/3.0 peak excess < half committed AND no must-not-regress failure | 15 / 12 |
| X-IMPROVED | not FIXES; MF(dev) improves by ≥2σ-equivalent AND instrument corroborates; no regression | 40 / 33 |
| X-NULL | neither; no regression | 30 / 35 |
| X-REGRESSED | any must-not-regress failure | 5 / 10 |
| gates | infra; determinism; canary kill | 10 / 10 |

Executor registers its columns in the prereg BEFORE training. P-T-style
coloring check per arm (the deconvolution recalibrated per checkpoint —
standard recipe; its band is a watched line, not a branch).

## Phase N4 — synthesis + morning deliverable

NIGHT-REPORT-3.md: MORNING SUMMARY (path taken, gate verdicts vs weights,
the three numbers, budget, provisional calls — ALL provisional);
verdict/interpretation separation; the O/AUG/TIDAL cross-arm reading
(role-shift vs architecture vs orientation — tonight's design separates
these); CASC probe verdict; drafted-NOT-run preregs: the second blind
shot (winning arm, JUDGE-2 as the untouched tier, needs reconvene
review), cascade training (if the probe surprises), and the
texture-transfer corrector update. Commit, push, STOP.

## Discipline (standing, all night)

MIG ≤2:59 pace arithmetic; JOBS.md pre-submission entries; pre-statements
with PENDING placeholders before reading ANY result; R12 verbatim;
grep-verify scripted edits; suites green before every submission.
**Subagent guardrails:** fan out for machinery construction, test
authoring, scoring, and analysis; NEVER for extra training variants,
never for adjudication (branch rules are applied by the main session,
mechanically); subagents return files/diffs — the MAIN session is the
single committer (no parallel git writes); worktrees if isolation is
needed. Refusals carried: no audit statistic in any loss; no χ/MF/PH
matching; no JUDGE-2 application to any generation. Budget: hard stop at
3.5 H100-h; queue stall >2h ⇒ park that leg, proceed elsewhere.

---

## RIDER — discretion charter (Fable executor; subagent-augmented night)

**The goal:** tomorrow's reconvene reads a maximally informative,
trustworthy record of WHICH principled lever moves the texture defect and
WHY — with the O/AUG pair separating data-availability from architecture,
and GATE-T separating orientation from everything else. A clean null with
a sharp mechanism is success. A muddy improvement is failure.

**Frozen:** branch rules and bars as written; GATE-T as written; the
one-probe-only status of CASC; JUDGE-2 quarantine; budget; the
no-audit-statistics-in-losses rule; prereg-before-run,
pre-statement-before-reading; single-committer.

**Granted:** implementation quality everywhere (the eigenframe
D4-covariance and the cascade sampler are genuinely hard — make them
excellent); subagent orchestration strategy (parallelize as you judge
best within the guardrails); diagnosis-on-surprise budget: 8 descriptive
runs / 1 H100-h (within the cap), each with a one-line JOBS.md intent;
opportunistic descriptive measurements labeled as such; the morning
report's analysis section — belief updates, competing mechanisms ranked,
the drafted preregs.

**Named temptations (refuse; note refusals in the report):** letting a
subagent's enthusiasm widen scope (extra arms, extra levers); softening
GATE-T to train TIDAL anyway; peeking at JUDGE-2 "just to see"; scoring
MF and calling it held-out; reading 1-seed improvement as adjudicable;
a synthesis that outruns the verdict tables.

---

## RIDER v2 (supersedes the charter above where they conflict; Andreas's
## explicit direction: real freedom to improvise, interpret, and adapt)

You are a strong, goal-driven model. Tonight that is the point, not a
risk to be contained. The orders above are the SPINE — the planned
experiments and their comparability structure — but the night belongs to
your judgment. What follows is the widened grant and the short list that
remains absolute.

**Widened grants:**
1. **Live reprioritization.** Reorder phases, reallocate seeds and budget
   between arms, descope or double down as evidence arrives (O lands NULL
   → maybe one AUG seed suffices and diagnosis deserves the rest; GATE-T
   fires at 8σ → TIDAL may deserve the extra seed). Record each pivot in
   one line (decision + reason) in the report's decision ledger.
2. **Adaptive probing without quotas.** The fixed diagnostic-run budget is
   replaced by judgment: run whatever descriptive probes the evidence
   calls for, within the GPU cap, each with a one-line JOBS.md intent.
3. **Design authority inside arms.** Eigenframe construction details,
   augmentation depth (how many pseudo-octave levels), cascade
   parameters, network-input plumbing — yours entirely. Commit a
   three-line rationale with each choice; no approval needed.
4. **Blind rule amendment.** You may sharpen branch rules, add branches,
   or add watched lines for any experiment PROVIDED the amendment is
   committed before that experiment's results are read (the A1/G2
   precedent). Post-hoc changes remain forbidden, forever.
5. **ONE improvisation slot.** If tonight's own measurements motivate an
   experiment nobody planned — including one extra training variant — you
   may take it: prereg with weights committed first, single-variable,
   within budget, canary discipline if it trains. One slot; choose it
   well; "the fix is obvious" is a reason to draft, "the measurement
   demands it" is a reason to run.
6. **Interpretation as a first-class deliverable.** The morning report's
   analysis section is yours without constraint: rank mechanisms, argue
   with the reconvene's weights, propose the next campaign. Verdict
   tables and interpretation stay in separate labeled blocks; inside the
   interpretation block, go as far as the evidence lets you.

**The absolutes (these are why the record is trusted; they do not bend):**
pre-statement before reading any result; R12 numbers-by-copy; tests-first
for new estimators; single-committer; JUDGE-2 stays quarantined — no
peeking, no exceptions; no audit statistic in any loss; the GPU hard cap
(raised to 4.0 H100-h to fund the improvisation slot honestly); morning
STOP with all calls provisional.

**On your disposition:** point the goal-drive at epistemic yield. The
best possible morning is not "an arm passed" — it is "we know which of
three causal stories is true and what to do about it." If you notice
yourself wanting a particular arm to win, write that sentence in the
report; it will be read as strength.
