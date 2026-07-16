# PLAN — Phase 2: the calibrated-substrate program (PROSPECTUS → plan on Andreas's go)

Drafted 2026-07-16 (reconvene + Andreas, from log/2026-07-16-note-solution-space-*).
STATUS: **prospectus with pre-registered predictions.** The phase-1 generator freeze
stands; nothing here reopens phase-1 attempts. Execution entry conditions in §0.

**Mode-2 outcome:** a generative substrate for cosmological fields whose conditional
law is *verifiably calibrated* (spread AND tails, per conditioning, per scale) at
trained scales — so the drift-dial extrapolation bet, the one question phase 1 never
got a clean shot at, is finally tested on its own merits. Either answer is a paper.

**Design principle:** the items interlock — (1) lognormal sandbox = the exact-truth
instrument; (2) starts as a model-free measurement; C-arms (vanilla-CFM control,
locality, energy-score, prior+posterior-sampling) are single-variable substrate
candidates ALL scored by the same frozen instruments against sandbox truth before
any real-field compute; (D) the extrapolation bet runs once, on the winner.

---

## §0. Entry conditions & sequencing (binding)

1. **Gate-0 kill-tests** (SPEC-novelty-collapse.md, cheap session) must return
   BEFORE any Stage-C arm runs: they price the collapse-law novelty AND the
   SR-audit-gap claim, and their reading list (AIFS-CRPS, Pacchiardi, TDS, Remy)
   directly informs C3/C4 design.
2. **The Option-1 audit paper takes calendar priority** (defense ~Sept; the paper is
   the thesis-relevant deliverable). Stages A and B are cheap (CPU/MIG-minutes) and
   may run opportunistically alongside writing; Stage C+ realistically begins
   post-defense / UVA-era. This plan is the program's prospectus either way.
3. Standing discipline, all stages: pull-before-preregister; one variable per arm;
   frozen instruments (scaledrift scorer untouched); grep-verify; pre-registered
   predictions WITH gate-branch weights (the 4b′-ii lesson); bars as RELATIVE
   residuals with pre-declared floors, never absolute-σ against tiny SEs (bar-miss
   #5 lesson); STOP at every gate for reconvene.

Budget cap for the full phase through Stage D: **25 H100-hours** (past evidence:
arms cost 6–8 MIG-minutes each at this scale; the cap is generous). Attention, not
compute, is the binding resource.

---

## §1. Stage A — the lognormal sandbox (item 1, re-scoped): exact conditional truth

**Build:** lognormal fields where TRUE p(fine | coarse) is available in closed form:
Gaussianize → conditional GRF sampling given the exact wavelet coarse projection
(linear algebra; the projection is linear in the Gaussianized field only if the
transform commutes appropriately — if exact closed-form conditioning proves messy
under the log map, the fallback is brute-force truth: rejection-free conditional
ensembles by fixing the coarse Gaussian modes and redrawing fine modes, which IS
exact for the Gaussian layer and induces the exact lognormal conditional). Deliver
per-coarse conditional ensembles (≥64 details per coarse, ≥256 coarses) + their
TRUE conditional statistics (var_slope, kurtosis, per-octave).

**Gate A (instrument calibration):** our frozen audit instruments, applied to the
TRUE ensembles, must recover the true conditional statistics within bootstrap
tolerance. FAIL → STOP (instrument bug hunt before anything else; nothing
downstream is interpretable).

- **P-A: 90%** — instruments recover exact truth. (If this fails, the entire
  phase-1c record needs re-examination — priced accordingly.)

Cost: CPU. Also delivers, permanently: the calibration test-bed any future
conditional-learning work (ours or reviewers') can be checked against.

## §2. Stage B — model-free measurements on real fields (stage-0 spirit)

**B1 — dependence range & shape (item 2, measurement before architecture):** on
gowerstreet (and sandbox as control), measure the predictability-saturation curve:
conditional variance of detail at a point given coarse context within radius r, as
r grows (k-NN / regression estimators on data, no generative model). Then the SHAPE
test: oriented (filament-aligned) context masks vs isotropic disks at matched area —
does orientation buy predictability? Deliverables: r* (saturation radius) per
octave; anisotropy verdict.

- **P-B1a: 70%** — saturation at r* well below map scale (strong locality,
  WC-RG-consistent).
- **P-B1b: 50%** — oriented contexts measurably beat isotropic at matched area
  (Andreas's filament hypothesis).
- Branch: no saturation below map scale (≤15% weight) → C2 dies AND the
  memorization account of phase 1 needs revisiting — report, reconvene.

**B2 — data inventory:** effective sample count from stride-shifted overlapping
crops (correlation-adjusted), on top of the 8× symmetry group. Cheap bookkeeping;
informs every C-arm's data recipe.

Cost: CPU / one MIG-hour.

## §3. Stage C — substrate arms (single-variable, sandbox-first, then real fields)

All arms: same architecture family unless the arm IS the architecture change; same
data recipe (augmentation + crops per B2); scored on (i) conditional calibration vs
EXACT sandbox truth (spread + kurtosis), (ii) end-to-end recursion calibration on
the sandbox (the compounding test, now against truth), then — only for arms passing
(i)+(ii) — (iii) gowerstreet with the frozen scorer, G-1c-style relative-residual
bars. Kill bar per arm: fails (i) on the sandbox → dead, no real-field compute.

- **C1 — vanilla CFM + augmentation (+ crops). The control; the un-run arm.** Full
  distribution model, no Gaussian crutch. Runs FIRST — if the collapse cure alone
  restores calibration including tails, the fancier arms become comparisons, not
  necessities.
  **P-C1a (trained-octave dispersion alive on sandbox): 60%. P-C1b (kurtosis at
  truth — the tails question): 45%. P-C1c (end-to-end recursion calibrated): 45%.**
- **C2 — locality-capped conditioning (receptive field = r* from B1; oriented iff
  B1b).** Runs only if B1a passes. Single variable vs C1: the conditioning range.
  **P-C2 (improves collapse onset AND calibration vs C1): 55%.**
- **C3 — energy-score/CRPS-trained sampler** (patched energy score per the
  Pacchiardi line; AIFS-CRPS "almost fair" variant considered — Gate-0 reading
  decides the exact form). Single variable vs C1: the objective.
  **P-C3 (calibrated spread from one-observation-per-condition data): 60%.**
- **C4 — per-octave UNCONDITIONAL prior + exact-projection posterior sampling**
  (TDS-style SMC primary for correctness; a cheap DPS-style variant recorded
  descriptively). The substrate reframe. Positioning per addendum-2: descent from
  Remy/Lanusse/Starck, machinery-advantage zero, tested for what it uniquely
  offers — conditional health without paired training.
  **P-C4a (trained-octave conditional calibration on sandbox): 65%. P-C4b
  (recursion calibrated — does hard-constraint conditioning resist the off-manifold
  cascade better than regression response?): 50% — this sub-question is novel
  territory and a finding either way.**

Order: C1 → (C2 | C3 | C4 as parallel single-variable arms against the C1 control).
Every arm pre-registers gate-branch weights before submission. Reconvene rules
between arms; no bundling ever.

## §4. Stage D — the bet (runs once, on the passing substrate)

**The decisive experiment phase 1 never cleanly reached:** on gowerstreet, with the
best Stage-C substrate, slide-the-edge — train octaves {3,4,...}, extrapolate the
COUPLING CURVE (not measured target-octave couplings: the deployment protocol) into
the held-out finest octave, generate, score against the hidden truth. Bars:
conditional calibration + held-out statistics (scattering covariance; starlet-ℓ1
added per the estimator-hardening list) at the extrapolated octave, relative
residuals with the sandbox-measured floor.

- **P-D (the dial extrapolates within bars): 40%** — the original project bet,
  priced with everything now known. Branch-complete: PASS → the method paper
  (descent framing, GOLCONDA/ST/LDT baselines per the phase-2 candidates already
  logged in reconvene-cprime2). FAIL → the audit paper gains its strongest chapter:
  the bet tested cleanly on a calibrated substrate and refused — a bounded,
  instrumented negative with nothing left to blame. AMBIGUOUS = FAIL (standing
  asymmetry).

## §5. Out of scope

Adversarial/texture-critic arms (Andreas's ruling, 2026-07-16). Constrained-
realization N-body ensembles (gold standard; requires the 3D-phase sims and an
owner for that data problem — revisit at UVA with GECO resources; the sandbox is
its dress rehearsal). Any phase-1 attempt revival. Any-scale claims, ever.

## FREE PERIPHERY

Estimator implementations for B1 (k-NN vs regression), sandbox conditioning
implementation (closed-form vs mode-redraw fallback), SMC particle counts and
schedulers for C4, energy-score patching geometry for C3, checkpoint/data layout,
plot styles. Document choices in preregs; the cores above are frozen.

## §6. Implementation & orchestration (how this actually runs)

**Topology: one repo, ONE active executor session at a time, arms sequential.**
Rationale from the phase-1 record: all race conditions came from parallel actors in
one repo; arms cost minutes of compute, so parallelism buys wall-clock nobody needs
and spends attribution everybody needs. The serialization point is judgment
(prereg + ruling), not GPUs.

- **Session per stage, not per arm.** A fresh executor session (compacted sessions
  re-anchor from disk) picks up at a stage boundary with the standing briefing pack:
  CLAUDE.md + this PLAN + the newest reconvene rulings (pull-before-preregister).
  Within a stage, the same session runs its experiments sequentially: one
  prereg → submit → readout → STOP cycle per experiment/arm.
- **Reconvene gates only at the named gates** (A, B, each C-arm readout, D) — not
  continuously. One paste-block per gate, as always.
- **Model tiering:** executor sessions = cheap/mid model (the arms are
  well-specified by this plan + their preregs); reconvene = strong model. The
  kill-test session (Gate 0) is its own separate cheap session and may run anytime —
  it only writes its own log file, so it cannot conflict.
- **Paper vs phase-2 in the same repo:** strictly one active session at a time.
  Realistic interleaving: paper skeleton + gallery + estimator hardening first (the
  current D4 session's context is valuable there); Stages A+B opportunistically
  after, same one-at-a-time rule; Stage C+ = a fresh dedicated session per stage,
  UVA-era.
- **Sanctioned parallelism (optional, Stage C only, if wall-clock ever matters):**
  git worktrees, one per arm, with the constraint that instruments/scorer are frozen
  on main and arms only ADD files (scripts/prereg/results under arm-named paths);
  merge after each harvest, reconvene rules on the merged state. Not recommended
  before UVA; never for Stage D (one experiment, one substrate, one run).
- **New code layout:** sandbox machinery under `sandbox/`, B1 estimators under
  `depmeasure/`, arms under `arms_p2/<arm>/` — additive, so the frozen phase-1
  record (scaledrift/, wfm/, results/) stays byte-stable and auditable.
- **Test gate:** the Stop-hook coverage fix (both test trees, correct envs — the
  pre-writing condition from the memo ruling) lands BEFORE any phase-2 code is
  written; new phase-2 estimators get tests-first as always, in a `tests_p2/` tree
  registered in the gate from day one.
