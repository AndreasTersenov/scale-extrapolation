# HANDOFF — scale-extrapolation, written 2026-08-08 for the low-token month

Read this FIRST in any new session. It replaces "ask the orchestrator".
Everything below is either binding (decided, do not relitigate) or an
explicitly scoped task you can do in ONE cheap session.

---

## 1. What this project is, in five lines

A generative method that adds fine detail to a coarse cosmological field,
one octave at a time, through an exact wavelet factorization: the coarse
scales you condition on are preserved EXACTLY, and one weight-tied
conditional flow-matching network does every scale. It is validated by a
pre-registered audit (marginals, starlet-ℓ1, peaks, spacing, topology) and
ships two tools: an exactly D4-equivariant group-averaged sampler and a
measured-transfer deconvolution that calibrates the finest octave at
deployment with no retraining (7/7 predictive record).

## 2. State of the science (binding — these are adjudicated, not opinions)

**Works, robustly:** exact coarse preservation; marginal statistics
(variance/kurtosis per octave, head-conditional and end-to-end); starlet-ℓ1
out-of-basis; symmetry/parity; spacing (pooled multi-stream convention);
the deconvolution calibration (7/7 across seeds, substrates, blind runs);
empty-beam physical floor (clean null, first live application).

**The two boundaries, both located at the SAME place** — the phase
organization of the texture at octave 1, which is *extrapolated* (never
trained) in every configuration:
- **Native-resolution peak excess +16–24%.** Immovable: survives all 24
  swept checkpoints, direct oct-1 training (oracle), every lever.
  This is the model-class residue.
- **Topology (Minkowski/Euler at mid-levels).** Corrected-selection scores
  3.66 (oracle, at-bar) / 3.99 / 4.18 vs a 3.5 bar. ~40% of the original
  excess was a selection artifact; the rest is real.

**Three principled levers tried and closed, each with a mechanism:**
pseudo-octave augmentation (retired — band-limited copies teach a
rolled-off finest-band law), cascade base at inference (dead — input
arrangement does not survive fixed weights), tidal-frame conditioning
(descoped at depth 1; its premise fires only at depth 2).
**Doctrine that came out of it:** output texture is a property of the
WEIGHTS. Corrections must act on training or on outputs, never on inputs.

**The organizing law:** every texture pathology scales with extrapolation
DEPTH, none with data availability alone.

**Two record corrections you must not lose (R48/R49):**
1. The stage-3 "fresh seeds are fragile" caveat was largely a bug (the
   checkpoint selector used the sandbox reference instead of gowerstreet).
   Under correct selection all picks cluster late. RETRACTED.
2. The declared-resolution peak claim ("peaks pass at σ_s ≥ 0.5 px") holds
   for the shipped blind config as measured, but does NOT generalize:
   2 of 3 corrected seed legs carry a significant excess (+4.6–7.8%, CIs
   exclude 0). The claim is CONFIG-SPECIFIC. The paper must say so.

## 3. The open decision (yours, not a session's)

The paper's framing. Three honest options, unchanged from R49:
- **A — narrow-and-ship:** claim what is true (exact scales, validated
  marginal/wavelet/spacing tiers, the calibration tool, the audit
  framework, certificates), present peaks/topology as located,
  mechanism-explained, certificate-quantified boundaries.
- **B — one more fix first:** the texture-aware checkpoint cage (select on
  topology at declared resolution). ~0.3 H100-h. Targets the residual
  selection gap directly. Needs a Goodhart-discipline prereg.
- **C — audit-as-product:** lead with the reliability framework, the
  generator as its case study. (Andreas leaned C on 2026-08-07.)

Reconvene recommendation stands: **B then ship under A**, with C as the
venue/abstract framing if aiming ML-methods. C and A are not exclusive —
C is a framing choice, A is a claim-scope choice.

**Nothing else is blocked on this.** Section 5's tasks are all
framing-independent.

## 4. How to work in the low-token month (operating rules)

The expensive pattern was: orchestrator session + executor session +
pre-registration ceremony. **Do not reproduce that.** It bought
trustworthiness during discovery; discovery is over.

Rules for cheap sessions:
1. **One task per session, from Section 5, by its ID.** Paste the task
   block verbatim as your prompt. Do not ask a session to plan.
2. **No subagents, no workflows, no fan-outs.** They were authorized once,
   for one night, deliberately.
3. **No new experiments beyond Section 5.** If a session proposes one,
   the answer is no — write it in `IDEAS-PARKED.md` instead.
4. **Keep pre-registration only where it still pays:** the ONE-SHOT blind
   run (T3) and anything that will appear as a paper claim. Everything
   else is writing or measurement on committed artifacts.
5. **The two absolutes survive:** every number in the paper is copied
   verbatim from a committed artifact (R12), and JUDGE-2 (persistent
   homology) stays quarantined until the one-shot.
6. **Verify cheaply:** `source env.sh && python -m pytest tests/ -q` and
   `JAX_PLATFORMS=cpu taskset -c 0-3 ~/wl-challenge-env/bin/python -m
   pytest tests_wfm/ tests_p2/ -q`. Both green before any commit.
7. **Login-node rule:** never run the suites or numerics on a login node
   without `taskset`; anything real goes through SLURM (see CLAUDE.md).

## 5. The task queue (each is one cheap session; ordered by value)

**T1 — conformal certification run** (cleared by R49; ~CPU only).
Spec: `log/2026-08-06-prereg-conformal-DRAFT.md` + amendment C-a in
`log/2026-08-07-reconvene-d1-adjudication.md`. Rank-uniformity test:
n≈50 held-out coarse fields disjoint from training/selection/blind sets,
m completions each, rank of the true statistic within the generated
ensemble, per peak bin and starlet scale; the trusted-octave zero-width
corollary; identical run on the corrected oracle for the certified
extrapolation-cost band. Include the declared-resolution peak bins (C-a) —
they quantify the config-specificity finding. Deliverable: artifact +
one log entry. This is the paper's certificate machinery and the highest
remaining value per token.

**T2 — the W2 decomposition theorem + negative lemma** (writing, no
compute). Prove: because the wavelet transform is an isometry, squared W2
error decomposes additively across octaves and the conditioned octaves
contribute exactly zero. Then the companion lemma: peak counts and Euler
characteristic are not L2-Lipschitz, so W2 provably cannot certify them —
which is why conformal (T1) is the complementary certificate, not a
duplicate. Deliverable: `paper/theory-w2.md` (statement, proof, and the
two-sentence version for the intro). Spec/context: `BRIEF-foundations.md`
item 3.

**T3 — the second blind shot** (ONE shot, pre-registered; ~0.3 H100-h).
Spec: `log/2026-08-06-prereg-d2-blindshot2-DRAFT.md` + amendments in R49
(pin the config first; pre-state that JUDGE-2 will likely FAIL — a pass
would be the surprise). Requires the config decision from Section 3.
Do NOT fire this until the paper's claim set is settled — it is the last
untouched judge and it fires once.

**T4 — paper claim-set update** (writing, no compute). Apply the R48/R49
corrections to `paper/`: 05-boundaries (retract checkpoint fragility;
restate the declared-resolution peak claim as config-specific with the
seed-leg numbers), 03-validation (corrected-selection table alongside the
bugged one), 04-mechanism (the depth law; the three closed levers with
their mechanisms; the weights-not-inputs doctrine), 07-appendix (the truth
bug as an instrument lesson; the corrected calibration scorecard).
R12 applies: copy numbers from `results_p2/d1_d1_*_verdict.json`,
`night3_*`, and the R48/R49 rulings only.

**T5 — texture-aware cage** (only if you choose option B; ~0.3 H100-h).
Single variable: replace the marginal-only checkpoint selection with one
that also sees topology at declared resolution. Goodhart discipline:
JUDGE-2 stays the untouched arbiter and must never be the selection
statistic. Prereg with weights before running.

**T6 — venue + title** (decision, no session needed). `paper/VENUE.md`
has both intros; `paper/00-title-abstract.md` has three titles.

## 6. What is deliberately NOT on the queue

The PM/N-body super-resolution ambition (gated on paired matched-IC data
you do not have — this is the next project, not this paper); cosmology
parameter conditioning; a second simulation family; 3D; the
texture-transfer corrector; cascade training; tidal at depth 2. All are
real and all are parked with their rationale in `BRIEF-foundations.md`
and the R47/R48 rulings.

## 7. Map of the record

- `PROJECT-EXPLAINER.md` — the codename-free scientific account (§8 is the
  PM application analysis; §10 the skeptic Q&A).
- `BRIEF-foundations.md` — the 18-agent research sweep: ranked ideas,
  now/next-paper/horizon, the two foundational items (theorem + conformal).
- `PLAN-phase3.md`, `NIGHT-ORDERS-{2,3}.md` — the campaign plans.
- `log/` — 49 numbered rulings; the ones that still bind are R43 (claim
  set), R47 (diagnosis), R48 (truth bug), R49 (D1 + the scope fork).
- `paper/` — the method-first skeleton (00→07), currently paused and
  carrying the pre-R48 claim set until T4 lands.
- `~/claude-notes/orchestration/reconvene-handbook.md` — how this campaign
  was run (patterns, scars, the calibration ledger).

## 8. The machinery (added at wind-down — use it, it saves tokens)

- `./check.sh` — the ONE verification command (both stacks, correct
  interpreters, cpu-pinned). Run before committing code.
- The Stop hook now runs the suites **only when executable files are
  dirty**. Writing turns skip it. (It used to run ~3 min of pytest on the
  login node after every turn, including doc-only turns.) The old
  unconditional version is kept at `.claude/hooks/test-gate.sh.full-backup`.
- `NUMBERS.md` — canonical paper figures, GENERATED from the artifacts by
  `scripts_p2/emit_numbers.py`. Never retype a number; regenerate:
  `~/wl-challenge-env/bin/python scripts_p2/emit_numbers.py > NUMBERS.md`.
  Anything it cannot find prints MISSING — it never guesses.
- `CLAUDE.md` (auto-loaded) is now a router: it points here, states the
  operating rules, and maps where things live. Keep it short — it is
  loaded into every session and long files there cost tokens forever.
- Historical campaign docs moved to `docs/archive/` so the root shows only
  live files (31 → 11).

## 9. Progress checklist (tick as you go — this is the shared state)

- [ ] T1 conformal certification run
- [ ] T2 W2 theorem + negative lemma (`paper/theory-w2.md`)
- [ ] T4 paper claim-set update (R48/R49 corrections)
- [ ] Framing decision A / B / C (Andreas)
- [ ] T5 texture-aware cage (only if B)
- [ ] T6 venue + title (Andreas)
- [ ] T3 the second blind shot (LAST, fires once)
- [ ] Full draft read + submission decision
