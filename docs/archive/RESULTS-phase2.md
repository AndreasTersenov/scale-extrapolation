# RESULTS — Phase 2 overnight run (2026-07-16, NIGHT-ORDERS)

# MORNING SUMMARY

## VERDICTS (rule-applied; mechanical)

| gate / prediction | rule outcome | standing weight | executor weight |
|---|---|---|---|
| **Gate A** (instrument calibration) | **PASS** — all octaves, both metrics (≤0.6% var_slope, ≤5.0% kurtosis) | P-A 90 hit | 92 hit |
| **P-B1a** (locality: r* ≤ 6 at oct 1+2) | **FIRES** — r*=0 by the rule, r*≈1 by eye (one 9% drop then flat to r=12) | 70 hit | 85 hit |
| **P-B1b** (anisotropy, amended diff-in-diff rule) | **does NOT fire** — z=0.91 (needs 3) | 50: NULL side | 35: leaned right |
| sandbox shape control "NULL" (executor 90) | **MISS as worded** — z=−3.09, systematic (split-half confirmed); design error caught by the control, verdict rule amended PRE-reading | — | 90 missed |
| **C1 sandbox branch** | **B-C1-TAILS both arms** (dispersion alive both levels 4.5–8.7%; kurtosis fails oct 2 at 18–21%; no collapse) | — | modal COLL 35 wrong; TAILS 25 fired |
| P-C1a (dispersion alive) | **HIT** | 60 hit | 55 hit |
| P-C1b (kurtosis at truth) | **MISS** | 45 miss | 30 leaned right |
| P-C1c (recursion calibrated) | **HIT** (compounding ≤1.4% on sandbox) | 45 hit | 25 miss (over-weighted compounding) |
| C1 gowerstreet-leg trigger | **PASS** → leg 2 ran (descriptive) | — | — |
| C1 gowerstreet expectation (registered pre-harvest) | "mild <10% compounding" fired (−9/−10% at oct 2) | — | modal 65% "reappears ≥15%" MISSED |
| **B2 crops** | landed via bounded fallback: **N_eff ≈ 13.7 ≈ parent count at EVERY stride** — crops buy ~nothing; the currency is parents | expectation "stride-64 useful ≥0.5" | 60 **MISSED** (measured 0.054) |

Chore 0 done (Stop hook gates both stacks, 14+32+21 tests green). Budget: ~0.45
H100-h GPU + CPU jobs, of the 3 H100-h cap. All preregs/amendments committed before
the numbers they govern; two pre-readout amendments (normconv truth; shape-null
rule) with timing disclosed inline.

## THE THREE NUMBERS THAT MATTER

1. **1.02–1.07 vs truth 1.070, flat through 20k steps** — plain conditional FM +
   8× D4 augmentation holds truth-level conditional dispersion in the naked
   ODE-pushforward channel on the exact-truth sandbox. No collapse, no variance
   head, no churn. (results_p2/c1_sandbox.png, left panel.)
2. **−27% → −9/−10%** — the gowerstreet oct-2 end-to-end compounding deficit,
   frozen 4a NLL-head generator vs tonight's plain-CFM generator, single variable
   = the sampling channel. Most of the phase-1 "compounding cap" was the NLL
   head's sampling noise, not an informational limit.
3. **r* ≈ 1 coarse pixel** — conditional predictability of gowerstreet detail
   saturates essentially at the nearest coarse ring (9% drop at r=1, flat to
   r=12); the amplitude channel adds only a slow ~4% by r=12.

## INTERPRETATION (executor's reasoning; separate from verdicts by design)

**Belief updates, with numbers.**
- *The collapse law was the whole head-level dispersion story.* Confirmed at
  truth-grade: cure the finite-data memorization (8× exact augmentation) and the
  pure ODE pushforward carries full conditional variance modulation (4.5–8.7% of
  exact truth). My residual belief in a structural ODE under-dispersion
  (P-C1a 55 vs standing 60) was too pessimistic.
- *The attempt-5 "informational compounding limit" needs re-scoping.* Ranked
  explanations for the phase-1 27% compounding, given tonight: (1) NLL-head
  sampling noise degrading the coarse manifold octave-over-octave — supported by
  the single-variable −27→−10 move and by the head's spatially-white conditional
  noise (the step-3 mechanism); (2) a residual ~10% genuinely informational /
  texture component on the real field (sandbox shows ≤1.4%, so it is
  field-structure-dependent); (3) drift per se — REFUTED as the driver (the
  sandbox drifts 1.23→0.75 and barely compounds). The attempt-5 discriminator
  measurement itself (end-to-end = drifted-input response) remains valid FOR THE
  NLL-HEAD GENERATOR it was run on; its generalization to "no train-side rescue
  can exceed ~0.75" does NOT transfer to the CFM channel — tonight's generator
  reached 0.90–0.91 without any anti-compounding lever. The reconvene should
  decide how RESHAPE-MEMO/paper language absorbs this: "a measured compounding
  cap OF THE VARIANCE-HEAD SAMPLER, consistent with an informational limit only
  in that channel" is the honest revision candidate.
- *Tails are the frontier, and they compound.* Head-conditional kurtosis −12% at
  oct 2 on gowerstreet (−18/−21% on the sandbox vs exact truth) growing to
  −26/−34% end-to-end. The ODE pushforward from a Gaussian base under-produces
  tail weight even when second moments are right — exactly C3's (energy-score /
  CRPS) target, and consistent with Gate-0's reading that calibrated-spread
  training is the mature answer to one-sample-per-condition.
- *The peak audit catches what dispersion calibration misses.* The near-calibrated
  C1 generator tilts the peak function UP at all thresholds (+5.4…+9.2σ), a
  different signature than the NLL-head's ±flip. Two independent generators, two
  different higher-order failure modes, both invisible to power-level checks —
  the audit-paper thesis now has a second exhibit.
- *The dial (Stage D's bet) changed shape.* At the extrapolated octave the
  scale-blind arm A (0.88 of real end-to-end; 0.98 head-conditional) BEATS the
  dial-conditioned arm B (0.84 / 0.90): the coarse input itself carries most of
  the scale information in this channel, and the OOD FiLM coordinate mildly
  hurts. The Stage-D question is no longer "does the dial extrapolate" but
  "does the dial add anything beyond the coarse field's own information at the
  extrapolated scale" — a cleaner, harder question.
- *Instrument trust.* The whole phase-1c verdict stack rests on a now-calibrated
  ruler (Gate A ≤0.6% on var_slope); the N=64 kurtosis wobble (21% at oct 3) and
  the raw-vs-normalized convention shift (−3…−19%) are the two bookkeeping
  cautions filed for any cross-record comparison.

**Named temptations check (rider):** the C1-gowerstreet result is exactly the
shape of finding that invites overselling ("the generator works!"). The verdict
tables say: dispersion near-calibrated, tails fail at 3–9σ equivalents, peaks
+5–9σ biased, extrapolated octave −12/−16% — a much better generator that still
fails the audit. Also: I resisted running a churn/late-checkpoint variant on the
tails failure (would have been variant N+1); it is drafted below instead.

**DRAFTED preregs (NOT run; for the reconvene to authorize/edit):**
1. **C3-draft (tails):** energy-score/CRPS-trained detail sampler (Pacchiardi
   patched energy score; AIFS-CRPS "almost fair" variant), single variable vs C1
   = the objective; sandbox-first with the C1 bars + kurtosis primary; prediction
   sketch: P(kurtosis within 15% of exact truth at oct 2–4) ~55%.
2. **C2-draft (locality):** conditioning receptive field capped at r=2 coarse
   pixels (B1-measured r*≈1 + margin), single variable vs C1; primary question is
   now capacity/data-efficiency (collapse onset vs training-set size at matched
   augmentation), since C1 already holds dispersion at 322 tiles; prediction
   sketch: P(no degradation vs C1) ~70%, P(measurably later collapse onset when
   augmentation is REMOVED — the clean capacity test) ~55%.
3. **Shape-control-draft:** matched-spectrum isotropic control for the anisotropy
   test — phase-randomized gowerstreet tiles (|FFT| kept, phases uniform) through
   the identical shape machinery; replaces the sandbox as the baseline in the
   diff-in-diff verdict.
4. **NLL-head forensic (cheap, descriptive):** regenerate from the FROZEN 4a
   checkpoints with the NLL head's noise injection DISABLED at generation (mean
   path only) and score end-to-end var_slope: if the deficit closes toward −10%,
   the channel-dependence attribution above is confirmed within the frozen
   phase-1 artifacts themselves.

**Open questions for the reconvene.**
- Does the RESHAPE-MEMO/paper wording on the compounding cap get revised now, or
  after the NLL-head forensic (draft 4)?
- C3 before C2 (tails are binding; locality is a capacity question)?
- Adopt the normalized-convention truth as the standard reference for all
  generator-side scoring (and re-state the coords dial in the same convention)?
- The r*-rule and running-peak-rule sharpness issues (both fired tonight in
  benign ways) — bar-design ledger entries #6/#7?
- B2 answered: crops are dead as a data lever (N_eff ≈ parent count at every
  stride, incl. DISJOINT tiles) — but the finding cuts deeper: the training set's
  law-level diversity is ~parents (~30), not ~tiles (322). Does this reprice the
  collapse-law "finite data" language, and does it promote constrained-realization
  ensembles / more parents over any within-parent augmentation for future arms?

---

## A — the lognormal sandbox & Gate A (exact conditional truth; instrument calibration)

Prereg log/2026-07-16-stageA-prereg.md; readout log/2026-07-16-stageA-readout.md;
figure results_p2/gateA.png; job 16490857 (first submission died on the POSIX-sh
`source` gotcha, JOBS.md).

- Built: fully synthetic lognormal sandbox (alpha=2.0, sigma_g=0.6, 128²; recipe +
  seeds in sandbox/recipe.py), 256 parents × 64 EXACT conditional redraws given the
  level-4 Haar Gaussian coarse (Hoffman–Ribak; validated tests-first against dense
  linear algebra at machine/statistical tolerance — tests_p2/test_sandbox_conditional.py).
- TRUE conditional statistics (raw convention): var_slope 1.259→0.809, kurtosis
  7.83→2.68 across octaves 1→4 (results_p2/sandbox_truth.json; batch-means SEs).
  Same-convention truth for generator scoring (per-tile normalized):
  var_slope 1.229→0.755, kurtosis 7.02→2.10 (sandbox_truth_normconv.json).
- **GATE A: PASS at every octave, both metrics** (var_slope rel err ≤0.6%, kurtosis
  ≤5.0%, bars max(5%/10%, 3·SE_rel)). P-A standing 90% and executor 92% — both hit.
- Descriptive: at N=64 (phase-1c production sample size) the instrument's oct-3
  kurtosis wobbles 21% low — phase-1c kurtosis FAILURES (50–60% deficits) are
  unaffected, but near-bar kurtosis PASSES should be read with this wobble in mind.
- **Measured convention effect (pre-readout amendment,
  log/2026-07-16-c1-amendment-normconv.md):** per-tile normalization (the wfm
  pipeline's world) shifts the estimand −3…−6% (var_slope) and −12…−19% (kurtosis)
  vs raw-field truth — comparable to adjudication bars; C1 was adjudicated against
  same-convention truth. Phase-1c itself was like-for-like (unaffected); any
  cross-convention comparison (stage-0 raw dial vs generator-side numbers) must
  mind this.

## B1 — dependence range & shape (model-free)

Prereg log/2026-07-16-stageB-prereg.md (+ pre-reading amendment
log/2026-07-16-stageB-amendment-shape-null.md); job 16491648; figure
results_p2/stageB.png; data results_p2/stageB1_{curves,shape}.json. Estimators
validated tests-first against EXACT Gaussian conditional variance
(tests_p2/test_depmeasure.py, test_shape.py — ridge and kNN within 10% of
closed-form truth at every radius).

- **Locality (P-B1a): FIRES — saturation within ~1 coarse pixel.** Gowerstreet
  V(r)/V(0) (mean channel, ridge): one ~9% drop from center-only to r=1, then FLAT
  to r=12 at every octave (oct-2 profile: 1.000, 0.911, 0.906, 0.904, 0.903, 0.894,
  0.902). By the pre-registered mechanical rule r*=0 (the 3·SE(r_max) threshold is
  as large as the whole drop — a bar-sharpness note, same class as bar-miss #5);
  by eye r*≈1. Either way r* ≤ 6 at octaves 1 AND 2 → **P-B1a fires** (standing
  70%, executor 85% — both hit). Sandbox control: same shape (18% drop to r=1,
  flat; r*=1 by the rule — its SEs are 6× smaller).
- **Methods finding:** kNN on annulus MEANS is flat at 1.00 on both fields — the
  near-context information is DIRECTIONAL (gradient-like raw-pixel combinations),
  not radial-mean; and the amplitude channel (|w| target) shows only a weak, slowly
  accumulating trend on gowerstreet (~4% by r=12, not clearly saturated — long-range
  variance correlations exist but are small).
- **Shape/anisotropy (P-B1b): does NOT fire.** The isotropic sandbox control fired
  at z=−3.09, catching a design error in my prereg (the "exchangeability ⇒ zero
  null" claim was wrong — the orientation classifier SELECTS locally-anisotropic
  noise; amendment committed before reading gowerstreet). Diagnostics (4 runs,
  JOBS.md): the negative baseline is systematic (split-half −0.19%/−0.29%;
  gate-off shrinks it — selection mechanism confirmed). Amended verdict
  (difference-in-differences vs the sandbox baseline): gowerstreet oct-2 amplitude
  channel +0.51% ± 0.56% → **z=0.91, NULL** at a measured ~0.5%-of-V sensitivity
  (w channel: −0.16%, z=−1.2). Standing 50% / executor 35% — the NULL outcome
  favors the executor's lean. Filament-transported conditional structure is not
  detectable at matched area on 128² tiles with this machinery; a matched-spectrum
  isotropic control (phase-randomized gowerstreet) is the drafted follow-up.

## B2 — crops inventory

Jobs: 16491648's B2 leg CANCELLED at 1:05 elapsed (executor call, rider grant 5 —
`iter_parent_maps` with max_shards=None reads ALL 256 gowerstreet shards ≈128 GB
before yielding; could not finish in budget); fallback job 16492950 (12 parents,
first 3 shards — bounded IO, physical-diversity caveat) COMPLETED. Data:
results_p2/stageB2_crops.json.

| stride | n_crops | N_eff (var_slope) | N_eff (detail_std) | N_eff_min / n |
|---|---|---|---|---|
| 128 (disjoint) | 132 | 15.4 | 13.7 | 0.10 |
| 64 | 252 | 15.4 | 13.7 | 0.054 |
| 32 | 984 | 15.2 | 13.8 | 0.014 |

- **The inventory answer: crops buy ~nothing, and the real currency is PARENTS.**
  N_eff_min ≈ 13.7 at EVERY stride — approximately the parent count (12) — i.e.
  even DISJOINT 128² tiles within one parent map are nearly redundant at the
  law-summary level, and stride-shifted crops add nothing beyond that. The
  pre-registered expectation (stride-64 N_eff/count ≥ 0.5, 60%) **missed
  decisively** (0.054).
- Scope: N_eff is defined on per-crop SUMMARY statistics (octave-2 var_slope,
  detail_std) — it measures between-tile statistical redundancy (law-level
  diversity), not pixel-level conditional-pair counts; and the fallback used 12
  parents from 3 shards (caveat). Both stated per prereg/fallback notes.
- Consequences: (a) C1's no-crops choice is validated post hoc; (b) the phase-1
  "322-tile" training set carries roughly PARENT-count (~30) effective diversity
  in law-level statistics — the collapse law's "finite data" is finiter than the
  tile count suggests, which sharpens the C2 capacity question and the
  constrained-realization (item-1) motivation for any future phase.

## C1 — vanilla CFM + augmentation (the un-run arm), sandbox leg

Prereg log/2026-07-16-prereg-c1-sandbox.md (branch weights pre-registered);
amendment (normconv truth) pre-readout; readout
log/2026-07-16-c1-sandbox-readout.md; verdict table
results_p2/c1_verdict_sandbox.json; figure results_p2/c1_sandbox.png; job 16491750,
config_hash ab9c175f59.

- **Branch: B-C1-TAILS (both arms).** No collapse signature — the oct-2
  head-conditional var_slope curve RISES to 1.02–1.07 (truth 1.070) and holds
  through 20k steps. Dispersion within bars at octaves 2,3,4 BOTH levels
  (head-conditional 4.9–6.8%; end-to-end 4.5–8.7%; bars 10–11.7%). Kurtosis FAILS
  at oct 2 (18.4–21.3% low vs 15–16.1% bars, both arms both levels; arm B also
  end-to-end oct 3).
- **Headline 1: the collapse cure works in the naked ODE-pushforward channel.**
  Plain conditional FM + 8× D4 augmentation, no variance head, reaches and HOLDS
  truth-level dispersion (phase-1 un-augmented: peak 0.97@2k → 0.75@10k). On a
  calibrated bed, phase-1's dispersion pathology is fully accounted for by
  finite-data mean memorization.
- **Headline 2: compounding is NOT generic to drifting fields.** The sandbox
  drifts (1.23→0.75) yet recursion from octave 4 costs ≤1.4% in var_slope at every
  trained octave (phase-1 gowerstreet comparator: −27% at oct 2). The
  "off-manifold texture" component of the informational-limit account is now the
  only surviving candidate for gowerstreet's compounding; the triggered
  gowerstreet leg is the direct discriminator.
- What still fails: tails (oct-2 kurtosis −19/−21%; extrapolated-octave
  DESCRIPTIVE: var_slope −7…−11%, kurtosis −22…−31% — the phase-1 pattern in
  miniature, far milder).
- Prediction verdicts: P-C1a standing 60 HIT / executor 55 HIT; P-C1b standing 45
  MISS / executor 30 leaned right; P-C1c standing 45 HIT / executor 25 MISS (I
  over-weighted compounding); branch: executor modal COLL 35 wrong, TAILS 25 fired.

## C1 — gowerstreet leg (descriptive; triggered by the sandbox dispersion pass)

Job 16491989, config_hash 6f9cdca680; readout
log/2026-07-16-c1-gowerstreet-readout.md; tables
results_p2/c1_descriptive_gowerstreet.json; maps results_p2/c1_maps.png; peaks
results_p2/c1_peaks_gowerstreet_descriptive.json. Registered expectation: my modal
"compounding reappears ≥15%" (65%) MISSED — the 20% "mild" branch fired.

- **Oct-2 end-to-end compounding: −27% (phase-1 NLL-head 4a) → −9/−10% (plain
  CFM), single variable = the sampling channel.** The attempt-5 "measured
  compounding cap" is substantially CHANNEL-DEPENDENT: ~17 of the 27 points
  belonged to the NLL head's sampling noise degrading the coarse manifold, only
  ~10% remains for the informational component on gowerstreet (and ≤1.4% on the
  in-class sandbox).
- **Extrapolated octave no longer broken:** end-to-end var_slope ratios 0.88/0.84
  (phase-1 P5: 0.55); head-conditional arm A at 0.98 of real. Arm A (scale-blind)
  BEATS arm B (dial) at the extrapolated octave — the OOD FiLM coordinate hurts in
  this channel; the Stage-D dial question changes shape.
- **Tails are the binding deficit and they compound:** oct-2 kurtosis −12%
  head-conditional → −26/−34% end-to-end.
- **DESCRIPTIVE peaks:** biased UP at ALL thresholds (+5.4…+9.2σ, no sign flip;
  vs the NLL-head generator's +14/−12 flip) — a near-dispersion-calibrated
  generator still fails the higher-order audit; second independent exhibit for
  the audit paper.
- Arm B's ckpt curve technically fires the running-peak rule on a RECOVERING 4k→6k
  transient (0.995→0.885→1.016) — the oscillation-vs-collapse rule weakness again;
  descriptive, filed to the bar-design ledger.

## C3 — patched-energy-score direct sampler (2026-07-17, R10): BLOCKED at the
## pre-training gate; GPU leg not run

STATUS: the prereg (b42edd9, approved R10 with conditions) was NOT executed. The
R10 condition-2 gate investigation measured a mechanism-level property of the
objective before any training spend: **the β=1 patched energy score learns
asymmetric tail weight and does not learn symmetric tail weight** — and Haar
detail marginals are exactly symmetric by construction (D4-augmented law: w ~ −w),
with kurtosis as the arm's PRIMARY bar.

Evidence (all held-out eval, production net + production patch config, 768-field
toys; figure results_p2/c3_gates.png; chain log/2026-07-17-c3-gate-design.md;
blocker + options log/2026-07-17-c3-blocker-symmetric-tails.md):

| toy | truth | recovered | job |
|---|---|---|---|
| exp skewness | 2.0 | 2.014 | 16605903 |
| exp excess kurtosis | 6.0 | 5.76 | 16605903 |
| t(5) excess kurtosis | 6.0 | **0.49 (flat 2.5k→4k)** | 16606069 |
| modulated-σ × t(5) pooled kurtosis | 5.96 | **1.10 (flat 2.5k→4k)** | 16606223 |

What survives: the full C3 implementation (arms_p2/c3/, runner/sweep/scorer/SLURM)
is committed, smoke-tested, gate-validated, runnable in minutes if the reconvene
amends the objective; the R10 condition-1 instrument (tail_q999) is validated with
its normconv truth (results_p2/sandbox_truth_q999.json: oct 1–4
5.974/5.532/5.103/4.591, SE ≤0.03); the two-moment + skewness gate itself PASSES
(the failure is specific to symmetric tails). Gate-design lessons logged
(capacity is part of a toy's validity regime; train-eval one-obs-per-conditioning
toys are memorization traps for proper-score training).

Reconvene options (executor's ordering in the blocker log): threshold-weighted
CRPS/ES > β<1-as-cheap-test > t-base flow (new prereg) > transform-space scoring >
run-anyway-as-negative. Any objective variant can be gated by the SAME t(5) field
toy in ~15 min CPU before a prereg amendment.

UPDATE (2026-07-17, R13 bake-off — log/2026-07-17-c3-bakeoff.md, figure
results_p2/c3_bakeoff.png): all three authorized candidates FAILED the
pre-registered selector (final held-out kurt ≥ 4.0 on both gates) — twcrps
3.01/3.57 (plus spurious skew ~+0.51 on a symmetric target), beta05 0.51/1.05,
tbase 3.56/2.98. twCRPS and t-base both CROSS the bar mid-training (5.14/4.80 and
4.11/5.13) then converge away under continued training — reaching the right tail
mass transiently is easy; HOLDING it is the hard problem. Both panels'
P(≥1 qualifies)=85 missed. ATTRIBUTION per the PRE-REGISTERED no-qualifier reading
(adversarial-sweep addendum, 9a43bb6, committed before the verdict): the suspect
is the SAMPLER/OPTIMIZER pathway, not the objective family — three structurally
different objectives (including plain L2-CFM, which actively removes the heavy
tails its t(5) base supplies) fail the same statistic the same way; reconciliation
section in the bake-off log. STOP per R13; next-step menu is
architecture/optimization-flavored and belongs to the reconvene.

UPDATE 2 (2026-07-17, R15 tail-dynamics diagnosis — log/2026-07-17-taildyn-
readout.md, figure results_p2/taildyn.png): **the rung-4 tail decay is
DATA-LIMITED.** At 1× toy data every run terminally collapses (final kurt
1.07–1.66 vs truths 4.15/5.96); at 8× every run ends at or near truth and HOLDS
(tbase composite 6.11/5.96 stable over steps 6k–12k; t5flat 4.32/4.15; twcrps
4.78 and 4.57) — rung 2's data-size cure reproduced at rung 4, the moment-ladder
hypothesis's causal signature (mechanical P1: TD-PARTIAL via the pinned onset
rule, whose dip-sensitivity under-reports the final-state pattern; rule lesson
logged). Joint near-truth checkpoints (dispersion ≤10% AND kurt ≥4.0) EXIST even
at 1× (P2 yes, both panels' weights hit). P3: the 1×-trained flow progressively
ERASES its heavy-tailed base (t-base and Gaussian-base outputs converge to the
same collapsed tails — base-independence as a memorization symptom); at 8× the
base's contribution survives to 12k. twcrps develops spurious skew that GROWS
with data (+0.8…+1.1) — disfavored as a lever. Next-arm decision at reconvene
with this in hand.

## C1-t — t(5)-base CFM + validation-selected early stop (2026-07-18, R17):
## SANDBOX C1T-CAL BOTH ARMS; gowerstreet binding-octave deficit closes

The first full pass of the campaign's arm series. Single variable vs C1 = base
distribution + pre-registered selection rule (Goodhart cage: val-32 selection /
test-32 adjudication, one pick). Readout log/2026-07-18-c1t-readout.md; figure
results_p2/c1t.png; prereg 6822ced (pre-submission).

- SANDBOX (adjudicated): **all 24 bars pass, both arms** (kurtosis primary +
  dispersion, octaves 2–4, both levels). Selected A@7500/B@2500; hc oct-2 kurt
  5.37/5.38 vs truth 4.917; q999 0.94–1.06. Texture: 32-field SEs widen kurtosis
  bars (22–38%); two octave-4 entries would miss the bare 15% floor; validation
  curves are noisy and both picks rode favorable spikes — held at TEST.
- ATTRIBUTION (zero cost): C1's Gaussian-base ceiling on the same protocol is
  kurt 4.29/4.59 at ANY checkpoint (truth 4.92) — the t base raises the
  reachable ceiling; selection places the pick; neither alone suffices.
- GOWERSTREET (descriptive): cage picks LATE (A@16000/B@18500 — no early decay
  to dodge on the heavier-tailed real field). e2e oct-2 kurtosis deficit
  −33.7→−4.0% (A) and −26.1→−8.0% (B) — **the standing halving prediction fired
  both arms** (reconvene 50/executor 45); dispersion at real (+1.9/+1.4% vs C1's
  −9.7/−9.2%). Remaining structure: extrapolated octave (−29/−40%) and octave 4
  (mixed; B worse than C1) — the Stage-D frontier.
- Scorecard: executor modal C1T-CAL 35 fired; all four registered lines on the
  winning side both panels; no gate branches fired.

## STAGE D — the bet, answered (2026-07-18): **D-PASS-BOTH — single-octave scale
## extrapolation WORKS, and works SCALE-BLIND.** R18 replication: 24/24.

The plan's final experiment (prereg a0b6e4c; readout
log/2026-07-18-stageD-readout.md; figure results_p2/stageD.png). Slide-the-edge
on gowerstreet, C1-t substrate, octave 2 held out, arm-B dial from the
deployment coupling curve (edge input error vs +2.9% / kurt +18%), selection
caged at octave 3.

- ALL 8 edge checks pass — under the formal bars AND the bare 10%/15% floors
  (A-hc-kurt 14.9% by 0.1%; every e2e number ≤7.9%). **P-D (PLAN standing 40%)
  PASSES.** Formal kurtosis bars inflated to ~57% by the noisy 32-field real
  reference — stated with the pass; the floors carry the verdict.
- **The dial adds nothing marginal**: pinned dial-beats metric FALSE (B 4.4% vs
  A 3.2% e2e kurt); the scale-blind arm extrapolates the edge from the coarse
  field alone — B1's r*≈1 confirmed at deployment level. The dial DOES change
  joint structure: peak-audit excess flips sign under B (−9/−17% at ν≥2.5) vs
  A's up-tilt (+13/+15%) — neither closes the audit.
- Two-octave extrapolation (oct 1) degrades (−44/−54% kurt): each unseen octave
  costs; the single-octave claim is the supported one.
- R18 replication (64 fresh fields, seed 20260720, frozen picks): **24/24 —
  C1T-CAL replicated**; arm A clean (0.0–12.4%), arm B tail-hot at oct 4
  (41% on a 56% SE-widened bar — the one bar-carried entry).
- Scorecard: executor modal D-FAIL-BOTH 55 large MISS — the fourth consecutive
  constructive under-confidence, now against an explicit R20 warning; the
  PLAN's original P-D 40 was the best-calibrated number in the room.

## Starlet-ℓ1 held-out characterization (2026-07-18, SPEC-starlet-l1):
## SL1-PASS all three legs; the tier question FIRES

Andreas's wl_stats_torch (starlet ≠ Haar; ℓ1 never in any design/selection
loop), verified + gated (5 tests_p2 gates; two instrument findings flagged:
unused gen2-reconstruct mismatch; zero-pad noise-plane border artifact →
plateau-normalization convention, rot90-exact). Prereg with executor lines
committed pre-scoring (fa56ff4); details log/2026-07-18-starlet-l1-readout.md;
figure results_p2/starlet_l1.png.

- **Arm A passes the 1σ-w-10%-floor at every scored scale on all three legs**
  (sandbox ≤2.0%; gowerstreet trained ≤1.5%; Stage-D edge headline +4.0% at
  the held-out octave). Arm B passes all scored scales too, but its off-binding
  pathologies show at descriptive octave 1: sandbox −11.4%, edge +23.9%.
- **P-SL1-blind FIRED** (rec 70 / exec 75): ℓ1 passes where the peak audit
  fails (+13/+15%) — even the field's most constraining summary sits below the
  placement tier; position-blind statistics cannot see where the extremes land.
  Descriptive: the edge 4px tail share tilts +5.9% rel, a sub-bar hint of the
  same excess. What remains is placement, not amplitude.
- Taxonomy panel: one statistic separates the generations — 4a graininess
  = +14.2% at 2px with tail share collapsed (energy without extremes); mu-only
  skeleton −84% at 2px; C1/C1-t both clean in totals, ranked only by tail
  share (~1.7σ). Robust under the survey-noise secondary (all ≤3.5%).
- Scorecard: first clean sweep — all five pre-registered lines on the right
  side, executor tighter on every line (first application of the R20
  recalibration; it held).

## Tests & infrastructure

- Chore 0 (154c30b): Stop-hook gate now covers BOTH stacks (tests/ under env.sh;
  tests_wfm/ + tests_p2/ under wl-challenge-env; pyproject lists all three trees —
  loud failure over silent skip; timeout 300→900 s; full gate green 14+32+…).
- tests_p2/ grew to 21 tests tonight (sandbox conditional sampler, B1 estimators
  incl. exact-GRF gates, shape-test nulls/controls) — all green under both stacks
  (pywt cross-checks auto-skip in the JAX env, run green under env.sh).
- SLURM: 5 jobs, all CPU except two MIG legs; cumulative GPU ≈ 0.4 H100-h of the
  3 H100-h cap. One t=0 infra failure (POSIX-sh `source`), documented.
