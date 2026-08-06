# 2026-08-05 — PREREG: NIGHT-ORDERS-3 texture campaign (committed BEFORE
# any machinery lands, any gate is read, or any job launches; R12;
# artifacts → results_p2/night3_*)

Governing: NIGHT-ORDERS-3.md + RIDER v2; R46/R47. Absolutes carried:
pre-statement before reading, R12 numbers-by-copy, tests-first,
single-committer, JUDGE-2 quarantine (zero applications to real or
generated stacks tonight), no audit statistic in any loss, GPU hard cap
4.0 H100-h, morning STOP with all calls provisional.

## N0 — harvest (already complete pre-orders)

The R47 order set ran and read out before NIGHT-ORDERS-3 arrived
(f206219): BL depth ladder, **O-NULL**, I instrument. Per N0's
pre-delegated logic the interpretation frame is set: **O-NULL ⇒
architecture-limit story favored; AUG runs anyway — the O/AUG pair
separates data-availability from architecture.** O's three unseparated
readings (capacity / cage-selection harvesting / step dilution) are
tonight's diagnostic targets. Decision ledger begins:

- DL-1: N0 skipped (already executed); its 0.12 H100-h counted against
  tonight's 4.0 cap, conservatively.

## MF status + JUDGE-2 (N1f)

The Minkowski judge is RECLASSIFIED as a development metric (we now
engineer against it). Every MF number tonight is labeled **MF(dev)**.
JUDGE-2 = persistent-homology Betti curves b0(ν)/b1(ν) (per-field
standardized superlevel sets; (4,8)-connectivity duality — components
4-conn, holes via 8-conn complement minus border; ν grid −3..3 step 0.5;
declared 0.5-px + native conventions; judge_T-style max|z| with tile
bootstrap). Tests-first on synthetic GRFs ONLY; committed FROZEN;
**applied to NO real or generated stack tonight** — its real split-half
null/bar computation is deferred to the drafted blind-shot-2 prereg
(the A4 pattern). Named temptation refused in advance: no "peeking to
see if it agrees with MF".

## GATE-T — orientation decoherence (mechanical; rec P(fire) = 60,
## **exec 40**)

Instrument (N1a, tests-first on synthetics): local-orientation alignment
between oct-1 texture and the coarse eigenframe.
u = oct-1 contribution field (frozen oct1_texture.bandlimit machinery);
θ_u from the structure tensor of u (Fourier/periodic smoothing, exact D4
commutation); θ_c + anisotropy from the Hessian eigenframe of
bandlimit(f); A_or = Σ w_u w_c cos 2(θ_u−θ_c) / Σ w_u w_c ∈ [−1,1],
D4-invariant by construction. Validation: oriented-vs-isotropic planted
synthetics discriminate at z > 5; D4 invariance at machine precision; SE
scaling ~√2. Smoothing scales are the builder's design periphery,
documented in-code, fixed before any real data is touched.

Rule (mechanical): GATE-T FIRES iff A_or(real) − A_or(gen) ≥ 3σ
(combined bootstrap SE) on the PRIMARY leg = trained
(l1pp_main_gen.npz adj1-3 pooled 96 vs arms_c1t_gowerstreet.npz real
32). Blind leg (stage3_blind_final.npz pooled 96 vs arms_stageD.npz
real) scored as descriptive corroboration only. NOT-FIRED ⇒ TIDAL
descoped tonight (the null is a mechanism finding: texture is
mis-organized in amplitude/topology but not orientation).
Executor reasoning for 40 vs rec 60: the amplitude-alignment statistic
(I instrument) read NORMAL on the primary trained leg (readout,
f206219); orientation is a different axis and may still decohere, but
the nearest measured neighbor came back quiet.
Result (verbatim, night3_gate_t.json; A-N3-1 rule): **NOT FIRED** —
primary trained leg A_real +0.1712±0.0128 vs A_gen +0.1355±0.0039,
|real|−|gen| = +0.0357, **z = +2.67 < 3**. TIDAL DESCOPED tonight (the
exec-40 side pays against rec 60). Descriptive corroboration: blind leg
A_gen +0.1114±0.0036, z = +4.50 — orientation decoherence exists and
TRACKS EXTRAPOLATION DEPTH (sub-bar at 1 extrapolated octave,
above-bar at 2), joining BL/MF/frag on the depth axis. Both real
stacks verified identical arrays (same 32 gowerstreet test tiles; the
equal A_real values are by construction). Sign note: locking is
POSITIVE (gradients lock to theta_c), the planted-synthetic negative
convention did not occur in data; the |·| rule (A-N3-1) covers both.
Mechanism note for the report: TIDAL's premise is real but sub-bar at
deployment depth 1 — a candidate for a depth-2 campaign, drafted not
run.

## AUG — design (RIDER v2 grant 3; the ×2 no-op theorem)

Registered BEFORE training, with an executable proof committed in-test:
under the Haar pyramid, ×2 downsampled copies are EXACT duplicates of
deeper original pairs — c_1(c_1(f)) = c_2(f) and d_1(c_1(f)) = d_2(f)
identically (multilevel DWT nesting) — so the orders' literal ×2
prescription adds only per-tile normalization jitter and would produce a
FAKE null. The arm therefore implements the mechanism the orders name
("trains ACROSS octave-role transitions") rather than the letter:
**band-limited FRACTIONAL rescaled copies**, which are NOT Haar-nested
and create genuinely new (coarse, detail) pairs at roles BETWEEN the
original octaves:

- g96 = fourier_resample(bandlimit(f), 128→96): role offset 0.415
  octaves; trained at its octaves {1,2,3} (sizes 48/24/12, all ÷4).
- g80 = fourier_resample(bandlimit(f), 128→80): offset 0.678; trained at
  {1,2} (40/20; its oct-3 size 10 is not ÷4 — excluded, noted).
- No-target-octave-leak BY CONSTRUCTION (copies are functions of
  bandlimit(f) alone) and IN-TEST (replace d_1(f) with noise → copies
  bit-identical). Composes with d4_augment (in-test). Caveat noted: the
  copies' finest band carries the resampling truncation rolloff (the
  degradation-model caveat, disclosed for interpretation).
- Trainer: 8 (stack, octave) slots cycled round-robin; per-octave std
  per (stack, octave) pool. **Steps 48000, ckpt-every 1000** — per-slot
  parity with the production recipe (48k/8 = 6000 vs baseline 20k/3 ≈
  6667); a 20k AUG run would undertrain every slot 2.7× and confound
  any null with dilution. Deviation registered (grant 3).
- Selection: unchanged rule, sel-octave 2 scored on the ORIGINAL stack's
  validation fields (a trained octave; same split, same score formula).
- e2e + final-config chain: IDENTICAL to baseline geometry (originals
  only; copies are training-only); std convention = std_from(gtrain,
  [2,3,4]) verbatim; fit target_octave 2 (the seed-leg deployment
  target); F2 sampler unchanged. The single variable vs the committed
  production model is the training distribution.
- Seeds 11 and 12; arm A (no cond vector); sandbox canary first
  (l1pp-canary-style dispersion check; kill: conditional-dispersion
  deficit > 40% at ckpt ≥ 4k).

## TIDAL — design (built regardless; trains iff GATE-T)

Eigenframe conditioning: features H(c) = (tr, a1, a2) = (Hxx+Hyy,
Hxx−Hyy, 2Hxy) of the Gaussian-smoothed (σ_H = 2 px, Fourier/periodic —
exact D4 commutation) Hessian of the conditioning coarse, standardized
per pool, CONCATENATED to the coarse channel (4-channel conditioning).
D4-covariance PROVEN in-test at machine precision: tr invariant;
(a1, a2) spin-2 (90° rotation → both negate; W-mirror → a1 fixed, a2
negates). F2 guarantee survives structurally: the sampler computes
features INSIDE the transformed frame — d = g⁻¹·model'(g·c) with
model'(x) = model(x, H(x)) is exactly the F2 form; asserted in-test on
the assembled sampler. Recipe otherwise verbatim (20k steps, octaves
{2,3,4}, cage unchanged); seeds 21 and 22; own sandbox canary. Sampling
via night3_tidal_white/final.py (same KEYS plumbing pattern).

## Branch rules (per arm; mechanical order gates → REGRESSED → FIXES →
## IMPROVED → NULL; ambiguity = negative; #11 bands)

| branch | rule | AUG rec / **exec** | TIDAL rec / **exec** |
|---|---|---|---|
| X-FIXES | MF(dev) declared ≤ 3.5 AND both native ν2.5/3.0 excesses < half committed (7.645/6.237%); no regression | 15 / **8** | 12 / **8** |
| X-IMPROVED | not FIXES; MF(dev) declared ≤ 2.24 AND max abs frag_z (ncomp+holes rows) ≤ 2.35; no regression | 40 / **30** | 33 / **32** |
| X-NULL | neither; no regression | 30 / **47** | 35 / **40** |
| X-REGRESSED | any must-not-regress failure | 5 / **5** | 10 / **10** |
| gates | infra (one bug-repair resubmission per job); determinism; canary kill | 10 / **10** | 10 / **10** |

IMPROVED operationalized (blind amendment, A1/G2 precedent): "≥2σ
MF(dev) improvement" = declared T ≤ 3.34 − 2×0.55 = **2.24** (baseline =
the trained-leg full-band reading, oct1fix_bl.json; σ = the committed
split-half null sd, stage3_mf_null.json); "instrument corroborates" =
max |frag z| over the ncomp and holes rows ≤ **2.35** (half the trained-
leg baseline max 4.7, oct1fix_instrument.json). Both computed on the
arm's pooled 96 finals vs gow real 32, declared convention.
**Arm branch = the WORSE of the two seeds' categories** (R-SEED-FRAGILE;
the 1-seed-improvement temptation refused in advance). Per-seed numbers
all reported. Must-not-regress battery = oct1fix_o_score.py's, verbatim
(marginals hc+e2e octaves 1–4, starlet leg, parity < 3, determinism);
P-T coloring per chain is a WATCHED line, not a branch.
Executor reasoning: O-NULL moves weight from FIXES/IMPROVED to NULL —
if direct oct-1 training can't render clean texture, a role-transition
augmentation that never sees oct-1 likely can't either; the residual
IMPROVED weight rides on the cage/dilution readings (AUG's denser role
ladder could regularize selection). TIDAL column stated unconditionally
but only spends if GATE-T fires.
Results: AUG **PENDING** · TIDAL **DESCOPED** (GATE-T not fired, z 2.67).

## CASC — inference-only transfer probe (NO cascade training tonight)

casc_base.py: log-normal-MRW-family modulated white seeds — ω a
log-correlated GRF (P_ω ∝ k⁻² in 2D; intermittency λ tuned on
synthetics by the builder, documented), M = exp(ω), ε = M·z with z
white Gaussian (ε is exactly white in second order; multifractal in
higher order), unit-variance normalized. Validation tests: whiteness
(flat ring spectrum within tolerance), D4-in-law, seeded determinism,
moment-scaling discrimination vs plain white at z > 5. Probe: ε
replaces the white Gaussian INPUT of the committed copula path (same
l1pp_filter.npz filt + z/x tables, same production ckpt, F2, oct-1
base only), finals with KEYS below, vs the committed adj baseline.
Scored (descriptive + one registered line): coloring C (watched),
I-instrument frag/alignment deltas, MF(dev).
Line: P(base phase structure measurably carried to output texture — any
frag or alignment |Δz| ≥ 3 vs adj baseline, matched convention):
rec 30 / **exec 22** (the transfer-function lesson cuts against:
the flow's spectral action was input-independent; the ODE pulls toward
the trained detail manifold; F2 averaging further decoheres seed
orientation. A carried result would be the night's most actionable
surprise).
Result: **PENDING**.

## CKPT-SWEEP — adaptive probe (grant 2; descriptive, adjudicates NO
## branch; feeds the morning lever recommendation)

e2e recursions (plain generate_recursive_tbase, the E3-comparable
convention) from dense checkpoints of: ckpt_oct1fix_oracle (armA,
{1,2,3,4}), ckpt_c1t_gowerstreet (armA production {2,3,4}),
ckpt_stage3_blind (seed 7, {3,4}); steps {2500..20000 step 2500}; scored
MF(dev) declared vs the matching real leg + frag profile + native peaks.
Pre-statements (committed before reading):
- Cage-selection reading TRUE ⇒ MF(dev) spread across ckpts > 2 with at
  least one PASS-category ckpt in a dir whose SELECTED ckpt fails.
- Capacity reading TRUE ⇒ flat-bad (all ckpts fail, spread ≲ 1).
- Dilution reading TRUE ⇒ oracle dir improves with steps (late best).
- Watched: rank correlation between the cage's selection score and
  MF(dev) across ckpts (a Goodhart exhibit if anti-correlated).
This probe is NOT the improvisation slot (no training). If it finds a
pass-category checkpoint, the morning deliverable includes a DRAFTED
(not run) selection-variant prereg.
Result: **PENDING**.

## KEYS registry (committed before any stream runs)

| tag | white (key, grng) | finals ((key, grng) ×3) |
|---|---|---|
| aug11 | (5301, 20260880) | (5311,20260881) (5312,20260882) (5313,20260883) |
| aug12 | (5401, 20260884) | (5411,20260885) (5412,20260886) (5413,20260887) |
| tidal21 | (5501, 20260888) | (5511,20260889) (5512,20260890) (5513,20260891) |
| tidal22 | (5601, 20260892) | (5611,20260893) (5612,20260894) (5613,20260895) |
| casc | — (committed l1pp filter; determinism gate in-probe) | (5711,20260896) (5712,20260897) (5713,20260898) |

No collisions with committed streams (checked against stage3_b_* KEYS).

## A-N3-4 — AUG design constraints found in-test (blind; committed
## before any AUG training runs)

Two constraints surfaced by the tests-first pass (the AUG subagent died
on a session limit mid-build; the main session finished the build):
1. **UNet mod-8 size constraint:** slot sizes must survive three exact
   halvings (skip-connection shape mismatch otherwise, caught by the
   trainer smoke test at 12²) ⇒ g96 trains octaves {1,2} (48/24), g80
   trains {1} (40); the registered {1,2,3}/{1,2} lists lose their
   smallest slots. Slot count 8 → **6**; per-slot-parity steps 48000 →
   **40000** (6 × 6667 ≈ baseline); ckpt-every 1000 (40 ckpts, matching
   the baseline count).
2. **Resample D4 form:** array-centered rot90/flips do not commute
   POINTWISE with the Fourier crop — the rotation center is half a
   pixel off the DFT origin, a different pixel-fraction on the two
   grids; the discrepancy is a pure TRANSLATION (magnitude spectra
   agree to 1.3e-7 relative, in-test). The pool's law is D4-exact via
   the trainer's own per-stack d4_augment; the test asserts the
   translation-invariant form. The ×2 no-op exhibit, no-leak test, and
   trainer smoke are green (6/6).

## A-N3-3 — CASC probe run 1 INVALID (numerics); the one licensed
## bug-repair (committed before the resubmission runs)

Run 18447918's gates passed (replay corr_min 0.99999999, determinism
exact) but the streams are numerically DEGENERATE (verbatim inspection:
values to ±2746.97, casc2 carries 16384 NaNs = one full map, stds
3.3/22.1/21.5 vs adj 0.99): the raw cascade seed's Gaussian-scale-
mixture marginal has heavy tails the copula z→t table (calibrated for
standard-normal input) cannot take. The scorer's "FIRED (max |Δ| =
19.76)" is therefore VOID — an explosion artifact, not carried phase
structure; recorded as INVALID, not as the registered line firing.
Repair (the single licensed bug-repair for this job): per-map
rank-Gaussianization of the cascade seed before the filter — exact
standard-normal marginals, cascade spatial ARRANGEMENT preserved
(monotone, permutation-equivariant ⇒ D4-in-law survives); verified
in-test (splice test extended: rank identity vs raw seed, marginal
mean/std, finiteness; 5/5 green; repaired base range ±7.3, std 0.996).
The registered line and its weights are UNCHANGED and will be read on
the resubmitted probe only.
Result (resubmission): **PENDING**.

## CKPT-SWEEP READOUT (verbatim, night3_ckpt_sweep_scores.json;
## job 18447330, 3:51)

| dir | selected | MF(dev) @sel-or-nearest | spread | min | pass-cat | Spearman cage↔MF |
|---|---|---|---|---|---|---|
| oracle | @5000 (bugged cage) | 4.54 @5000 | 2.07 | 2.89 @15000 | 3/8 | +0.38 (p 0.35) |
| prod | @16000 (clean cage) | 3.79 @15000 / 3.43 @17500 | 2.96 | 2.33 @20000 | 1/8 | +0.67 (p 0.07) |
| blind | @10500 | 8.62 @10000 | 7.61 | 5.28 @5000 | 0/8 | +0.50 (p 0.21) |

Pre-stated signatures: **CAGE-SELECTION fires** (oracle: spread > 2, 3
pass-category ckpts, selected fails at 4.54); capacity's flat-bad does
NOT hold for oracle/prod (some ckpts render near-clean declared MF);
dilution's clean monotone does not hold (late-better but non-monotone;
adjacent late swings 2.89↔4.80 exceed the 0.55 jitter — real
ckpt-to-ckpt texture variation). A-N3-2's P-55 line FIRES (@15000 2.89
< @5000 4.54). Blind (2 extrap octaves) never passes at any ckpt —
consistent with the BL depth ladder. The watched Goodhart line does NOT
fire: the CORRECT cage is weakly texture-protective (positive rho); the
harm was the wrong reference selecting early. Cross-cutting finding:
NATIVE PEAK EXCESS is checkpoint-ROBUST (+14–22% at every ckpt, every
dir) — it dissociates from MF along the checkpoint axis; the A8
two-signature structure deepens (topology: selection-/depth-driven;
peaks: model-class-robust).

## O-CORRECTED probe (adaptive, grant 2; pre-stated BEFORE running;
## descriptive — R47's O adjudication is NOT reopened tonight)

The oracle chain resampled at the corrected-cage pick @16000
(ckpt_oct1fix_oracle/armA_gowerstreet_s16000.pkl), final config,
ORACLE label (oct-1-measured fit target, target_octave 1, lo=1),
tag **oraclefix**, KEYS: white (5801, 20261300); finals
((5811, 20261301), (5812, 20261302), (5813, 20261303)). Scored
descriptively (entries + e2e marginals + starlet + parity/nn; hc
marginals unavailable at @16000 — noted): night3_oraclefix_score.py.
Registered expectations: MF(dev) declared ≤ 3.5 on finals — exec P 60;
native peak excesses REMAIN > half-targets — exec P 80 (checkpoint-
robust per the sweep). Meaning: both landing ⇒ O-NULL decomposes into
"topology was the cage artifact; the peak excess is the real
architecture residue" — the cleanest available account of R47's O.
Result: **PENDING**.

## Improvisation slot (RIDER v2 §5)

UNSPENT at prereg time. If spent, its own prereg section is appended
here with weights BEFORE its run (A1/G2). Candidate noted, not chosen:
a selection-cage variant motivated by CKPT-SWEEP — but MF-based
selection would burn MF's dev status further and needs the JUDGE-2 tier
to adjudicate; likelier a draft than a run.

## Budget ledger (cap 4.0 H100-h)

Spent: 0.12 (R47 set, DL-1). Projected: canaries ~0.05; AUG trainings
2 × ~0.20 (48k steps + 48-ckpt selection sweep); TIDAL 2 × ~0.12 (iff
gated); chains 4–5 × ~0.05; CKPT-SWEEP ~0.10; CASC probe ~0.05. Total
projected ≲ 1.3. Queue-stall rule: > 2h ⇒ park that leg.

## AMENDMENT A-N3-1 — GATE-T rule sign (blind; committed before the
## instrument touches any real or generated map)

The validated instrument's sign convention: texture elongated ALONG the
coarse eigenframe gives A_or < 0 (planted-coupled stack ≈ −0.90);
decoherence gives A_or ≈ 0. The pre-stated difference form assumed
positive locking. Corrected mechanical rule: GATE-T FIRES iff
|A_or(real)| − |A_or(gen)| ≥ 3σ (combined bootstrap SE) on the PRIMARY
trained leg — decoherence = loss of frame-locking MAGNITUDE. Signs of
both stack means reported descriptively. Line weights unchanged
(rec 60 / exec 40).

## AMENDMENT A-N3-2 — the TRUTH-REFERENCE BUG (disclosure + pre-stated
## corrected-cage recomputation; committed before reading any of it)

Discovered during tonight's slurm audit, before any night-3 result was
read: `run_c1t_arms.py --truth` defaults to the SANDBOX truth
(sandbox_truth_normconv.json), and three gowerstreet trainings omitted
the flag — **oct1fix_oracle_train.slurm (the O arm), stage3_seed.slurm
(stage-3 seeds 1 and 2)**. Their selection cages therefore scored
validation samples against the sandbox octave-2 reference (vs=1.0703,
kurt=4.9171) instead of the gowerstreet one (vs=1.0672, kurt=7.2738;
both quoted from the committed jsons). var_slope coincidentally matches
(0.3% apart); the KURTOSIS target was wrong by −32%. Clean (explicit
--truth): production seed 0, blind, stage-D, all sandbox legs. Log
evidence: oct1fix_oracle_train_18438237.log prints "selection reference
oct2 vs=1.070 kurt=4.917".
Consequences flagged for the morning reconvene (no retraction tonight —
that adjudication is theirs): (i) O-NULL's checkpoint (@5000) was chosen
by a mis-referenced rule — the cage-harvesting reading of O-NULL now has
a CONCRETE candidate mechanism; (ii) the stage-3 seed-fragility evidence
(picks @3500/@5500 vs seed0 @16000) mixes seed variation with a
different, broken selection rule. TEST-side numbers in those readouts
were scored against correct references and stand as measurements; what
is contaminated is WHICH checkpoint each leg shipped.
Fix-forward: night3_aug_train.slurm passes --truth explicitly (done
before any AUG submission).
Pre-stated recomputation (mechanical, CPU, from the committed
curve_val raw per-ckpt var_slope/kurtosis in the three selection
jsons): re-score the frozen selection formula against the CORRECT
gowerstreet reference. Registered expectations: corrected pick differs
from the committed pick in ≥1 of the three legs — exec P 75; under the
cage story the oracle's corrected pick shows better MF(dev) at the
nearest CKPT-SWEEP grid step than @5000 — exec P 55 (the sweep is the
arbiter; its pre-statements stand unchanged).
Result (verbatim, night3_cage_recheck.json / stdout): **ALL THREE
DIFFER** — oracle committed @5000 → corrected @16000 (corrected score
0.031; the committed pick RESCORES to 2.080, i.e. rejected under the
correct cage); seed1 @3500 → @19500 (0.044; committed rescores 1.920);
seed2 @5500 → @16000 (0.103; committed rescores 1.694). The exec-75
line fires. Structural reading (interpretation, labeled): under the
correct reference all legs pick LATE checkpoints clustered with clean
seed 0's @16000 — the stage-3 "picks-differ" seed-fragility line was
largely THE BUG (early ckpts are under-kurtotic ⇒ matched the sandbox's
low-kurt target); the oracle's bad-texture @5000 checkpoint is now a
concrete cage-artifact candidate, with the sweep as empirical arbiter.

## Sequencing

Prereg commit → N1 fan-out (5 subagents; files reviewed, suites green,
committed by main session only) → JUDGE-2 freeze commit → GATE-T
(pre-stated above) → canaries → trainings (AUG 11/12; TIDAL iff gate) →
CKPT-SWEEP + CASC in parallel → chains → scoring (A5 order per arm) →
NIGHT-REPORT-3.md with drafted preregs → commit, push, **STOP**.
