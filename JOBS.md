# JOBS — in-flight SLURM jobs & harvest instructions

Last updated 2026-07-17 (C3 gate investigation). Envs: `source env.sh` (scaledrift,
CPU, has pywt) for measurement; `~/wl-challenge-env/bin/python` (JAX) for training/
generation. GPU via SLURM only.

## STAGE D + R18 REPLICATION (2026-07-18) — CLOSED: **D-PASS-BOTH; P-D PASSES;
## 24/24 REPLICATED.** Readout log/2026-07-18-stageD-readout.md

- **16669982** COMPLETED (21:54): edge (oct 2) all 8 checks pass BOTH readings
  (formal bars AND bare floors; A-hc-kurt 14.9% by 0.1%); e2e kurt A 3.2% /
  B 4.4%; dial-beats-scale-blind FALSE (both panels missed) — extrapolation
  works SCALE-BLIND; dial flips the peak-audit sign at high ν; oct-1 two-octave
  extrapolation degrades (−44/−54%).
- **16669983** FAILED t=0 (env split: frozen scorer's pywt + JAX cannot share a
  process — runner split into sample/score phases); **16671189** COMPLETED
  (4:35) + local score: **24/24 bars pass on 64 fresh fields** (arm A clean
  0.0–12.4%; arm B tail-hot: hc-oct4 41% on a 56% bar, e2e-oct2 13.3% vs 15%).
- Scorecards: executor modal D-FAIL-BOTH 55 large MISS (4th consecutive
  constructive under-confidence, against R20's warning); PLAN's P-D 40 beat
  executor 27; repl modal full-pass 45 HIT. STOP; paper decision to Andreas.

- **16669982** GPU (MIG, 2:00, `scripts_p2/arms_stageD.slurm`): STAGE D
  slide-the-edge on gowerstreet — train {3,4}, edge oct 2 held out, arm-B dial =
  curve-extrapolated couplings (vs LINEAR 1.1836 / kurt LOG-LINEAR 10.5199 at the
  edge), selection caged at oct 3. Score → score_stageD.py; P-D rides arm B;
  AMBIGUOUS=FAIL.
- **16669983** CPU (def-lplevass, 16c/2h, `scripts_p2/c1t_replication.py`): R18
  64-fresh-field replication of the sandbox C1T-CAL at frozen ckpts (seed
  20260720). STOP at the combined readout.

## ARM C1-t (2026-07-18, R17) — CLOSED: sandbox C1T-CAL both arms; gowerstreet
## binding-octave deficit closes. Readout log/2026-07-18-c1t-readout.md

- **16666378** GPU sandbox COMPLETED (20:37): selected A@7500 B@2500; verdict
  **C1T-CAL both arms — all 24 bars pass** (hc oct-2 kurt 5.37/5.38 vs truth
  4.917; dispersion 0.2–8.5%; q999 0.94–1.06); trigger PASS. Texture: 32-field
  SEs widen kurtosis bars to 22–38% (two oct-4 entries above the bare 15% floor);
  noisy validation curves — picks on favorable spikes, held at TEST.
- **16666379** CPU attribution COMPLETED (16:17): C1 Gaussian-base ceiling
  4.29/4.59 at ANY ckpt vs truth 4.92 → the t base raises the ceiling, selection
  picks the window; both components necessary.
- **16666634** GPU gowerstreet COMPLETED (20:52, hash 27cc4a8f17): selected LATE
  (A@16000 B@18500 — the cage adapts). e2e oct-2 kurtosis deficit −33.7→−4.0%
  (A), −26.1→−8.0% (B) — **the halving prediction FIRED both arms**; var_slope at
  real (+1.9/+1.4%). Extrapolated octave keeps −29/−40%; oct 4 mixed (B worse).
  Descriptive; reconvene adjudicates. STOP at readout.

## R15 TAIL-DYNAMICS DIAGNOSIS (2026-07-17) — CLOSED: data-limited, mechanism in hand

- Jobs 16613873–880 (matrix as below) all COMPLETED 30–82 min, no resubmissions,
  no degenerates. **P1 mechanical: TD-PARTIAL** (tbase composite ratio 6.00 SHIFT;
  twcrps composite 1.80 NOSHIFT; secondaries split) — but the final-state pattern
  answers the R15 question outright: every 1× run terminally collapses
  (last-3-eval kurt 1.66/1.08/1.66/1.07) while every 8× run ends at/near truth
  (6.11/5.96, 4.32/4.15, 4.78/5.96, 4.57/4.15) and HOLDS for thousands of steps —
  the rung-4 decay is DATA-LIMITED (rung-2's cure reproduced one moment up).
  **P2: YES** — joint (disp ≤10% AND kurt ≥4.0) checkpoints exist at 1× (steps
  1000–3500 composite); at 8× the whole trajectory qualifies. **P3**: at 1× the
  trained flow ERASES its base (t-out and N-out converge to the same ~1.6); at 8×
  the base's tails survive (6.05 vs 0.42 at 12k). Side-finding: twcrps' spurious
  skew GROWS with data (+0.8…+1.1 at 8×) — disfavored. Readout
  log/2026-07-17-taildyn-readout.md; figure results_p2/taildyn.png. STOP; next-arm
  decision at reconvene.
- Submitted (def-lplevass, 16c/24G/1:30, `scripts_p2/tail_dynamics.py`, 12k steps,
  eval/500): **16613873** tbase/composite/n768, **16613874** tbase/composite/n6144,
  **16613875** tbase/t5flat/n768, **16613876** tbase/t5flat/n6144, **16613877**
  twcrps/composite/n768, **16613878** twcrps/composite/n6144, **16613879**
  twcrps/t5flat/n768, **16613880** twcrps/t5flat/n6144. P1 onset rule + P2 joint
  bars + P3 decomposition pinned in log/2026-07-17-prereg-tail-dynamics.md.

## R13 OBJECTIVE BAKE-OFF (2026-07-17) — CLOSED: NO QUALIFIER → STOP

- First submission 16609141–47 all FAILED at t=0 (3 s): the runner imports
  arms_p2.c1t before any wfm module, so wfm/__init__'s jax_flows sys.path shim
  was absent — shim duplicated into c1t/flow.py (1fbabaa; infra note:
  import-order-dependent path shims bite exactly like this).
- Resubmitted (def-lplevass, 16c/24G/1:30, `scripts_p2/bakeoff_c3.py`): all six
  COMPLETED 6–21 min — **16609897** twcrps/t5flat, **16609898**
  twcrps/composite, **16609899** beta05/t5flat, **16609900** beta05/composite,
  **16609901** tbase/t5flat, **16609902** tbase/composite.
- **Verdict (selector: final kurt ≥ 4.0 on BOTH gates): twcrps 3.01/3.57 FAIL
  (+ spurious skew ~+0.51 on the symmetric target, both toys); beta05 0.51/1.05
  FAIL; tbase 3.56/2.98 FAIL (Gaussianizes its base tails late).** twcrps and
  tbase cross the bar mid-training then converge away — the reconvene owns any
  checkpoint-rule discussion. Table + trajectories:
  log/2026-07-17-c3-bakeoff.md; figure results_p2/c3_bakeoff.png. No GPU leg;
  no prereg amendment; C3 objective family remains blocked.

## C3 PRE-TRAINING GATES (2026-07-17, R10 conditions) — arm BLOCKED, no GPU spent

- **Shape-capacity diagnostic, exp noise** (CPU, def-lplevass, 16c/24G/1h): **job
  16605903 COMPLETED (15:10)** — production net + patch config recovers held-out
  skew 2.014/2.0, kurt 5.76/6.0 (capacity confound of the tiny-net toy resolved).
  First submission 16605902 cancelled at t=0 — script lived on login-node /tmp,
  which compute nodes don't share (infra note: stage diagnostics via
  ~/links/scratch).
- **Shape diagnostic, symmetric t(5)** (same recipe): **job 16606069 COMPLETED
  (13:51) — BLOCKER**: held-out excess kurt plateaus at 0.49 vs truth 6.0 (flat
  2.5k→4k) under the identical config that recovers exp. β=1 patched ES does not
  learn symmetric heavy tails → C3 NOT submitted per the pre-registered rule.
  Evidence log/2026-07-17-c3-gate-design.md; blocker + reconvene options
  log/2026-07-17-c3-blocker-symmetric-tails.md.
- **Composite modulated-σ × t(5)** (same recipe): **job 16606223** — quantifies the
  delivered fraction of a pooled mixture+shape kurtosis (truth reference measured
  in-job: 5.86 train / 5.96 heldout). Result appended to the blocker log.
- **tail_q999 truth** (CPU, def-lplevass, 8c/64G/2h, `scripts_p2/truth_q999.py`):
  **job 16606155 COMPLETED (0:42)** — normconv q999 truth oct 1–4:
  5.974/5.532/5.103/4.591 (SE ≤0.03) → results_p2/sandbox_truth_q999.json
  (R10 condition-1 instrument reference).
- **C3 sandbox GPU leg** (`scripts_p2/arms_c3_sandbox.slurm`): NOT SUBMITTED
  (blocked at the gate). Implementation + scorers committed and smoke-tested;
  runnable within minutes of a reconvene decision on the objective.

## FORENSIC (2026-07-17, ruling R5)

- **NLL-noise forensic** (CPU, def-lplevass, 4c/16G/0:30,
  `scripts_p2/forensic_nllnoise.py` + `score_forensic.py`, prereg
  `log/2026-07-17-prereg-forensic-nllnoise.md`): mean-path regeneration from the
  FROZEN 4a ckpts, frozen-scorer readout. Branch weights: F-CLOSE 30 / F-MID 20 /
  F-COLLAPSE 40 / F-OVERSHOOT 5 / other 5. **Job 16581052 COMPLETED — branch
  F-OVERSHOOT both arms** (oct-2 mean-path var_slope 1.536/1.676 vs real 1.020;
  kurt 32–268; dstd ratio 0.28–0.39 = the pre-declared confound, quantitatively
  confirmed). Readout log/2026-07-17-forensic-readout.md: the 4a response is a
  mixture of an OVER-modulated μ-cascade and an UNDER-modulated σ-noise bath —
  information-exhaustion cannot produce this; R4 re-scoping confirmed in
  mechanism. Executor modal F-COLLAPSE missed.

## PHASE-2 OVERNIGHT (2026-07-16, NIGHT-ORDERS)

- **C1 sandbox leg**: **job 16491750 COMPLETED, hash ab9c175f59 — branch
  B-C1-TAILS both arms** (dispersion ALIVE both levels 4.5–8.7% vs 10% bars, NO
  collapse — curve rises to plateau; kurtosis fails oct 2 at 18–21%; recursion
  costs ≤1.4%). Readout log/2026-07-16-c1-sandbox-readout.md. Gowerstreet-leg
  trigger PASSED.
- **C1 gowerstreet leg** (triggered per prereg): **job 16491989** submitted
  (`scripts_p2/arms_c1_gowerstreet.slurm`, identical recipe, descriptive readout —
  morning reconvene adjudicates vs G-1c bars). Expected outcome (registered before
  harvest): head-conditional dispersion near the 4a σ-channel level (~0.96–1.0 of
  real at oct 2), end-to-end compounding REAPPEARS (deficit ≥15% at oct 2) — 65%;
  compounding mild (<10%) — 20%; other — 15%.
- **DIAGNOSTIC (rider budget, runs 1–4 of 6): sandbox shape-baseline stability —
  CLOSED: SYSTEMATIC.** Runs 1–2 (position seeds) were no-ops (all 32² positions
  already in use — predictable, my miss); run 3 gate-OFF shrank the baseline
  −0.155%→−0.072% (mechanism consistent: weaker selection, smaller effect); run 4
  split-half gave −0.19%/−0.29% (z −2.1/−3.2) on independent parent halves →
  the negative isotropic baseline is real. Amendment's difference-in-differences
  reading stands. Zero GPU spent.
- **stage-B B1+B2** (CPU, def-lplevass, 4c/24G/1:30, `scripts_p2/run_stageB.py`,
  prereg `log/2026-07-16-stageB-prereg.md`): **job 16491648 — B1 curves + shape
  COMPLETED (readouts in RESULTS-phase2.md §B1); B2 leg CANCELLED by executor at
  1:05 elapsed** — diagnosis: `iter_parent_maps` with max_shards=None round-robins
  ALL 256 gowerstreet shards (~500 MB each ≈ 128 GB IO) before yielding 30
  parents; could not finish in budget (infra note, rider grant 5).
- **B2 fallback** (CPU, def-lplevass, 4c/24G/0:45, `scripts_p2/run_b2_fallback.py`):
  **job 16492950** — 12 parents from the first 3 shards (bounded ~1.5 GB IO;
  physical-diversity caveat to be stated in the readout). Expected: stride-64
  N_eff/count ≥ 0.5 (60%, as pre-registered).

- **stage-A ensembles+truth+instrument** (CPU, def-lplevass, 4c/24G/1h,
  `scripts_p2/gen_sandbox_ensembles.py`, prereg
  `log/2026-07-16-stageA-prereg.md`): expected — generation ~minutes, truth
  ~10-15 min, instrument ~10 min; outputs results_p2/{sandbox_truth,
  gateA_instrument}.json + $SCRATCH arrays. Expected outcome: Gate A PASS (92%).
  **Job 16490857** COMPLETED — **GATE A PASS** (all octaves, both metrics; readout
  log/2026-07-16-stageA-readout.md). First submission 16490749 FAILED at t=0:
  `sbatch --wrap` runs POSIX-mode sh, where `source env.sh` without a slash searches
  only $PATH — fixed with absolute-path `. $REPO/env.sh` (infra note, rider grant 5).

## Ladder status — phase 1 COMPLETE (see RESULTS-toy.md)
- Rung (i) single-octave overfit — **GREEN** (`7045cea`).
- Rung (ii) two-octave recursion — **GREEN** (`9343214`).
- Rung (iii) GRF end-to-end null — **GREEN / P-NULL PASS** (job 15617056).
- Rung (iv) gowerstreet P4/P5/P6 — **CONCLUDED**: P4 PASS, **P5 HOLDS (robust break)**,
  P6 NOT demonstrated. 3 configs run (jobs 15627183 add, 15628956 FiLM, 15629332 big).
- Rung (v) transfer P13 — **CONCLUDED**: not demonstrated (same cap; arm B transfers an
  amplitude fix). `RESULTS-toy.md` written.

## IN FLIGHT
**None.** All phase-1c and phase-1d jobs harvested. **THE GENERATOR IS FROZEN**; the
phase-1d program (steps 1–4) is complete and `RESHAPE-MEMO.md` awaits Andreas's
decision. Gate-0 kill-tests run in a parallel session and gate all paper claims.

## HARVESTED (phase 1d, 2026-07-16)
- **16401105** (edge-2, f77178e6cf): pilot component 1 — P-edge PASS. First submission
  16401043 FAILED (frozen U-Net can't ingest 4×4 octave-5 pairs; amended to train {3,4}).
- **16401485** (self-similar control, 3c03bded83): 2.5% residual at the extrapolated
  octave — the cap is the field's drift, not the architecture.
- Steps 2c/3 were CPU-only (pilot battery, peak demo). Records in RESULTS-phase1c.md
  §phase-1d; figures pilot_validation/downstream_peaks/selfsim_control.png.

## HARVESTED (attempt 5 — self-conditioning, FINAL)
- **15762584/15762585** (p=0.5/1.0, hashes ffdeac4d4b/ccc86ccf1b): **branch B5 —
  lever FAILED with degradation** (oct-2 end-to-end 0.41–0.48 vs 4a's 0.73–0.77);
  the discriminator proved the compounding limit is INFORMATIONAL (4a's end-to-end =
  its drifted-input response 0.742≈0.746; honest conditional given drifted coarse ≈
  0.53). Generator frozen at calibrated heads + measured limit.
  `RESULTS-phase1c.md` §attempt-5, `results/readout_a5.png`.

## HARVESTED (attempt 4b′ — conditioning robustness)
- **15753842/15753843** (s_max 0.1/0.3, hashes 4f5bbe7b0f/9f56a059ad): **LEVER BAR
  FAILED** both values, both arms (oct-2 z 4.7–7.6 vs the fixed ceiling); project bar
  failed; stopped at the readout. Decomposition: corruption RAISED the heads' own
  ceilings (~at real at oct 2) and kurtosis, but s_gen=0 sampling never engages the
  robustness — end-to-end unmoved. Zero-training follow-up candidate: re-sample the
  existing s=0.3 ckpts (`data_cache/ckpt_4bp_s0.3/`) with pre-registered s_gen>0.
  `RESULTS-phase1c.md` §4b′, `results/readout_4bp.png`.

## HARVESTED (attempt 4a — augmentation diagnosis test)
- **15744601** (`scripts/arms_aug.slurm`, hash 1e61bd812a): collapse GONE within 20k
  (implied oct-2 var_slope flat at ~real; σ-share ~90%); literal onset rule mis-fires
  on the pre-peak warm-up dip (literal REFUTED / running-peak intent CONFIRMED) —
  **stopped at the readout, reconvene adjudicates the rule.** End-to-end scorer still
  ~7σ at oct 2: deficit moved to recursion compounding. `RESULTS-phase1c.md` §4a,
  `results/signature_4a.png`, `results/arms_aug_score.json`.

## HARVESTED (phase 1c, step 1 — Gaussian-NLL detail head)
Pre-registered `log/2026-07-11-prereg-1c-nllhead.md`.
- **15738957 HARVESTED**: GRF null **not cleanly preserved** — arm A z≈3.2 spurious
  var_slope at j=1; extrapolated-octave amplitude +14%/+43% (the exp-head OOD
  instability shows on GRF too). `results/pnull_nll_score.json`.
- **15738958 HARVESTED**: **G-1c FAILED → STOP at the gate (reconvene).** Dispersion
  bar 5–8σ short at oct 2–3 both arms; kurtosis ≈5σ; student-t fallback not triggered.
  See `RESULTS-phase1c.md` (verdict + mechanism: the mean memorizes finite data and
  starves ANY variance channel — third confirmation of the collapse law; exp head
  OOD-unstable at the extrapolated octave). No further generator variants without a
  new ruling.

## VARIANCE-FAITHFUL PROGRAM (reconvene-approved; unblocks P6/P13)
Ordered (a)→(b)→(c), each pre-registered. Success bar = trained-octave var_slope within 1σ
of real; frozen P6/P13 bars unchanged.
- **(a) SDE churn on existing checkpoints — DONE, insufficient.** Score identity
  s=(t·v−x)/(1−t) verified (`tests_wfm/test_score_identity.py`); churn saturates ~9–10σ short
  on the mean-collapsed 10k checkpoint. `log/2026-07-10-prereg-varfaithful-a-sde.md`.
- **(b) checkpoint sweep — DONE, insufficient alone.** var_slope PEAKS at ~2k steps and
  collapses with training (the dispersion-collapse curve). Peak 2k: oct2 3σ, oct4 within 1σ.
  `log/2026-07-10-prereg-varfaithful-b-ckptsweep.md`.
- **(a)+(b) — near-faithful, gate not clean.** 2k+churn4: oct2 1.9σ/oct3 0.6σ but oct4
  overshoots (global churn is uniform, deficit is octave-dependent). **BUT octave-1 P6 repair
  = 90%** here → the repair works once dispersion is restored.
- **(c) dispersion-regularized objective (Tweedie-mean-std matching) — DONE, INSUFFICIENT.**
  Implemented (`cfm_loss_dispersion`, `--lambda-disp`, tests green); swept λ∈{0.1,0.3,1.0}
  (job 15648042). Bar NOT met (oct2,3 still 6–9σ low). Diagnosis: the candidate matches the
  std of the Tweedie MEAN E[x1|x_t], which is structurally below the data std (total
  variance) and t-dependent → mis-specified. `log/2026-07-10-prereg-varfaithful-c-objective.md`.
- **(c') option 1 (t-consistent late-t penalty) — DONE, INSUFFICIENT** (job 15671262, tests
  green, `--disp-t-lo 0.6`, λ∈{0.3,1,3}). Bar NOT met (oct2,3 still 6–10σ low). Twice-confirmed:
  a training-time penalty on the deterministic model can't fix the ODE-pushforward
  under-dispersion.
- **(c') option 2 — HANDED TO RECONVENE for a design decision.** Fix must change the generative
  process (stochastic, learned conditional noise). (2a) hybrid learned-σ SDE = FM augmentation
  (free periphery, recommended); (2b) pure Gaussian detail head = a FROZEN-CORE change (replaces
  FM) → reconvene, not silent redesign. Details in `log/2026-07-10-prereg-varfaithful-c-objective.md`.

## Next (after the 2a/2b reconvene decision)
```bash
# 2a (recommended, within FM): add a per-pixel log-sigma head to ConditionalUNet; train by
# Gaussian NLL on r = detail-(x_t+(1-t)v): NLL=0.5*(r^2/e^{2g}+2g); sample via sample_sde with
# per-location noise scaled by e^{g}. Sweep, verify trained-octave var_slope within 1σ (det. ODE
# + learned noise) at oct 2,3,4 + GRF null, THEN re-run arms A/B (full H100, <=2:59) -> P6/P13.
```
Prior from (a)+(b): 90% octave-1 repair once dispersion is restored. Do NOT over-train (peak
~2k) or hand-tune per-octave churn.

## DONE (env facts)
GPU: MIG `h100_20gb` sees `CudaDevice(id=0)`; ~217 s (8k steps) / ~270 s (10k) / ~430 s
(25k, 48/96/192). Benign ptxas 12.6.77 clamping warning. CPU generation (transfer) via
affinity-pinned `~/wl-challenge-env` python.

## NEXT (after rung iii green)
Rung (iv): reuse `run_pnull.py` with `--field gowerstreet` (add gowerstreet's real stage-0
2-D running-coupling coordinate as arm B's `cond_by_octave`, replacing the placeholder
`scale_coord`), then `measure_generated.py` → P4/P5/P6. Arm A should BREAK at the
extrapolated octave (P5: var_slope/kurtosis far from real); arm B should repair ≥70% (P6).
This is the load-bearing run and wants a full H100 (`--gpus-per-node=1`, ≤2:59).
