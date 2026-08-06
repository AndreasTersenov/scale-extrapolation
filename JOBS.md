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

## 2026-07-25 — Phase A' M5 (parity checkpoint curve; R31 order 1)
- scripts_p2/parity_ckpt_sample.slurm: MIG h100_20gb, 45 min cap, inference only
  (e2e from committed ckpt_c1t_sandbox dense checkpoints, arms A+B, steps
  2500..20000 by 2500, 32 test tiles). Output results_p2/parity_ckpt_gen.npz +
  job log. Logged pre-submission.

## 2026-07-25 — Arms F/J sample phase 1 (R32 GO)
- scripts_p2/fj_sample_phase1.slurm: MIG h100_20gb, 59 min cap, inference only.
  Val e2e: sandbox arm A all 40 ckpts, arm B 2500-stride, gowerstreet A@16000;
  hc channel means at committed picks; equivalence gate inline. Outputs
  fj_val_gen.npz + fj_val_hc.json. Logged pre-submission.
- scripts_p2/fj_sample_phase2.slurm: MIG, 30 min cap, phase 2 (corrected test
  gens + joint-pick gens + gowerstreet echo). Logged pre-submission.

## 2026-07-25 — Arm F2 sample (R33; prereg 2026-07-25-f2-groupavg)
- scripts_p2/f2_sample.slurm: MIG, 30 min cap, inference only. Group-averaged
  e2e+hc at committed picks + gowerstreet echo. Logged pre-submission.
- 17424070 FAILED at arm B (full-batch FiLM cond vs per-group subset; arm A +
  identity gate clean); fix committed, resubmitting.

## 2026-07-25 — F2 disambiguation (R34 order 1)
- scripts_p2/f2_disambig.slurm: MIG, 20 min cap, inference only. F2 e2e arm A
  @7500 on the repl64 stream; reading rule pre-stated in the script docstring.
  Logged pre-submission.

## 2026-07-25 — Arm D training (parity-cure prereg, R32 approved; R34 GO)
- scripts_p2/arms_pcure_D.slurm x4 (ARM=n1s1/n8/n32/cens): MIG, 2:30 cap each,
  unmodified runner (execution reading in the F2 readout append). Outputs
  arms_pcure_<arm>.npz + pcure_selection_<arm>.json + dense ckpts. Logged
  pre-submission.
- 17436881 (n32) FAILED: OOM on the 20GB MIG slice (d4_augment of 10304 tiles
  on device). Resubmitting on a full H100 (<=3h keeps b1) — the one gate
  resubmission for this arm.
- 17437389 (n32 full-H100 resubmit) COMPLETED 20:18. pcure_curves.slurm:
  MIG, 90 min cap, val T_coef curves + F2-mode texture, all four arms.
- 17437678 (curves) FAILED: same n32 d4_augment OOM on MIG; resubmitting on a
  full H100 (the 17437389 pattern).

## 2026-07-25 — Program close (R35 orders 7a/7b; pre-stated in the readout log)
- pcure_confirm.slurm: full H100, 30 min, n32@joint-pick test e2e+hc.
- arms_pcure_D.slurm ARM=n1s2/n1s3/n1s4: MIG, seed mini-ensemble (option b,
  taken). All logged pre-submission.

## 2026-07-28 — Phase-3 Stage-0 sample (prereg 62db5f0 APPROVED + A1, reconvene
## review log/2026-07-28-reconvene-stage0-review.md; cleared to submit)
- scripts_p2/stage0_p3_sample.slurm: MIG h100_20gb, 59 min cap (b1 pool),
  zero training. F2 group-averaged e2e from committed ckpt_stageD (A@9000
  adjudicating, B@7500 descriptive; picks asserted at runtime) on the frozen
  Stage-D test tiles. Gates pre-stated in the script docstring: G1 identity
  (exact, binding) + G2 substrate-chain (rel <= 1e-3 vs committed gen_A).
  Outputs stage0_p3_gen.npz + stage0_p3_sample.json. One resubmission license
  (infra gate). Logged pre-submission.
- 17621159 SUBMITTED (MIG b1).
- 17621159 FAILED at G2 (threshold mis-design, disclosed in the readout
  execution note; corrected criterion pre-stated blind at 0ea7a0a).
- 17622970 RESUBMITTED (the one licensed infra resubmission).

## 2026-07-29 — NIGHT-ORDERS-2 N1 probe 5 (prereg 5e955b4, pre-run commit)
- scripts_p2/stage1_p3_replicates.slurm: MIG, 40 min cap, zero training.
  Replay gate (key 2400 stream must reproduce committed F2_gowA_e2e under
  the corrected-G2 criterion, asserted) + 2 fresh F2 streams (keys 2501/2502).
  Expected: gate PASS, two 32-map stacks -> stage1_p3_replicates.npz.
  One resubmission license (infra). Logged pre-submission.
- 17809642 SUBMITTED.

## 2026-07-29 — N1 diagnostic budget (RIDER §2; GATE-N1 NOT-CONFIRMED path;
## all CPU on committed artifacts, 0 GPU-h against the 1h diagnostic cap)
- D1 (intent: shape of the oct1 whiteness defect — full annular spectrum +
  absolute band powers real vs gen; expectation: gen deficit concentrated at
  low plane-k, high-k matching or excess). Designs the narrowed lever.
- D2 (intent: cross-scale detail-energy coupling K_12/K_23, NEW estimator
  tests-first; expectation: K_real > K_gen at the fine pair — the
  composition-effect hypothesis behind the transplant sign flip).
- D3 (intent: sandbox contrast — C octaves 1-3 on committed sandbox legs
  where peak counts show DEFICITS; expectation 50/50: whiteness present with
  deficit would BREAK the whiteness->excess causal chain).
- D4 (intent: replicate stability — C_oct1 + peak excess on rep1/rep2 when
  job 17809642 lands; expectation: both within 2SE of committed, P=80 per
  the prereg line).
- 17809642 COMPLETED 6:50 (replay gate PASS corr 0.99999999/ratio 1.000000; rep1/rep2 landed).

## 2026-07-30 — L1' execution (R39 approved; pre-statement 67d7304)
- l1p_canary.slurm: MIG 30 min, zero training. Replay gate (white base ==
  committed F2_A_e2e stream, asserted) + colored-oct1 canary stream.
  Expected: gate PASS, canary npz lands; kill criterion scored on CPU before
  any gowerstreet submission. One resubmission license (infra).
- 17939621 SUBMITTED.
- 17939621 COMPLETED (replay gate PASS corr 0.99999999; CANARY PASS — var_slope 4.7-7.5%; context note: oct2 kurtosis 17.3% vs 15.5% context bar, outside #11 band, recorded for the readout).
- l1p_main.slurm: MIG 45 min, gow replay + 3 adj + ablation + edge streams.
- 17941294 SUBMITTED.

## 2026-07-30 — L1'' (R40 pre-cleared; prereg with committed P-T)
- l1pp_canary.slurm: MIG 30 min. Replay gate + deconvolved canary stream.
  Expected: gate PASS; kill check on CPU before main; watched oct2 kurtosis.
- l1pp_main.slurm (after CANARY PASS): replay gate + 3 adj deconvolved +
  oracle ablation. Expected: gate PASS, 4 stacks. One resubmission each.
- 17951572 SUBMITTED (canary).
- 17951572 COMPLETED (replay PASS; CANARY PASS all bars — watched oct2 kurtosis 13.8% vs bar 17.9%, improved from L1-prime's 17.3%).
- 17952979 SUBMITTED (main).

## 2026-07-30 — R41 decision package (prereg committed)
- l1pp_decision_edge.slurm: MIG 25 min, the one negligible-GPU stream
  (deconvolved edge, per-checkpoint deployment recipe, replay gate).
- 17961764 SUBMITTED.

## 2026-07-31 — Stage 3 (prereg af34d93; R42 APPROVED CONDITIONAL)
- stage3_seed.slurm x2 (SEED=1,2): (a) seed-ensemble trainings, arm A only
  (additive --arms flag, default unchanged, grep-verified), MIG 1:45 cap.
  Expected: 20k-step runs + caged selections + dense ckpts. One
  resubmission each (infra).
- 17987316 (seed 1), 17987317 (seed 2) SUBMITTED.
- A4 EXECUTED: stage3_mf_null.json committed; recomputed values MATCH the
  prereg quotes EXACTLY (native 1.43/0.47/2.65; smoothed 1.48/0.55/2.87)
  -> (b) PRE-CLEARED per R42; proceeding dry-run -> blind -> one-shot.
- stage3_b_gpu.slurm (dry-run white + final phases): quarantined pipeline
  validation on committed stage-D A@9000; expected: determinism gate PASS,
  full chain completes; outputs stage3_dryrun_* (appendix only).
- 17987392 SUBMITTED (dryrun white).
- 17987506 SUBMITTED (dryrun final).
- dry-run chain COMPLETE (determinism exact; cal 1.6%; all phases +
  scorer ran; outputs QUARANTINED to appendix per prereg). A6 clean.
- stage3_blind_train.slurm: THE one blind training (seed 7, {3,4}, arm A).
- 17987660 SUBMITTED (blind training).
- 17987316/17 COMPLETED (picks: seed1 @3500, seed2 @5500 — vs seed0 @16000; the picks-differ line fires). 17987664/17987665 SUBMITTED (seed white phases).
- 17987821 (blind white), 17987822/17987823 (seed finals) SUBMITTED.
- 17988003 SUBMITTED (blind final — the one-shot's streams).

## 2026-08-05 — Phase-3b topo diagnosis (R46; prereg committed)
- topo_diag_hc.slurm: MIG 30 min, the licensed T1 hc generation (blind x3
  + stage-D context x1, real coarse at every octave). Expected: determinism
  gate PASS, 4 hc stacks. One resubmission (infra).
- 18402965 SUBMITTED.

## 2026-08-05 — R47 order set (prereg committed)
- BL EXECUTED (CPU): trained full-band 3.34 at-bar / BL 3.13 pass; blind BL
  4.15 FAIL (oct-2 extrapolated residual) / BL2 3.10 pass — defect tracks
  EXTRAPOLATION DEPTH; both registered lines lose informatively.
- oct1fix_oracle_train.slurm: MIG 2:45, the ONE oracle training {1,2,3,4}.
- 18438237 SUBMITTED.
- 18438237 COMPLETED 10:38 (timing VERIFIED consistent: all single-arm 20k trainings run ~9-10 min; pick @5000 score 0.981).
- 18440865 SUBMITTED (oracle white).
- 18442069 SUBMITTED (oracle finals).

## 2026-08-05 — NIGHT-3 (prereg 1536859)
- night3_ckpt_sweep.slurm: MIG 1:15, CKPT-SWEEP probe (grant-2, descriptive):
  24 e2e recursions, oracle/prod/blind x steps {2500..20000 step 2500};
  intent: separate cage-selection vs capacity vs dilution readings of O-NULL.
- 18447330 SUBMITTED (ckpt sweep).
- night3 CKPT-SWEEP COMPLETED 3:51 (cage signature FIRES; peaks ckpt-robust).
- stage3_b_gpu.slurm (O-CORRECTED white, tag oraclefix, oracle @16000, lo=1):
  intent: the oracle chain at the corrected-cage pick — decomposes O-NULL.
- 18447906 SUBMITTED (oraclefix white).
- CASC probe (MIG --wrap, 40 min): inference-only cascade-base transfer probe,
  l1pp chain verbatim, KEYS 5711-13; intent: does base phase structure survive
  fixed weights (rec 30 / exec 22).
- 18447906 COMPLETED (oraclefix white; determinism exact). Fit cal 2.1%,
  C_pred 0.7819±0.0374.
- CASC probe 18447918 COMPLETED 8:52.
- stage3_b_gpu.slurm (oraclefix finals).
- CASC run 1 INVALID (marginal-tail explosion through the copula; A-N3-3);
  rank-gauss repair committed; resubmitting (the one licensed repair).
- 18481064 SUBMITTED (CASC probe resubmission, repaired base).
- 18481033 SUBMITTED (oraclefix finals).
- AUG machinery finished by main session (subagent hit session limit);
  A-N3-4: 6 slots, 40k steps. night3_aug_canary.slurm: MIG 45 min, sandbox
  canary (dispersion kill <0.6 at s5000).
- 18481064 COMPLETED (CASC resubmission; gates exact; line NOT FIRED |D|=2.72).
- 18481033 COMPLETED (oraclefix finals; scored, MF 3.87 / peaks persist).
- 18481134 COMPLETED 5:14 (AUG canary PASS, dispersion 0.988).
- night3_aug_train.slurm x2 (seeds 11, 12; correct gowerstreet truth).
- 18481250 SUBMITTED (aug11 training). 18481251 SUBMITTED (aug12 training).
- 18481250/18481251 COMPLETED 12:30/12:20 (aug11/aug12; both pick @27000,
  scores 0.339/0.470, correct cage; slots 6x~6667).
- stage3_b_gpu.slurm x2 (aug11/aug12 whites).
- 18482912 / 18482913 SUBMITTED (aug whites).
- fits: aug11/aug12 done. 18482985 / 18482986 SUBMITTED (aug finals).
- 18482912/13 COMPLETED (aug whites; determinism exact). Fits 2.3/2.1%.
- 18482985/86 COMPLETED (aug finals). Starlet both PASS.
- AUG scored: BOTH SEEDS AUG-NULL (seed-stable; MF ~9.2, frag ~16.9,
  peaks sign-flip to -15%; aug11 C band MISS). Night GPU ~0.51 H100-h.

## 2026-08-06 — D1 corrected-selection re-ship (R48; prereg committed)
- tier-0 empty-beam violation instrument tests-first (5 green).
- 3 corrected-pick chains: d1_oracle @16000 (lo1,tgt1), d1_seed1 @19500
  (lo2,tgt2), d1_seed2 @16000 (lo2,tgt2). Ledger #17 reference assertion.
- 18574699/18574700/18574701 SUBMITTED (d1 whites: oracle/seed1/seed2).
- fits done (cal 1.7-1.9%). 18574959/18574960/18574961 SUBMITTED (d1 finals+hc).
- d1 starlet all PASS; d1 scored (ledger #17 assertion passed all 3):
  MF corrected 3.66/3.99/4.18 vs bugged 5.59/6.07/6.17 (~40% was the cage);
  declared-peak rule FAILS 0/3 (seeds significant +5-8% at 0.5px — NOT the
  bug); peaks persist +16-24% (fires); C lands all 3; tier-0 no violation.
  D1 readout appended. D2 + conformal drafts committed. STOP.
