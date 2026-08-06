# NIGHT-REPORT-3 — the texture campaign (2026-08-05/06; prereg
# log/2026-08-05-prereg-night3.md + amendments A-N3-1..4; all calls
# PROVISIONAL pending the morning reconvene)

## MORNING SUMMARY

**The night's headline is not an arm — it is a bug and its decomposition.**
A slurm audit before the first submission found that three committed
gowerstreet trainings (the R47 ORACLE, stage-3 seeds 1 and 2) had
cage-selected their checkpoints against the SANDBOX truth reference
(omitted `--truth`; kurtosis target wrong by −32%). Everything that
follows was reshaped by chasing this:

1. **The truth bug (A-N3-2, confirmed):** re-scoring the frozen selection
   rule against the correct reference moves ALL THREE picks
   (oracle @5000→@16000, seed1 @3500→@19500, seed2 @5500→@16000), and the
   shipped picks RESCORE to 1.7–2.1 (= rejected). Corrected picks cluster
   with clean seed 0's @16000. The stage-3 "picks-differ" seed-fragility
   line was largely the bug; R47's O-NULL checkpoint was chosen by a
   broken rule.
2. **CKPT-SWEEP (pre-stated signatures):** the cage-selection signature
   FIRES — the oracle dir has 3/8 pass-category checkpoints (best 2.89
   @15000, e2e read) while its shipped @5000 reads 4.54. Capacity's
   flat-bad signature does NOT hold; blind (2 extrapolated octaves) never
   passes at any checkpoint (0/8, 5.28–12.89 — depth beats selection).
   **Native peak excess is checkpoint-ROBUST (+14–22% everywhere)** — the
   two boundary signatures dissociate along the checkpoint axis.
3. **O-CORRECTED probe:** the oracle chain at @16000: MF(dev) declared
   **3.87** (vs 5.63 bugged; still > 3.5) and peaks **+15.2/+19.1%**
   (persist). C lands (7th consecutive). Account of R47's O-NULL:
   topology ≈ cage artifact + ckpt noise; peaks = real model-class
   residue that no selection fixes.
4. **GATE-T: NOT FIRED** on the primary trained leg (z = +2.67 < 3;
   rec 60 / exec 40 — exec side pays). TIDAL descoped. But blind-leg
   decoherence z = +4.50 (descriptive): orientation joins MF/frag/BL on
   the extrapolation-DEPTH axis. TIDAL machinery built + proven anyway
   (4/4 tests; full D4 channel-action table; F2-form exactness).
5. **CASC probe: NOT FIRED** (max |Δ| = 2.72 < 3; run 1 was INVALID —
   copula marginal-tail explosion — repaired under the one-repair
   license with per-map rank-Gaussianization, disclosed A-N3-3). Base
   spatial-arrangement structure does NOT survive fixed weights; the
   transfer-function lesson extends from spectrum to arrangement.
   rec 30 / exec 22 — exec closer.
6. **JUDGE-2 FROZEN** (persistent-homology Betti curves, (4,8) duality,
   GRF-only validation, 6 tests) and QUARANTINED — applied to nothing
   tonight; bar to be set from a real split-half null inside the
   blind-shot-2 prereg (A4 pattern).
7. **AUG arm:** ×2 copies proven a Haar-nesting NO-OP in-test (the
   registered exhibit); fractional band-limited copies (g96 {1,2},
   g80 {1}; UNet mod-8 constraint found in-test, A-N3-4) trained at
   seeds 11/12 with the CORRECT truth reference. Verdict: **PENDING**
   below.

Infra note, disclosed: the AUG and TIDAL build subagents died on an API
session limit mid-night (~4h stall); the main session finished the AUG
build (tests-first preserved) and the TIDAL subagent's completed files
tested green and were committed as-is.

## Verdict tables (verbatim from committed artifacts)

| item | registered line (rec / exec) | outcome |
|---|---|---|
| GATE-T fires | 60 / 40 | NOT FIRED (z +2.67) — exec pays |
| CASC carried | 30 / 22 | NOT FIRED (max abs Delta 2.72) — exec closer |
| A-N3-2 picks differ | — / 75 | FIRES (3/3 legs differ) |
| A-N3-2 sweep-better | — / 55 | FIRES (2.89@15000 < 4.54@5000) |
| O-CORRECTED MF ≤ 3.5 | — / 60 | does NOT fire (3.87) |
| O-CORRECTED peaks persist | — / 80 | FIRES (+15.2/+19.1%) |
| AUG arm branch (15/40/30/5/10 rec; 8/30/47/5/10 exec) | see prereg | **PENDING** |

CKPT-SWEEP (descriptive; night3_ckpt_sweep_scores.json): oracle spread
2.07, min 2.89@15000, 3/8 pass-cat, shipped 4.54@5000; prod spread 2.96,
min 2.33@20000, 1/8; blind 0/8, min 5.28. Spearman cage↔MF +0.38/+0.67/
+0.50 (correct cage weakly protective — the Goodhart watched line does
NOT fire).

## AUG arm (PENDING — filled verbatim at scoring)

- canary: PENDING
- seed 11: PENDING
- seed 12: PENDING
- arm branch (worse-of-seeds): PENDING

## Interpretation (executor's analysis; separate from verdicts above)

PENDING — written after the AUG readout.

## Decision ledger (RIDER v2 grant 1)

- DL-1: N0 already executed pre-orders; 0.12 H100-h counted to tonight.
- DL-2: CKPT-SWEEP probe launched first (grant 2) — highest
  information-per-GPU-minute given O-NULL's three unseparated readings.
- DL-3: truth-bug found in pre-submission slurm audit → A-N3-2 blind
  amendment + mechanical recomputation BEFORE reading the sweep.
- DL-4: GATE-T not fired → TIDAL descoped; budget shifted to the
  O-CORRECTED probe (grant 2) instead.
- DL-5: CASC run 1 invalid (numerics) → one licensed repair
  (rank-gauss), prereg'd (A-N3-3) before resubmission.
- DL-6: AUG/TIDAL subagents died (session limit) → main session
  finished AUG (the ordered arm has priority); TIDAL files tested and
  committed as machinery only.
- DL-7: AUG slot geometry corrected in-test (mod-8; A-N3-4): steps
  48k→40k, slots 8→6, registered before training.
- DL-8: improvisation slot NOT spent — the measurements motivating a
  variant (corrected-cage re-selection) are a RE-SHIP decision that
  belongs to the reconvene, not a new single-variable experiment; a
  drafted prereg stands in instead.

## Refusals (named temptations, refused)

- No JUDGE-2 peeking at any real or generated stack (including when it
  would have "settled" the O-CORRECTED reading).
- No softening of GATE-T (z 2.67 is sub-bar; TIDAL stayed descoped
  despite the blind leg's 4.50 making it tempting).
- No χ/MF matching, no topology statistics in any loss, anywhere.
- The CASC run-1 "FIRED |Δ|=19.8" was recorded as INVALID, not as the
  registered line firing (the number was an explosion artifact — reading
  it as signal would have been the muddy-improvement failure).
- MF used strictly as MF(dev); no held-out claims made on it.
- O re-adjudication NOT performed tonight (R47's branch stands until
  the reconvene rules; tonight's numbers are labeled descriptive).

## Budget

GPU (H100-h, MIG-rated): R47 set 0.12 (DL-1) + sweep 0.02 + oraclefix
white/finals 0.06 + CASC run1+run2 0.08 + canary 0.04 + AUG trainings
PENDING + AUG chains PENDING. Running total before AUG ≈ 0.32 of 4.0.

## Drafted preregs (NOT run; morning deliverables)

### D1 — corrected-selection re-ship (the truth-bug remediation; needs
### reconvene review)
Re-select oracle/seed1/seed2 from their EXISTING checkpoint grids under
the correct gowerstreet reference (zero training; night3_cage_recheck
already names the picks: @16000/@19500/@16000); re-run the three
final-config chains + full battery; re-state the stage-3 seed-ensemble
and O-arm conclusions with corrected picks; the R43 claim set is
re-examined where it cites seed fragility (the declared-res peak
fragility entry may be bug-inflated). Cost ≈ 0.2 H100-h. The
committed-artifact record keeps both selections (bugged + corrected)
for the calibration appendix.

### D2 — the second blind shot (JUDGE-2 as the unburned tier; needs
### reconvene review + a frozen bar)
Config: whatever the reconvene ships after D1 (production geometry,
corrected selection). Protocol mirrors R42's (b): A4-pattern null — the
JUDGE-2 bar computed from the real split-half null and committed BEFORE
any generated map is scored; one shot, no re-runs; MF retired to
development-only. JUDGE-2 remains untouched until then (quarantine held
tonight).

### D3 — TIDAL at depth 2 (the gate's own evidence)
GATE-T fired descriptively on the blind leg (z 4.50): orientation
decoherence is a DEPTH-2 phenomenon. If a depth-2 product is ever in
scope, TIDAL's machinery (committed, proven) trains under its original
branch rules with the gate re-scored at depth 2 as the primary leg.
Not viable for the current depth-1 deployment claim.

### D4 — cascade training (prior LOWERED by tonight's null)
The inference-only probe shows arrangement structure does not pass
through fixed weights; only joint training could put it there. Given
CASC-NULL + peaks-as-residue, this is third in line behind D1/D2; keep
drafted.

### D5 — texture-transfer corrector (R47 order 4, still draft-only)
The I-instrument + orientation axis now both measure oct-1 texture; the
corrector's target statistics exist and are validated. Its prereg still
requires its own reconvene review (standing).

## STOP

All calls above are provisional; the morning reconvene adjudicates:
(i) the O/seed re-adjudication under D1, (ii) whether AUG's branch
(below, once filled) changes the lever ranking, (iii) D2 scheduling.
No further experiments run tonight.
