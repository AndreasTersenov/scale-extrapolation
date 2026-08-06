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
7. **AUG arm: AUG-NULL, seed-stable — and mechanistically loud.** ×2
   copies proven a Haar-nesting NO-OP in-test (the registered exhibit);
   fractional band-limited copies (g96 {1,2}, g80 {1}; UNet mod-8
   constraint found in-test, A-N3-4) trained at seeds 11/12 with the
   CORRECT truth reference. Both seeds: marginals/starlet/parity PASS
   but MF(dev) ~9.2, fragmentation |z| ~16.9, and the native peak
   excess SIGN-FLIPPED to a ~−15% deficit; seed 11 additionally broke
   the P-T coloring band (first miss in eight chains). Role-transition
   training on band-limited degradations teaches the WRONG finest-band
   law — the strongest evidence yet for "no honest data-side fix at
   deployment depth".

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
| AUG arm branch (15/40/30/5/10 rec; 8/30/47/5/10 exec) | worse-of-seeds | **AUG-NULL** (both seeds; exec 47 pays vs rec 30) |

CKPT-SWEEP (descriptive; night3_ckpt_sweep_scores.json): oracle spread
2.07, min 2.89@15000, 3/8 pass-cat, shipped 4.54@5000; prod spread 2.96,
min 2.33@20000, 1/8; blind 0/8, min 5.28. Spearman cage↔MF +0.38/+0.67/
+0.50 (correct cage weakly protective — the Goodhart watched line does
NOT fire).

## AUG arm — **AUG-NULL (worse-of-seeds; both seeds NULL, seed-STABLE)**

- canary: PASS (dispersion 0.988; 18481134, 5:14).
- trainings 18481250/51 (12:30/12:20): both seeds pick **@27000** under
  the CORRECT cage (scores 0.339/0.470); slots 6 × ~6667 (parity as
  designed).
- seed 11: MF(dev) declared **9.34** (native 12.66); frag_max|z|
  **16.89**; native peaks **−15.46%±2.21 / −13.05%±2.05** (the excess
  SIGN-FLIPPED to a deficit); C 0.9672±0.0184 **FAILS its P-T band**
  [0.7194, 0.7917] — the first miss after seven consecutive hits;
  marginals PASS, parity 2.18, starlet PASS, nn 4.86±0.36.
- seed 12: MF(dev) declared **9.09** (native 12.85); frag 16.86; peaks
  −15.07/−14.18%; C 0.7678 LANDS; marginals PASS, parity 2.59, starlet
  PASS, nn 5.50±0.28.
- Arm branch: **AUG-NULL** (rec 30 / exec 47 — the exec column pays).
- Rule-wording flaw disclosed (did not bite): the FIXES "excesses <
  half" clause is one-sided — a large deficit satisfies it textually;
  MF gated the branch here. Tighten to |excess| in any reuse.

## Interpretation (executor's analysis; separate from verdicts above)

**The three causal stories, resolved as far as tonight can resolve
them.** The campaign entered the night with O-NULL and three unseparated
readings — capacity, cage-selection, step dilution. Tonight's evidence
reshapes that space:

1. **The topology signature (MF/fragmentation) is NOT an architecture
   wall.** The truth bug + sweep + O-CORRECTED chain show: the oracle's
   catastrophic 5.63 was mostly a mis-referenced selection cage (bugged
   pick rescores to 2.08 = rejected); at correct picks the oracle sits
   3.87 on finals with pass-category e2e checkpoints existing (2.89
   @15000). What remains above the bar is checkpoint-to-checkpoint
   texture variation (~±1 in T between adjacent late ckpts, 3.5σ of
   reading jitter) that the marginal-only cage is too weak to control
   (Spearman cage↔MF only +0.4–0.7). The honest residual disease at
   data-available scales is "selection cannot see texture", not "the
   network cannot render texture".
2. **The native peak excess IS the model-class residue.** It survives
   every checkpoint (+14–22% across all 24 sweep points), direct oct-1
   training (bugged +22.8/+24.8; corrected +15.2/+19.1), and is the one
   number no selection moves. AUG moved it — by OVERSHOOTING into an
   equal-magnitude deficit while wrecking topology, which is the
   exception that proves the rule: you can change the finest-band law
   only by changing what the finest-band role LEARNS, and the only
   band-limited-safe training data available teaches the wrong law
   (truncation rolloff ⇒ under-peaked, fragmenting texture). This is
   the empirical form of R47's structural reading: for true
   beyond-resolution deployment there is no honest data-side fix.
3. **Depth is the dominant real driver.** Blind (2 extrapolated
   octaves) never passes at ANY checkpoint (0/8, min 5.28); orientation
   decoherence is sub-bar at depth 1 (z 2.67) and above-bar at depth 2
   (z 4.50); BL's ladder said the same last session. Every texture
   pathology tonight scales with extrapolation depth, none with
   data-availability alone.
4. **The seed-fragility story needs re-writing.** Under the correct
   cage all four seeds' picks cluster late (@16000/@19500/@16000/@27000
   incl. AUG's). The stage-3 "picks differ wildly" line (@3500/@5500 vs
   @16000) was substantially the bug. What fragility remains is the
   genuine ckpt-axis texture noise in (1).
5. **Fixed weights are a phase bottleneck (CASC).** Seed arrangement
   does not reach the output (max |Δ| 2.72); together with L1″'s
   input-independent spectral action, the flow's output texture is a
   property of WEIGHTS, not of seeds — corrections must act on weights
   (training) or on outputs (the D5 corrector), never on inputs.
6. **Where I was wrong tonight, in the registered numbers:** my GATE-T
   40 was right (rec 60 wrong); my CASC 22 right (rec 30 wrong); my
   AUG NULL 47 right (rec 30 wrong); but my O-CORRECTED MF-≤3.5 at 60
   was wrong (3.87 — I over-trusted the sweep's e2e 2.89 transferring
   to the finals pipeline), and the prereg's modal O-FIXES 45 from R47
   was doubly wrong — for a reason nobody priced: the experiment
   itself was broken. Calibration lesson for the ledger: before
   weighing branches, VERIFY THE INSTRUMENTED EXPERIMENT (the cage's
   reference) — #16's "concentrated mechanisms fire" keeps winning
   only when the apparatus is sound.
7. **Lever ranking for the morning (my recommendation, not a
   decision):** D1 (corrected-selection re-ship — cheap, mandatory for
   record integrity) → D2 (blind shot 2 on JUDGE-2, after D1 ships the
   config) → D5 (corrector, the only remaining lever for the peak
   residue at deployment depth) → D3/D4 parked. AUG-style augmentation
   should be retired: its failure mode (teaching a rolled-off finest
   band) is intrinsic to band-limited-safe degradations, not a tuning
   accident.

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
2 × 12.5 min ≈ 0.12 + AUG chains (2 whites + 2 finals) ≈ 0.07.
**Night total ≈ 0.51 of the 4.0 cap.** The improvisation slot's funding
went unspent (DL-8).

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
