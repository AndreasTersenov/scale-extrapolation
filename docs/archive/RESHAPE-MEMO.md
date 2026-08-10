# RESHAPE MEMO — phase 1c → 1d closure and the pivot decision (for Andreas)

2026-07-16. Requested by the reconciliation ruling
(`log/2026-07-16-reconvene-reconcile-audits.md`). Everything below rides on
pre-registered, adversarially-gated measurements committed in this repo; the two
literature kill-tests (Gate 0, `SPEC-novelty-collapse.md`, parallel session) GATE all
novelty claims and are still pending. **Decision requested: pick a pivot option (§3)
and the paper branch (§4).**

## 1. What is banked (all pre-registered; figure pointers)

| Result | Status | Where |
|---|---|---|
| Between-scale drift of real fields: real, low-dimensional (2 numbers), field-general, null-clean | measurement, generator-independent | stage-0 RESULTS.md |
| P5 break: scale-blind weight-tied generator wrong at the extrapolated octave, 6–14σ, every config | confirmed, robust | RESULTS-toy.md; g1c/4a/edge-2 scores |
| The collapse law: finite-data mean-memorization starves conditional variance in any channel; CAUSALLY removed by 8× symmetry augmentation | confirmed in 3 channels + causal intervention; novelty pending Gate 0 | RESULTS-phase1c.md; signature_4a.png |
| Calibrated heads: given real coarse, var_slope ≈ real at oct 2 (0.996±0.041), kurtosis at real at oct 3 | banked (4b′ regularization gain) | readout_4bp.png |
| Conditioning drift is STRUCTURED, not additive (white noise ~2× less damaging per unit amplitude) | measured (4b′-ii gate-stop) | smatched_4bpii.png |
| Compounding cap: end-to-end = drifted-input response (0.742≈0.746); honest conditional ≈ 0.53; honesty-hurts inversion | measured cap + tested negative for train-side fixes (attempt 5, branch B5) | readout_a5.png |
| Lognormal fields drift like N-body (1.54→0.53) — the drift is generic to multiplicative fields | measured (step-4 design probe) | log/2026-07-16-prereg-step4-selfsim.md |
| The cap's boundary: on a scale-invariant in-class field the same architecture extrapolates to 2.5% (vs 39–65%) | measured (step-4 control) | selfsim_control.png |

## 2. Phase-1d results (this week; all pre-registered, readouts committed)

- **Step 1 — attempt 5 as discriminator:** already read out 2026-07-11 (race
  condition in the rulings surfaced and reconciled;
  `log/2026-07-16-step1-attempt5-reconciled.md`). Tested negative for train-side
  conditioning fixes; adopted wording: "measured compounding cap consistent with an
  informational limit."
- **Step 2 — the INVERTED validation pilot: the protocol catches everything.**
  P-edge PASS (extrapolation error edge-consistent, ratios 1.46/1.62);
  self-consistency detects the failure without ground truth (z=7.9 extrapolated,
  z=7.1 trained-octave compounding) and passes where the generator is good (oct-4
  z=0.7); held-out scattering rejects at 98–100% of channels (median |z|≈10); the
  battery also caught arm B's OOD blow-up re-emerging at the narrow second edge.
  Honest protocol lesson, measured: the deployable curve-referenced variant fails
  its own calibration on real fields — deployable checks need population-calibrated
  bands. (`pilot_validation.png`)
- **Step 3 — the downstream-bias demo (the usefulness sentence):** with P(k)-level
  checks passing (≤7%), the peak function is TILTED: +30% spurious low peaks (the
  white conditional noise's graininess) and −14/−23% missing extreme peaks (the
  kurtosis deficit) — 6–14σ, sign-flipping, threshold-dependent: the worst case for
  peak-based inference, and both mechanisms were flagged by our audit before a peak
  was counted. (`downstream_peaks.png`)
- **Step 4 — the self-similar control:** synthesized exactly scale-invariant,
  in-class cascade (couplings measured flat 0.56→0.49). RESULT: see §2a below.

### 2a. Step-4 readout: the boundary of the negative claim, measured

On a synthesized, exactly scale-invariant, in-class cascade (couplings flat 0.56→0.49)
the SAME frozen architecture extrapolates almost perfectly: **2.5% residual at the
extrapolated octave vs 39–65% on the drifting field**; trained octaves exact;
amplitude and kurtosis clean. (`selfsim_control.png` — the campaign's clearest single
figure.) The literal pre-registered 1σ bar fails at z≈5 because the control's
bootstrap SEs are ±0.002 — a bar-calibration miss, logged; the scope answer is
unambiguous: **the compounding cap is a property of drifting, off-manifold-textured
fields, not of the coarse-to-fine architecture** (with a measured ~2.5% finite-sample
floor). For the paper this converts the negative claim into a precise, bounded law:
"hierarchical conditional generation is statistically faithful where the between-scale
law is scale-invariant, and degrades with the measured drift — here is the audit that
tells you which regime you are in."

## 3. Pivot options (ordered; costs are wall-clock sessions + compute)

- **Option 1 (RECOMMENDED): the audit paper.** "A failure taxonomy and pre-registered
  audit protocol for scale-hierarchical generative emulators, demonstrated on weak
  lensing." Everything in §1–2 is its content; the pilot + downstream demo are its
  demonstrations; the self-similar control bounds the negative claims' scope. Cost:
  Gate-0 kill-tests (parallel, pending) + ~2–3 writing sessions + estimator hardening
  (nanmedian, population-calibrated SC bands); ≈ zero new compute. Venue ladder per
  the reconvene's two-branch pre-statement: NeurIPS-track only if Gate 0 leaves a
  novel mechanism claim; otherwise ML4PS / A&A-methods — honest and useful at either.
- **Option 2: the measurement paper** (stage-0 drift + P5 quantification + the 2-D
  coordinate). Safest, cosmology-native. RECOMMENDATION: fold into Option 1 as its
  empirical anchor section rather than split salami.
- **Option 3: generator program, next phase, in the regime where the economics bind
  (3D / hydro / Lyα).** The audits agree 2D WL was the right falsification domain and
  the wrong deployment domain. Months, new data, new sign-off; the informational-cap
  finding says the coarse fields themselves must carry more information (better
  fine-octave texture models), which is a research bet, not an engineering step.
  NOT recommended before Option 1 is banked; Option 1 is also its best prospectus.
- **Option 4: alliance track** (GOLCONDA baselines, LDT anchor, ST-judge) — phase-2
  items that slot INTO Option 1's discussion or follow it; coordination cost, not
  compute.

## 4. Risks and language discipline (binding, from the audits)

Gate 0 may rule the collapse law ADJACENT-KNOWN (β-NLL/exposure-bias lineages) — the
paper then says "we observe X, consistent with [refs], and add [causal test /
cross-channel invariance / field context]" per the SPEC's fallback sentence. Scope
words: one octave, measured couplings, 2D WL, 322 tiles, one seed; "RG-inspired";
never "certified", never "any-scale". The 4b′-ii/attempt-5 negatives are claims about
THIS architecture class in THIS data regime; the self-similar control (§2a) sets the
boundary of that statement.

## 5. Recommendation

Option 1 with Option 2 folded in, Gate-0-gated; Option 3 deferred to a new phase with
its own sign-off (and only if someone owns the 3D data problem); Option 4 slotted into
Option 1's discussion. The deliverable line both audits converged on: *the audit is
the asset; the generator was its first, well-instrumented casualty.*

---

## Appendix — the executor's big-picture audit (2026-07-16, committed per the ruling)

Reproduced from chat, lightly formatted; the reconciliation ruling adopted its four
constructive moves (downstream demo, inverted pilot, self-similar control, Seitzer).

**Lanusse-style critique.** "Train small, sample any scale" solves a problem cosmology
doesn't quite have: the expensive small scales are expensive because their PHYSICS
differs (baryons/feedback), which breaks the smooth scale-flow assumption exactly
where it matters; P5 is evidence for the skeptic. If target-resolution sims exist,
train directly; if not, the extrapolation is unvalidatable precisely where it is
sold — unless the validation story is the product. Evidence scale (one field, 322
tiles, 128², small UNet) supports methods claims only. Phase 1c's own measurements
confirmed the core of this critique (the informational cap).

**Mishra-Sharma-style critique.** P5 alone is near-tautological as a finding; its
value is the quantification for the wavelet/WC-RG school. The collapse law is the most
valuable object IF it survives the literature kill-test (β-NLL, exposure bias,
memorization-in-diffusion). The compounding-cap and honesty-hurts measurements are
sharp, falsifiable, and citable if they survive the exposure-bias lineage. And: no
downstream task, no utility — show a parameter-relevant bias P(k) misses and show the
audit flags it in advance.

**Verdict.** The project as pitched is dead and phase 1c killed it properly, with
measurements. The wreckage is not useless — it is mis-pointed: five attempts produced
a rather complete anatomy of why hierarchical generative models of fields fail
statistically, plus an unusually disciplined instrument and an unpiloted validation
architecture. The audit is the asset. Concrete moves: (1) reframe as the failure-
taxonomy/audit paper; (2) run the validation pilot INVERTED on the frozen failing
generator; (3) one downstream-bias demo (peaks/SBI); (4) optionally a self-similar
control to bound the negative claim. Do not salvage: "train small sample any scale",
P6 spin, generality beyond one field. Bias disclosure: the executor ran all five
attempts and has an incentive to call the wreckage valuable; the defense is that
every claim rides on pre-registered, adversarially-gated measurements.

---

## REVISION — 2026-07-17 (reconvene; evidence: C1 overnight run + NLL-noise forensic)

The §1–2 rows on the compounding cap and attempt 5 are re-scoped as follows
(original text preserved above for the record):

- **The "measured compounding cap" was a property of the variance-head SAMPLER, not
  of the information in the conditioning.** Forensic (log/2026-07-17-forensic-
  readout.md): the frozen 4a mean path alone OVER-modulates (1.54–1.68 vs real 1.02)
  with kurtosis 32–268; the production response (0.746) was that cascade diluted by
  its own ~2/3-variance white Gaussian bath. Plain CFM + augmentation (C1) reaches
  −9.7% compounding and 0.88 of real at the extrapolated octave with no
  anti-compounding lever at all.
- **The attempt-5 tested-negative is scoped to the NLL-head substrate.** Its
  measurements stand; its generalization is revoked.
- **The failure taxonomy now counts three measured mechanisms:** (1) finite-parent
  mean-memorization collapse — causally cured by exact-symmetry augmentation;
  (2) variance-head mixture dilution — the false "informational cap," exposed by the
  forensic; (3) Gaussian-base ODE tail deficit — the open frontier (C3 targets it).
  Plus two distinct downstream peak signatures (NLL-head sign-flip; C1 up-tilt),
  both invisible to power-level checks — the audit thesis, now with three exhibits.
- **Option-1's content is upgraded, not weakened:** the audit protocol caught and
  mechanistically resolved its own strongest negative. The pivot decision (paper
  branch) remains with Andreas, now waiting on C3's sandbox leg.
