# NIGHT-REPORT-2 — 2026-07-29 (NIGHT-ORDERS-2 execution; Fable executor)

## MORNING SUMMARY

**Path taken: N1 → GATE-N1 NOT-CONFIRMED → diagnosis path (no training ran
tonight).** The gate needed ≥2 octaves of clear coloring deficit; exactly ONE
octave clears (oct1, ~5σ, both legs). Per the orders this is the full-success
branch of the NO path: the mechanism came out sharp, the doctrine lever
survives in a NARROWED form (one-octave), and the diagnostic budget turned
the negative gate into a five-part mechanism profile. All committed
artifacts, zero training, both test suites green.

**The three numbers that matter:**
1. **GATE-N1: 1 clear octave of the required 2** — oct1 C_real
   0.7864±0.0100 vs C_gen 0.7237±0.0072 (z=+5.09); oct2 z=+1.35; oct3
   z=−1.00. Registered lines: executor 55, reconvene 60 — BOTH LOSE.
2. **The excess is pixel-scale graininess: +14.5% → −1.7% under 0.5-px
   smoothing** (and ν=1 carries +23–38%). The surplus peaks are not
   real-shaped structures; they are pixel-scale texture riding real
   structure.
3. **The cross-field causal test holds: sandbox oct1 whiteness z=+0.1
   (absent) exactly where peak counts show DEFICITS** — whiteness present ⟺
   excess present across both fields (gowerstreet: whiteness + excess;
   sandbox: neither).

**Budget:** ONE MIG job tonight (17809642, 6:50 → ≈0.03 H100-h); everything
else CPU on committed artifacts. Night total ≈0.03 of the 6-h cap; phase
total ≈0.43 of 10. Diagnostic budget: 4 of 6 descriptive runs used (D1–D4),
0 of the extra 1 GPU-h.

**Provisional call:** none to make — no N3 ran (gate closed it). Deliverable
is this mechanism report + three drafted preregs
(log/2026-07-29-draft-preregs-post-n1.md: L1 one-octave colored base with
executor weights, L2 loss-side direction, L3 coloring-taxonomy curve).

---

## VERDICT BLOCKS (mechanical; every number verbatim from committed json)

### GATE-N1 (results_p2/stage1_p3_probes.json /gate_n1 + /coloring)

| leg | octave | C_real | C_gen | z | status |
|---|---|---|---|---|---|
| trained | 1 | 0.7864±0.0100 | 0.7237±0.0072 | +5.09 | CLEAR |
| trained | 2 | 0.7737±0.0187 | 0.7428±0.0133 | +1.35 | no |
| trained | 3 | 0.7853±0.0412 | 0.8406±0.0373 | −1.00 | no (wrong sign) |
| edge | 1 | 0.7864±0.0100 | 0.7114±0.0097 | +5.38 | (clear) |
| edge | 2 | 0.7737±0.0187 | 0.7019±0.0174 | +2.81 | AT-BAR |
| edge | 3 | 0.7853±0.0412 | 0.7779±0.0236 | +0.15 | no |

**NOT-CONFIRMED** (clear octaves = {1}; rule requires ≥2; no at-bar octave
on the adjudicating trained leg). No training authorized; none ran.

### Probe verdicts (all descriptive)

- Height-resolved excess (trained leg): +14.5±3.1 / +11.1±2.8 / +10.3±3.8 /
  +7.2±4.9 % at ν=2.5/3/3.5/4 — declines with height.
- Smoothing response: σ=0.5px → −1.7±2.3% (ν2.5), −2.1±2.7% (ν3);
  σ=1px → −8.2±2.9%, −10.4±3.7% (overshoot to deficit); σ=2px noisy-null.
- Octave transplant (gen details in real pyramid, ν2.5/ν3): oct1
  −5.2±2.7/−16.6±2.1%; oct2 −2.9±3.0/−9.5±2.9%; oct3 +5.4±3.2/+1.4±3.1%;
  oct4 +4.4±3.3/+3.3±3.2%. NO single octave reproduces the +14.5% excess;
  fine-octave substitution REMOVES peaks.
- Spacing co-location: hybrid nn_T = 4.62/5.16/2.00/0.92 at oct1/2/3/4 —
  the nn residual lives at the fine octaves. Full-gen nn_T re-read 4.045 =
  committed 4.045 exactly (instrument identity).
- D1 spectrum shape (oct1, gen/real per annulus): 0.83–0.89 at all
  k = 1…24, → 1.00 at the Nyquist ring — a BROADBAND deficit closing at the
  pixel scale (equivalently: gen detail power tilts toward near-Nyquist).
- D2 envelope coupling: REFUTED as mechanism — K_gen ≥ K_real everywhere
  (trained z=−0.40/−0.37; edge z=−1.07/−2.59, gen higher).
- D3 sandbox contrast: C_oct1 z=+0.1 (both c1t and F2 arms — no whiteness);
  oct3 gen OVER-colored (z=−2.9/−2.1). Sandbox = deficit field.
- D4 replicates (job 17809642; replay gate PASS: corr_min 0.99999999,
  ratio_mean 1.000000, rel 6.97e-3): rep1 +15.86±3.12/+14.11±2.78,
  rep2 +15.54±3.21/+12.24±2.90 — 3-stream ν2.5 mean +15.29%, sd 0.72%.
  The reference for any lever run is now a 3-stream mean with priced
  generation-stream noise.

### Registered-line scorecard (night)

| line | P (exec/rec) | result |
|---|---|---|
| GATE-N1 CONFIRMED | 55 / 60 | **both LOSE** |
| 1-px smoothing cuts ν2.5 excess < half | 60 | HIT (killed at 0.5px) |
| excess grows with height (ν4 ≥ ν2.5) | 45 | did not occur (minority side right) |
| oct1 largest single-octave ν2.5 excess | 55 | LOSES (inverted: oct1 = deficit) |
| co-location argmax match | 50 | did not occur |
| replicates within 2·SE of committed | 80 | HOLDS |

### Test/gate integrity

New estimators tests-first before data: coloring_index (4 tests) — first
discrimination version FAILED pre-data and was corrected (see instrument
lesson below); scale_coupling (2 tests). Both suites green at report time.
Replay gate = the corrected-G2 criterion, second consecutive clean pass.

---

## INTERPRETATION (labeled: this section is analysis, not verdict)

**Mechanism ranking after tonight:**
1. **M-GRAIN (leading, all five probes consistent):** generated oct-1
   details on the real field carry a broadband ~15% power deficit below
   Nyquist — relatively MORE pixel-frequency texture. Composited across a
   full generated pyramid this creates many small, sharp, barely-threshold
   local maxima: the +14.5% count excess that dies under 0.5-px smoothing,
   is largest at low ν, and is absent on the field (sandbox) whose details
   are NOT whitened. Stream-stable (three PRNG streams).
2. **M-COUPLING (refuted):** the transplant sign-flip is NOT missing
   cross-scale envelope alignment — gen coupling matches or exceeds real.
3. **M-COMPOSITION (open):** WHY single-octave transplants give deficits
   while the full gen pyramid gives excess is not settled. Candidate: a
   gen-d1 in a real pyramid lowers mid-k power (fewer threshold crossings
   after per-field standardization) while full-gen graininess needs gen
   texture at several octaves to accumulate. A cheap descriptive follow-up
   is drafted in L3's family (NOT run tonight, diagnostic budget preserved).

**Belief updates with numbers:** P(whiteness premise as N2 specced it —
multi-octave) went 55 → measured ~0 (one octave only). P(oct-1 whiteness is
causally upstream of the count excess) rose to ~75 on the cross-field
association + smoothing kill + spectrum shape (association is two fields —
not proof; L3 would make it a curve). P(L1 one-octave colored base lands
IMPROVED-or-better | run) — executor draft: 50; the honest hedge is that
base→detail spectral transfer through the trained flow is unmeasured, and
that hedge is exactly what L1 tests.

**Instrument-ledger lessons (2):** (i) coloring statements about detail
coefficients must be made IN detail space — the field-coloring validation
synthetic failed pre-data because the Haar high-pass ~k² response cancels a
red field's P~k⁻² and decimation aliasing whitens fine-octave details of
red fields (test corrected before real data; disclosed in the prereg note).
(ii) The corrected-G2 corr/ratio criterion is now twice-validated as the
cross-run reproduction gate for recursive generations.

**Named-temptation ledger (RIDER):** the edge leg read oct1 clear + oct2
z=+2.81 AT-BAR — a favorable reading of the edge leg as corroborating
"two octaves" was available and refused: the gate's adjudicating leg is the
trained leg, where oct2 is z=+1.35, unambiguous. Also refused: treating the
sharp mechanism profile as license to run L1 tonight ("the fix is obvious")
— L1 is a draft with weights, nothing more.

**Open questions for the morning reconvene:**
1. Adopt L1 (one-octave colored base, D1-measured filter) as the Stage-2
   instantiation, with its drafted weights (CURED 10 / IMPROVED 40 / NULL
   30 / REGRESSED 10 / gates 10)?
2. Run L3's coloring-taxonomy curve first (CPU, half-day) to turn the
   two-field association into a predictor before spending GPU on L1?
3. GATE-N1 calibration note: both lines over-weighted CONFIRMED partly
   because neither separated "premise present" (true, 5σ) from "premise
   present at ≥2 octaves" (false) — worth a bar-ledger entry on gate
   wording vs premise wording?
4. Does the paper's mechanism section absorb the graininess decomposition
   now (it is committed, replicated, cross-field-tested) or wait for a
   lever result?

## Artifacts

results_p2/stage1_p3_probes.json (probes 1–4 + gate),
stage1_p3_diag.json (D1–D3), stage1_p3_replicates.npz/.json + 
stage1_p3_replicate_scores.json (probe 5 + D4), scripts_p2/{coloring_index,
scale_coupling,stage1_p3_probes,stage1_p3_diag,stage1_p3_replicates}.py,
tests/{test_coloring_index,test_scale_coupling}.py, drafted preregs in
log/2026-07-29-draft-preregs-post-n1.md, prereg + gate verdict in
log/2026-07-29-prereg-n1-mechanism.md. JOBS.md carries the diagnostic
ledger. Paper skeleton untouched; Stage 3 untouched; PLAN-phase3.md
untouched.
