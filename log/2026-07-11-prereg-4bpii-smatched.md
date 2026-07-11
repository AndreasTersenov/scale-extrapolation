# 2026-07-11 — 4b′-ii pre-registration: TERMINATED AT ITS OWN GATE (s_matched > s_max)

Per the ruling `log/2026-07-11-reconvene-4bprime.md`: before any generation, measure
the conditioning drift and derive s_matched; "if s_matched falls outside the trained
range (> s_max = 0.3), report and stop; do not extrapolate the conditioning."
**That condition fired. No generation job was submitted.**

## Estimators (periphery choice, fixed before computing)

1. **Primary — attenuation matching, in the corruption model's own units**
   (`scripts/measure_drift_smatched.py`): s_matched_j = the white-noise level s at
   which the s=0.3 head, fed CORRUPTED REAL coarse (c + s·std(c)·ε) with s_told = 0 —
   exactly the operating mode of the failed end-to-end run — reproduces the measured
   end-to-end var_slope at octave j. Grid s ∈ [0, 0.35], 3 noise seeds per point.
2. **Cross-check — pixel-aligned residual**: std(coarse_gen_j − coarse_real_j)/
   std(coarse_real_j) from the existing s=0.3 fields (same held-out octave-4 starts);
   upper-bound-flavored (contains the legitimate conditional sampling spread).

## Measured result (results/smatched_4bpii.json, results/smatched_4bpii.png)

- Attenuation matching: **no crossing within the trained range** at oct 2 for either
  arm (head only reaches 0.86/0.85 at s=0.35 vs targets 0.773/0.701); arm A oct 3
  crosses at 0.335 (> 0.3); arm B oct 3: no crossing ≤ 0.35.
- Aligned residual: 0.91–1.15 across arms/octaves — the generated coarse is as far
  from its paired real coarse as one real sample is from another.
- Also measured (would-have-been lever reference): the s=0.3 model's own ceiling with
  SEs — A: 0.996±0.041 / 0.748±0.026 / 0.489±0.017; B: 1.007±0.042 / 0.740±0.023 /
  0.471±0.014 (oct 2/3/4).

## Verdict: STOP (pre-named). What the measurement itself establishes

1. **The conditioning drift is not additive noise.** Per unit amplitude, white noise is
   far LESS damaging to the head's modulation than the actual generated-conditioning
   drift: mimicking the observed attenuation would need s ≈ 0.5+ (extrapolated) while
   the trained range ends at 0.3. Matched-level inference corruption therefore cannot
   be honestly engaged with these checkpoints — and the 4b′/4b′-ii risk note
   ("additive corruption may mis-model modulation-flattening drift") is now a
   MEASURED conclusion, not a hypothesis.
2. The damaging component is a structured, off-manifold texture mismatch: total
   aligned discrepancy ~1.0 (mostly legitimate sampling spread) with white-noise-
   equivalent damage ~0.5 — white noise mostly bounces off the head (it averages out
   over the receptive field), while the drift's structure does not.
3. Implication for the anti-compounding program (reconvene's call, not run): the
   corruption distribution must be drift-shaped, not white — the natural candidate is
   SELF-CONDITIONING (train on the model's own generated coarse, or corrupt by
   swapping in an alternative conditional sample of the coarse), i.e. the alternative
   named and deferred in the 4b′ prereg. Training-side corruption at much larger
   white s is the other option but risks severe attenuation.

## Prediction verdicts

Reconvene P-4b′ii-lever 50% → not adjudicable (stopped at gate); the gate firing was
itself an outcome neither side pre-weighted — logged for the scorecard as such.
