# SPEC — the explainer gallery (Andreas's request, 2026-07-16)

Purpose: paper- and talk-grade MAP-LEVEL visuals — "how it works, and how it compares
to truth." All generation from FROZEN checkpoints (4a gowerstreet; step-4 selfsim);
no training, no new measurements. Expected cost: one session + MIG-minutes (or CPU).

## Honesty rules (binding, before any figure is made)

1. **No cherry-picking:** every displayed field is the FIRST held-out index (index 0;
   where two are needed, indices 0 and 1), declared here before rendering. If a
   different index must be used (e.g. NaN), document why in the log.
2. **Shared color scale within every comparison row/column** (state per figure).
3. **Every generated panel is captioned as a conditional SAMPLE**, not a prediction
   of the specific truth map — same coarse start, different detail dice; pixel-wise
   agreement is NOT expected and the captions must preempt that misreading.
4. **Measured statistics quoted per panel** (var_slope at the panel's octave;
   kurtosis where relevant) — the numbers the eye should be checking against.
5. Held-out tiles only. Log the seeds. Commit the rendering script with the figure.

## FIG-G1 — "how it works": the ladder walkthrough

One held-out gowerstreet field. Left column: the real field represented at octaves
4 → 1 (progressive sharpening). Right column: the generator's coarse-to-fine pass
from the SAME octave-4 start, one row per octave, each row captioned "given the map
above, sample the next level of detail" with the running-coupling values used at that
step. Final row spans both columns: real vs generated at full resolution. (Refresh of
maps_ladder.png with the 4a generator and the coupling dial made visible.)

## FIG-G2 — "the boundary, in maps" (the money visual; companion to selfsim_control.png)

2×2 grid, one shared color scale per row:
- Row 1 (drifting field, gowerstreet): REAL at the extrapolated octave | generated at
  the extrapolated octave (4a frozen). Captions quote var_slope 1.12 vs ~0.68–0.71
  and kurtosis — the visible symptom: generated texture too uniform, extremes missing.
- Row 2 (self-similar control, same architecture): REAL extrapolated | generated
  extrapolated. Captions quote 0.558 vs 0.544 — visually indistinguishable is the
  expected reading.
Title states the law: "where the between-scale law is scale-invariant the method
works (bottom); where it drifts, the drift is the error (top)."
Optional third column: the |real − generated| residual at matched normalization, if
it reads cleanly; drop it if it confuses (conditional samples differ legitimately).

## FIG-G3 — "where the peaks go" (the downstream bias, physically)

Real vs generated (4a frozen, gowerstreet) full-resolution maps, same field as
FIG-G2 row 1. Overlay markers: peaks ≥ 2.5σ circled on BOTH maps (one marker style),
peaks in 1.0–1.5σ band marked differently on the generated map only if count differs
visibly. Side inset: the measured peak-count bars at ν = 1 and ν = 3 (from
downstream_peaks.json, no recomputation). Caption: "power-level checks pass at ≤7%
on this generator; the peak function is biased +14σ (spurious low peaks) to −12σ
(missing extremes) — circled: what the map is missing."

## FIG-G4 (optional, if cheap) — the ensemble view

Nine generated samples from one coarse start (3×3, 4a generator) around the single
real field (center panel highlighted or shown left). Shows what "sampling the
conditional" means to a non-ML audience. Reuse the maps_4a_diversity machinery.

## Deliverables

results/gallery_{ladder,boundary,peaks,ensemble}.png + a one-entry log with seeds,
indices, checkpoint hashes. Commit script + figures together. STOP after rendering —
no new claims, no README edits; the figures feed the paper skeleton after Andreas's
pivot confirmation.
