# SPEC (DRAFT) — the placement experiment: is positional structure the moment ladder's next rung?

**Status: DESIGN OPEN until Phase 0 of TAKEOVER-BRIEF.md — the executor is
invited to critique and improve this design BEFORE pre-registration. After the
prereg commits, everything freezes as usual.**

## The question

The calibrated generator's one open audit failure: extreme peaks appear at
nearly the right rate and amplitude, but their SPATIAL arrangement is biased at
the ~15% level (position-aware peak statistics fail while every position-blind
statistic — including starlet-ℓ1 — passes). Two candidate explanations:

- **H-data (the moment-ladder hypothesis):** joint/positional structure is the
  next rung — cross-coefficient dependence, with the harshest effective-data
  requirement (currency: independent parents). Prediction: placement error
  shrinks causally with training-set parent count, as dispersion (rung 2) and
  tails (rung 4) did.
- **H-arch:** conditional regression through this sampler cannot express the
  needed joint law regardless of data (an architectural/objective ceiling).
  Prediction: placement error is flat in parent count. Named principled
  alternative if this branch fires: joint structure from an unconditional
  prior via posterior sampling (solution-space note, item 4).

## Draft design (critique me)

Arena: the exact-truth lognormal sandbox — conditional ensembles are exact, so
placement statistics have TRUTH references, and parent count is a free dial.

1. **Instrument first.** Define placement statistics measurable against exact
   conditional ensembles; candidates (executor may substitute better ones with
   argument): peak nearest-neighbor distance distribution; peak two-point
   function at small separations; environment-conditioned peak rate (peaks per
   coarse-quantile bin — "do extremes land where the environment demands");
   seam statistics at octave boundaries (to catch transition artifacts as a
   separate confound). Validate each instrument on truth ensembles (does it
   recover truth-vs-truth null within bootstrap?) before touching any model.
2. **The causal intervention.** Train the final-recipe generator at 1× and 8×
   parent counts (sandbox parents are free), matched total steps and selection
   protocol; measure placement error vs truth at both. The known rung-2/rung-4
   signatures ran through exactly this harness — reuse it.
3. **Branches (draft weights, to be re-registered by the executor):**
   PL-DATA (placement error shrinks ≥2× at 8×; H-data): reconvene 45.
   PL-FLAT (no material shrink; H-arch): 30.
   PL-PARTIAL (shrinks but saturates above truth-consistency): 20.
   Gate branches (instrument fails truth-null; degenerate run): 5.
   Include the meaning of each null IN the prereg (house rule).
4. **Real-field echo (descriptive only):** score the committed real-field
   generations with the validated placement instruments — no new training —
   so the sandbox verdict has a real-data anchor.

## Confounds the design must address (from the draft author's own doubts)

Selection-rule interaction (the checkpoint cage optimizes marginal statistics —
could it be selecting AGAINST placement? cheap check: placement error across
the checkpoint curve, descriptive); octave-seam artifacts vs genuine placement
bias (instrument 1's seam statistic separates them); estimator power (placement
statistics on 32–64 fields are noisy — budget the reference noise in the bars,
bar-ledger #9).

## Cost

CPU/MIG-minutes throughout (sandbox-scale training + scoring). No real-field
training. STOP at the readout; the reconvene adjudicates; either branch feeds
the paper's frontier section and the next-phase (PM application) design.
