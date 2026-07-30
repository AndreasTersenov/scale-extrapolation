# SCOPE FORK — the peak tier (R41; decision belongs to Andreas)

**Where the record stands (all measured, all committed):** the one-octave
extrapolated maps pass marginals, starlet-ℓ1, parity, and the spacing
statistic (whose committed T≈4 values probe-1 just showed were stream
maxima — white-stream mean 3.15 ± 0.95, config differences within scatter).
The one open tier: a +14% peak-count excess at native pixel resolution,
now measured to be (i) pixel-scale (dies by 0.5-px smoothing), (ii)
independent of the oct-1 power spectrum at fixed weights (three coloring
doses, nine streams, oracle included), (iii) NOT the lattice defect (F2),
and (iv) unreachable by any inference-only spectral lever — while the
deconvolution methodology that established this is itself a validated,
deployment-pure calibration tool (oracle prediction hit at 0.04σ).

**The new decision measurement (l1pp_decision_sweep):** every config × leg
crosses zero between 0.28 and 0.54 px of Gaussian smoothing. At σ_s = 0.5
px the DECONVOLVED configuration passes the declared-resolution rule (both
ν CIs include 0, panel not sign-consistent) at BOTH the trained leg and
the stage-D edge leg (and the oracle stream); the white-base configuration
does NOT — its edge crossing sits at 0.28 px and overshoots to −4.0% by
0.5. The deconvolved base aligns the edge crossing with the trained-leg
ones (0.28 → 0.49 px). No single σ_s passes all five combos, so the
across-config candidate is None (strict prereg rule) — the coherent
declared resolution exists exactly and only for the deconvolved config.

## Option A — declared-resolution close, then Stage 3 (≈2 H100-h, days)

Ship the deconvolved base as the final configuration. Claim: peak counts
calibrated at declared resolution σ_s ≥ 0.5 px (FWHM ≈ 1.2 px) at trained
scales AND the extrapolated edge; native-resolution counts declared
outside the validated domain, with the mechanism located (pixel-scale
phase texture, spectrum-independent — the record's third clean
dissociation). Note: WL peak analyses smooth at or above this scale in
practice, so the declared domain covers actual use. Remaining work is
Stage 3 as planned (seed ensemble, one-shot blind re-run with smoothed-
peak entries pre-registered, the frozen Minkowski judge as the untouched
tier). Risks: a reviewer may read the declared resolution as evasion —
answered by the mechanism section; the declared-resolution pass is a
5-point grid result at n=32–96 — Stage 3's blind re-run tests it once,
cleanly.

## Option B — training-side redesign (L2 family; ≳2.5–4 H100-h compute,
## mostly calendar + design risk)

The only remaining lever family for native-resolution counts: loss-side
penalties (spectral/phase) or training through octave 1 — both touch the
one-model recipe and the founding bet's framing, both need a doctrine
debate, preregs, sandbox canary, ≥2 seeds. The target (sub-0.5-px phase
texture) currently has NO committed instrument measuring it — the
mechanism hunt would start by building one. The record's own calibration
lesson (R41): measured dose-responses beat mechanism hopes; there is no
measured dose-response supporting any specific L2 lever yet.

## Recommendations (recorded separately)

**Executor (Fable): Option A.** The declared domain covers the field's
actual smoothing practice; the boundary is stated with a located mechanism
rather than a shrug; the shipped config carries a validated calibration
story (the L1→L1″ arc); and Option B attacks a sub-resolution artifact at
open-ended cost — it remains available as follow-up work AFTER a shipped
method paper, with the phase-domain instrument as its natural first step.
This also lands better than PLAN-phase3's own written fallback: peaks pass
WITH a declared resolution, not not-at-all.

**Reconvene:** (slot — to be added at review.)

**Andreas decides.** Budget state: phase spend ≈ 0.56 of 10 H100-h.
