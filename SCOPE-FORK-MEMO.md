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

**Reconvene: Option A, concurring** (added at review, 2026-07-31; package
verified — sweep crossings, 0.5-px pass pattern, panels, nn stream-scatter
all re-checked against artifacts; the probe-2 pooling bug's in-script
disclosure audited: caught pre-commit by its own inconsistent output,
corrected logic sound). Four reasons, one caveat:
(1) the mode-2 goal — "a method you can use for higher-order-statistics
mocks" — is met honestly by A: the field computes peak statistics on
smoothed maps, the declared domain covers that practice, and the boundary
comes with a located mechanism and the record's third clean dissociation,
not a shrug. (2) Option B attacks a sub-half-pixel artifact with NO
committed instrument and NO measured dose-response — the exact
mechanism-hope pattern the R41 calibration note warns against; it remains
available as post-paper follow-up with the phase-domain instrument as its
natural first step. (3) The deconvolved base is the right final config on
the evidence: it is the ONLY config with a coherent declared resolution at
both legs (edge crossing 0.28 → 0.49 px), and it carries the validated
calibration arc as its story. (4) Probe 1's deflation of the spacing
frontier (committed T≈4 was a stream MAXIMUM; white-stream mean 3.15 ±
0.95, config differences within scatter — the max-vs-mean confound's third
appearance, now at stream level) removes the strongest residual-mystery
motivation for B; the paper's spacing row will be restated on pooled
multi-stream measurements. CAVEAT, stated plainly: the declared-resolution
pass rests on a 5-point grid at n=32–96 maps; Stage 3's pre-registered
one-shot blind re-run (with smoothed-peak entries) is the real test, and
if it fails, the fallback is PLAN-phase3's written scope shrink.

**Andreas decides.** Budget state: phase spend ≈ 0.56 of 10 H100-h.
