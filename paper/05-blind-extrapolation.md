# §5 — The blind extrapolation test (Stage D: the founding bet)

## 5.1 The deployment protocol [WRITE from prereg]

Train octaves {3,4}; octave 2 is the HELD-OUT edge; selection caged at octave
3 (the finest TRAINED octave — no deployment-side choice touches the edge);
the dial arm's couplings CURVE-EXTRAPOLATED from trained-visible octaves only
(no measured target-octave information); AMBIGUOUS = FAIL.
<!-- src: log/2026-07-18-prereg-stageD.md -->

## 5.2 The verdict (D-PASS-BOTH; the bare-floor discipline)

| arm | level | var_slope (rel err) | kurtosis (rel err) |
|---|---|---|---|
| A (scale-blind) | head-conditional | 0.915 (6.0%) | 5.139 (14.9%) |
| A | end-to-end | 1.019 (4.7%) | 5.846 (3.2%) |
| B (curve dial) | head-conditional | 0.999 (2.6%) | 6.182 (2.4%) |
| B | end-to-end | 1.049 (7.9%) | 6.304 (4.4%) |

Real edge references: var_slope 0.973, kurtosis 6.040.
<!-- src: stageD_verdict.json, verbatim -->

MANDATORY STATEMENTS (attach wherever this result is stated):
- The formal 3·SE bars inflated to 56–59% (32-field reference-side SE) and
  are near-vacuous; the result is adjudicated on the BARE floors (10%/15%),
  where all 8 checks still pass — arm A's head-conditional kurtosis inside
  by 0.1%. The end-to-end numbers are unambiguous; the conditional-level
  margin is razor-thin. Bar-ledger #9 disclosed.
- Replication context from §4.3 (the 64-field sandbox replication) stated.

## 5.3 The dial finding (the founding hypothesis dies so the mechanism lives)

Dial-beats-scale-blind: FALSE (|e2e kurt deficit| B 4.4% vs A 3.2%).
Single-octave extrapolation works SCALE-BLIND — §2.3's r*≈1 confirmed at
deployment level: the coarse conditioning itself carries the scale
information. The drift is real and measured (§2.2 stands untouched); the
architecture does not need to be told about it at one octave. The dial is
not inert: it flips the joint/morphological excess sign (below).
<!-- src: stageD_verdict.json dial_beats_scaleblind; R22 -->

## 5.4 The held-out basis and the open tier (R27 wording throughout)

- Starlet-ℓ1 at the edge: +5.4/+4.0/+3.0/+1.8% (2–16px) vs a 10% floor; the
  Stage-D edge row +4.0% is the headline; survey-noise convention robust
  (all totals within ±3.5%). <!-- src: starlet_l1_edge.json -->
- The joint/morphological residual, WITH ERROR BARS AND DISCLOSURE (both
  mandatory): arm A peak-count excess +13.1%±3.1% (ν=2.5) and +14.6%±3.1%
  (ν=3), z≥4.2; the real edge reference comprises THREE parent simulations;
  the per-parent excesses are sign-consistent ([+10,+9,+20]% and
  [+21,+15,+9]%); ν=1 rows are parent-dominated and not headlined
  (bar-ledger #10). The dial flips the sign: B −9.4%±2.6% / −17.3%±2.2%.
  <!-- src: audit_peak_ci.json -->
- The earned tier sentence (R27 form): even the field's most constraining
  marginal statistic cannot see the morphological excess. Literal placement
  is untested pending the placement experiment (§6.1).
- Taxonomy panel: one statistic separates the diseases (NLL-head graininess
  +14.2% at 2px with COLLAPSED tail share 0.0237 vs 0.0620; μ-only skeleton
  −83.8%; C1 vs C1-t below total-ℓ1 resolution, ranked only by the 4px tail
  share). <!-- src: starlet_l1_taxonomy.json; fig starlet_l1.png -->

## 5.5 The cost curve

Two-octave extrapolation degrades to −44/−54% kurt (vs the fully-trained
substrate's −29/−40 at its extrapolated octave): each octave of extrapolation
costs; the validated domain is ONE octave, with the second octave's cost
measured. <!-- src: stageD readout descriptive block -->

Figs: stageD.png (near-paper-ready 4-panel), starlet_l1.png,
c1t_maps_peaks.png + the new peak-CI/per-parent panel [HARDEN-FIG].
