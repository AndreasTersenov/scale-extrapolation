# PLAN — Stage-0 for RG-consistent generation ("train small, sample any scale")

**Project claim (context):** a wavelet-factorized flow-matching generator with
scale-shared (tied) weights — RG fixed-point as architectural prior — trained at ≤128²,
generating at 4–16× that resolution, with certified multi-scale statistics. Per the
novelty sweep, the load-bearing contribution is NOT the architecture (wavelet FM exists:
arXiv:2605.16573; 1D tied-weight equivariance proof: arXiv:2605.17582; theory ancestor
WC-RG: arXiv:2207.04941) but **P5/P6**: the predicted breakdown of non-Gaussian
statistics in the first extrapolated octave, and its repair by scale-conditioning
("running couplings"). Background: `~/claude-notes/brainstorms/2026-07-09-dl-project-directions.md`
(§D4, v1.1, v2) and `...novelty-sweep-RESULTS.md` (§D4). D6b (single-realization mode)
is deferred until after stage-0 + toy.

**Stage-0 purpose:** P5 is a claim about the FIELDS, not about any network. Measure the
scale-drift of conditional wavelet statistics directly on data. No generative model, no
training.

---

## FROZEN CORE — do not modify (pre-registered 2026-07-09)

### Measurement M1 — scale-drift of conditional wavelet statistics

For each field class, wavelet-decompose (2D DWT, ≥5 octaves) many maps and estimate,
per octave j: (a) the marginal PDF of normalized detail coefficients; (b) the
conditional statistics of detail coefficients given the coarse field at that scale
(binned conditional means/variances/skewness — the objects WC-RG models per scale);
(c) cross-octave coefficient couplings (correlation of |w_j| with |w_{j+1}| at aligned
positions). Then quantify DRIFT between octaves: a distance (e.g. W1 between per-octave
conditional PDFs) as a function of octave separation. A weight-tied network can only be
exactly right if these objects are octave-invariant; the drift profile is the exact
gap the "running couplings" conditioning must absorb.

### The control ladder (all data already at `/project/rrg-lplevass/shared/wl_chall_data/`)

1. **GRF (`GRF_HF`) — null gate.** For a power-law GRF, suitably normalized wavelet
   statistics are scale-invariant: measured drift ≈ 0 within estimator noise. If the
   pipeline shows drift on GRF, the pipeline is buggy — fix before proceeding. (For a
   non-power-law spectrum, the Gaussian conditional structure is still analytic —
   whatever drift the spectrum implies is computable in closed form; verify against it.)
2. **Lognormal (`lognormal`) — analytic control.** Known pointwise non-Gaussianity;
   drift expected, partially computable; estimator sanity check in the non-Gaussian
   regime.
3. **N-body (`gowerstreet*`) — the measurement.** The physical scale-dependence of
   non-Gaussianity (quasi-linear → nonlinear transition) is the thing D4's repair must
   capture.
4. **Cross-check field (choose one, implementer's pick):** a second real field with
   different physics — e.g. a JHU turbulence database slice, or any local hydro/κ map —
   to test whether the drift profile's SHAPE is field-specific or qualitatively
   universal. (Qualitative agreement → method is general; disagreement → interesting,
   report it.)

### Pre-registered predictions

- **P9a (70%):** GRF drift consistent with zero (after the analytic spectrum correction);
  N-body maps show drift between adjacent octaves that is large compared to estimator
  noise (operationally: >3σ and >10% in the chosen distance) across the
  train-vs-extrapolation octave range.
- **P9b (55%):** the drift is LOW-DIMENSIONAL: ≥80% of the cross-octave variation in the
  conditional statistics is captured by 1–3 smooth functions of scale ("running
  couplings" exist and are few). This is the load-bearing bet for P6's repair being a
  small conditioning, not a full per-scale model.

### Kill / gate criteria

- **K-M1a:** GRF null fails after reasonable debugging → estimator problem; do not
  proceed to interpretation until fixed.
- **K-M1b (project gate):** if N-body drift across the intended extrapolation range is
  NOT significant (P9a false), then naive weight-tying already suffices → the P5/P6
  paper does not exist as scoped → STOP, report; pivot options (certification-only
  angle, or fold effort into D1/D6a) are decided at reconvene.
- If P9a true but P9b false (drift high-dimensional): project lives but gets harder —
  report the effective dimensionality; the toy-phase conditioning design will need it.

### Out of scope for stage-0 (do NOT build)

Any generative model or training; 512² generation; TRACE/typicality certificates;
D6b single-realization mode; comparisons to arXiv:2507.01707 (that's toy-phase).

---

## FREE PERIPHERY — implementer's choice

Wavelet family (start db4 or Haar; check robustness with one alternative), binning
schemes, distance metric details (W1 vs KS vs L2 on normalized PDFs — justify in log),
number of maps (enough that estimator noise ≪ measured drift; bootstrap error bars
required), torch (`~/software/wl_stats_torch` may be reused) vs pywt vs jax, GPU usage.

## Logging (required)

Same convention as the sibling repos: `log/YYYY-MM-DD-<slug>.md` entries
(hypothesis → setup → expectation → result → updated belief). Final `RESULTS.md`:
the drift-vs-octave-separation curves per field class with error bars, the GRF null
verdict, P9a/P9b verdicts with numbers, gate status, and — if alive — the empirical
"running coupling" functions extracted (the toy-phase design spec). Written for a
reconvene session that has read PLAN.md but not the code.
