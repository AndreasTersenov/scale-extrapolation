# §4 — The audit and the mechanisms (how every §2 element was earned)

FIRST PARAGRAPH (binding placement): positioning vs Schanz — "extending the
conditional-diversity check of Schanz et al. (width-only, single
conditioning, qualitative tails) into a calibrated, scale-resolved,
pre-registered audit with downstream propagation." Rouhiainen cited as the
summary-stats-only contrast. <!-- cite: Schanz 2310.06929, Rouhiainen
2311.05217; gate0 §4 -->

## 4.1 The audit protocol (instruments before adjudications)

- The exact-truth sandbox: lognormal; conditional GRF sampling exact;
  every instrument calibrated before adjudicating any model (Gate-A:
  var_slope ≤0.6%, kurtosis ≤5.0%). <!-- src: gateA_instrument.json -->
  Fig F4.
- The three tiers (R27 wording verbatim): power; conditional marginals
  (var_slope, kurtosis, q999); joint/morphological structure (peak-count
  excess at fixed marginal calibration). Plus the held-out basis
  (starlet-ℓ1) and — NEW, the phase-3 addition — the untouched judge
  (Minkowski functionals; frozen pre-cure, validated on synthetics only,
  first applied at the one-shot; §5.2).
- The protocol's teeth without a good generator (the inverted pilot):
  self-consistency z=7.9 vs 0.7; held-out scattering rejects at 98–100% of
  channels; the deployable variant fails its own calibration — deployable
  checks need population-calibrated bands. <!-- src: RESHAPE §2 step 2 -->
  Fig F5.
- The downstream demonstration: with P(k)-level checks passing (≤7%), the
  peak function is tilted 6–14σ, sign-flipping — both mechanisms flagged by
  the audit before a peak was counted. <!-- src: RESHAPE §2 step 3 --> Fig F6.
- Audit of the auditors: starlet package findings; the coloring-index and
  Minkowski validation corrections (appendix A.4) — the protocol treats
  validators as auditable objects.

## 4.2 Disease I — conditional-variance collapse ("collapse law",
paper-internal name)

BINDING WORDING, VERBATIM (Gate-0 claim-1 bottom line; quote in full where
the claim is made): "We measure a training-time dispersion-collapse curve in
the conditional-variance channel of a coarse→fine wavelet cascade —
conditional dispersion peaks early and decays as the conditional mean
memorizes — show it is channel-invariant (identical across the FM-ODE
pushforward, a dispersion-penalty target, and an explicit Gaussian-NLL head),
localize it with an exact Var(μ|bin)+E[e²ᵍ|bin] decomposition, and show 8×
exact-symmetry augmentation eliminates it; this is consistent with
steady-state conditional-diversity collapse [CADS; pix2pix; Mathieu; GenCast]
and with memorization growing with training and shrinking with data [Gu;
Kadkhodaie; Somepalli; Carlini], and is mechanistically distinct from the
β-NLL gradient pathology [Seitzer], which requires inverse-variance weighting
our L2 channel does not have."
<!-- src: log/2026-07-16-novelty-collapse.md claim-1 bottom line -->
- Never claimed as new; Seitzer discrimination its own paragraph.
- Cure at truth grade: CFM + 8× D4 holds dispersion 1.02–1.07 vs truth
  1.070, flat through 20k steps. <!-- src: RESULTS-phase2.md #1 -->
- Figs F7, c1_sandbox.

## 4.3 Disease II — our own repair was the poison (the mixture artifact)

[Carried verbatim from the pre-phase-3 skeleton — narrative, forensic
numbers (μ-path over-modulation 1.536/1.676 vs 1.020; kurtosis 32.5/78.3 vs
6.7; production response 0.746 = cascade diluted by a ~2/3-variance white
bath <!-- src: forensic_nllnoise.json; R8 -->); head retired, −27% → −9/−10%
with the sampling channel the single variable; CLAIM-2 BINDING WORDING with
the 07-17 scope clause attached to every quotation; the structured-not-
additive drift flagship paragraph with its citation block; the lesson
sentence.] <!-- cite: Ning 2301.11706 §5.3, Li 2305.15583, Ning 2308.15321,
Huszár 1511.05101, Ho 2106.15282; src: smatched_4bpii.png -->
Figs F8, F9.

## 4.4 Disease III — tail starvation, and the moment ladder as a SCHEDULE

[Carried: the pass-through signature; twCRPS retired; the causal 1×/8× test
(terminal collapse vs at-truth 6.11/5.96, 4.32/4.15, 4.57/4.15); base
erasure (P3 verbatim R16); the cure = t(5) base + caged selection, both
components necessary; sandbox 24/24 + repl64; real-field binding-octave
deficit closes −33.7→−4.0% / −26.1→−8.0%.] <!-- src: taildyn_*.json;
c1t_verdict_sandbox.json; c1t_repl64.json; c1t_gow_descriptive.json -->
NEW (phase-2/3 upgrade): the ladder matured into a SCHEDULE — transient
defects (nonzero detail-channel means, corr(H,V)) rise and fall along
training, and the marginal-optimal cage harvests whatever defect is live at
its pick; caged picks land anywhere on the trajectory (16000/3500/5500/
10500/9000 across this paper's runs <!-- src: selection jsons -->), which
is why the robustness table prints the seed spread (§3.4) and why selection
is part of the method. Figs F10, F11.

## 4.5 The lattice-parity arc, the transfer function, and the three
## dissociations (the phase-3 mechanism block)

- **Lattice parity ✂ peaks (dissociation 1).** The sampler's lattice
  defect (transient channel means + corr(H,V), schedule-harvested) is
  KILLED by exact D4 group-averaged sampling (F2; parity_T 1.00, coef-T
  2.18 at the edge) while the peak excess moves 0.16σ/0.39σ — the cure
  worked on its target; the excess is orthogonal.
  <!-- src: stage0_p3_verdict.json; f2_verdict.json -->
- **The transfer function of weight-tied extrapolation (measured object).**
  The flow's spectral action on the never-trained octave is a nearly
  input-independent multiplicative filter: T(k) rising 0.66 → 1.06 across
  the band, two-input consistency T_col/T_white median 0.981
  (IQR 0.967–1.029). The octave-1 "whiteness" defect IS this transfer
  applied to a white base. <!-- src: l1p_transfer_analysis.json -->
- **Base coloring ✂ peaks (dissociation 2).** Coloring the base toward the
  real detail spectrum moves the output coloring by −24.8σ (the
  wrong-target inversion the transfer model predicts) while the peak
  excess moves 0.02σ. <!-- src: l1p_verdict.json -->
- **Octave-1 spectrum ✂ peaks (dissociation 3).** Deconvolving the
  measured transfer restores the coloring (P-T lands; the oracle lands ON
  real at 0.04σ) — and the excess still does not move (Δ 0.34/−0.31σ;
  oracle stream +15.75%/+12.91% at restored-real coloring). Three coloring
  doses, nine streams, same ~+14%: the native peak excess is INDEPENDENT
  of the octave-1 power spectrum at fixed weights.
  <!-- src: l1pp_verdict.json -->
- **Where the excess lives.** It is pixel-scale (0.5-px smoothing kills
  +14.5% → −1.7% <!-- src: stage1_p3_probes.json morphology -->), tracks
  the FIELD not the scale (real trained legs +14%, edge +12–13%; sandbox
  legs DEFICITS <!-- src: audit_peak_ci.json; stage0_p3_verdict.json -->),
  is dose-responsive in the cross-model taxonomy on the real field
  (Gaussian-base +5.7σ whiteness → +24.6% excess; over-colored arm B →
  deficits <!-- src: stage1_p3_l3_taxonomy.json -->), and after the three
  dissociations is located in the PHASE / higher-order structure of the
  generated fine details. This located mechanism is the §5.1 boundary.
- The L1→L1′→L1″ arc as the audit-guided-design exhibit: naive target →
  inversion → system identification → validated calibration (5/5). One
  page, told as method development, with the wrong-target step INCLUDED
  (the deconvolution algebra was available at design time and two review
  layers missed it — appendix scorecard).
Figs: F17 mechanism panel (detail planes + spectra), F18 smoothing sweep,
F19 L3 taxonomy scatter, F15 map gallery.
