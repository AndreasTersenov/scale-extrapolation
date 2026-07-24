# §4 — Anatomy and therapy: the moment ladder

Framing sentence [WRITE]: three diseases found in sequence by the §3
instruments; the connecting mechanism ("moment ladder" — introduced as a
PAPER-INTERNAL name, Gate-0 discipline) is that finite-parent training
starves conditional structure moment by moment, each rung with a causal
data-size signature.

## 4.1 Disease I — conditional-variance collapse ("collapse law",
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
- The phenomenon and the monotone curve are NEVER claimed as new.
- Seitzer discrimination gets its own paragraph (positive result).
- Cure verification at truth grade: plain CFM + 8× D4 holds conditional
  dispersion 1.02–1.07 vs truth 1.070, flat through 20k steps, naked ODE
  channel. <!-- src: RESULTS-phase2.md "three numbers" #1; fig c1_sandbox -->
- Figs: signature_4a.png (the decomposition), c1_sandbox.png.

## 4.2 Disease II — our own repair was the poison (the mixture artifact)

- Narrative [WRITE from explainer §5.3]: variance head bolted on → months of
  "fundamental limit"-shaped results → forensic on the FROZEN model.
- Forensic numbers: μ-path-only regeneration over-modulates (oct-2 var_slope
  1.536/1.676 vs real 1.020) with kurtosis 32.5/78.3 vs real 6.7, at the
  pre-declared amplitude confound (detail_std ratio 0.38/0.39 = √(1−σ-share)).
  Production response 0.746 = that cascade diluted by a ~2/3-variance white
  Gaussian bath. <!-- src: forensic_nllnoise.json; R8 -->
- Head retired; compounding deficit moves −27% → −9/−10% with the sampling
  channel as the single variable. <!-- src: RESULTS-phase2.md #2 -->
- CLAIM-2 lives here. BINDING WORDING VERBATIM (Gate-0 claim-2 bottom line),
  and the 07-17 scope clause ATTACHES to every quotation: "of the
  variance-head sampler; the information required for calibrated modulation
  is measurably present in the conditioning (forensic 2026-07-17)".
  The structured-not-additive drift measurement (signature vi) is the
  flagship surviving novelty and is stated HERE with its own paragraph
  (audit finding 3.2 — it was homeless in the earlier skeleton):
  matched-amplitude white noise is ~2× less damaging than the real
  conditioning drift — in contrast to the isotropic-Gaussian error model of
  the exposure-bias lineage. <!-- cite: Ning 2301.11706 §5.3 (opposite claim
  in their domain), Li 2305.15583, Ning 2308.15321, Huszár 1511.05101, Ho
  2106.15282; src: smatched_4bpii.png; novelty-collapse claim-2 table -->
- Lesson sentence (kept verbatim-adjacent from explainer): the most
  convincing "fundamental limit" of the project was an artifact of its own
  instrumentation.
- Figs: nll_diagnosis.png / nll_sigma_maps.png (forensic panel),
  smatched_4bpii.png.

## 4.3 Disease III — tail starvation (the ladder's second measured rung)

- The signature: four of six objective/sampler pairs PASS THROUGH near-truth
  tail states and converge away (trajectories verbatim in fig taildyn.png);
  proper-score objectives are not the cure (twCRPS retired: skew pathology
  worsens with data +0.79…+1.11 while t-base stays clean +0.014…+0.023).
  <!-- src: R14–R16; bakeoff/taildyn JSONs -->
- The causal test at rung 4: every 1× run terminally collapses (final kurt
  1.07–1.66); at 8×, three of four runs sit AT truth (6.11/5.96, 4.32/4.15,
  4.57/4.15, last-3-eval means) and hold flat; the fourth holds at 80%.
  <!-- src: taildyn_*.json; R16 -->
- The mechanism gem (P3, adopted verbatim R16): at 1× the trained flow
  converges to BASE-INDEPENDENCE (base erasure); at 8× the base's tail
  contribution survives with a stable gap. Base erasure is a data-starvation
  symptom, not a property of the flow.
- The cure at fixed data: Student-t(ν=5) base (extremes by default) + the
  caged, pre-registered validation-selected checkpoint. Attribution honest,
  both components necessary: the Gaussian-base ceiling never reaches truth
  (4.29/4.59 vs 4.917); selection picks within the window.
  <!-- src: c1_tails_val.json / c1t readout; R18.iii -->
- Verification: sandbox first full pass, all 24 bars both arms (32-field
  caveat stated; 64-field replication 24/24 with A clean everywhere ≤12.4%,
  B's oct-4 over-tails 41.0% riding a 56.0% bar — named, direction OVER);
  real-field binding-octave deficit closes −33.7→−4.0% / −26.1→−8.0%
  (halves-prediction fired), dispersion at the real value (+1.9/+1.4%).
  <!-- src: c1t_verdict_sandbox.json, c1t_repl64.json, c1t_gow_descriptive.json -->
- Figs: taildyn.png, c1t.png, maps_channels.png.

## 4.4 The ladder, stated [WRITE, half page]

Rung 2 (dispersion) and rung 4 (tail weight) each causally data-limited, same
lever; N_eff ≈ parents says why rare-event structure starves first; the
selection cage is a caged recipe treating a symptom (principled-vs-recipe
audit from explainer §10 quoted); §6 carries the joint/morphological rung
question to the placement experiment.
