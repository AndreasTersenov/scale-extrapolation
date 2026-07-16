# 2026-07-16 — Gate-0 novelty kill-test: collapse law + informational compounding limit

Executed per `SPEC-novelty-collapse.md` (pre-registered as REQUIRED before any paper
claim, cprime2 addendum 4; scope extended 2026-07-16 to both audit claims). Protocol:
adversarial — try to prove we are scooped; every quote below was DIRECT-FETCHED this
session (arXiv abs pages, arXiv/ar5iv HTML full texts, or the arXiv PDF read page-by-page
where HTML failed); no from-memory quotes remain. Ambiguity resolved toward "scooped."

**Fetch provenance / failures (honesty):** Seitzer 2203.09168 — ar5iv fatal-error,
OpenReview bot-blocked, alphaxiv 403; resolved by reading the arXiv PDF directly
(pp. 1–8), all Seitzer quotes are from the PDF. Schanz 2310.06929's load-bearing
quotes were verified by a second independent full-text fetch (identical wording).
Bonavita 2309.08473 was fetched (abstract only); it contains no blur-specific sentence
on the abs page, so area 1 rests on GenCast's full text. Everything else fetched clean.

---

## CLAIM 1 — the collapse law (signatures i–iv)

Signatures under test: (i) conditional dispersion peaks early and decays monotonically
with training as the mean memorizes; (ii) finite-data causality (8× exact-symmetry
augmentation eliminates the collapse); (iii) channel-invariance (FM-ODE pushforward,
dispersion-penalty target, explicit Gaussian-NLL head); (iv) exact decomposition
Var(μ|bin) + E[e^{2g}|bin] showing the μ-term absorbing the data's variance.

### Verdict table — claim 1

| Area / source | (i) monotone-with-training | (ii) finite-data causal | (iii) channel-invariance | (iv) decomposition | Ruling (overall) | Verbatim quote pointer |
|---|---|---|---|---|---|---|
| 1. GenCast 2312.15796 (abs+HTML) | not found ("Not found in this paper" on training-dynamics question) | not found | n/a | not found | ADJACENT-BUT-DISTINCT: steady-state blur documented, no dynamics/causality | "They are typically trained to minimise mean squared error (MSE), and as a result tend to produce blurry forecast states, especially at longer lead times, rather than a specific realisation" (Intro); spread/skill defined for calibration (Results) |
| 2. Cascaded diffusion, Ho 2106.15282 (abs+ar5iv) | not found | not found | not found — "does not analyze the super-resolution model's conditional variance or connect compounding error to variance underestimation" | not found | ADJACENT: fixes compounding, never characterizes head-level variance starvation | "conditioning augmentation is effective because it alleviates compounding error in cascading pipelines due to train-test mismatch, sometimes referred to as exposure bias" (§1); augmentation = "adding Gaussian noise (forward process noise)... randomly applying Gaussian blur" (§3) |
| 3. CADS 2310.17347 (HTML v4) | not found ("does not discuss how diversity changes across training epochs") | PARTIAL — dataset-size link stated, correlational, steady-state | not found | not found | ADJACENT; **phenomenon-level statement subsumed** (closest single sentence in the literature) | "the model establishes a near one-to-one mapping from the conditioning signal to the generated images, thereby yielding limited diversity" (§3; pointer corrected §1→§3 at reconvene spot-check); "conditional diffusion models trained on smaller datasets...tend to have limited variation" |
| 3. Memorization: Gu 2310.02664 (abs+ar5iv) | PARTIAL SCOOP (mirror curve, replication channel): "we notice that the memorization ratio increases with more training epochs" (§2, Fig 1d) | PARTIAL — "memorization behaviors tend to occur on smaller-sized datasets"; augmentation deliberately EXCLUDED: "without applying data augmentation (to avoid any ambiguity regarding memorization)" (§2) | not found (replication ratio only, "not sample diversity or model variance") | not found | ADJACENT, theoretical skeleton present | "the typical training objective of diffusion models... has a closed-form optimal solution that can only generate training data replicating samples" (abstract); also: "conditioning training data on uninformative random labels can significantly trigger the memorization" |
| 3. Kadkhodaie 2310.02557 / Somepalli 2212.03860 / Carlini 2301.13188 (abs ×3) | not found | PARTIAL — dataset-size interventions on memorization exist | not found | not found | ADJACENT (sample-replication axis, not conditional-variance axis) | "two DNNs trained on non-overlapping subsets... learn nearly the same score function... when the number of training images is large enough" (Kadkhodaie); "how factors such as training set size impact rates of content replication" (Somepalli) |
| 3b. Seitzer β-NLL 2203.09168 (PDF pp.1–8) | DISTINCT — their worsening is the gradient-weighting effect: "This effect worsens as training progresses" (§3, culprit 2); their Fig 1 tracks RMSE (mean), not dispersion | not found — no data-size ablation in the paper; pathology persists with abundant data (feature-space granularity) | DISCRIMINATES ours: their mechanism requires inverse-variance weighting; our L2 FM-ODE channel has none and collapses identically | not found | ADJACENT-BUT-DISTINCT, mechanisms measurably different | "the NLL loss scaling down the gradient of poorly-predicted data points relative to the well-predicted ones, leading to effectively undersampling the poorly-predicted data points" (Contributions); "the variance quickly shrinks in these areas to match the reduced MSE" (§3.2); their headline harm is "subpar mean fits" (§1) |
| 3b′. Stirn & Knowles 2212.09184 (abs) | not found | not found | not found | not found | ADJACENT (same family as Seitzer: variance corrupts the MEAN — mirror of ours) | "optimizing network parameters via log likelihood gradients can yield suboptimal mean and uncalibrated variance estimates" |
| 4. Posterior collapse, Lucas 1911.02469 (abs) | not found | not found | not found | not found | SAME WORD, DIFFERENT PHENOMENON (latent→prior via likelihood local maxima) | "We explain how posterior collapse may occur in pPCA due to local maxima in the log marginal likelihood" |
| 4. cGAN: pix2pix 1611.07004 (ar5iv); Mathieu 1511.05440 (abs) | not found | not found | not found | not found | ADJACENT folklore ancestors: conditional stochasticity dies at steady state | "the generator simply learned to ignore the noise" + "we observe only minor stochasticity in the output" (pix2pix §3.1); "the inherently blurry predictions obtained from the standard Mean Squared Error (MSE) loss function" (Mathieu) |
| 5. MultiscaleFlow 2306.04689 (abs+ar5iv); CTFM 2505.00632 (abs+HTML) | not found | not found | not found | not found | UNRELATED — no dispersion/collapse discussion anywhere incl. footnotes (both checked) | MultiscaleFlow's only sample-quality claim: "Multiscale Flow samples and test data agree well in terms of the power spectrum and pixel probability distribution function" (§4.4) — i.e., exactly the validation level our downstream demo indicts |
| 6. Free lens: perception–distortion, Blau & Michaeli 1711.06077 (abs); Huszár 1511.05101 (abs) | not found | not found | not found | not found | ADJACENT theory ancestors ("mean-optimal ⇒ distribution-poor" proven abstractly) | "we prove mathematically that distortion and perceptual quality are at odds with each other" (Blau); scheduled sampling's objective "is improper and leads to an inconsistent learning algorithm" (Huszár) |
| 6′. 2025–26 FM diversity-collapse (2602.05951, 2606.27371, 2411.16171 surfaced; first two fetched) | not found ("does not explain its root causes or document whether it increases with training") | not found | not found | not found | ADJACENT: phenomenon now common knowledge in FM; signatures absent | "they suffer from diversity collapse when generating multiple samples under the same conditioning" (2606.27371); "failure modes... including distributional collapse and instability" (2602.05951) |

### Bottom line — claim 1 (binding wording)

A paper may claim: "We measure a **training-time dispersion-collapse curve** in the
conditional-variance channel of a coarse→fine wavelet cascade — conditional dispersion
peaks early and decays as the conditional mean memorizes — show it is
**channel-invariant** (identical across the FM-ODE pushforward, a dispersion-penalty
target, and an explicit Gaussian-NLL head), localize it with an **exact
Var(μ|bin)+E[e²ᵍ|bin] decomposition**, and show **8× exact-symmetry augmentation
eliminates it**; this is consistent with steady-state conditional-diversity collapse
[CADS; pix2pix; Mathieu; GenCast] and with memorization growing with training and
shrinking with data [Gu; Kadkhodaie; Somepalli; Carlini], and is mechanistically
distinct from the β-NLL gradient pathology [Seitzer], which requires inverse-variance
weighting our L2 channel does not have." The **phenomenon** (conditional variance
starved by strong conditioning/limited data) and the **monotone memorization-with-
training curve** (in the replication channel) may NOT be claimed as new. What survives
is: signatures (iii) and (iv) as measurements/diagnostics, the augmentation test as
the first *causal* elimination in a conditional-variance channel, and the scientific-
field cascade setting — i.e., "collapse law" as a paper-internal name for a
synthesized, quantified instance, never as a claimed new discovery.

---

## CLAIM 2 — the informational compounding limit (signatures v–vii)

Signatures under test: (v) end-to-end statistical response equals the head's response
to drifted input (attenuation matching); (vi) the conditioning drift is STRUCTURED,
not additive — matched-amplitude white noise is ~2× less damaging; (vii) honesty-hurts
inversion — exposing the model to its true conditioning uncertainty worsens
pushed-forward statistics vs naive over-trust.

### Verdict table — claim 2

| Source | (v) attenuation matching | (vi) structured ≠ additive | (vii) honesty-hurts | Ruling | Verbatim quote pointer |
|---|---|---|---|---|---|
| Ho 2106.15282 (ar5iv) | not found | not found — "compare Gaussian noise versus Gaussian blur but do not explicitly analyze structured versus white noise corruption" | not found | ADJACENT (we imported their fix; they never measure the drift) | augmentation forms: "adding Gaussian noise... randomly applying Gaussian blur" (§3) |
| Ning 2301.11706 (ar5iv, incl. App. A.3) | PARTIAL — they estimate per-t error amplitude ν_t | **OPPOSITE CLAIM in their domain**: "we empirically verified that, for a given t, e_t is normally distributed: e_t ∼ N(0, ν_t²I)" (§5.3) | PARTIAL SCOOP (per-step analog): DDPM-y (perturbed input WITH consistent target) — "Tab. 7 shows that DDPM-y is even worse than DDPM" (App. A.3); the over-trust asymmetric design (perturbed input, original target) is what works | ADJACENT on (v),(vii); (vi) unaddressed-and-opposite | perturbation is white Gaussian: y_t = √ᾱ_t x₀ + √(1−ᾱ_t)(ε+γ_t ξ), ξ∼N(0,I) (§5.1 Eq. 6) |
| Li 2305.15583 (HTML v5) | ADJACENT — the time-shift move IS effective-corruption matching within a chain: "there might be an alternate time step ts that potentially couples better with x̂_{t−1}" | not found — error treated as variance magnitude only; Gaussianity inherited: "Previous works... have empirically and statistically affirmed that the network prediction error of DPM follows a normal distribution" | not found | ADJACENT on (v); nothing on structure | §1 core claim quoted left |
| Ning 2308.15321 (abs) | ADJACENT — analytic sampling-distribution model; per-step error as root cause | not found | not found | ADJACENT (their fix scales the output DOWN — over-trust direction) | "attribute the prediction error at each sampling step as the root cause of the exposure bias issue"; "Epsilon Scaling... scaling down the network output" |
| Huszár 1511.05101 (abs) | n/a | n/a | THEORY ANCESTOR: scheduled sampling's objective "is improper and leads to an inconsistent learning algorithm" | ADJACENT (proves the objective is biased; does not show the statistics-level inversion) | abstract |
| MAD 2307.01850 (abs); Shi 2509.16499 (abs) | n/a | n/a | not found | ADJACENT, different axis (variance shrinkage across self-consuming TRAINING generations, not across octaves of one cascade) | "future generative models are doomed to have their quality (precision) or diversity (recall) progressively decrease" (MAD); "Prior work primarily characterizes this collapse via variance shrinkage" (Shi) |
| CADS 2310.17347 (HTML) | n/a | uses ADDITIVE Gaussian corruption of conditioning at inference (annealed) — the additive model again | not found (their inference-time noise HELPS diversity in their domain) | ADJACENT to 4b′/4b′-ii machinery, not to (vi)/(vii) | "the conditioning signal is perturbed using additive Gaussian noise combined with an annealing strategy" (§method) |

### Bottom line — claim 2 (binding wording)

A paper may claim: "In a scale-recursive cascade on cosmological fields we **measure**
that (a) the end-to-end statistical response equals the frozen head's response to its
own drifted conditioning (0.742 ≈ 0.746), an attenuation-matching protocol in the
spirit of, but distinct from, time-step recoupling [Li 2305.15583] and epsilon scaling
[Ning 2308.15321]; (b) the conditioning drift is **structured, not additive** —
matched-amplitude white noise is far less damaging (s_matched > the trained range
everywhere) — in contrast to the per-step Gaussian error model measured in image
diffusion [Ning 2301.11706 §5.3]; and (c) train-side exposure to realistic conditioning
worsens pushed-forward statistics (honest conditional ≈0.53 < over-trust ≈0.74),
consistent at the per-step level with the DDPM-y ablation [Ning 2301.11706 App. A.3]
and with the impropriety of scheduled sampling [Huszár 1511.05101]." Signature (vi) is
the strongest surviving element — no fetched source makes the structured-vs-additive
distinction, and the exposure-bias line explicitly models the error as isotropic
Gaussian; (v) and (vii) survive only as measured-in-our-setting instances with the
above citations mandatory. The word "informational limit" must stay MEASURED
("a measured compounding cap consistent with an informational limit"), per the adopted
wording — nothing fetched contradicts that framing, and nothing anticipates it.

---

## EXTRA CLAIM (verified as instructed): do the cosmological SR-by-diffusion papers validate conditional calibration, or only summary statistics?

**Split verdict — the blanket "summary statistics only" claim is FALSE and must never
appear in our paper.**

- **Schanz, List & Hahn 2310.06929 (full text fetched twice, quotes identical):
  PARTIALLY VALIDATES CONDITIONAL CALIBRATION.** They generate ensembles conditioned
  on one LR field and compare width against a true conditional ensemble: "we compute
  kernel density estimates based on 100 HR and SR simulations conditioned on the LR
  simulation" (§4.2, Fig. 6); "the width of the SR distribution is also consistent
  with HR"; and they report the tail deficit qualitatively: "Generally, the
  distributions produced by our diffusion model are somewhat more Gaussian than the
  true HR distributions." Scope limits (what they do NOT do): one conditioning field,
  2D proof-of-concept, no z-scored calibration bars, no per-scale (octave-resolved)
  conditional analysis, no propagation into downstream statistics (their validation
  stats are P(k) + Δ²(k) covariance; "No bispectrum, pixel histograms, or other
  higher-order statistics mentioned" for the conditional-width analysis beyond the
  KDE panels).
- **Rouhiainen et al. 2311.05217 (full text fetched): SUMMARY STATISTICS ONLY.**
  Validation = power spectrum, bispectrum projection, one-point PDF, void size
  function (§5); diversity is shown as sample-to-sample cross-correlation ("At
  nonlinear scales, the cross-correlation coefficient drops to nearly 0", §6) with
  **no truth-referenced conditional spread**: "do not assess whether the conditional
  distribution's spread or tails match theoretical expectations given the
  conditioning" (fetch verdict: not found in paper).

**Consequence for the audit-paper pivot (Option 1):** the conditional-calibration
audit is NOT virgin territory — Schanz et al. already run an embryo of it (width-only,
single conditioning, qualitative tails). Our surviving contribution is the
*systematized protocol*: pre-registered per-octave calibration with bootstrap bars,
tail/kurtosis quantification, the inverted no-truth-at-target pilot, and the
downstream peak-function propagation — plus the failure taxonomy. Position as
"extending the conditional-diversity check of [Schanz] into a calibrated,
scale-resolved audit," never as introducing conditional validation to the field.

---

## Sources fetched this session (26)

arXiv 2312.15796 (abs+HTML), 2106.15282 (abs+ar5iv), 2203.09168 (abs+PDF pp.1–8),
2301.11706 (abs+ar5iv ×2), 2305.15583 (HTML v5), 2308.15321 (abs), 1511.05101 (abs),
2310.17347 (HTML v4), 2310.02664 (abs+ar5iv), 2310.02557 (abs), 2212.03860 (abs),
2301.13188 (abs), 1611.07004 (ar5iv), 1511.05440 (abs), 1911.02469 (abs), 2212.09184
(abs), 1711.06077 (abs), 2307.01850 (abs), 2509.16499 (abs), 2306.04689 (abs+ar5iv),
2505.00632 (abs+HTML), 2310.06929 (abs+ar5iv ×2), 2311.05217 (abs+ar5iv), 2602.05951
(abs), 2606.27371 (abs), 2309.08473 (abs; no blur quote on abs page — not used).
Surfaced but not fetched (not load-bearing): 2411.16171, 2310.05264, scheduled-sampling
original (Bengio et al. 1506.03099 — covered via Ho's and Huszár's characterizations).

## Updated belief

Both claims survive in NARROWED form; neither survives as originally worded. The
collapse-law *phenomenon* is thoroughly pre-known at steady state and its
training-dynamics mirror exists in the memorization literature; our additive content
is the channel-invariance, the decomposition diagnostic, the causal augmentation test
in a variance channel, and the setting. The compounding-limit claim survives mostly
intact, with signature (vi) (structured ≠ additive) the sharpest surviving novelty —
the adjacent literature explicitly models the drift as isotropic Gaussian. The
fallback-paper branch is NOT needed, but the audit-paper framing must cite Schanz's
conditional-width check as prior art and drop any "first conditional validation"
implication. Gate-0 is now CLEARED for paper drafting under the bottom-line wordings
above.
