# SPEC — novelty kill-test: the collapse law AND the informational compounding limit

**For a cheap-model session. Read this file first; repo context ONLY if needed:
RESULTS-phase1c.md + log/2026-07-11-reconvene-4a.md. Budget: one session,
web-search + direct-fetch, NO code. Deliverable: log/YYYY-MM-DD-novelty-collapse.md.
This test was pre-registered as REQUIRED before any paper claim (cprime2 addendum 4);
scope extended 2026-07-16 to cover both audit claims (reconcile-audits ruling).**

## Claim 2 under test (added 2026-07-16): the informational compounding limit

In hierarchical (coarse→fine cascaded) conditional generation: (v) the end-to-end
statistical response equals the head's response to drifted input (attenuation
matching); (vi) the conditioning drift is STRUCTURED, not additive — white noise of
matched amplitude is ~2× less damaging than the actual generated-conditioning drift;
(vii) the "honesty-hurts" inversion: exposing the model to its true conditioning
uncertainty at generation time WORSENS pushed-forward statistics vs naive over-trust.
Sweep additionally: diffusion exposure-bias literature (Ning et al. 2301.11706 input
perturbation; Li et al. exposure-bias diffusion line; scheduled-sampling ancestors) —
is (vii) documented? Is the structured-vs-additive distinction (vi) made anywhere?

## Claim 1 under test (ours, exactly)

**The collapse law:** in conditional generative models trained on finite data,
memorization of the conditional MEAN starves the conditional VARIANCE — in whatever
channel carries it (deterministic-ODE pushforward of flow matching, dispersion-penalty
targets, explicit Gaussian-NLL variance heads). Signatures we claim: (i) conditional
dispersion peaks early in training and decays MONOTONICALLY as the mean memorizes;
(ii) the mechanism is finite-data (CAUSAL test: 8× data via exact-symmetry
augmentation eliminates the collapse over the same horizon); (iii) channel-invariance
(three channels confirmed); (iv) exact decomposition Var(μ|bin) + E[e^{2g}|bin] showing
the μ-term absorbing the data's variance. Context: 2D cosmological fields,
coarse→detail wavelet conditionals, 322-tile training set.

## Kill-test protocol (adversarial: prove we are scooped)

For each area: fetch abstracts/PDFs, quote verbatim, rule
IDENTICAL / SUBSUMES / ADJACENT-BUT-DISTINCT / UNRELATED per signature (i)–(iv).
From-memory quotes are DRAFT until fetched. Ambiguity = assume scooped.

Sweep areas, priority order:
1. **Probabilistic forecasting blur** — the closest big literature: ML weather models
   regress to the conditional mean and lose ensemble spread (GenCast 2312.15796,
   Graphcast-era critiques, CRPS-trained ensembles). Do they document the
   MONOTONE-WITH-TRAINING collapse and a data-size causal test, or only the
   steady-state blur?
2. **Exposure bias / cascaded conditioning** — scheduled sampling lineage; cascaded
   diffusion conditioning augmentation (Ho et al. 2106.15282, Imagen). We IMPORTED
   their fix for compounding — the question is whether they also characterize the
   head-level variance starvation (our law) or only the train-test mismatch.
3. **Conditional diffusion/FM variance miscalibration** — searches: "conditional
   diffusion underdispersion", "flow matching variance collapse", "diversity collapse
   conditional generation", "condition memorization diffusion". Also memorization-in-
   diffusion literature (Somepalli et al., Carlini et al.) — they show sample-level
   memorization with dataset-size dependence; do they connect it to CONDITIONAL
   VARIANCE starvation?
3b. **Heteroscedastic regression variance collapse — Seitzer et al., "On the
   Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural
   Networks" (β-NLL, arXiv:2203.09168) and follow-ups** — the sharpest known
   neighbor for the σ-head starvation (our signature iv). Rule per signature: do
   they show the MONOTONE-with-training decay (i)? finite-data causality (ii)?
   channel-invariance (iii)? Their mechanism is gradient-weighting pathology — is
   ours (mean memorization) the same thing in different words or measurably
   distinct?
4. **Posterior collapse (VAE) + mode collapse (cGAN)** — the folklore ancestors. Rule
   carefully: same phenomenon or same word?
5. **CTFM (2505.00632) and MultiscaleFlow (2306.04689)** specifically (we build on
   their setting): any dispersion-collapse discussion, esp. footnote-level.
6. One free lens of your choosing; report what surfaced.

## Output

Verdict table (signature × area × ruling × quote pointer). Bottom line, three
sentences max: what exact sentence about the collapse law may appear in a paper
("we observe X, consistent with [refs], and add Y"), where Y is only what survived.
If Y is empty, say so plainly — the fallback-paper branch absorbs that outcome fine.
