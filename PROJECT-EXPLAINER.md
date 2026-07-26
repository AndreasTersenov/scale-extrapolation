# Statistically Faithful Scale Extrapolation for Cosmological Fields
### A self-contained scientific account of the project — goal, method, reasoning, results, uses
*(2026-07-24. Written deliberately without internal run/arm codenames; every quantitative
claim traces to a pre-registered, committed measurement in this repository.)*

---

## 1. The scientific goal

Modern cosmology consumes enormous numbers of simulated maps — for covariance
matrices, for testing analysis pipelines, for training inference networks. The
information that current surveys chase lives increasingly in the *non-Gaussian*,
small-scale structure of these maps: the variability from region to region, the
heavy tails, the rare bright peaks. High-resolution simulations that get this
structure right are expensive; low-resolution ones are cheap.

The project asks two questions, which turn out to be inseparable:

1. **Generation.** Can a generative model trained *only* on affordable,
   coarse-resolution fields produce the next finer level of structure — beyond
   anything it saw in training — with statistics faithful enough for scientific use?
2. **Trust.** At a scale where no ground truth exists, how would you ever *know*?
   What does validation even mean for beyond-data generation?

The second question turned out to carry the project: the validation machinery we
built to answer it ended up diagnosing and curing the generator itself.

---

## 2. The object we build

### 2.1 The scale ladder

A field $x$ (e.g., a $128^2$ weak-lensing convergence map) is decomposed with an
orthogonal wavelet transform into a coarse approximation and detail coefficients
at each dyadic scale ("octave" = one factor-2 refinement):

$$x \;\longleftrightarrow\; \{\,c_J,\; d_J, d_{J-1}, \dots, d_1\,\},$$

where $c_J$ is the coarsest map and $d_j$ the detail needed to go from scale $j$
to scale $j-1$. Because the transform is orthogonal and exactly invertible, the
probability of a field factorizes *exactly* into a chain of conditionals:

$$p(x) \;=\; p(c_J)\;\prod_{j=J}^{1} p\!\left(d_j \,\middle|\, c_j\right).$$

Generation is then recursive: given a coarse map, sample this octave's detail
from the learned conditional, assemble the finer map, repeat. **One weight-tied
network** represents every rung's conditional — the same rule applied at every
scale. To generate *beyond* the training resolution, you apply the same rule one
more time.

### 2.2 The conditional model

Each rung's conditional $p(d_j \mid c_j)$ is a genuine probability distribution
— the universe rolls dice at small scales, and one coarse map admits many valid
detail fields. We model it with **conditional flow matching**: a network
$v_\theta(x_t, t, c)$ is trained to regress the velocity of a probability path
from a base distribution to the data,

$$\mathcal{L} = \mathbb{E}_{t,\,x_0,\,x_1}\big\|\,v_\theta(x_t, t, c) - (x_1 - x_0)\,\big\|^2,
\qquad x_t = (1{-}t)\,x_0 + t\,x_1,$$

and sampling integrates the learned ODE from base noise $x_0$ to a detail field.
In its final, working form the model has two deliberate modifications, each the
cure of a measured disease (Section 5): the base distribution is *heavy-tailed*
(Student-t) rather than Gaussian, and the training checkpoint is chosen by a
strict, pre-registered validation rule rather than "train until done."

### 2.3 What "faithful" means here

Success is never judged per image — a conditional *sample* is not a prediction
of one specific truth map. Faithfulness means the **conditional law** is right,
measured by (in increasing order of difficulty):

- second moments and their environmental modulation — does detail variance
  respond to the coarse environment the way real detail does?
- marginal tail weight (excess kurtosis, extreme quantiles) — are the rare,
  bright fluctuations produced at the right rate?
- joint / positional structure — do the extremes land *where* the environment
  demands them (peak counts and their spatial statistics)?

These three tiers became the backbone of the audit (Section 4), and the campaign
demonstrated that each tier fails invisibly to the tier below it.

---

## 3. Why this design — and not another

**Why not generate full high-resolution images directly?** Because the ladder
converts one huge generative problem into a chain of small, statistically
similar ones, and because the factorization above is *exact*: coarse-graining a
generated map returns its conditioning identically, by construction. In
pixel-space cascades (the standard super-resolution pattern) consistency between
levels is merely learned and can silently drift.

**Why wavelets specifically?** Three load-bearing properties. (i) *Criticality*:
the detail coefficients are exactly the new degrees of freedom of each octave —
no redundancy, so independently sampled details always assemble into a valid
field. (ii) *Statistical homogeneity*: detail coefficients are near-zero-mean,
localized, and similar from octave to octave, which is what makes one weight-tied
network per ladder sensible. (iii) *The physics language*: coarse-graining is the
renormalization-group operation; the mathematics of the wavelet-conditional
renormalization group even proves that these per-scale conditionals have
short-range structure — a theorem our own locality measurement (Section 5.1)
confirmed empirically at deployment level. The *specific* wavelet (we use Haar)
is an implementation detail; the exact, critically-sampled factorization is not.
Redundant transforms popular elsewhere in weak lensing (e.g., the starlet) are
excellent *measurement* bases — we use one as an independent validation
instrument — but cannot serve as generation coordinates without losing the exact
factorization.

**Why a learned conditional rather than the alternatives?** The field has three
schools. *Statistic-matching* (scattering-transform / maximum-entropy models)
generates fields that reproduce chosen summary statistics — interpretable,
data-efficient, but bounded by the statistics you thought to enforce.
*Conditional super-resolution* (GAN/diffusion pairs of low- and high-resolution
sims) is powerful but definitionally interpolation: it requires training
examples *at the target resolution*, which is exactly what we assume we cannot
afford. *Posterior sampling with learned priors* solves inverse problems at a
fixed resolution. Our approach — a learned conditional law, weight-tied across
scales, applied once beyond the data — is the only one of the four that even
attempts validated extrapolation, and it captures the full joint law rather than
enforced summaries. The price is that everything rests on trust, which is why
the audit is half the project.

**Could it be done another way?** Partly, and we measured the comparisons where
it mattered: training objectives that score whole predictive distributions
(energy scores / CRPS, the weather-forecasting solution) turned out to be
mechanistically blind to symmetric heavy tails — the very statistic our fields
need — which we established with fifteen-minute toy experiments before spending
GPU time. An explicit per-pixel variance head (our own first fix) proved
actively harmful. The final design is not the one we started with; it is the one
that survived.

---

## 4. The methodology — how the campaign was run

Every experiment in this project followed one discipline:

- **Pre-registration.** Predictions with stated confidences, decision rules
  written *branch-complete* before any run (including what a null result would
  mean), and success bars derived from *measured references* — never set by feel.
  Ambiguity counts as failure.
- **Cheapest-first falsification.** Model-free measurements before any model;
  closed-form toys before GPU arms; a pre-training sanity gate on every new
  component. Two GPU allocations were saved in a single day by such gates, and
  both stops yielded a mechanism, not just avoided cost.
- **An exact-truth sandbox.** A synthetic lognormal field where the *true*
  conditional ensembles are computable in closed form (conditional Gaussian
  sampling under the log map). Every measurement instrument was calibrated
  against exact truth (to $\lesssim 1\%$) before adjudicating any model, and
  every candidate model faced the sandbox before real data.
- **The three-tier audit.** Power-spectrum-level checks, then conditional
  marginals (variance modulation, kurtosis, extreme quantiles), then positional
  structure (peak counts and placement) — plus a *held-out statistic* in a
  different basis (the starlet $\ell_1$-norm, computed with independent,
  published code) that was never used in any design decision.
- **Blind extrapolation tests.** The decisive protocol hides the finest
  available octave entirely — training, model selection, everything is blind to
  it — extrapolates into it using only the trend measured at coarser scales, and
  only then compares against the hidden truth.

---

## 5. What we did, step by step, and what each step showed

### 5.1 Founding measurements (model-free)

Before any network: we measured how the conditional law $p(d_j\!\mid\!c_j)$
*changes* across scales in gravitational simulations. Findings: the change
("drift") is real and large (several $\sigma$ per octave — a scale-blind model
is measurably wrong), but **low-dimensional and smooth** — two components
capture most of it — and **exactly zero on Gaussian control fields**. Later we
added two more: conditional *locality* — the detail at a point is predictable
from the coarse field within roughly one coarse pixel, and adding context
beyond that adds almost nothing — and a sobering data-accounting result: the
*effective* sample size of a training set is approximately the number of
independent parent simulations, not the number of tiles cut from them.

### 5.2 Disease I: conditional collapse (and its causal cure)

Trained naively, the model's conditional *variance* dies: dispersion peaks
early in training, then decays monotonically as the network memorizes the
conditional mean. We showed this in three different architectural channels, and
localized it with an exact decomposition,

$$\operatorname{Var}(d \mid \text{bin}) \;=\; \operatorname{Var}\!\big(\mu(c) \mid \text{bin}\big)
\;+\; \mathbb{E}\!\left[\sigma^2 \mid \text{bin}\right],$$

which showed the mean term absorbing the data's variance while the stochastic
term starved. The *cause* was established causally: expanding the effective
training set eightfold using exact symmetries of the field (which provably
preserves the conditional law) **eliminates** the collapse. The phenomenon
itself is known in the generative-modeling literature; our additions are the
channel-invariance, the decomposition diagnostic, and the causal data-size test.

### 5.3 Disease II: our own repair was the poison

Our first fix bolted an explicit variance head onto the model. Months of
"fundamental limit"-looking failures later, a forensic experiment on the frozen
model — regenerating with the noise term switched off — revealed that the mean
path alone *over*-modulates with enormously heavy tails, and the production
output was that signal drowned in the head's own white-noise bath. What had
looked like an information-theoretic ceiling was arithmetic: a mixture of two
opposite miscalibrations. The head was removed; plain flow matching with the
data cure restored dispersion to the real value and cut cross-scale error
compounding by two-thirds. Lesson worth stating: the most convincing
"fundamental limit" of the project was an artifact of its own instrumentation.

### 5.4 Disease III: the moment ladder and the tails

With variance healthy, the tails were still tame — and we found the mechanism:
during training, models repeatedly *pass through* the correct heavy-tailed
state and then converge away from it, exactly as the variance had. Higher
statistical moments starve in sequence — a "moment ladder": each moment order
receives a weaker training signal, and finite data lets memorization eat it.
Again the cure was causal: with $8\times$ the toy training data the tail decay
is *absent*, not delayed; and a decomposition experiment showed a data-starved
flow actively *erases* the tails its own base distribution supplies, while a
data-rich one preserves them. The practical cure at fixed data: give the flow a
**heavy-tailed (Student-t) base**, so extremes are the default rather than
something to be manufactured against the gradient, and select the training
checkpoint by a pre-registered validation rule that harvests the model inside
its healthy window. With both, the model passed *all* pre-registered
calibration bars — dispersion and tails, conditional and end-to-end, replicated
on fresh fields.

### 5.5 The blind test of the founding question

The decisive experiment: hide the finest octave; extrapolate into it under the
deployment protocol (the scale-trend itself extrapolated, no measured
information from the hidden scale; model selection blind to it); compare.
**Result: pass** — dispersion within $\sim5$–$8\%$, tail weight within
$3$–$4\%$ of the hidden truth. And with a twist that revised the project's own
founding hypothesis: the explicit "scale dial" we had built (conditioning the
network on the measured drift coordinates) added *nothing* at one octave — a
scale-blind model extrapolates equally well, because the local coarse structure
itself carries the scale information (the locality measurement, confirmed at
deployment level). The drift is real; telling the network about it is
redundant. Meanwhile the field's most constraining practitioner statistic — the
starlet $\ell_1$-norm — passes on the extrapolated maps within $4\%$.

### 5.6 What still fails, precisely

Three honest boundaries. (i) *Placement*: the total number and amplitude
distribution of extreme peaks is now nearly right, but their spatial
arrangement is still biased at the $\sim15\%$ level — the third audit tier
remains open, and notably the starlet $\ell_1$, being position-blind, passes
while it fails: even the field's most constraining summary cannot see where
extremes land. (ii) *One octave*: extrapolating a second octave degrades
sharply (tail errors of tens of percent) — the validated domain is one octave,
with the cost of the second now measured. (iii) *Scope*: 2D gravity-only
fields, one simulation family, one training-set size; every claim carries these
words.

---

## 6. What exists at the end

1. **A generator** that, on held-out real simulation fields, produces
   next-octave detail with calibrated conditional dispersion, calibrated
   marginal tails and extreme quantiles, and a passing starlet-$\ell_1$ — one
   octave beyond its training data, under a deployment-blind protocol.
2. **An audit protocol** — instruments calibrated on exact truth, three tiers
   of statistics, held-out-basis checks, downstream propagation — that found
   three distinct failure mechanisms invisible to standard validation, in our
   generator and (by direct literature check) unexamined in published ones.
3. **Mechanisms**, not just fixes: the moment ladder with causal data-size
   evidence at two rungs; the mixture-artifact forensic; the locality result;
   the parents-not-tiles accounting. Each constrains how anyone should train
   and validate such models.
4. **A fully pre-registered record**: every prediction (both human-adjacent and
   model) scored, including a measured streak of under-confidence after the
   failure era — the collaboration's own calibration as data.

---

## 7. What this is actually useful for

Honestly tiered. Nothing here has *guarantees* — but neither do simulations;
science runs on characterized error within declared domains, and that is
precisely what the audit supplies.

- **Covariance mocks (usable now).** Higher-order-statistics analyses need
  thousands of mocks; mocks need *known* fidelity, not perfection, and
  covariances are second-order forgiving. Fine-scale completion of cheap coarse
  mocks, shipped with the audit numbers as an error budget, is better than the
  lognormal mocks in standard use.
- **Pipeline stress-testing.** Realistic small-scale texture for testing
  mass-mapping, deblending, and peak-detection pipelines.
- **Learned priors in inverse problems.** As a fine-scale prior inside
  posterior-sampling reconstructions, where the data likelihood corrects
  residual imperfection.
- **The audit itself, exported.** The emerging emulator/super-resolution
  literature validates at the power-spectrum level; every generator our
  three-tier audit touched failed somewhere above that level. "Here is how your
  emulator is lying to you, and the cheap instrument that catches it" is useful
  to the field immediately.
- **Not yet:** direct insertion into precision inference chains, or multi-octave
  extrapolation. The audit says no, and listening to it is the product.

---

## 8. The particle-mesh idea: super-resolving fast simulations for inference

The proposal: particle-mesh (PM) simulations are fast but inaccurate at small
scales; N-body/hydrodynamical simulations are accurate but slow. Could this
machinery upgrade PM outputs — *probabilistically, with correct statistics* —
such that cosmological inference on the upgraded maps is unbiased?

**This is not a tangent; it is arguably the natural destination of the project.**
But it differs from what we validated in one structural way, and honesty about
it defines the research program:

**What changes.** Our validated task is *self*-super-resolution: the coarse
input is a coarse-grained version of the same physics. A PM field is not that —
its small scales are not merely absent but *wrong* (suppressed power near the
mesh scale, incorrect collapsed structures), and even its intermediate scales
carry distortions. The task becomes conditional **domain translation plus
resolution extension**: learn $p(\text{fine truth} \mid \text{PM field})$,
which must both correct and extend. The good news: this fits the identical
mathematical scaffold — condition the ladder on the PM field instead of the
self-coarse field — and the pairing problem has a beautiful solution native to
cosmology: **run both simulators from identical initial conditions.** Matched
PM/N-body pairs by construction — precisely the "multiple realizations per
condition" design this project already identified as the gold-standard data
strategy, and the parents-not-tiles accounting tells us the budget currency:
independent initial-condition realizations, not map area.

**What our results say transfers.** Locality (small-scale correction should
depend on the local PM environment — the same theorem-plus-measurement that
made scale-blind extrapolation work); the moment ladder (tails will starve
first and need the data/base/selection cures — PM translation will hit the
identical diseases); the audit (transfers unchanged, and is the missing
ingredient in the existing literature: PM-correction and super-resolution
emulators exist — GAN- and diffusion-based, including recent work with partial
conditional validation — but none ships conditional-calibration audits, and
none tests inference-level bias).

**The two honest gaps between "correct statistics" and "correct inference."**
(1) *Cosmology dependence*: the conditional law depends on cosmological
parameters $\theta$. A generator trained at one cosmology is valid there;
inference explores a parameter range, so the model must be trained
$\theta$-conditioned (simulation suites spanning cosmologies exist, including
the one we already train on) and audited *across* the prior range — otherwise
its errors imprint directly as posterior bias. (2) *The joint tier*: our own
results show marginal calibration does not imply positional calibration, and
peak-based inference would inherit exactly the residual our audit still flags.
Therefore the acceptance test must be run at the *inference* level: infer
$\theta$ from super-resolved maps of held-out true simulations and require the
posterior to be unbiased and correctly calibrated (coverage tests) — our
downstream-bias demonstration, promoted from statistic level to posterior
level. That test is the definition of success for this application, and it is
buildable from components this project already has.

**Verdict.** Feasible as a research program, with the audit as its spine and
paired-initial-condition suites as its data design; the one-octave validated
domain and the placement residual are the current hard boundaries; the payoff —
PM-speed simulations with audited, inference-grade small scales — would matter
to essentially every field-level analysis in the survey era. It is the right
flagship application for the next phase of this work, where high-resolution
truth is genuinely unaffordable and the economics that motivated this project
actually bind.

---

## 9. Closing statement

We set out to make a generative model extrapolate resolution, and spent most of
the campaign discovering — with pre-registered, causal, mechanism-level
measurements — why such models fail statistically and how to repair them. The
result is a generator that passes a blind extrapolation test one octave beyond
its data, an audit that looks three tiers deeper than standard practice and
caught every disease including our own instrumentation artifact, and a set of
mechanisms (the moment ladder, locality, parents-as-currency) that outlive this
particular model. The honest one-sentence summary: **trustworthy beyond-data
generation is achievable one validated step at a time — provided the validation
machine is at least as sophisticated as the generator it audits.**

---

## 10. Questions a skeptic should ask (added 2026-07-24, from Andreas's review)

**Does weight-tying assume fractal structure?** No — exact self-similarity would
make the rung conditionals identical, and our founding measurement says they are
not. The assumption is weaker: the conditionals form a smooth, low-dimensional
family one network can represent (near-self-similarity of hierarchical
clustering, with measured departures). Adequacy was measured, not presumed — and
it must degrade where new physics enters; the claim never extends past the
measured regime.

**Isn't extrapolation speculation?** The economic foundation is a sample-size
asymmetry: training needs thousands of fields, validating summary statistics
needs tens. The deployment pattern is therefore: train on cheap simulations,
validate on a SMALL held-out set of expensive ones (far too few to train on),
and never claim beyond the octave that validation reached. The blind edge test
is the rehearsal of exactly this. One octave past all validation truth, the
support is indirect (scale-generic mechanism, self-consistency monitors, theory
anchors) — stated as such, always.

**"Weight-tied across scales"?** One network, one parameter set, every rung —
which is simultaneously statistical sharing, the near-scale-covariance bias, and
the entire license for extrapolation: applying the shared rule one octave beyond
training is defined; applying a per-octave model there is not.

**Are the two drift components arbitrary and permanent?** Neither: the dimension
fell out of a principal-component measurement (not imposed), the components have
physical identities (modulation slope, tail weight) — and the final model does
not consume them at all. The scale-blind result demoted the drift fit from model
input to monitoring instrument, where a descriptive parameterization is adequate.
It is a measured property of four octaves of gravity-only fields, not a law.

**Were the fixes principled or recipes?** Mixed, honestly: symmetry augmentation
is exact group theory (principled); removing the variance head deleted a hack;
the heavy-tailed base is principled in kind (measured structure into the model
class) with a convenience choice of tail index; checkpoint selection is a caged
recipe treating a symptom. Every fix sits on a measured mechanism with a causal
test — and the audit is what makes recipe-grade fixes safe (verify the result,
never trust the recipe). The principled agenda for the next phase: data design
(paired/constrained ensembles — the cure that attacks the cause), bases fitted
to measured marginals, and a real theory of the moment ladder.

**Is the placement residual another data limitation?** Plausibly — positional
structure is the next rung of the moment ladder, where the effective-data
requirement (parents, not tiles) bites hardest. The hypothesis is cheaply
falsifiable: the exact-truth sandbox supports the same causal data-size test
that confirmed rungs two and four, applied to placement statistics. If data →
constrained ensembles are the designated cure; if not → the
unconditional-prior/posterior-sampling reframe, where joint structure comes from
the prior's full law, is the principled candidate.

**If only one octave, why the ladder at all?** Generation is recursive across
all trained octaves (the cross-scale compounding results were multi-rung);
what is single-octave is the extrapolation. For one octave of extension a
non-laddered pair model is a legitimate competitor — but has no defined claim
beyond its trained resolutions. The weight-tied ladder is what converts "apply
the rule again" into a testable scientific statement; the multi-octave promise
is real, unrealized, and priced (the second octave's degradation is measured).

---

## Revision note (2026-07-24, takeover audit — compliance with the adjudicated record)

Two statements in §5 are tightened to match the binding rulings; the substance
is unchanged.

1. §5.5's pass summary ("dispersion within ~5–8%, tail weight within 3–4%")
   quotes the end-to-end numbers (dispersion 4.7%/7.9%, tail weight 3.2%/4.4%
   for the two arms). Per the Stage-D ruling, the following attaches wherever
   that result is stated: the conditional-level (head-conditional) tail check
   of the reference arm passed its 15% floor with a 0.1% margin (14.9%) — the
   end-to-end pass is unambiguous; the conditional-level margin is razor-thin.
2. §5.6's placement residual ("biased at the ~15% level") now carries error
   bars (results_p2/audit_peak_ci.json, computed from the committed Stage-D
   fields): peak-count excess +13.1%±3.1% and +14.6%±3.1% at the two scored
   thresholds, consistent in sign and magnitude across all three independent
   parent simulations of the test set.

## Revision note 2 (2026-07-24, reconvene ruling R27 — this entry governs the wordings below)

The takeover audit's objection 4.1 was accepted and extended: a peak COUNT is
a joint/morphological statistic (local-maximum density rides the correlation
structure) but does not measure WHERE peaks sit. Wherever this document says
the residual is in the extremes' "spatial arrangement" (§5.6), that the third
tier asks whether extremes "land where the environment demands them" (§2.3),
or that the starlet contrast shows a position-blind statistic missing a
positional failure (§5.6, §10), the adopted wording governs instead:

- What is measured: **joint/morphological structure — a peak-count excess at
  fixed marginal calibration** (+13.1%±3.1% / +14.6%±3.1% at the deployment
  edge, arm A; parent-robust).
- The starlet contrast: **even the field's most constraining marginal
  statistic cannot see the morphological excess** — the demonstrated tier gap
  is marginal-vs-morphological, not position-blind-vs-position-aware.
- Literal placement (environment-conditioned peak rates, peak clustering) is
  **untested** pending the placement experiment's instruments.

And one finding promoted by the same ruling (audit 2c, accepted 4.2): the
joint/morphological residual is **field-structure-dependent** — on the
exact-truth sandbox the same reference arm shows a significant peak *deficit*
(−9.4%±1.7% at the top threshold, replicated on 64 fresh fields) where the
real field shows an *excess*. Any cure demonstrated on the sandbox licenses
"joint structure is data-limited in this model class," never "the real-field
excess is thereby cured" — the real-field echo measurement is load-bearing.

---

## 11. Epilogue — the lattice saga and the schedule
*(appended 2026-07-25, integrated with the main text 2026-07-26; same conventions —
no internal codenames, every number from a pre-registered committed measurement)*

### 11.1 The question we asked, and how it refused to be answered

After the blind test of §5.5, one failure from §2.3's third tier remained open:
the generator over-produces extreme peaks on the real field even though every
tier-two (marginal) statistic is calibrated. We hypothesized this was the
moment ladder of §5.4 reaching its next rung — joint structure starving for
data as variance (§5.2) and tails (§5.4) had — and designed a causal
experiment in the style of §4: instruments validated on the exact-truth
sandbox first, pre-registered branches, bars from measured references. The
experiment's own discipline intervened twice before any training compute was
spent. First, its position-pure instruments had to demonstrate they could see
the known residual — and they could not: on the sandbox, the residual is not
positional at all. The experiment stopped at its own gate. Second, the gate's
confound instrument — built merely to separate grid artifacts from genuine
placement signal — found something real: generated fields prefer particular
pixel PARITIES for their peaks (odd-odd positions at ~30% versus the correct
25%, at up to $10\sigma$). A checkerboard bias invisible to the power
spectrum, to every marginal statistic of §2.3, to peak counts, and to the
starlet $\ell_1$ of §5.5 — a FOURTH tier of failure, discovered by the
instrument nobody expected to matter.

### 11.2 The mechanism hunt: a symmetry violated by the model that the data obeys

The fields of §2 are statistically invariant under the dihedral group
$G = D_4$ (the four rotations and four reflections of the square): for every
$g \in G$,

$$p(g \cdot x) = p(x), \qquad\text{hence}\qquad p(d \mid c) = p(g \cdot d \mid g \cdot c),$$

and the wavelet detail channels $(d^H, d^V, d^D)$ of §2.1 transform among
themselves under $g$ with definite signs (a reflection negates $d^H$ or $d^V$;
a 90° rotation exchanges them). Two immediate consequences for the TRUE
conditional law: every channel mean vanishes, and isotropy forbids
cross-channel correlation,

$$\mathbb{E}\!\left[d^{k}\right] = 0 \;\; (k = H,V,D), \qquad
\operatorname{Corr}\!\left(d^{H}, d^{V}\right) = 0 .$$

The generated coefficients violate both: channel means at $z$ up to $-17$ at
the finest octave, and $\operatorname{Corr}(d^H, d^V) = +7.5\sigma$ — and a
transplant test proved causality (swapping only the finest octave's
coefficients moves the output checkerboard). Note the subtlety this exposes
about §5.2's cure: the symmetry augmentation makes the TRAINING DISTRIBUTION
exactly $G$-invariant, but a network at a finite training step is under no
obligation to be equivariant — data symmetry is an attractor of training, not
a property of every checkpoint. Indeed the defect is an optimization
TRANSIENT: the channel-mean trajectory $\mu(t)$ rises, peaks mid-training, and
decays toward zero — and the checkpoint-selection rule of §2.2, tuned to
marginal statistics, had been choosing $\hat{t}$ at the defect's PEAK, at
every data size tested. (The defect also predates the final model: it was
always present in the mean pathway, and the retired variance head's noise bath
— the instrument artifact of §5.3 — had been concealing this too. Its second
concealment.)

### 11.3 The schedule: the moment ladder acquires a time axis

Write $W_m(N) = \{\, t : |S_m(\theta_t) - S_m^\star| \le \varepsilon_m \,\}$
for the "healthy window" of statistic $S_m$ — the training times $t$ at which
the model (trained on $N$ effective parents, §5.1) holds $S_m$ within its bar.
The moment ladder of §5.4 said which $S_m$ starve; the epilogue's finding is
WHEN: tails are healthy EARLY ($W_{\text{tails}}$ opens and closes before the
symmetry cleans), lattice symmetry is healthy LATE, and at the standard data
size the windows are DISJOINT,

$$W_{\text{tails}}(N_1) \,\cap\, W_{\text{sym}}(N_1) \;=\; \varnothing,$$

demonstrated mechanically: the best joint checkpoint fails the tail bar at
$-23.9\%$ against a $15\%$ bar. Worse, single-tier selection is actively
adversarial to the other tiers: $\hat{t} = \arg\min_t \mathcal{L}_{\text{marginal}}$
landed at $\arg\max_t \mu(t)$ for BOTH defect layers. Selection is part of the
model. Abundant data widens the windows — at $32\times$ the joint defect
reached its lowest test value of the campaign (3.48 vs the 3.0 bar):
overlap is PROXIMATE but not demonstrated (§11.5).

### 11.4 The cures, in ascending principledness — ending in a guarantee

1. **Subtract the offsets.** A per-channel constant $\hat{\mu}$, estimated on
   validation generations, removed at sampling: free, costs no marginal bar,
   halves the defect — and its residual isolated the correlation layer, which
   no constant can touch.
2. **Restore the symmetry exactly.** Sample through a random group element:

$$d \;=\; g^{-1}\!\cdot f_\theta\!\left(g \cdot c,\; z\right), \qquad g \sim \mathrm{Uniform}(D_4),$$

   i.e. the sampling law becomes the group average
   $p_{\text{sym}}(d \mid c) = \tfrac{1}{|G|}\sum_{g} g^{-1}\!\cdot p_\theta(\,\cdot \mid g\cdot c)$.
   This carries a small theorem worth stating: any statistic $S$ that
   transforms in a sign representation of $G$ — $S(g \cdot d) = \chi(g)\,S(d)$
   with $\chi$ non-trivial, which is exactly the case for the channel means,
   the cross-channel correlation, and the pixel-parity contrast — satisfies
   $\mathbb{E}_{p_{\text{sym}}}[S] = 0$ identically, while every $G$-INVARIANT
   statistic ($\chi \equiv 1$: all the calibration bars of §2.3) is exactly
   preserved. Annihilation of the defects and safety of the calibration are
   both guaranteed by symmetry, not hoped for. Zero training, same cost per
   sample; verified clean (one component sat at the detection bar and resolved
   as noise under replication). There is a pleasing bookend here: the SAME
   group $G$ cured rung two by symmetrizing the DATA (§5.2) and cures tier
   four by symmetrizing the SAMPLER — the project's founding doctrine (§3, put
   measured structure into the construction rather than asking training to
   find it) applied one final time.

### 11.5 The data question, answered honestly

Does data merge the windows? Measured carefully: data lowers the late defect
plateau, and at $32\times$ the joint defect reached 3.48 against the 3.0 bar —
the campaign's lowest, a window PROXIMATE but not demonstrated. Two safeguards
shaped this verdict. The built-in seed replication revealed the single-seed
baseline had been the WORST of five seeds (ensemble $11.1 \pm 3.4$, range
$[6.9, 15.3]$) — seed variance comparable to the data effect, retiring the
headline ratios (the paired, within-curve comparisons survive; cf. §10's
recipe-vs-principle audit). And the scoring code's first version was caught
implementing a looser rule than pre-stated — corrected BEFORE adjudication,
both versions preserved. One data result is decision-grade and feeds §8
directly: at matched total cost, INDEPENDENT initial conditions beat multiple
realizations per condition by a factor $1.4$ as the currency of joint
structure — the particle-mesh program of §8 should buy independent
simulations first.

### 11.6 What remains, and what the epilogue means

One residual survives everything: on the real field (not the sandbox), peak
SPACINGS carry a small, parent-consistent signal ($T \approx 4$) that outlived
the variance-head removal, the constant correction, and exact symmetrization —
the project's one genuinely unexplained statistic, the named frontier for §7's
next phase. And the epilogue's meta-lesson completes the answer to §1's second
question (how would you ever know?): the audit found a defect nobody
suspected, in coordinates nobody was watching, hidden by an instrument
artifact (§5.3), harvested by our own selection rule (§11.3) — and the cure
with the best properties was neither a bigger model nor more data but a
symmetry restored exactly (§11.4). Validation machinery at least as
sophisticated as the generator is not overhead; it is where the discoveries
come from.
