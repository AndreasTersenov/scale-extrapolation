# BRIEF — foundations & innovation sweep (2026-08-05)

Provenance: reconvene-run research workflow (Andreas-authorized), 18 agents:
5 lens-specific sweeps (guarantees / ML frontier / cosmology deployment /
distribution-free UQ / adjacent fields) -> 12 deduped architecture-compatible
ideas -> adversarial referee per idea (citation existence verified via search;
cooking/duplication screens) -> single synthesis. The filter killed nothing
outright but demoted two ideas to near-dead with specific reasons (one novelty
claim already published: arXiv:2510.08929). All ideas below passed the
exact-consistency compatibility screen. Status: INPUT to reconvene planning —
nothing here is authorized until it enters an order set.

Surviving ideas (referee-strengthened versions live in the workflow record):
- Provable end-to-end cascade error bound via conditional Wasserstein distances and Haar orthogonality
- Conditional strong log-concavity certificate per octave (Guth-Lempereur-Bruna-Mallat program)
- Hard physical constraints by construction: mirror/reflected flow on the convex feasible set {coarse fixed} ∩ {kappa >= -kappa_empty}
- Distribution-free conformal certification of downstream inference statistics
- Per-octave flow-matching convergence rates with measured constants (Benton-Deligiannidis-Doucet import)
- Conditioning-noise augmentation for the extrapolated octave (cascade exposure-bias treatment)
- Autoguidance at the finest octave: guide the flow with an early checkpoint of itself (inference-only)
- Phase-structured base measure: WPH/multiplicative-cascade base noise at the extrapolated octave
- VAR-style long-range context: give the tied conditional a global receptive field over the full coarse pyramid
- Conformal wrapper: distribution-free validity certificates for downstream statistics of generated maps
- Paired cheap-simulator conditioning corpus: retrain the cascade on FastPM/COLA coarse octaves with shared-seed N-body truth
- Posterior-level deployment gate: pre-registered SBI coverage-and-bias test on generated maps

---

# Research brief: from validated cascade to foundational method

## 1. Ranking

Scored on (impact on the PM-to-N-body field-level ambition) x (principledness / guarantee value) / cost. First, one piece of housekeeping: the two conformal proposals (guarantees lens and ml-frontier lens) are the same idea filed twice — the rank/PIT tier of one and the oracle-floor band tier of the other compose into a single program. I merge them below and rank once.

1. **Paired cheap-simulator corpus (COLA/FastPM matched seeds).** The only idea that touches the actual ambition rather than the current surrogate of it. Everything validated so far is super-resolution of downsampled truth; deployment conditions on a simulator whose coarse scales are themselves wrong. The Stage 0 instruments (per-octave Wiener transfer, cross-correlation r_j(k) defining the trusted/generated boundary) are distribution-free, CPU-only, require no retraining, and extend the exact epistemic move — measure the deployed system's error, correct at inference — that already has a 5/5 record. Highest impact, principled core, moderate cost.
2. **Conformal certification (merged).** Converts the blind-shot protocol from "passed k judges once" into finite-sample, distribution-free, per-(statistic, octave) certificates, without burning a frozen judge per question. The band gap against the oracle arm is a certified cost of extrapolation — the method's signature claim turned into a number. Near-zero risk given your background; low cost; the guarantee is exactly the kind the taste rules ask for.
3. **Tier-1 W2 decomposition theorem.** Two lines of proof (diagonal coupling on trusted scales + Haar isometry), no constants, and a statement structurally unavailable to any method whose synthesis operator is not an isometry. It is the formal identity behind "the finest octave is the entire error budget." The honest negative lemma (peaks and Euler characteristic are not L2-Lipschitz, so W2 cannot certify audit outcomes) is itself valuable: it explains why conformal is the complementary certificate, not a redundancy.
4. **Posterior-level deployment gate.** Cheap, pre-registerable now, and it prices the two surviving defects in the only currency a survey cares about: sigma of parameter bias, reported as the survey area A_1sigma at which bias reaches one posterior sigma. The three-arm design (raw / transfer-repaired / oracle) prices both the repair and the extrapolation in deployment units in a single blind shot.
5. **CSLC certificate (GLBM program).** The deepest diagnostic on the list: a theory-grounded, per-octave measurement of extrapolation cost (kappa_j^model vs kappa_j^data), with GLBM having demonstrated the data-side property on convergence maps already. It also supplies the principled justification for weight tying (WCRG locality) and for the pseudo-octave arm. Medium cost, and more diagnostic than certificate after the referee's honest trimming — hence below the pure-guarantee items.
6. **Conditioning-noise augmentation on top of arm (b).** The best-replicated robustness intervention in the cascade literature, applied at exactly the train/test gap arm (b) creates. Low cost, no guarantee, high mechanistic plausibility. The referee's prediction that the standalone version does nothing is a built-in falsification and I agree with it.
7. **Physical-constraint tiers 0-2 (kappa floor, mass conservation).** Tier 0 and the violation-rate instrument are essentially free ledger items; Tier 2 rejection sampling gives an exact support guarantee with a one-line proof if violations are rare. Ranked here rather than higher because the guarantee, while exact, is about physical validity rather than the defects currently blocking the ambition. Tier 3 (mirror retraining) is a horizon item, gated.
8. **Autoguidance at the finest octave.** Very cheap, inference-only, and the referee's version converts Karras's untestable assumption into a measured go/no-go gate using the oracle arm. Its payoff is speculative but the gate costs almost nothing and the composition with T(k) re-measurement is machinery only this method has.
9. **WPH phase-structured base.** The most direct attack on the diagnosed root cause (phase coherence supply), with an explicit max-entropy characterization. But it is properly gated on probe (d), and Stage 1 success is genuinely uncertain. Stage 0 (scale-regularity of WPH covariances across trained octaves) is a cheap kill-test and worth running regardless — it is a T(k)-style instrument for phase.
10. **Per-octave FM convergence rates.** Marginal. The novelty claim is dead (arXiv:2510.08929 already covers Student-t bases), the composed bound will likely be vacuous through its Gronwall factor, and the genuinely useful number — held-out finest-octave FM loss minus the oracle's — free-rides on tonight's arms and needs none of the theorem apparatus. Keep as a time-boxed appendix at most.
11. **VAR-style global context.** Most marginal. MVAR undercuts the framing (full-pyramid attention is largely redundant), the intervention is expensive, and it is doubly contingent (defect scale must exceed the receptive field AND the tidal-frame arm must fail). The free Step 1 — measure the effective receptive field against the defect's correlation length — is worth doing as pure diagnosis; everything past it stays shelved.

## 2. Now / next paper / horizon

**Now (weeks; interleaves with tonight's arms).** The Tier-1 W2 theorem plus the mass-conservation lemma and the negative lemma — a few days of writing, done once, cited forever. The merged conformal program, calibrated the moment the oracle arm finishes, since the oracle is its floor. Constraint Tier 0-2 (kappa_empty measurement, violation instrument, rejection wrapper if warranted, with the T(k)-before-feasibility ordering rule stated). Conditioning augmentation retrained on top of arm (b) with its three pre-registered predictions. The autoguidance alignment gate and the receptive-field measurement, both free diagnostics on the oracle arm and existing checkpoints. Pre-registration (not yet execution) of posterior-gate Stage 1.

**Next paper.** Paired-corpus Stage 0 and the Stage 1 retrain on Wiener-corrected PM octaves, carrying whichever texture fix survives the current campaign — this is the paper that claims the ambition rather than the surrogate. Posterior-gate Stage 1 executed as part of it, Stage 2 declared as intent. The CSLC certificate program as the theory section: data-side replication of GLBM on your family, model-side gap as the measured extrapolation cost, with the sub-Gaussian-tails adjudication of the Student-t base as a free falsifiable claim. WPH Stages 0-1 if probe (d) leaves the question open. The FM-rates appendix only if its numbers turn out non-vacuous.

**Horizon.** Mirror-coordinate retraining (Tier 3), only if the violation instrument shows frequent, void-localized violations — and note it presupposes arm (b), the only mechanism exposing the tied network to near-boundary regimes. The global-context branch, only if both of its gates trip. WPH Stages 2-3. And the weaknesses no surviving idea addresses: cosmology-parameter conditioning (posterior-gate Stage 2 forces this), a second simulation family, and 3D. Those are the real horizon.

## 3. The two that most strengthen foundations

**The Tier-1 W2 theorem.** First step is proof, not experiment: write the two-line argument (diagonal coupling on the exact conditioning, Haar orthogonality transporting squared W2 additively, trusted octaves contributing zero), then immediately the negative lemma showing peak counts and Euler characteristic are not L2-Lipschitz. The pair of statements — what the isometry certifies and what it provably cannot — is the honest foundation. The first numerical step is the per-octave budget figure via sliced-W2 on (condition, detail) pairs at the held-out octave, with the Chemseddine et al. metric relation stated explicitly and no naive conditional-W2 claims.

**The conformal program.** First experiment: for n ≈ 50 fresh-seed calibration coarse fields (strictly disjoint from training, checkpoint selection, and the frozen PH blind set), draw m generated completions each, and compute the rank of the true statistic within the generated ensemble for peak counts per bin and starlet-l1 per scale. Under a correct conditional law the ranks are exactly uniform — a finite-sample, assumption-free test of precisely what SBI consumers need. Run it identically on the oracle arm the morning it finishes; the band gap is the first certified cost-of-extrapolation number the project produces. The weekend proposition — zero-width certificates for any statistic of trusted octaves, because those octaves are exact by construction — is the sentence that separates this method from every monolithic generator.

These two compose deliberately: the theorem certifies where the error budget lives in field space; conformal certifies its downstream consequences for exactly the statistics the theorem proves it cannot reach.

## 4. Connections, including to tonight's arms

The oracle arm (a) is the quiet load-bearing element of half this brief: it is the conformal floor, the autoguidance alignment reference, the FM-loss gap baseline, and the third arm of the posterior gate. Whatever else tonight yields, the oracle earns its compute four times over.

The pseudo-octave arm (b) is the substrate for three proposals: conditioning augmentation applies on top of it, mirror Tier 3 presupposes it, and the CSLC certificate predicts and explains its success (it trains the tied network inside the conditioning regime it must extrapolate into). If (b) helps, the augmented-vs-unaugmented pair also becomes the primary good/bad pairing for autoguidance — a degradation axis aligned exactly with the defect.

The tidal-frame arm (c) is the hand-crafted special case of the global-context pathway and composes with WPH Stage 2's coarse-aligned conditional base; the receptive-field measurement adjudicates between "architecture problem" and "training-distribution problem" for all three at once. Probe (d) gates WPH: cascade fixes topology, stop; neither helps, coherence-supply is falsified and both die; WPH helps where cascade does not, phase constraints are confirmed as the missing ingredient.

One scheduling constraint runs through everything: judge economy. The frozen persistent-homology judge is a single bullet. The composed repair arm (b + augmentation, possibly + autoguidance at a pre-gated w) should be assembled first and shot once; conformal exists precisely so that most future questions never need a judge.

## 5. What would make this foundational

An emulator produces maps that pass the tests someone thought to run; a foundational method states, with proof or finite-sample guarantee, which claims about its output are certified, which are measured, and which are open — and its architecture is what makes those statements possible. Here the Haar isometry pins the error budget entirely to extrapolated octaves by a two-line theorem, conformal calibration converts that budget into distribution-free certificates per statistic and scale, and the oracle gap prices extrapolation itself. The endgame is a generator shipped not with validation plots but with a certificate ledger — exact coarse scales, exact symmetry, exact support, certified statistic bands, and a measured price of every scale it invented.
