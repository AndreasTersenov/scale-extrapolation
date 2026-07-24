# 2026-07-24 — TAKEOVER AUDIT (Phase 0 of TAKEOVER-BRIEF.md)

Written by the incoming writing-and-frontier session after the full reading list
(explainer incl. §10; RESHAPE + revision; rulings R8–R26; Gate-0 kill-test
bottom lines; both specs) and an independent verification pass over the raw
artifacts. Findings are sorted by the brief's authority split. Numbers below are
verbatim from committed artifacts; the new ones are from
`results_p2/audit_peak_ci.json` (script `scripts_p2/audit_peak_ci.py`,
validation test `tests_p2/test_audit_peak_ci.py`, committed with this memo).

## 1. Verification report (R12 pass)

Re-verified from raw JSONs/npz, no discrepancies: stageD_verdict.json (all 8
edge checks, A-hc 14.9% vs 57.6% formal bar, peaks +13.1/+14.6% A and
−9.4/−17.3% B, branch D-PASS-BOTH, dial_beats_scaleblind false);
c1t_repl64.json (24/24; worst entry B-hc-oct4 41.0% vs 56.0% bar; A worst
12.4%; B-e2e-oct2 13.3% vs 15% floor); c1t_verdict_sandbox.json (A hc-oct2
kurt 5.368 vs truth 4.917 = +9.2%); c1t_gow_descriptive.json (deficit halving
−33.7→−4.0% / −26.1→−8.0% at oct 2; B oct-4 worsening −32.7→−41.5%);
starlet_l1_edge.json (totals +5.4/+4.0/+3.0/+1.8%; 4px tail share 0.1680 vs
0.1587±0.0045); forensic_nllnoise.json (μ-only var_slope 1.536 vs real 1.020,
kurt 32.5, detail_std ratio 0.383 = the pre-declared amplitude confound);
taildyn (1× finals 1.07–1.66 collapsed, 8× finals at truth under the last-3-eval
convention). Both test stacks green (tests/ 14 pass; tests_wfm+tests_p2 all pass,
2 skips). The record's numbers are what the record says they are.

## 2. New measurements (descriptive, committed artifacts only — no new runs)

**2a. The peak-audit residual now has error bars — and survives.** The ~15%
joint-structure residual was quoted everywhere without error bars, in a repo
whose hard rule is that such a claim doesn't count, and it motivates the entire
placement experiment. From audit_peak_ci.json (field-bootstrap, n_boot=5000):
Stage-D edge, arm A: +24.7%±3.9% (ν=1), +13.1%±3.1% (z=+4.2, ν=2.5),
+14.6%±3.1% (z=+4.7, ν=3). Arm B: +37.5%±4.4%, −9.4%±2.6%, −17.3%±2.2%. The
residual and the dial's sign-flip are both real, not noise.

**2b. But the audit's own instruments inherit N_eff ≈ parents.** The
truth-vs-truth split-half null on the real edge fields fires at ν=1
(+19.3%±6.6%, z=2.9): the 32-tile test reference is internally heterogeneous
because it comes from THREE parent simulations (verified: tiles_pnull.npz is
parent-ordered, 11 tiles/parent, edge-continuity test; stage-D real =
tiles[298:330] exactly, corr 1.0). Field-bootstrap SEs therefore understate
reference uncertainty — the campaign's own accounting result, applied to its
own error bars. The rescue: generation is paired to the same parents, and the
per-parent excesses are sign- and magnitude-consistent where it matters —
arm A ν=2.5: [+10%, +9%, +20%]; ν=3: [+21%, +15%, +9%]; arm B ν=3:
[−13%, −17%, −21%]. The adjudication-relevant thresholds are parent-robust;
the ν=1 row is parent-dominated (one parent contributes +54%/+69%) and should
be de-emphasized in any figure.

**2c. The sandbox shows the tier-3 residual too — with the OPPOSITE sign in
the reference arm.** C1-t sandbox arms vs held-out truth tiles: arm A
−4.9%±2.1% (ν=2.5), −9.1%±2.1% (z=−4.4, ν=3); arm B −11.9/−13.1%. Replicated
on the 64 fresh fields (seed 20260720): A −6.8%±1.7% (z=−4.0), −9.4%±1.7%
(z=−5.6); B −13.5/−14.2%; split-half nulls clean (|z|≤0.6). So: the placement
experiment's arena DOES exhibit a significant joint-structure failure (it can
discriminate H-data from H-arch), but the real field shows an EXCESS of peaks
where the sandbox shows a DEFICIT (arm A). Consequences in §4.2.

**2d. Split provenance, verified and gap noted.** Gowerstreet train/test are
parent-disjoint (train = parents 0–24, test = parents 27–29) — the referee's
cheapest leakage attack fails, and the repro appendix can now say so with an
executable check. Two margins to disclose: the selection (val) set shares
parent 24 with train (9 of 32 tiles), a mild optimism channel for checkpoint
picks (adjudication is on test — verdicts unaffected); val and test share
parent 27 (1 val tile / 10 test tiles). Gap: the script that built
tiles_pnull.npz is not in the repo — the committed-recipe discipline
(make_c1_data.py style) should be backfilled.

**2e. Evidence-scale correction.** Gowerstreet training is 266 tiles from 30
parents (run log 16666634: "266 train / 32 val / 32 test"); the "322 tiles"
in RESHAPE's scope words is the SANDBOX training count. R25's binding scope
list already omits the number; the paper must not inherit 322 for the real
field.

## 3. Within-authority findings — adopted for WP1/WP2, documented here

**3.1 Skeleton structure (the spec invites this).** The spec's §3 (moment
ladder) and §5 (audit-guided arc) tell the same experiments twice at different
altitudes — the bake-off trajectories, the data-size causal tests, and C1-t
each appear in both. I will restructure to the explainer's proven arc (it
survived Andreas's skeptic review in exactly this shape): measurements →
audit protocol and the three tiers (instruments before adjudications, so each
later claim lands on a calibrated instrument) → one "diseases and cures"
section (collapse; the instrumentation artifact; the tail ladder — with the
moment ladder as the connecting mechanism and its name in the section title,
keeping the mechanism prominent for ML readers) → Stage D → discussion. This
is the audit-first spine with the rescue arc told inside the mechanism
section; spec §§1,2,6,7,8 are otherwise kept as drafted.

**3.2 The claim-2 flagship is currently homeless in the skeleton.** Gate-0's
strongest surviving novelty — structured-not-additive conditioning drift —
appears in no section of the spec skeleton. It will be placed in the
disease-II subsection (its substrate era) with the 07-17 scope clause verbatim.

**3.3 Peak-audit figures gain CIs and the per-parent panel** (2a/2b) — added
to the [HARDEN-FIG] ledger. The starlet/nanmedian/SC-bands chores stand.

**3.4 Reproducibility appendix additions:** the split protocol + parent
accounting (2d, 2e); the tiles_pnull builder backfill; the reference-noise
lesson (2b) as bar-ledger #9's second exhibit; the audit-of-validators
paragraph as already planned.

**3.5 Explainer compliance revision** — appended, dated, committed with this
memo: §5.5 now carries the R21-mandated A-hc 0.1%-margin statement (it stated
the Stage-D result without it); §5.6's residual now carries error bars.

**3.6 WP2 design improvements (adopted into the prereg draft; the prereg
itself still STOPs for reconvene review before submission):**
1. *Sequencing:* Phase A = instrument validation on exact-truth ensembles PLUS
   descriptive scoring of the committed 1×-regime artifacts (partly done — 2c
   is its first row), so every bar in the prereg comes from measured
   references; Phase B = the causal runs. Bars from references with floors,
   final-state statistics (bar-ledger #8), reference-side noise budgeted (#9).
2. *Position-pure instruments:* count-matched / rate-normalized variants of
   each placement statistic, so the placement signal is orthogonal to residual
   marginal miscalibration; the environment-conditioned peak rate is the
   discriminator between "extremes in the wrong PLACES" and "wrong peak
   morphology/fragmentation"; the seam statistic separates octave-boundary
   artifacts (kept from the draft).
3. *A third branch with a mechanism:* PL-SELECTION — placement error at the
   caged checkpoint materially worse than the checkpoint-curve minimum at the
   same data size. The draft lists this as a confound check; it is a
   first-class outcome (the cure would be a joint selection criterion, not
   more data and not a new model class).
4. *Curve, not line:* sandbox parents are free and training is MIG-minutes —
   add a 32× or 64× point so a slow logarithmic shrink cannot masquerade as
   PL-FLAT. Two points cannot distinguish "flat" from "slow"; the tail-dynamics
   lesson (onset rules vs final states) generalizes.
5. *Seed-robustness rider at zero cost:* the 1× arm is itself a fresh-seed
   replication of C1-t at sandbox scale; score its marginal bars descriptively
   to answer the single-training-seed referee question.
6. *Optional currency arm (reconvene decides scope):* 8× parents × 1 detail
   vs 1× parents × 8 exact constrained realizations per coarse (free on the
   sandbox) at matched total fields — discriminates WHICH data axis is the
   operative currency for the joint rung, and directly sets the paired-IC
   data design of the PM program (explainer §8).

**3.7 Venue view (recommendation only; Andreas decides).** Astro-methods
first (A&A/OJA): the evidence scale (one 2D field family, 30 parents, small
UNet) is under-powered for an ML-venue empirical bar; Gate-0 narrowed the ML
novelty to citation-heavy measured instances plus one flagship; the audiences
that can immediately use the audit protocol and the taxonomy panel are
emulator builders in astro. The both-ways §1 draft the spec requests will
still be written.

**3.8 Frontier alternatives considered and not preferred.** Jumping straight
to the posterior-sampling reframe (skips the cheap discriminating measurement
that would justify it); a two-octave rescue arm (no mechanism hypothesis yet —
premature); a PM-translation pilot (needs paired data and its own phase).
Placement-first stands: cheapest, mechanism-guaranteed under every branch, and
it properly gates the posterior-sampling pivot.

## 4. Reconvene-gated flags (objections and new evidence; nothing changed unilaterally)

**4.1 "Spatial placement" overstates what peak counts measure** (touches
R19/R22 adopted texture). A peak COUNT is sensitive to the joint law (local-
maximum density rides the correlation structure) but blind to WHERE peaks sit;
"their spatial placement is not [calibrated]" asserts more than the measured
statistic can. Proposed paper wording: "joint/morphological structure
(peak-count excess at fixed marginal calibration)" for what is measured, with
literal mis-placement (environment-conditioned rates, peak clustering) stated
as untested until the placement experiment's instruments run. Load-bearing:
WP2's hypotheses and instruments are phrased around exactly this distinction;
if the reconvene prefers the R19 wording, WP2's instrument set should still
include the discriminator (3.6.2) so the wording becomes measured either way.

**4.2 The sandbox does not rehearse the real-field residual's sign** (new
evidence, 2c — bears on the draft spec's framing before prereg). The draft
treats the sandbox as the arena for "the ~15% placement failure"; on the
sandbox the reference arm's failure is a peak DEFICIT, not the real field's
EXCESS. The experiment remains well-posed (H-data vs H-arch on the sandbox's
own significant residual), but the prereg must carry a transfer clause: a
PL-DATA verdict licenses "joint structure is data-limited in this model
class," NOT "more parents cure the gowerstreet up-tilt" — the real-field echo
leg is therefore load-bearing, not decorative, and the branch meanings should
say so. I request the reconvene's read before I draft the prereg around this.

**4.3 Peak-reference disclosure for the paper** (touches adjudicated Stage-D
texture): the edge peak audit's real reference is 3 parents; the ν=2.5/3
signals are parent-robust (2b) but the disclosure sentence should accompany
the numbers wherever they appear, same discipline as the A-hc margin; the ν=1
row should not be figure-headlined.

**4.4 RESHAPE scope-words correction for the record** (2e): "322 tiles" is
sandbox-only; gowerstreet is 266 train tiles / 30 parents / one seed. R25's
list already omits the number; flagged so the memo's successors don't quote it.

## 5. Checked and NOT objected to (the anti-rubber-stamp ledger)

Gate-0 bottom-line wordings (tight, correctly narrowed, correctly citation-
mandated); the bare-floor adjudication of Stage D (stricter than the formal
bars, the honest direction); the arm-A-reference ruling and its starlet
confirmation; the one-octave scope sentences; the tier ordering incl. the 4px
tail-share nuance; the placement experiment's core causal design (the 1×/8×
harness is the right instrument — my changes are refinements, not redesign);
the calibration-scorecard-as-finding framing. I re-derived none of these to
disagree performatively; each was checked against its artifact.

## 6. Failure-mode self-check (per the brief)

Rubber-stamping risk: my §1 pass found zero numerical discrepancies, which
could mean shallow checking — the mitigation is that the pass surfaced
process-level findings instead (2b, 2d, 2e), including one (2b) that cuts
against a number I myself had just strengthened. Performative-disagreement
risk: 4.1 is the closest call; it stays because WP2's instrument design
changes depending on the answer, i.e., it is load-bearing, not rhetorical.

## 7. STOP

Per the brief's stop rule: reconvene-gated flags (§4) are raised, so I stop at
the end of Phase 0. State on stop: WP1 structure decided up to the 4.1 ruling
(restructure per 3.1 is within authority and will proceed on resume); WP2
prereg improvements enumerated (3.6) and awaiting the 4.2 read; all §2
artifacts committed; both test stacks green.
