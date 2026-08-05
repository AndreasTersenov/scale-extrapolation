# 2026-08-05 — RECONVENE R47: Phase-3b diagnosis adjudicated — the
# fragmentation is OCT-1 EXTRAPOLATED TEXTURE; lever menu chosen: band-limit
# check + the oracle-at-oct-1 arm + the phase-texture instrument

Ruled on: the READOUT appended to log/2026-08-05-prereg-topo-diagnosis.md
(d09544f) + topo_diag_* artifacts. Same-day git log; queue empty.

## Verification

R12 spot-checks ALL MATCH (T_hc 47.27/67.20, stage-D hc context 27.31,
e2e re-emit 6.3513 identical to the committed E3; T2 singles 11.30/10.19/
8.50/6.65 with shift fractions +1.150/−0.91/−0.40/−0.25; cumulative-≤2
16.32/0.49; stream fractions 1.150/1.133/1.26; T3 detail-skew z ≤ 1.3
everywhere). Suites re-run: tests/ 48 pass (incl. the 4 new skew-scorer
tests); tests_wfm+tests_p2 pass (exit 0). Determinism gate exact on the
one MIG job (0.02 H100-h).

## Adjudication

1. **T2 is ACCEPTED as the diagnosis:** the mid-level fragmentation is an
   octave-1 generated-detail-texture property — substituting only oct-1
   generated details into fully-real pyramids reproduces MORE than the
   full χ(0) shift, stream-stable; misalignment pushes the opposite sign
   (measured, with the cumulative-≤2 cancellation as its exhibit). The
   topology defect and the native peak excess now share a located locus
   (oct-1 texture) on the axis the spectrum provably does not reach.
2. **T1 is VOIDED as posed, with an instrument lesson (ledger #15):**
   hc-composed MAPS are confounded by cross-scale misalignment (skew
   collapse z = −8.5 is the smoking gun); per-octave hc STATISTICS remain
   valid, hc maps do not follow from them. The deployment-split question
   is additionally moot: e2e IS octave-4-conditioned deployment mode —
   there is no more-conditioned mode to rescue topology; the defect is
   intrinsic to extrapolated fine-octave generation. Closed.
3. **T3 REFUTED; scorecard, both columns:** my 55 on detail under-skew
   LOSES to the executor's registered parity argument (20) — signed
   detail skew vanishes in law for both stacks; the hypothesis was
   ill-posed and the counter-argument was available at design time
   (exchangeability ⇒ symmetric differences). Their map-level 55 also
   loses (z = −2.4, sub-bar). My single-octave line 40 FIRED above both
   columns — **calibration note #16: single-located-cause outcomes have
   now fired above both columns' priors repeatedly (collapse, parity,
   coloring, this); weight concentrated-mechanism branches higher.**

## The structural reading (adopted)

Octave 1 is extrapolated in EVERY audited product (train-octave sets
never include it). Both surviving defects — native-resolution count
excess and mid-level fragmentation — are properties of the TEXTURE that
weight-tied extrapolation produces at the first untrained octave on the
real field. The spectrum of that texture is calibrated (5/5 tool); its
phase organization is not. The honest question "can the topology issue be
solved" therefore splits: (i) for products where target-octave training
data EXISTS, solving it means training through oct-1 — measurable now;
(ii) for true beyond-resolution deployment, no honest data-side fix
exists; the lever is texture-transfer correction (spectral-deconvolution
analog) — instrument first.

## Orders (this order set ≤ 1 H100-h)

1. **BL — band-limit check (CPU, committed maps, runs first):** reconstruct
   trained-leg + blind maps WITHOUT oct-1 details; score the frozen MF
   judge (identically band-limited real reference; declared-resolution
   smoothing convention preserved). Registered line: P(both legs pass at
   the frozen bar) = 80. If PASS: topology joins the declared-domain
   family with a crisp band statement AND confirms no mid-octave residual
   hides behind oct-1's dominance.
2. **O — the ORACLE-AT-OCT-1 arm (the direct solve where data exists):**
   ONE training of the production recipe with train-octaves {1,2,3,4}
   (everything else verbatim; caged selection unchanged; canary bars
   standing; ~0.5 H100-h), final-config sampling (F2; deconvolution
   recalibrated to the new checkpoint per the standard recipe), scored on:
   MF declared + native, native peaks, and the full must-not-regress
   battery. Labeled ORACLE (target-octave data is unavailable in true
   beyond-resolution deployment — the arm quantifies the COST OF
   EXTRAPOLATION and becomes the recommended config when fine-scale data
   exists). Branches (rec): **O-FIXES 45** (MF declared ≤ 3.5 AND native
   ν2.5/3.0 excess < half the committed values, no regression) /
   **O-PARTIAL 25** (exactly one of the two targets) / **O-NULL 20**
   (neither — an architecture/capacity finding, NOT a failure of the
   diagnosis) / O-REGRESSED 5 / gates 5. Executor column at prereg.
3. **I — the oct-1 phase-texture instrument (tests-first, CPU):** the
   F20-style component/hole decomposition SCOPED to oct-1 detail
   contributions + one cross-scale alignment statistic (executor's design
   periphery; validated on synthetics before real data). Purpose:
   mechanism section, the O-arm's texture verification, and the target
   for any future texture-transfer corrector.
4. **Drafted, NOT run:** the texture-transfer extrapolation corrector
   (measure the instrument's statistic octave-by-octave in the trained
   range, model its transfer, extrapolate one octave, correct — the
   spectral-calibration playbook on the phase axis). Post-instrument,
   post-O; its prereg would need its own reconvene review.
5. Standing refusals carried: χ-matching post-hoc; topology losses before
   the instrument exists.

STOP at the readout. Paper remains paused (Andreas's direction); the
eventual claim structure now in sight: full calibration incl. topology at
data-available scales (if O lands), extrapolation with a quantified,
located texture cost at the beyond-data octave.
