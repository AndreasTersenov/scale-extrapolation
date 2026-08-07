# 2026-08-07 — RECONVENE R49: D1 corrected-selection re-ship adjudicated —
# the truth bug explained ~40% of topology but NOT the peak claim; R43
# finding-3 SPLIT and partially RETRACTED; the declared-resolution peak
# claim is config-specific; D2/conformal preregs reviewed; a scope
# decision goes to Andreas

Ruled on: the D1 READOUT (log/2026-08-06-prereg-d1.md, 9be358b) + the two
DRAFT preregs (7233b9f) + d1_* artifacts. Same-day git log; queue empty.

## Verification

R12 independently re-derived from the verdict artifacts (not the readout
prose): declared-peak ν2.5/ν3.0 excesses with 95% CIs — oracle
+2.5%[−1.7,+6.9] / +4.7%[−0.1,+9.8] (both include 0); seed1
+4.6%[+0.3,+9.0] / +6.6%[+1.7,+11.9] (both EXCLUDE 0); seed2
+6.7%[+2.4,+11.0] / +7.8%[+2.8,+13.1] (both EXCLUDE 0). MF(dev) 3.66 /
3.99 / 4.18 (bugged 5.59/6.07/6.17); native peaks +16.7/+19.5,
+20.1/+22.2, +22.9/+24.2; C lands all three; tier-0 no violation. ALL
MATCH the readout. Suites re-run at adjudication: tests/ 62 pass,
tests_wfm+tests_p2 pass (exit 0). Ledger-#17 reference assertion fired
green on all three chains (the bug cannot recur silently).

## Ruling

**1. The ~40% topology-from-selection finding is ACCEPTED** (consistent
2.3–2.4 absolute MF drop on all three legs from correct selection alone,
at the full final-config chain). This completes R47's O decomposition:
the oracle's ~5.6 was mostly a mis-referenced cage. **But topology does
not clear the bar even corrected** (oracle at-bar 3.66; seeds 3.99/4.18
fail) — the residual "the marginal-only cage cannot see texture" gap is
real and is the legitimate D6 lever.

**2. R43 finding-3 is SPLIT; one half is RETRACTED, one half STRENGTHENED**
(binding record correction, both selections retained):
- The *checkpoint* seed-fragility ("fresh seeds pick early @3500/5500 and
  fail") — **RETRACTED as a bug artifact.** Under correct selection the
  picks cluster late (@16000/@19500/@16000, with clean seed 0 @16000).
- The *declared-resolution peak claim* — **DOES NOT SURVIVE the
  seed-ensemble under correct selection.** Two of three corrected seed
  legs carry a significant declared-resolution (0.5-px) excess (CIs
  exclude 0: seed1 +4.6/+6.6%, seed2 +6.7/+7.8%; oracle marginal at
  +2.5/+4.7% with CIs including 0). Both reconvene (60) and executor (58)
  bet the bug explained the old failure; **both LOSE** — the failure is
  not the bug, it is the model-class peak residue seen natively (+16–24%)
  only *partially* attenuated by smoothing, not eliminated.

**This is the consequential finding of the night, stated without
softening:** the paper's headline peak claim — "declared-resolution
peak counts pass at σ_s ≥ 0.5 px" — held for the single shipped {3,4}
blind config as measured (R43 E2 was real: −1.1/−4.0%), but it is
**config-specific and does not generalize across seeds under correct
selection.** The honest claim shrinks. See the scope decision below.

**3. Calibration scorecard.** Reconvene 60 and executor 58 both wrong on
the same reasoning error (assuming the bug was the cause) — a genuine
SHARED miss, logged with full prominence; #17b ("verify the apparatus")
protects against broken instruments but not against a correct instrument
measuring a real residual we hoped was an artifact. Native-peaks-persist
(rec 85 / exec 88) FIRED — the residue was always the honest read.
Tier-0 clean null on first live application: the empty-beam certificate
row is born clean.

## The DRAFT preregs (D2 + conformal) — reviewed, one AMENDMENT each,
## then HELD pending Andreas's scope decision

- **D2 (blind shot 2, JUDGE-2):** structurally sound (A4-pattern committed
  null bar, one shot, oracle-gap as held-out topology cost). **Amendment
  D2-a:** the blind config must be pinned by the scope decision below —
  D2 cannot fire until we decide WHAT ships. Also: JUDGE-2 will very
  likely FAIL (corrected topology is 3.66–4.18 on MF; PH is a different
  but correlated instrument) — pre-state that expectation now so the shot
  is informative rather than disappointing; a JUDGE-2 pass would be the
  surprise, not the base case.
- **Conformal certification:** the highest-value item on the board and
  independent of the scope decision. **Amendment C-a:** add the
  declared-resolution peak bins to the certified statistics explicitly —
  the rank-uniformity test will QUANTIFY the config-specificity finding
  (do the trusted-octave statistics certify at zero width while the
  extrapolated-octave peaks show non-uniform ranks?). This turns finding-2
  from a caveat into a measured certificate. Cleared to run independently
  of the scope call.

## The scope decision (Andreas's — the honest fork, stated with a
## recommendation)

D1 changed the paper's spine: the declared-resolution peak claim is now
config-specific, and topology does not clear even corrected. Three honest
framings:

- **A — narrow-and-ship (recommended).** The claim set becomes: exact
  trusted scales; validated marginals, starlet-ℓ1, spacing across seeds;
  the *shipped blind config's* declared-resolution peak pass as a
  single-config demonstration (not a seed-general claim); the peak excess
  and topology as *located, mechanism-explained, certificate-quantified*
  boundaries; the deconvolution calibration (7/7 now) and the audit
  methodology as the contributions. Add the conformal certificates and
  the W2 theorem. This is a true, complete methods paper whose headline
  is the audit framework and the calibration tool, with generation
  quality honestly bounded.
- **B — fix-then-ship.** Take the D6 texture-aware cage (select on MF at
  declared resolution) as one more single-variable experiment — it
  targets exactly the residual selection gap (~40% of topology) and might
  pull topology under-bar and shrink the declared-peak excess. Cost ~0.3
  H100-h + a Goodhart-discipline prereg (selecting on a near-judge
  statistic; JUDGE-2/PH stays the untouched arbiter). Delays ship ~a
  week; could materially strengthen the claim.
- **C — reframe to the audit as the product.** Lead with the framework
  (pre-registration, frozen judges, located mechanisms, certificates)
  demonstrated ON this generator including its honest failures; generation
  is the case study. Strongest ML-methods framing; weakest astro-utility
  framing.

**Reconvene recommendation: B then A** — run the texture-aware cage once
(it is the one untried lever that directly attacks the residual, and D1
proved selection matters at ~40%), and regardless of its outcome ship
under framing A with the conformal certificates. C is the venue framing,
decided later. But this is a scope call with calendar weight (defense) —
it is Andreas's.

## Orders

1. **Conformal certification: RUN** (amendment C-a; independent of scope).
2. **D6 texture-aware cage: prereg DRAFTED for Andreas's B/A decision**
   (Goodhart discipline; JUDGE-2 untouched; single variable).
3. **D2: HELD** until the scope decision pins the config; expectation
   pre-stated (JUDGE-2 likely fails).
4. Paper stays paused; the W2 theorem + the finding-2 correction are
   queued for the writing session that follows the scope decision.

Budget: D1 ≈ 0.2 H100-h; phase ≈ 2.8 of 10. STOP for Andreas's scope
decision + the conformal run.
