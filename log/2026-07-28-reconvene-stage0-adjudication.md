# 2026-07-28 — RECONVENE R37: Stage-0 adjudication — S0-UNCHANGED CONFIRMED;
# the excess is field-level texture, not an extrapolation defect; Stage 1
# ordered field-first; Stage-2 colored base is the live lever

Ruled on: log/2026-07-28-stage0-p3-readout.md (commit 0ba2de4) + artifacts.
Same-day git log taken; no other movement in the repo since my review 8204251.

## Verification (all checks performed independently)

1. **R12:** every adjudicating number in the readout re-checked against
   results_p2/stage0_p3_verdict.json — edge A ν=2.5 +12.446%±3.094
   (readout +12.45±3.09 MATCH), ν=3.0 +12.937%±2.999 (+12.94±3.00 MATCH);
   ci95 bounds, z, per-parent panels, B rows, G1/G2 diagnostics all MATCH.
2. **Branch mechanics re-derived:** rν/2 boundaries (6.57%, 7.31%); both eν
   above them by 5.9%/5.6% while the #11 half-SE bands are 1.55%/1.50% — no
   at-bar entry, thresholds agree, A1 CIs entirely positive (no flip).
   Δ-reductions recomputed: 0.155σ / 0.391σ. **S0-UNCHANGED is the unique
   mechanical outcome.**
3. **G2 correction audit (the sensitive spot):** timeline from git — machinery
   12:12 → job 17621159 dies AT the gate (its log contains zero scored
   quantities; G1 never ran) → PENDING pre-statement 12:22 → corrected
   criterion committed 12:32 → resubmission recorded 12:33 → readout 12:48.
   The correction was blind to every adjudicating number, disclosed in place,
   and consumed the one licensed resubmission. **Legitimate instrument
   repair, credited.** Lesson for the instrument ledger: a gate comparing
   post-recursion outputs must budget for multiplicative amplification of
   per-op float noise (4 octaves of 80-step ODE chains turn 1e-3-grade noise
   into ~1e-2 diffs); corrected-G2's corr/ratio criteria are the right shape.
4. **Test integrity:** diff since my review = one ADDED file
   (tests/test_stage0_adjudicate.py, +72) — no assertion lightening. Suite:
   tests/ 33 pass (env.sh stack, re-run at adjudication); tests_wfm+tests_p2
   re-run at adjudication: ALL PASS (exit 0, login-node cpu convention).
5. **Labeling caveat (recorded):** the "committed pre-F2 gen_A" trained-leg
   row (before_gowA +14.06/+15.31) is a FRESH scoring of committed generation
   files by the frozen scorer, not a previously-adjudicated number — valid as
   texture under R12 (maps committed, scorer frozen), labeled as such here.

## Ruling

**S0-UNCHANGED stands.** The peak tier does not close for free. The
dissociation is total and elegant: on the SAME maps, the F2 sampler's target
defect is dead (parity_T 1.00, T_coef_oct1 2.18, both < 3) while the peak
excess moved 0.16σ/0.39σ. The cure worked on its target; the excess is
orthogonal to it. The review's stated doubt ("the parity→edge-peak-excess
causal link was never established") is now a measurement, not a doubt.

**The reframe that matters (adopted as the Stage-1 framing):** the count
bias tracks the FIELD, not the scale —

- real field (gowerstreet), trained scales: +14.5%±3.1 / +11.1%±2.8 (F2),
  +14.1/+15.3 (pre-F2) — same size as the edge;
- real field, edge: +12.4 / +12.9 (F2);
- sandbox substrates, trained scales: −4.9 to −9.4% (committed).

On the real field, extrapolation ADDS essentially no peak error — the
excess is a field-level joint-texture mismatch present at every scale, with
sign flipping between substrates. Consequences: (a) the method paper's
extrapolation claim is STRENGTHENED (the residual is not
extrapolation-specific — this row belongs in the final-config audit table
with exactly that sentence); (b) Stage 1's mechanism profiling gains a
cheaper, better-sampled venue: trained scales on the real field, where data
is plentiful; (c) per the pre-stated branch meaning, Stage 2's colored
heavy-tailed base is the live lever — the base sees the same texture at all
scales, which is precisely the regime a measured-autocorrelation base
addresses.

## Registered-line scorecard (both columns)

- **Modal branch: BOTH MISS.** Executor S0-SHRUNK 43, reconvene S0-SHRUNK 38;
  fired UNCHANGED at 25/30 — reconvene's heavier UNCHANGED better calibrated,
  neither column modal-correct. Logged with standard prominence.
- **nn edge persists (P=85/85): LOSES** — nn_T 2.57 at the edge, arm A.
  Scope discipline: FIRST edge nn reading ever; the named tier-3 frontier
  (T≈3.5–4.1) lives on trained legs and stands; instrument power shown by
  arm B's edge 4.76. The frontier narrative gains a wrinkle: the residual is
  absent (or sub-threshold) exactly where extrapolation happens.
- **starlet-ℓ1 holds (P=90/90): HIT** — edge and trained legs pass, frozen
  scorer, fresh streams.
- **B worse wherever they differ (P=75/70): LOSES on the strict quantifier**
  (B smaller-magnitude at ν=2.5; cleaner oct3/4 marginals). Bar-ledger #12
  (new): comparative descriptive lines must declare per-statistic scope —
  universal quantifiers over statistic sets keep losing on one entry.

## Orders

1. **Stage 1 (mechanism profiling) is authorized** with this readout's
   numbers as its measured references, POSED FIELD-FIRST: profile the
   real-field excess at TRAINED scales (peak morphology, octave attribution,
   detail autocorrelation per PLAN-phase3) alongside one edge leg for
   continuity — the mechanism is a property of the field/base mismatch, not
   of the extrapolation step. Executor preregs with weighted branches +
   gates; reconvene lines at review; STOP before running.
2. Stage-2 (colored heavy-tailed base) spec may be DRAFTED in the same
   session (design work, no compute) so the Stage-1 readout can flow
   directly into it — but no Stage-2 training is authorized until the
   Stage-1 harvest.
3. The marginal/starlet/spacing/peak rows of this readout are designated the
   method paper's final-configuration audit table (per the prereg's
   pre-statement); paper/ remains frozen until the Stage-1 readout.

Budget: ≈0.4 H100-h spent of the phase's 10 (two MIG jobs, one gate-failed).
