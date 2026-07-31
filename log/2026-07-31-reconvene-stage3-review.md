# 2026-07-31 — RECONVENE R42: Stage-3 prereg review — APPROVED CONDITIONAL;
# (a) cleared; (b) one-shot BLOCKED on amendment A4 (the MF null has no
# committed artifact — R12); scoring-order and dry-run riders added

Reviewed: log/2026-07-31-prereg-stage3.md (af34d93). Same-day git log; no
code changes since the last green suite run; queue empty.

## Design audit — what is APPROVED as written

- **(a) seed ensemble:** recipe verbatim, survivorship labeling of seed 0,
  per-seed deployment-recipe deconvolution with own P-T bands (the
  transferred sigma_lnT prior properly disclosed), determinism-gate
  substitution for replay (fresh weights cannot replay — correct), #14
  conditional meanings on R-SEED-FRAGILE. Cleared to run after A4 lands
  (A4 is cheap and (a)'s trainings can queue meanwhile — see order).
- **(b) protocol:** Stage-D verbatim with the final config; **the
  oct-3-rescaled blind target is the review's highlight** — using the
  oct-2 ring would have leaked the held-out edge octave into the base
  calibration; the two-octave rescaling is deployment-pure, blind-clean,
  and licensed by the measured near-invariance. Credited. Matched-
  treatment smoothing (both stacks), corrected panel convention, the
  A1-guard for over-smoothing named inside PARTIAL, native-MF
  expected-fail pre-statement, dry-run quarantine: all sound. Branch
  table checked for completeness: every entry-outcome combination maps to
  exactly one branch.
- **(c)** staged caveat text: accurate; wordsmithing deferred to WP.
- Seeds/keys disjointness verified against the committed stream registry.

## Amendment A4 (BINDING — (b) blocked until it lands): the MF null artifact

The prereg freezes the Minkowski bar against "the measured real split-half
null (native 1.43/0.47/2.65; smoothed 1.48/0.55/2.87, 20 seeded splits)" —
**no committed artifact contains these numbers** (searched results_p2/ and
history). The bar's DERIVATION is exactly right (measured reference,
ledger #3/#8; scoring real tiles does not violate the judge freeze — the
freeze quarantines GENERATIONS, and reference-side null calibration is
required bar-craft). But under R12 the numbers cannot enter the record
without a committed source — this is the campaign's founding scar and the
one-shot's bar is the last place it may recur.

**A4:** commit the null computation as script + artifact
(results_p2/stage3_mf_null.json, seeded splits recorded), then verify. If
the recomputed numbers match the prereg's quoted values exactly, (b) is
PRE-CLEARED to launch with no further round-trip — the match note goes in
JOBS.md. If they differ AT ALL, STOP: the 3.5 bar is re-derived from the
committed null (same 3.7-null-sd construction) and the delta is disclosed
before the blind training launches.

## Amendments A5–A6 (binding, cheap)

- **A5 (scoring order, structural):** for BOTH (a) and (b): determinism/
  identity gates → coloring C + P-T band → adjudicating entries. The C
  context must exist on disk before any peak/marginal number is computed
  (the L1″ pattern, twice validated).
- **A6 (one-shot hygiene):** after the dry-run passes, ANY edit to the
  sampling/scoring pipeline invalidates it — re-run the dry-run before
  the blind training launches. Edit → dry-run → launch, no exceptions.

## Reconvene weights and lines (registered now)

- (a): **R-ROBUST 50 / R-SEED-FRAGILE 40 / gates 10** (exec 55/35/10).
  Lines: all-3-seeds pass declared-resolution rule 55 (exec 65 — the
  committed crossings sit at 0.41–0.49 px, close to the 0.5 boundary, and
  cross-seed crossing variance is unmeasured); all-3 C in own P-T band 65
  (exec 70); picks-differ 50 (co-sign).
- (b): **B3-PASS 40 / B3-PARTIAL 35 / B3-FAIL 15 / gates 10** (exec
  45/30/15/10). Entry lines: E1 85 (co-sign); E2 65 (exec 70 — fresh seed
  + fresh T + the oct-3-target recipe delta vs the committed edge
  evidence); E3 62 (exec 70 — the untouched tier is untouched precisely
  because it can catch what the designed-against tiers miss); E4 starlet
  90; native-MF echo 75 (co-sign).

## Order

1. Execute A4 now (CPU, minutes). (a)'s two trainings may be SUBMITTED
   immediately (they are independent of A4); their scoring waits for
   nothing but themselves.
2. On A4-match: (b) dry-run → (A6 discipline) → the one blind training →
   the one-shot adjudication scored per A5. On A4-mismatch: STOP as above.
3. STOP at the Stage-3 readout for the final reconvene adjudication of
   the phase.

Budget confirmed ≈1.1 H100-h of the ~2 allocated; phase ≈0.56 spent.
