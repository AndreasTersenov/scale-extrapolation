# 2026-08-06 — PREREG (DRAFT, NOT RUN): D2 — the second blind shot,
# JUDGE-2 (persistent homology) as the unburned tier (R48 order 2)

STATUS: **DRAFT for reconvene review.** Comes to reconvene WITH the D1
readout. Nothing here runs until reconvene clears it AND the shipped
post-D1 configuration is fixed. JUDGE-2 quarantine is ABSOLUTE until the
moment defined below — no application to any real or generated stack
before then, this shot included until its null bar is committed.

## Why a second blind shot

The Minkowski judge is now a development metric (we engineered against it
all of phase-3b). The audit thesis needs a tier that has NEVER seen a
generated map. JUDGE-2 (judge2_betti.py, FROZEN 2026-08-05: b0 4-conn /
b1 (4,8)-duality Betti curves, 13-threshold superlevel grid,
per-field-standardized, GRF-validated, quarantined) is that tier. One
bullet; fired once, here.

## The shipped configuration (fixed by reconvene AFTER D1)

The single config the reconvene ships post-D1 — production geometry
({3,4} deployment or the corrected-selection production model as
adjudicated), corrected selection (ledger #17 reference assertion), F2
sampler, measured-transfer deconvolved oct-1 base. The oracle arm at its
corrected pick (@16000) is the paired data-available reference (its own
JUDGE-2 reading is the extrapolation-cost comparison, scored in the SAME
shot under the same committed bar).

## A4-pattern null bar — COMMITTED BEFORE ANY GENERATED MAP IS SCORED

Mirror stage3_mf_null.py exactly, with judge2_T in place of judge_T:
`night3_judge2_null.py` (to be written, tests-first not required — it
only *invokes* the frozen judge) computes, on the 32 committed
gowerstreet real test tiles:
- 20 seeded 16v16 half-splits, split rng default_rng(100+s).permutation(32),
  judge2_T(n_boot=400, seed=500+s), in BOTH native and declared (0.5-px)
  domains; every split's (split_seed, judge_seed, T) recorded verbatim.
- Committed artifact results_p2/d2_judge2_null.json.
- **Bar (frozen at commit):** T_JUDGE2 bar = null max over the 20 splits
  in the declared domain (the conservative one-shot bar, matching the
  MF-judge's stage3 precedent of quoting {mean, sd, max}); at-bar band
  ±0.25·sd; worse governs; ONE shot, NO re-run. Native reported
  descriptive.

Sequencing lock (A4): the null artifact + this frozen bar are committed
in a SEPARATE commit BEFORE the generation stack is loaded by the D2
scorer. The scorer refuses to run if d2_judge2_null.json is absent or
uncommitted (git-clean check on the null file).

## The shot (one generation stack, scored once)

- Generate the shipped config's three final streams (pooled 96) with
  FRESH registered keys (committed in this prereg's KEYS block at run
  time), determinism gate asserted, replay gate vs the committed chain.
- Score judge2_T(declared) of pooled-96 vs the 32 real tiles ONCE.
  PASS iff T ≤ bar and not at-bar. Native descriptive.
- Same for the corrected oracle stack (extrapolation-cost comparison):
  report T_oracle alongside; the gap (T_shipped − T_oracle) is the
  first held-out topology cost-of-extrapolation number.
- No other JUDGE-2 call, ever, on any other stack. If the shot is
  spent, JUDGE-2 is burned; a THIRD tier would need a new frozen judge.

## Weights (rec PENDING reconvene; exec pre-stated for review)

- P(shipped config passes JUDGE-2 at the declared bar): exec **45** —
  MF-declared landed 3.13–3.34 at trained scales for the shipped family
  (BL), but JUDGE-2's b1 hole-curve is a DIFFERENT statistic and the
  peak/topology residue is real; genuine coin-flip-ish, slight lean to
  fail given the residue.
- P(oracle passes JUDGE-2): exec **40** (3.87 MF-declared at @16000; the
  topology residue is smaller but present).
- P(gap > 0, shipped worse than oracle on JUDGE-2): exec **70** — every
  depth-ordered instrument put the extrapolating config above the
  data-available one.

## What this buys the paper

The certificate ledger's held-out topology row: "generated maps pass a
persistent-homology test never used in development, at a bar set by the
real data's own split-half null; the price of extrapolation is X±Y in
that test's units." Burns exactly one judge; conformal (drafted
separately) covers everything else without a judge.

## STOP — draft only. Reconvene fixes the shipped config + rec weights;
## the null bar is committed before the shot; the shot is single.
