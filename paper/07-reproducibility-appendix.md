# Appendix — Reproducibility and the audit of the audit

## A.1 The pre-registration ledger [WRITE: table]

Every experiment: prereg commit hash → run/job id → readout log → ruling.
Source: log/ directory (preregs committed pre-submission throughout; the two
pre-readout amendments disclosed inline where they occurred). Config hashes
verbatim from run logs (e.g. Stage D 16669982; C1-t 16666378/16666634 hash
27cc4a8f17).

## A.2 Data provenance and splits (audit 2d/2e, accepted R27)

- gowerstreet: 266 train / 32 val / 32 test tiles from 30 parent maps
  (parent-ordered storage, 11 tiles/parent — verified by edge-continuity;
  test parents {27,28,29} DISJOINT from training parents). Disclosed margins:
  the validation (selection) set shares one parent with training (9/32
  tiles); val and test share one parent at a 1-tile margin. Adjudication is
  on test; verdicts unaffected.
- sandbox: 322 train (seed 20260718) + 64 held-out (20260719) + 64
  replication (20260720) — disjoint seed streams by construction.
- [HARDEN]: backfill the committed builder recipe for tiles_pnull.npz
  (make_c1_data.py style).

## A.3 The bar-design ledger (disclosed, numbered, verbatim from rulings)

#8 decay questions get final-state statistics (onset rules mislead);
#9 reference-side SEs budgeted when bars are set (32-field references
inflate 3·SE bars to near-vacuous);
#10 bootstrap over the exchangeable units — parents, not tiles — when
references are hierarchical (audit finding 2b).

## A.4 Auditing the validators

The held-out-basis instrument's own package: gen2 reconstruction mismatch on
an unused path; zero-padded noise plane breaking periodicity/D4 of shipped
SNR (interior-plateau convention adopted, matches the analytic table,
rot90-exact at 4.5e-15). Reported upstream. The protocol treats validators
as auditable objects. <!-- src: starlet_l1_instrument.json -->

## A.5 The scorecard (hits AND misses, both roles) [WRITE: table]

Full prediction table from the rulings incl.: the no-qualifier bake-off miss
(reconvene 85), the taildyn onset-rule miss (both roles), the four
consecutive under-confidence misses on constructive outcomes (R24), the
first modal hit (C1T-CAL 35), the starlet clean sweep, and the ~15% residual
circulating without error bars charged to the whole record judge-first
(R27 scorecard). The environments run: env split (pywt/JAX process
separation) noted as infra.

## A.6 Test gates

Both stacks green at every session end (Stop-hook enforced): tests/ (DWT
round-trip at machine precision; executable GRF null; √N estimator
consistency; symmetry), tests_wfm + tests_p2 (sandbox conditional truth,
starlet instrument, peak-CI instrument, gate registrations).
