# 2026-07-16 — C1 PRE-READOUT AMENDMENT: truth reference must share the
# normalization convention (measured, committed before any C1 adjudication)

While preparing the C1 scorer (training job 16491750 in flight, no readout seen),
a convention mismatch was caught and MEASURED: the wfm pipeline trains, generates
and scores in PER-TILE normalized fields (normalize_tiles), while
results_p2/sandbox_truth.json is the raw-field truth. On the 256 sandbox parents:

| oct | raw var_slope | norm var_slope | Δ | raw kurt | norm kurt | Δ |
|---|---|---|---|---|---|---|
| 2 | 1.0985 | 1.0636 | −3.2% | 5.52 | 4.87 | −11.8% |
| 3 | 0.9571 | 0.9151 | −4.4% | 3.92 | 3.29 | −16.1% |
| 4 | 0.8085 | 0.7604 | −6.0% | 2.81 | 2.27 | −19.3% |

The shift is comparable to the pre-registered bars (10% dispersion / 15% kurtosis):
adjudicating normalized-convention generations against raw-field truth would be
biased AGAINST the generator by up to a full bar.

## Amendment (bars unchanged in form; reference values swap convention)

1. The C1 bars are adjudicated against **sandbox_truth_normconv.json** — the SAME
   estimand on the SAME 16384 exact ensemble fields, each per-field normalized
   (deterministic map of exact samples ⇒ still exact truth), same batch-means SEs.
2. The DEGENERATE amplitude check compares generated detail_std to the detail_std
   of the arms npz's own normalized `real` stack (same convention by construction).
3. Nothing else changes: bar formulas, branch definitions, weights, and predictions
   stand exactly as pre-registered.

## Note for the morning reconvene (no action tonight)

Phase-1c adjudications were like-for-like (gen and real both scored under
normalize_tiles from the same npz), so no retroactive issue. But stage-0 raw-field
coupling values and generator-side normalized values differ by 3–19% on the sandbox
— any cross-convention comparison (e.g. coords dial vs generator-side measurements)
should mind this. The coords dial passed to arm B uses raw-truth values, mirroring
phase 1 (which used stage-0 raw measurements) — consistent with the frozen 4a
protocol, so left untouched tonight.
