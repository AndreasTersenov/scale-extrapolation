# 2026-08-01 — RECONVENE R46: PHASE 3b OPENED (Andreas's direction: the
# paper waits; try to SOLVE the topology boundary) — three-probe diagnosis
# ordered, zero-to-negligible GPU, diagnosis before any lever

Authorization: Andreas, 2026-08-01 ("we are not yet at the point of writing
the paper... do you think there's any way to solve the topology issue" +
agreement to diagnosis-first). This supersedes R43's experiments-closed
state. The paper skeleton stays as committed (its boundary wording already
says OPEN); no paper edits during 3b.

## The target

The E3 defect: T_MF 6.35 at declared resolution, carried by V2/Euler at
ν≈0 — generated maps fragment (more islands, fewer holes, worse on the
underdense side; survives smoothing → NOT the pixel-scale peak texture).

## Probes (executor preregs expectations for each BEFORE computing;
## reconvene lines registered here; STOP at the readout)

**T1 — the deployment split (the decisive probe).** Score the FROZEN MF
judge (declared resolution adjudicating vs the frozen 3.5 bar + #11 band
0.25; native descriptive) on HEAD-CONDITIONAL maps (real coarse +
generated details) vs the committed E2E maps, blind-config streams first,
stage-D lineage as context. If hc map stacks are missing at any leg,
generate them inference-only from committed checkpoints (≤0.05 H100-h,
prereg'd). MEANING, pre-stated: the deployment use-case (PM-conditioned
super-resolution) provides the coarse field — if conditioned mode passes
while self-generated mode fails, the boundary re-scopes to "self-generated
mode only" and the application claim strengthens without any model change.
| branch | rec | meaning |
|---|---|---|
| T1-CONDITIONED-PASS (hc ≤ 3.5, e2e fails) | 25 | defect is coarse/compounding-propagated; deployment story improves |
| T1-BOTH-FAIL | 55 | fragmentation lives in the generated details themselves; levers below |
| T1-MIXED / at-bar | 15 | worse category governs; one disambiguation licensed (fresh streams) |
| gates | 5 | infra, one resubmission |

**T2 — octave attribution, χ-scored (the transplant instrument).** Real
pyramid + generated details at octave j only (j = 1..4), AND the
cumulative variant (real above j, generated at-and-below j) — MF/χ(ν)
curves at declared resolution per hybrid. Registered line: P(a single
octave's substitution reproduces ≥ half the full-gen χ(0) shift) = 40
(the peak transplants showed composition effects; topology may compound
the same way).

**T3 — the skewness hypothesis (reconvene hypothesis, entered under house
rule 1: a hypothesis to test, not a directive).** Per-octave
detail-coefficient skewness, real vs generated (all committed stream
sets, bootstrap SEs over tiles), plus map-level skewness at declared
resolution. Registered line: P(generated details under-skewed at ≥2 of
octaves 2–4 at ≥3σ) = 55. The χ asymmetry (island excess + hole deficit,
deeper underdense trough) is a structure-skewness signature; the marginal
suite audited variance and kurtosis, never skew — this is an unexamined
axis. IF confirmed: the drafted lever is a skewed base via the committed
copula machinery — but its prereg MUST begin with a marginal-shape
transfer measurement (two-point, L1′-style) BEFORE any cure claim: the
spectral transfer was multiplicative; marginal-shape transfer through
fixed weights is unmeasured. Draft only at this stage.

## Discipline

All scoring CPU on committed maps; the only licensed GPU is the T1 hc
generation if needed (≤0.05 H100-h). Frozen instruments only (the MF
judge, the DWT machinery); any NEW estimator (skewness scorer) is
tests-first. R12 throughout; pre-statement before reading; artifacts →
results_p2/topo_diag_*. Refused in advance, for the record: post-hoc
χ-matching of maps (Goodhart — poisons the judge permanently);
topology-aware training losses before this diagnosis lands (no measured
dose-response — the L1 lesson at full price). STOP at the readout for
reconvene adjudication; the lever menu is chosen there.
