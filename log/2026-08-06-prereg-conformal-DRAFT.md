# 2026-08-06 — PREREG (DRAFT, NOT RUN): conformal rank-uniformity
# certification (R48 order 3; BRIEF-foundations #2)

STATUS: **DRAFT for reconvene review**, drafted alongside D2. Burns NO
judge. Becomes the paper's certificate machinery. Nothing runs until
reconvene clears it and the shipped post-D1 config is fixed.

## The claim being certified

Under a correct conditional law p(fine | coarse), for a held-out coarse
field the rank of the TRUE completion's statistic within an ensemble of
GENERATED completions is uniform on {0, …, m}. Rank-uniformity is a
finite-sample, distribution-free test of exactly what an SBI consumer
needs — no frozen judge, no asymptotics, per (statistic, scale). This is
the same epistemic move as the L1″ transfer calibration (measure the
deployed system, certify distribution-free), extended from spectrum to
downstream statistics.

## Calibration set (the disjointness is load-bearing)

n ≈ 50 fresh gowerstreet coarse fields at octave 4, STRICTLY DISJOINT
from: the training tiles, the checkpoint-selection validation split, and
(when D2 runs) the JUDGE-2 blind set. Concretely: draw from gowerstreet
tiles NOT in `tiles_pnull.npz`'s train/val/test partition — requires a
fresh tile extraction (make_* script) with a recorded disjoint index
set, committed as the calibration manifest BEFORE any rank is computed.
If insufficient fresh fields exist, n is reduced and the achievable
rank-resolution (1/(m+1)) and per-bin CI are stated honestly, not
padded.

## Protocol

For each of the n calibration coarse fields c_i:
1. Draw m (≈ 100) generated fine completions from the shipped config
   (F2 sampler, deconvolved oct-1 base), conditioned on c_i.
2. Reconstruct each to a full map; compute the target statistics:
   - peak counts per ν bin (the audit's own peak instrument bins),
   - starlet-l1 per scale (frozen scorer scales).
3. The TRUE completion of c_i (the real fine field it came from) gives
   the reference statistic value.
4. rank_i(stat) = #{generated completions with stat < true} (ties
   split by a fixed jitter seed), ∈ {0, …, m}.

Aggregate over i: the rank histogram per (statistic, bin/scale).
Under a correct law it is uniform.

## Certification tests (committed bars)

- **Rank-uniformity:** per statistic, a distribution-free uniformity
  test on the n ranks — the discrete PIT. Primary: the reduced-rank
  chi-square on B≈10 equiprobable rank-bins, and the coverage of the
  central 1−α interval (does the true stat fall in the central 90% of
  the generated ensemble at rate 90%?). Bars committed here before
  reading: coverage within ±(binomial 2σ at n=50); chi-square p not
  below 0.01 (flag over-/under-dispersion direction — under-dispersion
  = generator too narrow, the phase-1 disease's signature at the
  statistic level).
- **Trusted-octave zero-width corollary (the separating sentence):**
  for any statistic that depends ONLY on octaves 2–4 (exact by
  construction — coarse fixed, those details are the real conditional),
  the certificate is zero-width: the generated ensemble's trusted-scale
  statistics equal the truth up to sampling. Stated and checked as a
  sanity anchor (a trusted-scale statistic must show rank-uniformity
  trivially / degenerate ranks), separating this method from any
  monolithic generator.

## The extrapolation-cost band (identical run on the corrected oracle)

Run the WHOLE protocol identically on the corrected-pick oracle
(@16000, data-available): its rank-uniformity is the FLOOR. The gap
between the shipped config's rank-calibration and the oracle's — per
statistic — is the certified cost of extrapolation in
distribution-free, per-statistic units. This is the number the method
is built to produce: not "passed a judge" but "certified to rank-
calibration X, versus the data-available floor Y, at the invented
octave."

## Machinery to build (tests-first when it lands, NOT tonight)

- conformal_ranks.py: rank computation per (field, statistic), the
  disjoint calibration manifest loader, the uniformity + coverage tests;
  validated on synthetics where the conditional law is known (ranks
  provably uniform) and where it is deliberately mis-specified
  (under-dispersed generator → detected).
- A fresh disjoint-tile extraction with a committed index manifest.

## Weights (rec PENDING reconvene; exec pre-stated)

- P(shipped config passes trusted-scale zero-width anchor): exec **90**
  (exact by construction; a fail would indicate a pipeline bug, valuable).
- P(shipped config shows detectable finest-scale under-dispersion in
  peak-bin ranks): exec **65** (the peak excess + the phase-1 dispersion
  history predict a too-narrow finest-scale ensemble).
- P(oracle rank-calibration strictly better than shipped on ≥1
  statistic): exec **75** (depth ordering again).

## Why it composes with D2 and the W2 theorem

The W2 theorem (assigned to the writing session) certifies WHERE the
error budget lives (extrapolated octaves, by Haar isometry) but provably
CANNOT bound peak counts / Euler characteristic (not L2-Lipschitz — the
negative lemma). Conformal certifies exactly those non-Lipschitz
downstream statistics, distribution-free. JUDGE-2 (D2) is the single
held-out topology bullet; conformal is why most future questions never
need a judge. Three complementary certificates, no redundancy.

## STOP — draft only. Reconvene fixes config + rec weights + n; the
## disjoint calibration manifest is committed before any rank is read.
