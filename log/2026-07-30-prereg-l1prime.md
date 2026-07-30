# 2026-07-30 — PRE-STATEMENT: L1′ execution (R39 approval; committed BEFORE
# any result — canary included — is read; R12 throughout)

Approved spec (R39 (a)+(b), the delta memo verbatim): inference-only
colored-t base at octave 1 on the COMMITTED checkpoints; oct2rescaled filter
ADJUDICATES; oct1-measured = ONE labeled descriptive ablation stream outside
the branch table. Zero training. Artifacts → results_p2/l1p_*.

## Streams and seeds (recorded now)

| leg | ckpt | filter (oct-1 base) | key / grng | role |
|---|---|---|---|---|
| sandbox canary | sandbox A@7500 | sandbox oct2rescaled | 3301 / 20260801 | must-not-regress gate |
| sandbox replay | sandbox A@7500 | WHITE (t-base) | 2200 / 20260728 | identity gate vs committed F2_A_e2e |
| gow stream 1–3 | gow A@16000 | gowerstreet oct2rescaled | 3401-3/20260802-4 | ADJUDICATING |
| gow ablation | gow A@16000 | gowerstreet oct1-measured | 3405 / 20260805 | labeled DESCRIPTIVE |
| gow replay | gow A@16000 | WHITE | 2400 / 20260730* | identity gate vs committed F2_gowA_e2e (*grng 20260728+2, the committed stream) |
| edge continuity | stage-D A@9000 | gowerstreet oct2rescaled | 3406 / 20260806 | descriptive |

New plumbing (sample_base_fn / gen_groupavg_base) is validated BEFORE any
job by a tests_p2 equivalence test: with the white base it must equal
sample_tbase EXACTLY (same key usage), so the replay gates test the
committed chain, not the new code. Filter-source caveat, stated now: the
oct2 ring shape is measured on training tiles' octave-2 planes; octave 2 is
a TRAINED octave on the adjudicating (c1t) substrate — deployment-pure
there — but is stage-D's held-out edge, so the edge continuity leg carries
an information caveat and stays descriptive (it already was).

## Order of scoring (binding)

1. Canary marginals (env.sh; e2e couplings octaves 1–4 vs sandbox truth,
   standing bars as context). KILL: var_slope rel err > 50% at any scored
   octave (2–4) → stop, no gowerstreet submission, diagnose. Result: PENDING.
2. Identity gates (replay criterion, twice validated: corr_min ≥ 0.99,
   |ratio_mean − 1| ≤ 5e-3, rel max-abs ≤ 5e-2). Results: PENDING (sandbox),
   PENDING (gowerstreet).
3. **A2 FIRST, before any peak number is read:** pooled C over the 3
   adjudicating streams (96 maps), octave 1. TRANSFER iff C ≥ 0.7551 AND
   (C − 0.7237) ≥ 3·SE_pooled. Per-stream values shown; ablation stream's C
   reported alongside (descriptive). Result: PENDING.
4. Peaks (frozen bootstrap_excess): e_new = pooled-96-vs-32 excess with
   bootstrap ci95 (both sides resampled) at ν ∈ {2.5, 3.0}; per-stream
   excesses + stream sd reported; ν=1 descriptive (#10). Reference (A3):
   r = +15.290% (ν2.5), +12.474% (ν3.0); SE_ref = 3.12% / 2.80% (the
   committed single-stream bootstrap SEs — conservative, shared-real
   conservatism noted). Δν = (rν − eν)/hypot(SE_new, SE_ref). Result:
   PENDING.
5. Must-not-regress set: marginal suite (no catastrophe), starlet-ℓ1
   trained + edge legs on L1′ maps (frozen scorer, fresh leg indices),
   parity_T < 3, identity gates. nn WATCHED (frozen instrument; the
   co-location result predicts movement if the graininess is the shared
   mechanism). Results: PENDING.

## Branch table (mechanical; weights carried, rec/exec)

| branch | rule (both ν unless stated) | weights |
|---|---|---|
| N-CURED | both pooled ci95 include 0; no regression | 10/10 |
| N-IMPROVED | not cured; both Δ ≥ 2; no regression | 35/40 |
| N-NULL | both Δ < 2 (mixed → worse governs); no regression; **A2 splits: NULL-WITH-TRANSFER (M-GRAIN wounded) vs NULL-WITHOUT-TRANSFER (L2 becomes live)** | 35/30 |
| N-REGRESSED | any must-not-regress failure, regardless of peaks | 10/10 |
| N-FLIPPED / gates | either ci95 entirely below 0 (A1 guard); identity/infra (one licensed resubmission per job) | 10/10 |

#11 half-SE at-bar bands on the Δ=2 boundary (|eν − (rν − 2·hypot)| ≤
0.5·SE_new → at-bar, worse governs); ONE fresh-PRNG disambiguation
pre-authorized. Executor's registered descriptive lines: P(A2 TRANSFER) =
55 (the flow may not carry base spectra into details — untested); P(nn_T
drops below 3 on the adjudicating streams | TRANSFER) = 40; P(ablation
stream shows larger C than adjudicating streams) = 75 (oracle filter ≥
deployment filter).

## Sequencing (pre-stated)

Canary → (pass) → one gowerstreet+edge job → score in the order above →
readout appended here → **STOP for reconvene adjudication.** Budget ≲0.4
H100-h; two MIG jobs expected ≈0.06.

---

## READOUT (appended; every number verbatim from l1p_* artifacts)

1. **Canary: PASS** (kill did not fire; var_slope rel 7.5/4.7/7.2% at
   octaves 2/3/4). Context note, recorded: oct2 kurtosis rel 17.3% vs its
   15.5% context bar (outside the #11 band; the committed white-base F2
   sandbox leg passed it) — the colored base slightly inflates sandbox oct2
   kurtosis error. Context only; not a branch quantity.
2. **Identity gates: both PASS** under the replay criterion — sandbox
   corr_min 0.99999999 / ratio 0.999999 / rel 1.42e-3; gowerstreet
   corr_min 0.99999999 / ratio 1.000000 / rel 6.73e-3.
3. **A2 (scored first, per the binding order): NO-TRANSFER — INVERTED.**
   C_pooled = 0.6030 ± 0.0049: not only below the 0.7551 transfer
   threshold, but **24.8σ BELOW the white-base baseline 0.7237**. Per
   stream: 0.6189/0.6056/0.5845; ablation (oct1-measured oracle filter):
   0.6295 — also inverted. Registered line P(TRANSFER)=55 LOSES, and in an
   unlisted direction (the record's recurring lesson).
4. **Peaks vs A3: NULL, cleanly.** Pooled (96 maps) ν2.5 +15.19%±2.93 vs
   reference +15.290% → Δ = 0.02σ; ν3.0 +13.67%±2.64 vs +12.474% →
   Δ = −0.31σ. Per-stream +16.13/+13.86/+15.57 (ν2.5); ablation +15.62;
   edge continuity +15.14/+15.95 (vs committed white-base stage-D F2
   +12.45/+12.94 — directionally worse, descriptive); ν=1 pooled +21.11%
   (#10, never headlined). No at-bar entry, no flip.
5. **Must-not-regress: NO regression.** No marginal catastrophe;
   parity_T = 1.27; starlet-ℓ1 trained + edge legs PASS (frozen scorer,
   legs 8/9); identity gates PASS. WATCHED: nn_T = 3.33 (committed
   white-base streams: 4.045 — same instrument, different streams;
   descriptive). coef_T_oct1 = 2.35.
6. **BRANCH (mechanical): N-NULL-WITHOUT-TRANSFER** (weights rec/exec
   35/30 on NULL, A2-split). Registered lines: ablation-C-larger 75 → HIT
   (0.6295 > all three adjudicating streams); nn-conditional void (no
   transfer).

### DESCRIPTIVE addendum (labeled; outside the branch table):
### the two-point transfer analysis (l1p_transfer_analysis.json)

The night's two input spectra (white C=1.0; colored C≈0.74) give a
two-point measurement of the flow's oct-1 spectral action. Per-ring
transfer T(k) = S_out/S_in is CONSISTENT across the two inputs to ~3%
(T_col/T_white median 0.981, IQR [0.967, 1.029]): the extrapolating flow
acts as a nearly input-independent multiplicative filter, T(k) rising
0.66 → 1.06 from k=2 to k=30 — a blue-tilting transfer. Consequences:
(i) the whiteness defect IS the transfer function of weight-tied
extrapolation on this field (with a white base, output = T·white); (ii)
L1's filter TARGET was mis-specified — we colored the base toward the
real-detail spectrum when cancellation requires pre-emphasis AGAINST the
transfer: the deconvolved base target S_real/T has C = 1.10, red of white,
the OPPOSITE side from the filter we ran (0.74); the multiplicative model
predicts the observed inversion (predicted colored-out C 0.527 vs measured
0.589, ring convention). (iii) **The pre-stated MEANING of
NULL-WITHOUT-TRANSFER ("the flow does not carry base spectra into
details") is contradicted by the measured mechanism** — the flow carries
them at ~unit ring-wise fidelity; the literal A2 rule (moved TOWARD real
at 3σ) correctly read NO because the carried spectrum was aimed at the
wrong target. The branch verdict stands as the mechanical outcome; its
attached meaning needs reconvene amendment before L2 is declared the live
lever — a measured-deconvolution L1″ (inference-only, base filter =
S_real/T from the committed two-point measurement, ~0.03 H100-h) is the
cheap falsifiable next move, with the two-point consistency as its
registered premise. DRAFT ONLY — nothing further runs tonight.

### Execution accounting

Two MIG jobs (17939621 canary 8 min; 17941294 main 18 min) ≈ 0.06 H100-h
of the ≲0.4 budget; zero training; both test suites green; the A2-before-
peaks order enforced structurally (l1p_a2.json written before any peak
number was computed).

**STOP — reconvene adjudication of the readout, the meaning amendment, and
the L1″ draft.**
