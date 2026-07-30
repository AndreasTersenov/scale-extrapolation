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
