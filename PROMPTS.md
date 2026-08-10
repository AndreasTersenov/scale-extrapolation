# PASTE-READY PROMPTS — one per task, no planning required

Copy a block verbatim into a fresh session in ~/software/scale-extrapolation.
Each is self-contained. Do them one at a time, in any order except: T4 after
T1 (so the certificates can enter the paper), T3 last.

---

## T1 — conformal certification run (do this first; CPU only)

```
Read HANDOFF.md, then log/2026-08-06-prereg-conformal-DRAFT.md and the
"DRAFT preregs" section of log/2026-08-07-reconvene-d1-adjudication.md
(amendment C-a). Implement and run the conformal rank-uniformity
certification exactly as specified: n≈50 held-out coarse fields disjoint
from training, checkpoint selection, and the blind sets; m generated
completions each; the rank of the true statistic within its generated
ensemble, per peak bin and per starlet scale; include the
declared-resolution (0.5-px) peak bins per amendment C-a; report the
trusted-octave zero-width corollary; run the identical test on the
corrected oracle (d1 oracle @16000) for the certified
extrapolation-cost band. Tests-first for any new estimator. Numbers only
from committed artifacts. Commit the artifact + a log entry
log/YYYY-MM-DD-conformal-readout.md stating what is certified, at what
width, and what is not. Do not start any other task. Both test suites
green before committing.
```

## T2 — the W2 theorem + negative lemma (writing, no compute)

```
Read HANDOFF.md and BRIEF-foundations.md (item 3). Write paper/theory-w2.md
containing: (1) the proposition that because the wavelet transform is an
isometry, the squared W2 generation error decomposes additively across
octaves and the conditioned octaves contribute exactly zero, with its
proof (diagonal coupling on the exact conditioning + Haar orthogonality);
(2) the negative lemma: peak counts and the Euler characteristic are not
L2-Lipschitz, so W2 cannot certify them — hence the conformal certificates
are complementary, not redundant; (3) a two-sentence plain version for the
introduction. Be honest about any gap in the argument rather than papering
over it. No experiments. Commit.
```

## T4 — paper claim-set update (writing, no compute; after T1)

```
Read HANDOFF.md §2 (binding science state) and the rulings
log/2026-08-06-reconvene-night3-harvest.md (R48) and
log/2026-08-07-reconvene-d1-adjudication.md (R49). Update paper/ to the
corrected claim set: 05-boundaries — retract the checkpoint
seed-fragility caveat (bug artifact) and restate the declared-resolution
peak claim as CONFIG-SPECIFIC with the corrected seed-leg numbers;
03-validation — add the corrected-selection table alongside the bugged
one; 04-mechanism — add the depth law, the three closed levers with their
mechanisms, and the weights-not-inputs doctrine; 07-appendix — the truth
bug as an instrument lesson plus the corrected calibration scorecard. If
T1 has landed, add its certificates to the validation section. Every
number copied verbatim from a committed artifact with a src pointer
comment. No new claims beyond the rulings. Commit.
```

## T5 — texture-aware cage (ONLY if Andreas picks option B)

```
Read HANDOFF.md §3 and §5-T5. Pre-register (commit before running) a
single-variable experiment: replace the marginal-only checkpoint selection
with a rule that also sees topology at declared resolution, applied to the
existing checkpoint grids. Weighted branches, mechanical rules,
ambiguity = negative. Goodhart discipline is binding: JUDGE-2 (persistent
homology) is never the selection statistic and stays quarantined; state
explicitly in the prereg how you separate "selection can now see texture"
from "we optimized the metric". Then run it (~0.3 H100-h), score with the
frozen battery, and write the readout. STOP after the readout.
```

## T3 — the second blind shot (LAST; one shot, ever)

```
Read HANDOFF.md, log/2026-08-06-prereg-d2-blindshot2-DRAFT.md, and the D2
amendments in log/2026-08-07-reconvene-d1-adjudication.md. The
configuration to test is: <FILL IN — the config Andreas ships>. Commit the
prereg first, including the A4-pattern JUDGE-2 bar computed from the real
split-half null BEFORE any generated map is scored, and the pre-stated
expectation that JUDGE-2 will likely FAIL (a pass is the surprise).
Then run it exactly once — no re-runs, no second look — score, and write
the readout. This is the last untouched judge in the project; treat the
quarantine as absolute until the moment of scoring.
```

---

## If a session proposes something not on this list

Say no, and have it write the idea into IDEAS-PARKED.md with a date and a
paragraph. That is the whole protocol.
