# 2026-07-25 — ARMS F/J READOUT: **F-PARTIAL** (the defect has two layers; the
# mean layer is cured free, a cross-channel correlation survives) and
# **J-TRADEOFF** (the schedule bind is now mechanically demonstrated). The
# real-field spacing residual SURVIVES the symmetry fix — it is a second,
# independent joint residual.

Prereg log/2026-07-25-prereg-parity-cure.md (approved R32, weights registered
there). Jobs: phase-1 17422062 (8:19), phase-2 17422379 (1:49) — MIG,
inference only, logged pre-submission. Artifacts: fj_val_gen.npz,
fj_val_hc.json, fj_offsets.json, fj_joint_pick.json, fj_test_gen.npz,
fj_test_hc.json, fj_verdict.json. All numbers verbatim (R12). Machinery
tests-first (tests/test_fj_machinery.py, 6 tests): the channel-mapping test
caught that the generator's H/V detail channels are SWAPPED relative to the
measurement transform — an untested identity mapping would have corrected the
wrong channels. Equivalence gate: corrected recursion with zero offsets
reproduces the committed sampler EXACTLY (max abs diff 0.0). Stability gates:
arm A worst_ratio 1.81 (clear), gowerstreet A 1.32 (clear); arm B 2.14 —
fires, B stays descriptive as prereg'd.

## F — the DC correction (sandbox arm A adjudicating): **F-PARTIAL**

- **Marginals: ALL bars pass** (hc + e2e, octaves 2–4; worst margin e2e oct-3
  var_slope 8.0% vs 10.0%) — the correction spends nothing of the tail
  rescue (F-BREAKS did not fire).
- **T_coef oct-1: 15.3 → 7.5** (a 2.04× drop; ≥2× per the branch rule);
  oct-2: 4.0. **Parity T: 6.2** — the output residual remains.
- **The mechanism sharpens — the defect has TWO LAYERS:** after correction
  the surviving oct-1 components are corr(H,V) z = +7.5 and signV z = +3.5;
  the channel MEANS are gone (as designed — a constant subtraction removes
  exactly the mean layer). The residual parity is driven by a cross-channel
  CORRELATION the mean-fix cannot touch: no constant offset can remove a
  joint H–V dependence. F-PARTIAL's pre-registered meaning fires precisely:
  "not a pure DC offset — higher spatial structure in the bias; the M2 corr
  components say where."
- Arm B (descriptive): marginals pass; T_coef1 7.0; parity 2.9 — B's OUTPUT
  parity cleans under correction even as its coefficient statistic stays hot.

## J — joint selection (arm A): **J-TRADEOFF**

The frozen criterion picks step **16000** (marginal cage: 7500); val-side
T_coef 15.6 → 5.3. Test adjudication at 16000: **head-conditional oct-2
kurtososis 3.743 vs truth 4.917 (−23.9% vs the 15% bar) — FAILS.** By 16k the
tails have decayed (the moment ladder's rung-4 schedule) while the lattice
defect is only partly cleaned (test T_coef 5.2). At 1× the kurtosis-healthy
and symmetry-healthy checkpoint sets are DISJOINT: the schedule bind (R32's
named finding) now has a mechanical exhibit — 7500 = tails pass / defect
maximum; 16000 = defect reduced / tails fail. Reading note (as scored): the
J-WINDOW exists-quantifier is witnessed by the criterion's own val-side pick;
no test-side sweep of the curve was run (the test set is not for browsing).

## The corrected real-field echo (descriptive, LOAD-BEARING per R27/R32)

Gowerstreet arm A, before → after correction (vs the 32 real test tiles,
parent conventions disclosed as standing):
- T_coef oct-1: 7.5 → 4.6; oct-2: 4.3 → 3.1; parity T: 2.5 → 1.6.
- Marginals (descriptive): e2e oct-2 kurtosis 5.80 → 6.40 vs real 6.04 (the
  −4.0% deficit becomes +6.0% — sign flips, magnitude similar, inside every
  standing bar); var_slope 0.991 → 0.992 (+1.9% vs real, unchanged).
- **nn spacing statistic: T = 3.5 → 4.1 — the genuine position-pure spacing
  residual SURVIVES the symmetry fix.** The real field carries TWO
  independent joint-structure residuals: the lattice defect (partly curable,
  fully mechanistic) and a spacing anomaly that is NOT a parity artifact.
  Transfer clause honored: nothing here claims the real-field peak excess is
  cured; the echo's role was exactly to test that, and the answer is no.

## Scorecard (registered weights → outcomes)

| line | executor | reconvene | fired |
|---|---|---|---|
| F-CLEAN / F-PARTIAL / F-BREAKS / gate | 45/30/15/10 | 55/22/13/10 | **F-PARTIAL** — executor closer; the judge's F-CLEAN lean priced the mean layer, not the correlation layer |
| J-WINDOW / J-TRADEOFF / gate | 35/50/15 | 30/55/15 | **J-TRADEOFF** — both on the winning side; reconvene closer |
| stability gate (A) | 10 | 10 | did not fire (B's fired, descriptive) |

## Consequences for D (queued next per the prereg's order)

D's question is now two-layered: does data (8×/32×) shrink, at the caged
pick, (a) the mean layer (F showed it is also curable by calibration — D
tests whether data prevents it) and (b) the corr(H,V) layer (which no
constant cures — if data cleans it, the data lever reaches where calibration
cannot; if not, it is the first candidate for a genuinely architectural
residual). The A′ M5 curve says both decay by 17.5k at 1× — D-WIDEN asks
whether 8× tails stay healthy long enough to harvest that clean region
(reconvene 45 / executor 40). The corrHV component will be reported per arm
alongside T_coef. STOP — the reconvene adjudicates F/J; D submits after,
per the prereg's own order.
