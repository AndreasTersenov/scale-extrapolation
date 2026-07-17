# 2026-07-17 — PREREG: R15 tail-dynamics diagnosis (3 probes, CPU only)
# Committed BEFORE submission; R12 rule: every readout number is a verbatim copy
# from results_p2/taildyn_*.json.

Authorization: log/2026-07-17-reconvene-bakeoff.md R15. Runner:
scripts_p2/tail_dynamics.py. Eight jobs: {tbase, twcrps} × {t5flat, composite} ×
{n=768 (1×), n=6144 (8×)}, ALL at 12k steps / warmup 1.2k / cosine-to-12k /
lr 1e-3 / seed 2 / batch 16 / production net / eval every 500 (24 points, K=4,
held-out n=256 per the bake-off protocol). Dispersion (binned_sigma_maxrel,
validated tests_p2) and, for tbase, the Gaussian-base-through-the-same-flow
decomposition are logged at every eval.

## Design note (deviation from "same configs", justified)

The bake-off's 4k runs cannot serve as onset baselines for a ≥2× shift question:
the window is too short to see a shifted onset, and stretching the window reshapes
the cosine schedule (a confound). The causal design therefore re-runs the 1×
controls at the SAME 12k schedule as the 8× runs — data size is the ONLY
difference within each comparison pair. Onset ratios are computed within-schedule.

## Mechanical rules (pinned)

- **Decay onset** (per run): the eval step of the running maximum of held-out
  excess kurtosis, PROVIDED some later eval falls ≥25% below that maximum
  (a confirmed decay). No confirmed decay by 12k → onset censored at >12000.
- **P1 shift ratio** (per candidate, composite gate PRIMARY, t5flat secondary):
  onset(8×)/onset(1×). Censored 8× onset with a confirmed 1× onset ≤6000 counts
  as ratio ≥2 (SHIFT). A 1× run with NO confirmed decay makes that comparison
  unadjudicable (AMBIG input).
- **P2 joint-viability** (tbase runs; composite PRIMARY): does ANY eval step have
  disp_maxrel ≤ 0.10 AND kurt ≥ 4.0 simultaneously? (Bars: the C1/C3 dispersion
  convention and the R13 selector bar — continuity, not new inventions.)
- **P3** is descriptive: base-vs-flow tail gap = kurt(t-base out) −
  kurt(N-base out) per eval; ν is architecturally FIXED at 5 (no learnable df
  parameter exists), so drift of the realized tail order across checkpoints is
  attributable to the learned map.

## Branches and weights

**P1 (causal test; adjudicated on the composite gate, both candidates):**

| branch | definition (mechanical) | reconvene | executor |
|---|---|---|---|
| TD-SHIFT | ratio ≥2 for BOTH candidates | 60 (as "shift ≥2×") | **50** |
| TD-PARTIAL | ratio ≥2 for exactly one | — (folded) | **15** |
| TD-NOSHIFT | ratio <2 for both | 25 | **18** |
| TD-AMBIG | censoring/no-1×-decay prevents adjudication | 15 | **17** |

Executor reasoning: the gate-design investigation already showed a data-size-
dependent transit (n=192 overshoots fast; n=768 slower/lower) and the morning's
B2 finding (N_eff ≈ parents) supports rung-4's effective data being small — but
the 12k-schedule 1× baseline may not reproduce a clean ≤6k onset (hence AMBIG
17), and β=1's flat-t5 run showed objective-dependent dynamics (no transit at
all), so the mechanism may not be purely data-limited.

**P2 (joint near-truth checkpoint, tbase composite 1×):** reconvene 55% yes;
executor **60%** yes (dispersion recovers early and holds in every modulated-σ toy
so far; the kurt ≥4 window at 1× spanned ~1.5–2.5k in the bake-off; overlap
likely). Secondary (descriptive): same question on the 8× run.

**P3 expectation (descriptive, registered):** executor 65% on "flow-contracts-
base-tails" — N-base output kurt stays ≪ t-base output kurt early, and the gap
SHRINKS over training (the flow learns to undo the base's tails rather than the
base failing to inject them).

**Gate branches (the R15 process rule):** P(≥1 job needs resubmission for
infra reasons): **15** (one t=0 class already caught today); P(any run
DEGENERATE — NaN/amplitude blowup — requiring exclusion): **5**.

## Deliverables

Eight taildyn_*.json artifacts; readout log/2026-07-17-taildyn-readout.md with
the P1 verdict table (onsets + ratios, verbatim), the P2 joint-viability answer
with the qualifying checkpoint(s), the P3 decomposition read, and a trajectory
figure. STOP at the readout — the next-arm decision is the reconvene's, with
this diagnosis in hand.
