# 2026-07-17 — R15 TAIL-DYNAMICS READOUT: the rung-4 decay is DATA-LIMITED
# (8× data holds tails AT truth for 12k steps); P2 joint checkpoint EXISTS;
# P3: the 1×-trained flow erases its base — 8× preserves it

Jobs 16613873–880, all COMPLETED (30–82 min), no resubmissions, no degenerates
(both gate branches: not fired). Prereg log/2026-07-17-prereg-tail-dynamics.md
(69f9a99, committed pre-submission); every number below is a verbatim copy from
results_p2/taildyn_*.json; figure results_p2/taildyn.png.

## P1 — the rung-4 causal test

Mechanical verdicts by the PINNED onset rule (running-max step confirmed by a
≥25% later drop):

| candidate | gate | onset 1× | onset 8× | ratio | verdict |
|---|---|---|---|---|---|
| tbase | composite (PRIMARY) | 500 | 3000 | 6.00 | SHIFT |
| twcrps | composite (PRIMARY) | 2500 | 4500 | 1.80 | NOSHIFT |
| tbase | t5flat (secondary) | 500 | 500 | 1.00 | NOSHIFT |
| twcrps | t5flat (secondary) | 2500 | censored | ≥4.8 | SHIFT |

**Primary branch: TD-PARTIAL** (exactly one candidate ≥2× on the composite gate)
— executor's 15-weight branch fires; executor modal TD-SHIFT 50 and reconvene 60
both MISS at the strict both-candidates pairing.

**The final-state pattern the onset rule under-reports (descriptive, and the
substantive answer to R15's question).** Last-3-eval mean held-out kurtosis:

| run | 1× final | 8× final | truth |
|---|---|---|---|
| tbase composite | 1.66 | **6.11** | 5.96 |
| tbase t5flat | 1.08 | **4.32** | 4.15 |
| twcrps composite | 1.66 | 4.78 | 5.96 |
| twcrps t5flat | 1.07 | **4.57** | 4.15 |

Every 1× run terminally collapses; at 8× three of four runs sit AT truth and hold
there flat for thousands of steps (tbase composite: 5.84–6.28 over steps
6k–12k), the fourth reaches 80% of truth and holds. At 8× the decay is not merely
delayed — within this window it is largely ABSENT. This is rung 2's cure
reproduced at rung 4 by the same causal lever (data size), which is what the
moment-ladder hypothesis predicted. Rule-design lesson for the ledger: the pinned
onset rule reads transient DIPS as decay onsets (tbase t5flat 8× "onset 500" is a
dip the run recovers from to finish at truth) and cannot distinguish dip-and-
recover from terminal collapse — a final-state statistic adjudicates this
question more faithfully; logged for the reconvene's rule library, not
re-adjudicated tonight.

## P2 — checkpoint viability: YES (executor 60, reconvene 55 — both hit)

tbase composite 1×: **6 qualifying evals (steps 1000–3500)** with disp_maxrel
≤ 0.10 AND kurt ≥ 4.0 jointly (e.g. step 1000: kurt 4.43, disp 0.085). tbase
t5flat 1×: 4 (steps 2500–4000). At 8×: 24/24 (composite) and 19 (t5flat) — the
entire trajectory is jointly viable. A pre-registered validation-selected
early-stop is therefore a live lever at 1×, and at 8× no early-stop is even
needed within 12k.

## P3 — attribution: the flow ERASES its base at 1×; 8× PRESERVES it

ν is architecturally fixed at 5 (no learnable df parameter). Decomposition
(kurt of t-base output / kurt of N-base output through the SAME checkpoint):

- 1× composite: 500: 5.66/0.31 → 3500: 4.47/0.58 → 6500: 1.65/1.09 → 11000:
  1.66/1.58 — the two bases CONVERGE to the same ~1.6 endpoint: the trained map
  becomes base-INDEPENDENT, reproducing a fixed lighter-tailed conditional
  regardless of what enters. (Registered "flow-contracts-base-tails" 65 — fired,
  with this sharper form: contraction proceeds to base erasure.)
- 8× composite: t-out 5.4–6.3 vs N-out 0.26–0.48 throughout; final 6.05/0.42 —
  the base's tail contribution SURVIVES training, stable gap to 12k.

Base erasure is thus a symptom of the data-starved regime (consistent with
memorization of the finite conditional ensemble), not an intrinsic property of
the flow — the cleanest mechanistic link yet between the moment-ladder decay and
effective data size.

## Candidate-relevant side-finding (descriptive)

twcrps' spurious skew on symmetric targets WORSENS with data: 1× 12k-schedule
finals are small (+0.051/+0.050) but 8× runs at +0.79…+1.11 (composite) and
+0.24…+0.59 (t5flat), persistent to 12k. Whatever the next-arm decision, the
chained score carries a data-scaling asymmetry pathology the t-base flow does
not (t-base final skews: +0.014…+0.023 at 8×).

## Scorecard

P1: TD-PARTIAL — executor 15 fires (modal TD-SHIFT 50 miss); reconvene 60 miss
at the strict pairing, though the substantive claim their weight expressed (the
mechanism is data-limited) is what the final-state pattern confirms — scoring
ambiguity noted rather than resolved in either side's favor. P2: both hit. P3
expectation: fired, sharpened. Gate branches (infra 15 / degenerate 5): neither
fired.

STOP at this readout per R15 — the next-arm decision (early-stop lever vs
data-scaled arm vs architecture change) is the reconvene's, with this diagnosis
in hand. Executor's one-line read for that discussion: the diagnosis favors
data-side levers (the 8× augmentation analogue — e.g. more parents/tiles or
D4-beyond augmentation for rung 4) or the t-base + validation-early-stop combo;
twcrps is disfavored by its skew pathology.
