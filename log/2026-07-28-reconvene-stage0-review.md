# 2026-07-28 — RECONVENE: Stage-0 prereg review (Phase 3) — APPROVED with
# amendment A1; reconvene lines registered; executor cleared to run

Reviewed: log/2026-07-28-prereg-stage0-phase3.md (commit 62db5f0). Same-day
git log of this repo taken before ruling (stale-view rule). No Stage-0
artifacts exist; the STOP was honored. The running cluster job 17613540
belongs to cosmo-sonar (WorkDir checked) — not this project.

## R12 verification

The prereg's reference numbers were checked verbatim against
results_p2/audit_peak_ci.json, stage_d leg, arm A:
ν=2.5 excess 13.1286% se 3.1042% (prereg: +13.13±3.10 — MATCH);
ν=3.0 excess 14.6172% se 3.0778% (prereg: +14.62±3.08 — MATCH).

## Amendment A1 (binding): the sign-flip guard

Measured motivation, not feel: the committed trained-scale legs are already
significantly NEGATIVE — repl64 arm A: −6.79%±1.69 (z=−4.01) at ν=2.5,
−9.37%±1.69 (z=−5.55) at ν=3.0; sandbox32 arm A: −4.93%±2.14 (z=−2.31),
−9.07%±2.08 (z=−4.36) <!-- src: audit_peak_ci.json -->. If the F2 average
removes a lattice-manufactured peak component at the edge, overshoot into a
significant deficit is a live outcome. Under the committed rules a
significant deficit would satisfy "eν < rν/2 AND Δν ≥ 2" and masquerade as
S0-SHRUNK — the wrong meaning entirely.

**Rule:** if either eν's 95% CI lies entirely below 0, none of
S0-CLOSED/SHRUNK/UNCHANGED applies. Record **S0-FLIPPED**, apply the
pre-authorized single disambiguation (one regeneration, fresh sampler PRNG),
re-apply once; if the deficit persists, STOP and report it as its own
finding (it would mean the edge excess and the trained-scale deficit share a
mechanism the symmetrization exposes — Stage-1 material, but a different
Stage 1 than either SHRUNK or UNCHANGED implies).

This is the 10th unlisted-branch instance priced BEFORE the run rather than
after — the ledger working as intended.

## Note N1 (recorded, no rule change)

Δν = (rν−eν)/hypot(sν, sν_ref) treats the two excess measurements as
independent, but both are scored against the same real test tiles, so the
real-side bootstrap variance is common-mode. hypot therefore overstates the
SE of the reduction and biases the SHRUNK/UNCHANGED boundary toward
UNCHANGED. Direction-safe (ambiguity = negative); recorded now so the
readout does not relitigate a visually large reduction that lands Δν < 2.

## Reconvene weights (registered at review, per standing rule)

| branch | executor | reconvene |
|---|---|---|
| S0-CLOSED | 22 | 12 |
| S0-SHRUNK | 43 | 38 |
| S0-UNCHANGED | 25 | 30 |
| S0-FLIPPED (A1) | — (unpriced) | 10 |
| gates | 10 | 10 |

Reasoning in brief: the post-diagnosis-cure record (5-for-5) argues for
CLOSED/SHRUNK mass, but F2 was diagnosed against the parity defect, and the
parity→edge-peak-excess causal link was never established — the edge showed
only weak level-1 parity (T=2.4) while the excess is z≈4.2–4.7. Full
closure therefore needs the coarser block-level bias (T=3.7/4.1) to have
been the peak producer; possible, not modal. The nn spacing residual's
survival through four cures says genuine un-modeled joint structure exists
at the edge; the excess plausibly shares in it → UNCHANGED gets real mass.

Descriptive expectations co-signed: nn persists P=85 (agree, 85); starlet-ℓ1
edge leg holds P=90 (agree, 90); arm B worse where they differ P=75 (mine:
70 — the group average may compress inter-arm differences).

## Clearance

Prereg APPROVED as committed + A1. The executor may submit the one MIG job
per the prereg (zero training; artifacts → results_p2/stage0_p3_*; frozen
instruments only; identity gate re-asserted on the Stage-D substrate; seeded
group assignments recorded). STOP at the readout stands — reconvene
adjudicates before any sequencing.
