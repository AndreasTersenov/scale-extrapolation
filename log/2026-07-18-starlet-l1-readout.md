# 2026-07-18 — STARLET-ℓ1 READOUT: **SL1-PASS on all three legs (arm A, every
# scored scale) — and the tier question FIRES: the field's constraining
# position-blind statistic does not flag the known peak-placement excess.**

Prereg log/2026-07-18-prereg-starlet-l1.md (fa56ff4, committed before scoring).
No SLURM job — CPU-light scoring from frozen committed npz stacks. Artifacts:
results_p2/starlet_l1_{sandbox,gowerstreet,edge,taxonomy}.json,
starlet_l1_instrument.json, figure results_p2/starlet_l1.png. All numbers
verbatim from those JSONs (R12).

## Verdict table (arm A adjudicates; total-ℓ1 rel. error vs the 1σ-w-10%-floor
## bar; all bars floored at 10.0%)

| leg | 2px (oct 1, descr.) | 4px (oct 2) | 8px (oct 3) | 16px (oct 4) | scored |
|---|---|---|---|---|---|
| sandbox truth-ref | −2.0% | +1.1% | +0.0% | −0.1% | **PASS** |
| gowerstreet trained | −0.9% | +0.6% | +1.5% | −0.1% | **PASS** |
| Stage-D edge | +5.4% | **+4.0% (THE EDGE)** | +3.0% | +1.8% | **PASS** |

Arm B (descriptive): passes all scored scales on all three legs (sandbox
−8.5/−5.1/−2.0; gowerstreet +1.8/+2.6/+0.8; edge −2.3/−1.9/−0.9) but its two
known off-binding pathologies reappear in the held-out basis at the
descriptive octave-1 scale: sandbox 2px **−11.4%** (the tail-hot arm's
fine-scale deficit) and edge 2px **+23.9%** (the two-octave-extrapolation
blow-up) — both outside the 10% floor. Consistent with R19's "arm A =
reference".

## The tier question (P-SL1-blind): FIRED

Arm A passes every scored scale on legs 2 AND 3 while the committed peak audit
fails there (stageD_verdict.json arm A: +13%/+15% at ν=2.5/3). The paper gains
the pre-registered sentence: **even the field's most constraining summary sits
below the placement tier — position-blind statistics cannot see where the
extremes land.** Honest descriptive nuance: the edge 4px tail share (ℓ1 in
|SNR|≥3 bins) runs 0.1680±0.0020 (A) vs real 0.1587±0.0045 — a ~+5.9% rel
tilt, same direction as the peak excess, visible but below the scored bar; the
amplitude histogram sees a *hint* of the excess that peak counts amplify to
+13/+15%. §4's tier ordering stands unrevised.

## Taxonomy panel (descriptive): the statistic separates the diseases

Total-ℓ1 rel. vs the 64-field real reference (arm A of each generation):

| generation | 2px | 4px | 8px | 16px |
|---|---|---|---|---|
| 4a NLL-head | **+14.2%** | −2.5% | −2.0% | +2.2% |
| 4a mu-only skeleton | **−83.8%** | **−57.7%** | **−22.1%** | −1.0% |
| C1 (Gaussian base) | +1.8% | +0.9% | +0.1% | −1.3% |
| C1-t | +0.8% | −0.4% | +0.3% | −0.2% |

- The 4a graininess disease has a clean ℓ1 signature: +14.2% total at 2px with
  the tail share COLLAPSED (0.0237 vs real 0.0620) — excess amplitude carrying
  no extremes. The mu-only skeleton shows the complement: fine-scale amplitude
  lives in the sampler's noise term (−84% at 2px).
- C1 vs C1-t — the tail rescue — is BELOW this statistic's resolution in
  totals (both ≤1.8%); only the 4px tail share ranks them (real 0.1252±0.0040,
  C1-t 0.1154±0.0031, C1 0.1094±0.0019): C1-t closer to real, same ordering
  the kurtosis bars established, but only at ~1.7σ between the two arms here.

## Secondary convention (survey-like noise, σ_n = 2σ_real, descriptive)

All totals within ±3.5% (gowerstreet A: −0.4/−0.1/+2.4/+1.9; edge A:
+0.2/+1.0/+3.3/+3.5; B edge 2px note: the +23.9% noiseless excess is
noise-dominated away to +1.8%). Conclusions robust to the thesis-world
convention.

## Instrument findings (for Andreas, from starlet_l1_instrument.json)

1. `verify_installation.py` fails its own Starlet check: `reconstruct(gen2=True)`
   error 1.41 on a unit-variance GRF — the gen2 filter doesn't match the gen1
   forward transform (which round-trips at 4.4e-16 via sum-of-scales). Unused
   by ℓ1; the package's own suite ("45 passed, 1 skipped") avoids it.
2. `get_noise_levels` uses zero-padded linear convolution while the forward
   transform is periodic: on unmasked sim tiles the shipped SNR plane carries
   border artifacts (noise-plane cv 0.107 at 2px; rot90 commutation error up
   to 8.8 in SNR units) and would fail a D4 gate. Adopted convention: interior-
   plateau scalar normalization (matches the analytic B3 table
   0.891/0.201/0.0855/0.0412 at σ=1; rot90-exact at 4.5e-15). A patch making
   noise propagation periodic (or masking borders) would make the shipped SNR
   D4-clean for sim use.

## Scorecard (verdict vs pre-registered expectations)

| line | reconvene | executor | outcome |
|---|---|---|---|
| P-SL1-trained | 65 | 80 | FIRED — both hit; executor tighter |
| P-SL1-edge | 55 | 70 | FIRED — both hit; executor tighter |
| P-SL1-blind | 70 | 75 | FIRED — both hit |
| gate: install/env failure | 10 | 2 | didn't fire |
| gate: second primary forced | 5 | 5 | didn't fire (plateau choice was a pre-scoring refinement INSIDE the declared primary, not a second primary) |

First application of the R20 recalibration (constructive branches priced high)
— it held. The five-line sheet is the first clean sweep of the campaign.

## What this adds to the paper (§4/§6)

§4 gains the taxonomy panel (one statistic, four generations, three diseases
separated) and the tier sentence above. §6 gains: the calibrated substrate
passes the field's standard constraining summary out-of-basis at every scored
scale including the deployment edge (headline +4.0% against a 10% floor), with
conclusions stable under the survey-noise convention. The residual frontier is
unchanged and now sharper: what remains is placement, not amplitude. STOP;
next assignment to Andreas/reconvene.
