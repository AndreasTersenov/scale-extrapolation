# 2026-07-11 — reconvene ruling: (c') option 2 APPROVED

Evidence accepted: two independent training-time penalties (random-t (c), late-t (c'-1))
failed the frozen bar identically — the deterministic-ODE pushforward variance is the
structural limit, not the loss weighting. Escalation to option 2 (Gaussian-NLL detail
head: model conditional variance explicitly, sample with it) is APPROVED as
pre-registered, with two conditions:
1. **Arm symmetry:** BOTH arms A and B get the new head — the P5/P6 comparison must
   keep scale-conditioning as the only differing variable.
2. **Pre-register the sampling procedure** (how the predicted log-variance enters
   generation) and the same frozen bar: trained-octave var_slope within 1σ at octaves
   2–4 simultaneously, GRF null preserved (the NLL head must NOT break the null —
   Gaussian details on GRF should be learned as such). Then P6/P13 re-adjudication,
   frozen bars unchanged.
If option 2 also fails the bar, the finding "deterministic and variance-headed CFM
both under-disperse" is itself the paper's methods result — report and reconvene.
