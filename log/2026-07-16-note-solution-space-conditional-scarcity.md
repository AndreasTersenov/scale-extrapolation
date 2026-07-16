# 2026-07-16 — NOTE: the solution space for one-sample-per-condition (brainstorm w/ Andreas)

The root problem, named: each coarse map appears ONCE in training; conditional
spread exists only via generalization across similar environments; training
dynamics erode it (the collapse law). Candidate solution families for any FUTURE
generator phase (Option-3 prospectus; the freeze stands — pre-registered arms only):

1. **Constrained-realization conditional ensembles (gold standard; needs own sims →
   3D phase).** Fix large-scale IC modes, randomize small-scale modes, re-run: true
   samples of p(fine|coarse) by construction. Even a few hundred coarses × ~5
   details each gives measured conditional spread for training AND validation truth.
   A training set DESIGNED for conditional learning.
2. **Locality / receptive-field dial (cheapest, theory-backed — WC-RG short-range
   theorem).** Local coarse ENVIRONMENTS repeat hundreds of times even when maps are
   unique; a receptive field larger than the physical interaction range is pure
   memorization capacity. Arm: conditioning-range sweep vs collapse onset +
   end-to-end. MIG-minutes.
3. **Overlapping crops** (stride-shifted tiling of parent maps): ~10–50× correlated
   law-preserving pairs; stacks with the 8× symmetry augmentation.
4. **Proper scoring rules (CRPS / energy score) instead of NLL/regression** — the
   probabilistic-weather answer to literally this problem (one observation per
   condition, calibrated ensembles required; GenCast lineage — fetch-verify).
   Contained objective swap; already inside the Gate-0 kill-test reading scope.
5. **Unconditional prior + inference-time conditioning (posterior sampling).**
   Coarse-graining is a known exact LINEAR operator → detail-given-coarse is a
   linear inverse problem. Train an unconditional model of fine tiles (every tile =
   i.i.d. marginal sample — the pairing scarcity dissolves); condition at sampling
   via DPS-style guidance or SMC (guarantees available; guidance bias is measurable
   by our existing instruments). Prior memorization remains (same cures apply).
6. Honorables: texture-critic adversarial finetuning (blandness fix; mode-seeking
   vs calibration tension — needs the audit as a leash); early-stop at dispersion
   peak + capacity control (small, stackable). Target-side noise: rejected
   (re-invents diffusion, biases the law).

Ranking for a future phase: un-run vanilla-CFM+augment arm (see companion note) →
locality dial → energy score → prior+posterior-sampling → constrained ensembles.
Credit: Andreas's brainstorm (multiple-fine-per-coarse; smart augmentations) —
items 1 and 3 are his directly; 2 fell out of pressing on WHY generalization works
at all.
