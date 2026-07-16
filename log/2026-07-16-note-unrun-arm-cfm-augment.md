# 2026-07-16 — NOTE: the un-run arm (vanilla CFM + augmentation), surfaced by Andreas

Andreas's methodology question ("isn't the model supposed to model the actual
conditional? mean + Gaussian spread feels unjustified") exposed a sequencing gap in
the attempt tree:

1. Vanilla CFM (full-distribution model, no Gaussian restriction) collapsed →
   retreat to the explicit Gaussian-NLL head (cprime2 ruling, with the pre-registered
   kurtosis check that later FAILED, confirming the family is too tame).
2. The collapse's actual cure (8× D4 symmetry augmentation, causal test, 4a) was
   found AFTER the retreat — and was only ever applied to the RESTRICTED model
   (all 4a-era arms ran --nll-head --augment).
3. **Vanilla CFM + augmentation was never run.** The collapse law is
   channel-invariant and the cure removes the mechanism's cause, so it is a live
   possibility that the unrestricted model, given the data fix, keeps both its
   dispersion AND its heavy tails (kurtosis) — no Gaussian crutch, no student-t
   needed.

Status: the generator remains FROZEN (attempt-5 finality clause; this note reopens
nothing). Filed as the FIRST pre-registered arm of any future generator phase
(RESHAPE-MEMO.md Option 3 prospectus), ahead of the student-t head. Credit: the gap
was found by Andreas pressing on the model-class justification — the mode-2 challenge
doing its job again.
