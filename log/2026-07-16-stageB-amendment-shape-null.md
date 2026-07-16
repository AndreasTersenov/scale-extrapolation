# 2026-07-16 — Stage B PRE-READING AMENDMENT: the shape-test null is NOT zero;
# P-B1b verdict becomes difference-in-differences against the sandbox control

**Timing disclosure:** the Stage-B job (16491648) has finished the shape section.
The ONLY shape numbers seen at amendment time are the two SANDBOX control lines that
scrolled in the job tail — `sandbox oct2 absw: z=-3.09`, `sandbox oct2 w: z=-1.13`.
The gowerstreet shape numbers exist on disk but have NOT been read; this amendment
is committed before reading them.

## What the control caught (and my error)

The prereg claimed Δ_align = V_misaligned − V_aligned is "exactly zero in population
under isotropy (exchangeability)". That argument is WRONG, and the isotropic sandbox
control measured it: z = −3.09 (absw). What 90°-exchangeability actually gives is
the CROSS-CLASS equality E[V|class 0] = E[V|class 2] — it does NOT make the
within-class aligned-vs-misaligned difference vanish, because conditioning on the
orientation classifier SELECTS positions whose local noise texture is genuinely
anisotropic even in an isotropic field. On such selection-oriented texture the
along-structure direction is locally smoother/more redundant, which can favor the
MISALIGNED mask — a negative Δ_align with no transported structure present. The
in-test isotropic null (64 fields) passed only for lack of power; the estimand's
true isotropic baseline is a small negative number.

## Amendment (rule change BEFORE reading gowerstreet; bar form otherwise kept)

- **P-B1b fires iff** (gowerstreet octave 2, absw channel):
  Δ_align^gow − Δ_align^sandbox > 3·sqrt(SE_gow² + SE_sand²)
  — anisotropy beyond the isotropic-selection baseline, sandbox as the measured
  control. (The original ">3·SE vs 0" reading is reported alongside, labeled with
  this amendment.)
- The prereg's "sandbox control NULL (90%)" prediction is scored as a MISS against
  its literal wording (z=−3.09 is not null) — logged as such; the miss is the
  estimand-design error documented above, caught by the control as designed.
- Limitation stated for the readout: sandbox and gowerstreet differ in field
  structure, so the selection-baseline subtraction is approximate (same machinery,
  same gating, same masks — but selection-effect size need not transfer exactly).
  The morning reconvene may prefer a matched-isotropy control (e.g. Gaussianized
  gowerstreet — phase-randomized tiles) as a follow-up; drafted in the summary,
  not run tonight.
- test_shape docstrings claiming an exact exchangeability null are corrected in the
  same commit (the tests' numeric tolerances already accommodated the small
  baseline; their gates remain valid).
