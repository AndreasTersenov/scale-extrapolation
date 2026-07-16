"""Phase-2 Stage B1: model-free measurement of the conditional dependence range/shape.

Numpy/scipy only. Two complementary estimators of the predictability-saturation curve
V(r) = per-band conditional detail variance given the coarse context within radius r:
ridge on raw patch pixels (linear predictability; exactly validatable on a GRF) and
k-NN on annulus-summary features (leading nonlinear gain; also exactly validatable on
a GRF because annulus means are linear functionals). See predictability.py docstring
for the estimand's precise definition and its pre-declared limits.
"""
from .predictability import (
    analytic_grf_vr,
    disk_offsets,
    knn_vr,
    predictability_curve,
    ridge_vr,
)
from .masks import (
    ellipse_offsets, elongated_offsets, match_area, orientation_class,
)
