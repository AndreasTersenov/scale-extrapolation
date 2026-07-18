"""Stage D deployment-protocol dial: extrapolate the gowerstreet COUPLING CURVE
from trained-visible octaves into the held-out edge (no measured target-octave
couplings — PLAN-phase2 §4).

Mechanical, pinned in the Stage-D prereg BEFORE results: per coupling component
(var_slope, kurtosis), fit BOTH a linear and a log-linear curve in octave index on
the trained-visible octaves j in {5, 4, 3} (all computable from training-scale
data), pick the form with the smaller in-sample residual sum of squares, and
extrapolate to j = 2 (the held-out edge) and j = 1. Measured values at j >= 3 are
kept verbatim. Writes data_cache/running_couplings_stageD.json and prints every
choice for the record.
"""
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = json.load(open(os.path.join(REPO, "data_cache", "running_couplings.json")))
FIELD = "gowerstreet"
FIT_J = np.array([5.0, 4.0, 3.0])
OUT_J = (2, 1)

meas = {int(j): v for j, v in SRC[FIELD].items()}
out = {str(j): list(meas[j]) for j in (5, 4, 3)}
report = {}
for ci, name in enumerate(("var_slope", "kurtosis")):
    y = np.array([meas[int(j)][ci] for j in FIT_J])
    lin = np.polyfit(FIT_J, y, 1)
    rss_lin = float(np.sum((np.polyval(lin, FIT_J) - y) ** 2))
    lg = np.polyfit(FIT_J, np.log(y), 1)
    rss_log = float(np.sum((np.exp(np.polyval(lg, FIT_J)) - y) ** 2))
    form = "linear" if rss_lin <= rss_log else "log-linear"
    for j in OUT_J:
        val = (float(np.polyval(lin, j)) if form == "linear"
               else float(np.exp(np.polyval(lg, j))))
        out.setdefault(str(j), [None, None])[ci] = val
    report[name] = {"form": form, "rss_linear": rss_lin, "rss_loglinear": rss_log}
    print(f"{name}: {form} (RSS lin {rss_lin:.4g} vs log {rss_log:.4g}) -> "
          f"j=2: {out['2'][ci]:.4f}, j=1: {out['1'][ci]:.4f}")

with open(os.path.join(REPO, "data_cache", "running_couplings_stageD.json"),
          "w") as f:
    json.dump({FIELD: out, "_meta": {"rule": "per-component linear vs log-linear "
              "by in-sample RSS on j={5,4,3}; measured j>=3 verbatim; "
              "extrapolated j=2,1", "report": report}}, f, indent=1)
print("wrote data_cache/running_couplings_stageD.json")
