"""Stage-B milestone figure: predictability saturation (the locality result) and the
shape-test verdict with its isotropic baseline."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
curves = json.load(open(os.path.join(REPO, "results_p2", "stageB1_curves.json")))
shape = json.load(open(os.path.join(REPO, "results_p2", "stageB1_shape.json")))

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

# --- panels 1+2: V(r)/V(0) per octave, gowerstreet and sandbox
for ax, name in zip(axes[:2], ("gowerstreet", "sandbox")):
    for j, colr in zip("1234", ("tab:blue", "tab:orange", "tab:green", "tab:red")):
        row = curves[name][j]
        grid = sorted(int(r) for r in row["w"].keys())
        v0 = row["w"]["0"]["ridge"]
        vals = [row["w"][str(r)]["ridge"] / v0 for r in grid]
        ses = [row["w"][str(r)]["ridge_se"] / v0 for r in grid]
        ax.errorbar(grid, vals, yerr=ses, fmt="o-", color=colr, ms=3,
                    label=f"octave {j} (mean channel)")
        a0 = row["absw"]["0"]["ridge"]
        avals = [row["absw"][str(r)]["ridge"] / a0 for r in grid]
        ax.plot(grid, avals, "--", color=colr, alpha=0.45)
    ax.axhline(1.0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel("coarse-context radius r (coarse pixels)")
    ax.set_ylabel("V(r) / V(center only)")
    ax.set_title(f"{name}: conditional variance vs context radius\n"
                 "(solid: detail; dashed: |detail| amplitude channel)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    ax.set_ylim(0.78, 1.06)

# --- panel 3: shape-test deltas relative to V, with the sandbox baseline
labels, vals, errs, colors = [], [], [], []
for name, j, t in (("gowerstreet", "2", "absw"), ("sandbox", "2", "absw"),
                   ("gowerstreet", "2", "w"), ("sandbox", "2", "w")):
    r = shape[name][j][t]
    V = r["V_aligned"]
    labels.append(f"{name[:4]} oct{j} {t}")
    vals.append(100 * r["delta_align"] / V)
    errs.append(100 * r["delta_align_se"] / V)
    colors.append("tab:red" if name == "gowerstreet" else "tab:gray")
ax = axes[2]
xs = np.arange(len(labels))
ax.bar(xs, vals, yerr=errs, color=colors, capsize=4, alpha=0.8)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(xs)
ax.set_xticklabels(labels, rotation=20, fontsize=8)
ax.set_ylabel("aligned-mask advantage  (% of V)")
ax.set_title("shape test: aligned vs misaligned contexts\n"
             "(gray = isotropic sandbox control = the measured baseline)",
             fontsize=10)
ax.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = os.path.join(REPO, "results_p2", "stageB.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
