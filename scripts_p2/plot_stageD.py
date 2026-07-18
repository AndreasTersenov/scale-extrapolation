"""Stage D + R18 replication figure. Parses committed jsons only (R12).

(a) THE EDGE: octave-2 verdict — generated vs real, both arms, both levels, with
    the bare floors shown (the honest bars; the formal SE-widened bars are
    stated in-caption);
(b) selection curves at octave 3 (the cage, edge untouched) with picks;
(c) peak audit at the edge: the dial flips the high-threshold sign;
(d) R18 replication: all 24 rel-errors vs their bars (log-log), everything
    under the line.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")
V = json.load(open(os.path.join(RES, "stageD_verdict.json")))
SEL = json.load(open(os.path.join(RES, "c1t_selection_stageD.json")))
REF3 = json.load(open(os.path.join(RES, "gowerstreet_val_ref_oct3.json")))
REPL = json.load(open(os.path.join(RES, "c1t_repl64.json")))

fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.2))

ax = axes[0][0]
real_vs = V["levels"]["A"]["head-conditional"]["var_slope"]["real"]
real_k = V["levels"]["A"]["head-conditional"]["kurtosis"]["real"]
labels, vals_vs, vals_k, colors = [], [], [], []
for arm, color in (("A", "tab:purple"), ("B", "tab:red")):
    for lv, tag in (("head-conditional", "hc"), ("end-to-end", "e2e")):
        labels.append(f"{arm} {tag}")
        vals_vs.append(V["levels"][arm][lv]["var_slope"]["value"])
        vals_k.append(V["levels"][arm][lv]["kurtosis"]["value"])
        colors.append(color)
x = np.arange(4)
ax2 = ax.twinx()
ax.bar(x - 0.18, vals_vs, 0.36, color=colors, alpha=0.55)
ax2.bar(x + 0.18, vals_k, 0.36, color=colors)
ax.axhline(real_vs, color="k", lw=1.2, ls="--")
ax2.axhline(real_k, color="k", lw=1.4)
ax.axhspan(real_vs * 0.9, real_vs * 1.1, color="tab:green", alpha=0.10)
ax2.axhspan(real_k * 0.85, real_k * 1.15, color="tab:green", alpha=0.10)
ax.set_xticks(x, labels)
ax.set_ylabel("var_slope (left bars, dashed line = real)")
ax2.set_ylabel("kurtosis (right bars, solid line = real)")
ax.set_title("THE EDGE (held-out octave 2): all 8 checks pass\neven at the bare "
             "10%/15% floors (green bands)", fontsize=10)

ax = axes[0][1]
for arm, color in (("A", "tab:purple"), ("B", "tab:red")):
    cv = SEL[arm]["curve_val"]
    steps = sorted(cv, key=int)
    ax.plot([int(s) / 1000 for s in steps], [cv[s]["kurtosis"] for s in steps],
            "-", color=color, lw=1.5, label=f"arm {arm} (oct-3 val kurtosis)")
    sel = SEL[arm]["selected_step"]
    ax.plot(sel / 1000, cv[str(sel)]["kurtosis"], "*", ms=16, color=color,
            mec="k", label=f"arm {arm} PICK @{sel}")
ax.axhline(REF3["truth"]["3"]["kurtosis"], color="k", lw=1.3,
           label=f"real oct-3 val ref {REF3['truth']['3']['kurtosis']:.2f}")
ax.set_xlabel("training steps (k)")
ax.set_ylabel("validation kurtosis, octave 3")
ax.set_title("The cage, moved off the edge: selection sees\nonly octave 3 — "
             "octave 2 stays deployment-blind", fontsize=10)
ax.legend(fontsize=7.5)
ax.grid(alpha=0.25)

ax = axes[1][0]
nus = ("1.0", "2.5", "3.0")
x = np.arange(3)
for i, (arm, color) in enumerate((("A", "tab:purple"), ("B", "tab:red"))):
    exc = [V["peaks"][arm][nu]["excess"] * 100 for nu in nus]
    ax.bar(x + (2 * i - 1) * 0.19, exc, 0.38, color=color, label=f"arm {arm}")
ax.axhline(0, color="k", lw=1.2)
ax.set_xticks(x, [f"ν = {n}" for n in nus])
ax.set_ylabel("peak-count excess vs real (%)")
ax.set_title("Peak audit at the edge: the dial FLIPS the high-ν sign\n"
             "(A keeps the up-tilt; B over-corrects) — neither closes it",
             fontsize=10)
ax.legend(fontsize=8.5)
ax.grid(alpha=0.25, axis="y")

ax = axes[1][1]
for arm, marker in (("A", "o"), ("B", "s")):
    for lv, color in (("head-conditional", "tab:blue"), ("end-to-end", "tab:orange")):
        for j in ("2", "3", "4"):
            for m in ("var_slope", "kurtosis"):
                r = REPL["levels"][arm][lv][j][m]
                ax.plot(r["bar"] * 100, r["rel_err"] * 100, marker, color=color,
                        ms=6, alpha=0.75)
lim = [0, 60]
ax.plot(lim, lim, "k--", lw=1.2, label="rel. error = bar")
ax.fill_between(lim, lim, [lim[1]] * 2, color="tab:red", alpha=0.06)
ax.set_xlim(*lim); ax.set_ylim(*lim)
ax.set_xlabel("pre-registered bar (%)")
ax.set_ylabel("observed relative error (%)")
ax.set_title("R18 replication, 64 fresh fields: all 24 checks\nunder the line — "
             "C1T-CAL replicated (circles A, squares B;\nblue hc, orange e2e)",
             fontsize=10)
ax.legend(fontsize=8.5)
ax.grid(alpha=0.25)

fig.suptitle("STAGE D — the original bet, answered: single-octave scale extrapolation WORKS on the calibrated "
             "substrate under the deployment protocol, and works SCALE-BLIND (the dial adds nothing marginal; "
             "B1's r*≈1 confirmed at deployment). Plus the required 64-field replication.",
             fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
out = os.path.join(RES, "stageD.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
