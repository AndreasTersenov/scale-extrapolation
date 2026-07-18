"""Arm C1-t readout figure. Parses committed jsons only (R12).

(a) sandbox VALIDATION selection curves (oct-2 kurtosis) for C1-t arms A/B with
    the picks marked, overlaid with C1's Gaussian-base checkpoints on the SAME
    protocol (the base-effect + selection-effect in one panel);
(b) sandbox verdict: kurtosis per octave, both levels, vs exact truth (15% band);
(c) gowerstreet e2e kurtosis deficits by octave: C1 vs C1-t (the deficit-halving
    question, answered per arm);
(d) gowerstreet validation selection curves — the cage picks LATE here (the rule
    adapts per regime).
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")
SEL_S = json.load(open(os.path.join(RES, "c1t_selection_sandbox.json")))
SEL_G = json.load(open(os.path.join(RES, "c1t_selection_gowerstreet.json")))
ATTR = json.load(open(os.path.join(RES, "c1_tails_val.json")))
VERD = json.load(open(os.path.join(RES, "c1t_verdict_sandbox.json")))
GOW = json.load(open(os.path.join(RES, "c1t_gow_descriptive.json")))
TRUTH = json.load(open(os.path.join(RES, "sandbox_truth_normconv.json")))["truth"]
GREF = json.load(open(os.path.join(RES, "gowerstreet_val_ref.json")))["truth"]["2"]

fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))

ax = axes[0][0]
for arm, color in (("A", "tab:purple"), ("B", "tab:red")):
    cv = SEL_S[arm]["curve_val"]
    steps = sorted(cv, key=int)
    ax.plot([int(s) / 1000 for s in steps], [cv[s]["kurtosis"] for s in steps],
            "-", color=color, lw=1.6, label=f"C1-t arm {arm} (t base)")
    sel = SEL_S[arm]["selected_step"]
    ax.plot(sel / 1000, cv[str(sel)]["kurtosis"], "*", color=color, ms=17,
            mec="k", label=f"arm {arm} PICK @{sel}")
    at = ATTR[arm]
    ax.plot([int(s) / 1000 for s in sorted(at, key=int)],
            [at[s]["kurtosis"] for s in sorted(at, key=int)], "o--",
            color=color, alpha=0.45, ms=4, label=f"C1 arm {arm} (Gaussian base)")
ax.axhline(TRUTH["2"]["kurtosis"], color="k", lw=1.4,
           label=f"exact truth {TRUTH['2']['kurtosis']:.2f}")
ax.set_title("SANDBOX validation curves (oct-2 kurtosis):\nthe t base raises the ceiling; "
             "the cage picks inside the window", fontsize=10)
ax.set_xlabel("training steps (k)")
ax.set_ylabel("validation kurtosis (K=4)")
ax.legend(fontsize=7)
ax.grid(alpha=0.25)

ax = axes[0][1]
octs = [2, 3, 4]
tv = [TRUTH[str(j)]["kurtosis"] for j in octs]
ax.errorbar(octs, tv, yerr=[TRUTH[str(j)]["kurtosis_se"] for j in octs],
            fmt="ko-", lw=2, capsize=3, label="exact truth")
ax.fill_between(octs, [v * 0.85 for v in tv], [v * 1.15 for v in tv],
                color="tab:green", alpha=0.15, label="±15% floor")
for arm, color in (("A", "tab:purple"), ("B", "tab:red")):
    hc = [VERD["levels"][arm]["head-conditional"][str(j)]["kurtosis"]["value"]
          for j in octs]
    e2 = [VERD["levels"][arm]["end-to-end"][str(j)]["kurtosis"]["value"]
          for j in octs]
    ax.plot(octs, hc, "s--", color=color, label=f"arm {arm} head-conditional")
    ax.plot(octs, e2, "^:", color=color, alpha=0.7, label=f"arm {arm} end-to-end")
ax.set_xticks(octs)
ax.set_xlabel("octave")
ax.set_ylabel("excess kurtosis")
ax.set_title("SANDBOX verdict at the selected ckpts:\nC1T-CAL — all 24 bars pass",
             fontsize=10)
ax.legend(fontsize=7)
ax.grid(alpha=0.25)

ax = axes[1][0]
octs4 = ["2", "3", "4", "1"]
x = np.arange(len(octs4))
w = 0.2
for i, (arm, color) in enumerate((("A", "tab:purple"), ("B", "tab:red"))):
    c1 = [abs(GOW[arm][j]["kurtosis"]["c1_rel_deficit"]) * 100 for j in octs4]
    ct = [abs(GOW[arm][j]["kurtosis"]["rel_deficit"]) * 100 for j in octs4]
    ax.bar(x + (2 * i - 1.5) * w, c1, w, color=color, alpha=0.35,
           label=f"C1 arm {arm}")
    ax.bar(x + (2 * i - 0.5) * w, ct, w, color=color,
           label=f"C1-t arm {arm}")
ax.set_xticks(x, [f"oct {j}" + (" (extrap)" if j == "1" else "") for j in octs4])
ax.set_ylabel("|e2e kurtosis deficit| vs real test (%)")
ax.set_title("GOWERSTREET: the binding-octave tail deficit closes\n"
             "(oct 2: 33.7→4.0 / 26.1→8.0 %); extrapolated octave keeps a gap",
             fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.25, axis="y")

ax = axes[1][1]
for arm, color in (("A", "tab:purple"), ("B", "tab:red")):
    cv = SEL_G[arm]["curve_val"]
    steps = sorted(cv, key=int)
    ax.plot([int(s) / 1000 for s in steps], [cv[s]["kurtosis"] for s in steps],
            "-", color=color, lw=1.6, label=f"arm {arm}")
    sel = SEL_G[arm]["selected_step"]
    ax.plot(sel / 1000, cv[str(sel)]["kurtosis"], "*", color=color, ms=17,
            mec="k", label=f"arm {arm} PICK @{sel}")
ax.axhline(GREF["kurtosis"], color="k", lw=1.4,
           label=f"real val reference {GREF['kurtosis']:.2f}")
ax.set_title("GOWERSTREET validation curves: on the real field the\ncage picks LATE "
             "(no early decay to dodge at this regime)", fontsize=10)
ax.set_xlabel("training steps (k)")
ax.set_ylabel("validation kurtosis (K=4)")
ax.legend(fontsize=7.5)
ax.grid(alpha=0.25)

fig.suptitle("Arm C1-t — t(5)-base CFM + pre-registered validation-selected early stop: "
             "sandbox C1T-CAL both arms (all bars); gowerstreet binding-octave deficit halves and more",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(RES, "c1t.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
