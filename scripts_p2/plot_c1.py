"""C1 sandbox readout figure: the dispersion-vs-training curve against exact truth
(the collapse question, answered on a calibrated bed) + the bar chart of all
pre-registered checks."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COND = json.load(open(os.path.join(REPO, "results_p2", "c1_conditional_sandbox.json")))
VERD = json.load(open(os.path.join(REPO, "results_p2", "c1_verdict_sandbox.json")))
TRUTH = json.load(open(os.path.join(REPO, "results_p2",
                                    "sandbox_truth_normconv.json")))["truth"]

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

# --- panel 1: the checkpoint curve (the collapse-law question)
ax = axes[0]
for arm, color in (("A", "tab:blue"), ("B", "tab:orange")):
    curve = {int(k): v["var_slope"] for k, v in COND[arm]["curve_oct2"].items()}
    curve[20000] = COND[arm]["final"]["2"]["var_slope"]
    steps = sorted(curve)
    ax.plot([s / 1000 for s in steps], [curve[s] for s in steps], "o-",
            color=color, label=f"arm {arm} (head-conditional oct 2)")
t = TRUTH["2"]["var_slope"]
ax.axhline(t, color="k", lw=1.5, label=f"exact truth (norm conv) {t:.3f}")
ax.axhspan(t * 0.9, t * 1.1, color="tab:green", alpha=0.15, label="±10% bar")
ax.set_xlabel("training steps (k)")
ax.set_ylabel("conditional var_slope, octave 2")
ax.set_title("vanilla CFM + augmentation:\ndispersion vs training (collapse gone?)",
             fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

# --- panels 2+3: per-octave var_slope and kurtosis, both levels, vs truth
for ax, metric, bar in ((axes[1], "var_slope", 0.10), (axes[2], "kurtosis", 0.15)):
    octs = [2, 3, 4]
    tv = [TRUTH[str(j)][metric] for j in octs]
    tse = [TRUTH[str(j)][metric + "_se"] for j in octs]
    ax.errorbar(octs, tv, yerr=tse, fmt="ko-", lw=2, capsize=3, label="exact truth")
    ax.fill_between(octs, [v * (1 - bar) for v in tv], [v * (1 + bar) for v in tv],
                    color="tab:green", alpha=0.15, label=f"±{int(bar*100)}% floor")
    for arm, color in (("A", "tab:blue"), ("B", "tab:orange")):
        hc = [COND[arm]["final"][str(j)][metric] for j in octs]
        hcse = [COND[arm]["final"][str(j)][metric + "_se"] for j in octs]
        ax.errorbar(octs, hc, yerr=hcse, fmt="s--", color=color, capsize=3,
                    label=f"arm {arm} head-conditional")
        e2 = [VERD["levels"][arm]["end-to-end"][str(j)][metric]["value"]
              for j in octs]
        ax.plot(octs, e2, "^:", color=color, alpha=0.7,
                label=f"arm {arm} end-to-end")
    ax.set_xticks(octs)
    ax.set_xlabel("octave")
    ax.set_title(f"{metric} vs exact truth (final ckpt)", fontsize=10)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.25)

fig.suptitle("Arm C1 sandbox leg — plain conditional FM + D4 augmentation, no "
             "variance head; adjudicated against exact conditional truth",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(REPO, "results_p2", "c1_sandbox.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
