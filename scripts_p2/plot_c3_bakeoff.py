"""R13 bake-off figure: six held-out kurtosis trajectories against the selector
bar. Parses results_p2/bakeoff_*.json directly (R12: no hand-copied numbers)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")

STYLE = {"twcrps": ("tab:blue", "twCRPS (chain τ=2, a=4)"),
         "beta05": ("tab:orange", "β = 0.5"),
         "tbase": ("tab:purple", "t(5)-base CFM")}

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
for ax, toy, title in ((axes[0], "t5flat", "flat σ × t(5) gate"),
                      (axes[1], "composite", "modulated σ × t(5) composite gate")):
    for cand, (color, label) in STYLE.items():
        d = json.load(open(os.path.join(RES, f"bakeoff_{cand}_{toy}.json")))
        traj = d["trajectory"]
        steps = sorted(traj, key=int)
        ax.plot([int(s) / 1000 for s in steps], [traj[s]["kurt"] for s in steps],
                "o-", color=color, label=label)
        ax.plot(int(steps[-1]) / 1000, traj[steps[-1]]["kurt"], "o",
                color=color, ms=11, mfc="none", mew=2)
    d0 = json.load(open(os.path.join(RES, f"bakeoff_twcrps_{toy}.json")))
    ax.axhline(d0["truth_ref_heldout"]["kurt"], color="k", lw=1.5,
               label=f"held-out data reference {d0['truth_ref_heldout']['kurt']:.2f}")
    ax.axhline(4.0, color="tab:red", lw=1.5, ls="--", label="selector bar 4.0 (at 4k)")
    ax.set_xlabel("training steps (k)")
    ax.set_ylabel("held-out excess kurtosis")
    ax.set_title(title, fontsize=10.5)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25)

fig.suptitle("R13 objective bake-off — NO candidate holds the bar at the pinned 4k readout "
             "(ringed markers): twCRPS and t-base CROSS it mid-training, then converge away; "
             "β=0.5 never approaches. Production net, held-out eval.",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.90])
out = os.path.join(RES, "c3_bakeoff.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
