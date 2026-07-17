"""R15 tail-dynamics figure: the rung-4 decay is data-limited, made visible.
Parses results_p2/taildyn_*.json directly (R12: no hand-copied numbers).

Row 1: held-out kurtosis vs steps, 1x (thin) vs 8x (thick) vs truth — the four
       candidate x gate panels; every 1x run terminally collapses, 8x holds at
       (or near) truth.
Row 2: P3 base-vs-flow decomposition (tbase composite): at 1x the two bases
       converge to the same collapsed endpoint (base ERASED); at 8x the gap
       survives. Last panel: twcrps' spurious skew grows WITH data.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")


def load(cand, toy, n):
    d = json.load(open(os.path.join(RES, f"taildyn_{cand}_{toy}_n{n}.json")))
    tr = d["trajectory"]
    steps = sorted(tr, key=int)
    return ([int(s) / 1000 for s in steps], tr, steps,
            d["truth_ref_heldout"]["kurt"])


fig, axes = plt.subplots(2, 4, figsize=(17.5, 8.0))

for ax, (cand, toy) in zip(axes[0], (("tbase", "composite"), ("tbase", "t5flat"),
                                     ("twcrps", "composite"), ("twcrps", "t5flat"))):
    for n, lw, alpha in ((768, 1.4, 0.65), (6144, 2.6, 1.0)):
        x, tr, steps, truth = load(cand, toy, n)
        ax.plot(x, [tr[s]["kurt"] for s in steps], "-", lw=lw, alpha=alpha,
                color="tab:purple" if cand == "tbase" else "tab:blue",
                label=f"n = {n}" + (" (8×)" if n == 6144 else " (1×)"))
    ax.axhline(truth, color="k", lw=1.3, label=f"truth {truth:.2f}")
    ax.set_title(f"{cand} — {toy}", fontsize=10.5)
    ax.set_xlabel("training steps (k)")
    ax.set_ylabel("held-out excess kurtosis")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

ax = axes[1][0]
x, tr, steps, truth = load("tbase", "composite", 768)
ax.plot(x, [tr[s]["kurt"] for s in steps], "o-", ms=3, color="tab:purple",
        label="t(5)-base output")
ax.plot(x, [tr[s]["gaussbase_kurt"] for s in steps], "s-", ms=3,
        color="tab:gray", label="GAUSSIAN-base output, same flow")
ax.axhline(truth, color="k", lw=1.3)
ax.set_title("P3 at 1×: the flow ERASES its base\n(both bases → the same collapsed tails)",
             fontsize=10)
ax.set_xlabel("training steps (k)")
ax.set_ylabel("held-out excess kurtosis")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

ax = axes[1][1]
x, tr, steps, truth = load("tbase", "composite", 6144)
ax.plot(x, [tr[s]["kurt"] for s in steps], "o-", ms=3, color="tab:purple",
        label="t(5)-base output")
ax.plot(x, [tr[s]["gaussbase_kurt"] for s in steps], "s-", ms=3,
        color="tab:gray", label="GAUSSIAN-base output, same flow")
ax.axhline(truth, color="k", lw=1.3)
ax.set_title("P3 at 8×: the base's tails SURVIVE\n(stable gap to 12k)", fontsize=10)
ax.set_xlabel("training steps (k)")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

ax = axes[1][2]
for toy, ls in (("composite", "-"), ("t5flat", "--")):
    for n, lw, alpha in ((768, 1.4, 0.6), (6144, 2.6, 1.0)):
        x, tr, steps, _ = load("twcrps", toy, n)
        ax.plot(x, [tr[s]["skew"] for s in steps], ls, lw=lw, alpha=alpha,
                color="tab:blue",
                label=f"{toy} n={n}")
ax.axhline(0.0, color="k", lw=1.3, label="truth (symmetric) 0")
ax.set_title("twcrps pathology: spurious skew\nGROWS with data", fontsize=10)
ax.set_xlabel("training steps (k)")
ax.set_ylabel("held-out skewness")
ax.legend(fontsize=7)
ax.grid(alpha=0.25)

ax = axes[1][3]
x, tr, steps, _ = load("tbase", "composite", 768)
ax.plot(x, [tr[s]["disp_maxrel"] for s in steps], "o-", ms=3, color="tab:green",
        label="1× dispersion error")
x8, tr8, steps8, _ = load("tbase", "composite", 6144)
ax.plot(x8, [tr8[s]["disp_maxrel"] for s in steps8], "o-", ms=3, lw=2.4,
        color="tab:olive", label="8× dispersion error")
ax.axhline(0.10, color="tab:red", ls="--", lw=1.3, label="10% bar")
ax.set_title("P2: dispersion holds ≤10% nearly everywhere\n→ joint near-truth checkpoints exist",
             fontsize=10)
ax.set_xlabel("training steps (k)")
ax.set_ylabel("max per-bin σ rel. error")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

fig.suptitle("R15 tail-dynamics diagnosis — every 1× run terminally collapses (thin); every 8× run holds at/near truth for 12k steps (thick): "
             "the rung-4 tail decay is DATA-LIMITED, and the 1×-trained flow erases even the heavy-tailed base it was given",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(RES, "taildyn.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
