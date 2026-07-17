"""C3 pre-training gate figure: what the patched energy score learns and what it
does not (the blocker, made visible).

Parses the diagnostic job logs verbatim (copied into results_p2/ for provenance —
no hand-transcribed numbers): production net + production patch config, 768-field
toys, held-out eval throughout.

  Panel 1: exp (asymmetric) toy — skew AND kurtosis recover to truth.
  Panel 2: t(5) (symmetric) toy — kurtosis plateaus at ~0.5 vs truth 6; the same
           config, the same budget. THE blocker.
  Panel 3: composite modulated-σ × t(5) — the delivered fraction of a pooled
           mixture+shape kurtosis (the sandbox estimand's regime).
"""
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")


def parse(fname, tag):
    steps, skew, kurt = [], [], []
    pat = re.compile(rf"\[{re.escape(tag)}\] step\s+(\d+).*?"
                     rf"(?:skew\s+([-\d.]+)\s+)?kurt\s+([-\d.]+)")
    for line in open(os.path.join(RES, fname)):
        m = pat.search(line)
        if m:
            steps.append(int(m.group(1)))
            skew.append(float(m.group(2)) if m.group(2) else np.nan)
            kurt.append(float(m.group(3)))
    return np.array(steps), np.array(skew), np.array(kurt)


def parse_truth_ref(fname):
    for line in open(os.path.join(RES, fname)):
        m = re.search(r"truth reference.*heldout ([\d.]+)", line)
        if m:
            return float(m.group(1))
    raise ValueError("no truth reference line")


s_e, sk_e, ku_e = parse("diag_c3_capacity_16605903.out", "patch8-prodnet")
s_t, sk_t, ku_t = parse("diag_c3_capacity_t5_16606069.out", "patch8-prodnet-t5")
s_m, _, ku_m = parse("diag_c3_t5mod_16606223.out", "t5mod")
truth_mod = parse_truth_ref("diag_c3_t5mod_16606223.out")

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

ax = axes[0]
ax.plot(s_e / 1000, ku_e, "o-", color="tab:green", label="recovered kurtosis")
ax.plot(s_e / 1000, sk_e, "s--", color="tab:olive", label="recovered skewness")
ax.axhline(6.0, color="k", lw=1.5)
ax.axhline(2.0, color="k", lw=1.0, ls="--")
ax.text(0.1, 6.15, "truth kurt 6.0", fontsize=8)
ax.text(0.1, 2.15, "truth skew 2.0", fontsize=8)
ax.set_title("ASYMMETRIC tails (exp noise):\nboth shape moments recovered", fontsize=10)
ax.set_xlabel("training steps (k)")
ax.set_ylabel("held-out shape statistic")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

ax = axes[1]
ax.plot(s_t / 1000, ku_t, "o-", color="tab:red", label="recovered kurtosis")
ax.axhline(6.0, color="k", lw=1.5)
ax.text(0.1, 6.15, "truth kurt 6.0", fontsize=8)
ax.axhspan(0, 1.2, color="tab:red", alpha=0.10)
ax.text(1.6, 0.75, "1-D mechanism limit (~1.1)", fontsize=7.5, color="tab:red")
ax.set_ylim(-0.3, 7.0)
ax.set_title("SYMMETRIC tails (t(5) noise): kurtosis\nplateaus at 0.49 — THE BLOCKER",
             fontsize=10)
ax.set_xlabel("training steps (k)")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

ax = axes[2]
ax.plot(s_m / 1000, ku_m, "o-", color="tab:purple",
        label="recovered pooled kurtosis")
ax.axhline(truth_mod, color="k", lw=1.5,
           label=f"truth (mixture + shape) {truth_mod:.2f}")
ax.set_title("composite modulated σ × t(5):\nmixture part delivered, shape part missing",
             fontsize=10)
ax.set_xlabel("training steps (k)")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

fig.suptitle("C3 pre-training gate (R10 condition 2): production net + production patch config, "
             "768-field toys, HELD-OUT eval — the β=1 patched energy score learns asymmetry "
             "(first-order CRPS signal) but not symmetric tail weight (second-order)",
             fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.90])
out = os.path.join(RES, "c3_gates.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
