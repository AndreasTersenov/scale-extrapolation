#!/usr/bin/env python
"""Ground-up intuition visuals for the scale-extrapolation project (real fields)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scaledrift import octave_pair, octave_conditional_moments
from scaledrift.wavelet import octave_wc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results")
K, BLUE, VERM = "#000000", "#0072B2", "#D55E00"
plt.rcParams.update({"font.size": 10})

gow = np.load(os.path.join(REPO, "data_cache", "tiles_pnull.npz"))["gowerstreet"]
grf = np.load(os.path.join(REPO, "data_cache", "tiles_pnull.npz"))["grf"]
gen = np.load(os.path.join(RES, "arms_2k_c4.0.npz"))            # gen_A, gen_B, real (extrap octave=1)


def norm(t):
    return (t - t.mean()) / t.std()


# ================= Figure 1: what a wavelet octave actually is =================
fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
f = norm(gow[3])
det, coarse = octave_pair(f, 2)                # det (3,32,32), coarse (32,32)
mag = np.abs(det).mean(0)
ims = [(f, "RdBu_r", "a real N-body κ map (128×128)\nweak-lensing convergence"),
       (coarse, "RdBu_r", "COARSE at octave 2 (32×32)\nthe large-scale field"),
       (mag, "inferno", "|DETAIL| at octave 2 (32×32)\nsmall-scale fluctuations")]
for a, (img, cm, ttl) in zip(ax, ims):
    vlim = np.abs(img).max() if cm == "RdBu_r" else None
    im = a.imshow(img, cmap=cm, vmin=-vlim if vlim else None, vmax=vlim)
    a.set_title(ttl, fontsize=10); a.set_xticks([]); a.set_yticks([])
    plt.colorbar(im, ax=a, fraction=0.046)
fig.suptitle("What we measure: split each map into a COARSE field and its DETAIL at each scale (octave).\n"
             "The question — does the detail's statistics depend on the coarse value, and does that change across octaves?",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(os.path.join(RES, "intuition_1_objects.png"), dpi=130, bbox_inches="tight")
plt.close(fig)


# ============ Figure 2: the physical finding — fan-out (real data) ============
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
octs = [1, 2, 3, 4]
cmap = plt.cm.viridis(np.linspace(0, 0.85, len(octs)))
for a, tiles, name in [(ax[0], gow, "N-body (gowerstreet)"), (ax[1], grf, "Gaussian field (GRF) — the null")]:
    maps = [norm(t) for t in tiles[:60]]
    for j, col in zip(octs, cmap):
        m = octave_conditional_moments(maps, j, n_bins=10)
        a.plot(m["c_center"], m["var"], "-o", ms=4, color=col, label=f"octave {j}")
    a.set_xlabel("coarse field value (std units)")
    a.set_ylabel("Var( detail | coarse )")
    a.set_title(name); a.grid(alpha=0.25); a.legend(fontsize=8, title="scale")
ax[0].annotate("curves FAN OUT across octaves\n→ non-Gaussianity RUNS with scale\n(this is what a scale-blind model must miss)",
               (0.5, 0.02), xycoords="axes fraction", fontsize=9, color=VERM,
               ha="center", va="bottom")
ax[1].annotate("curves COLLAPSE (octave-invariant)\n→ nothing runs with scale",
               (0.5, 0.9), xycoords="axes fraction", fontsize=9, color=BLUE, ha="center")
fig.suptitle("The physical finding (measured on REAL fields): in N-body data the conditional detail variance\n"
             "depends on scale (octave); in a Gaussian field it does not. A weight-tied generator assumes it never does.",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(os.path.join(RES, "intuition_2_fanout.png"), dpi=130, bbox_inches="tight")
plt.close(fig)


# ========= Figure 3: the experiment & result — fields + break/repair =========
fig = plt.figure(figsize=(13, 8.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=0.32, wspace=0.15)
# top: sample fields
rows = [("real held-out", gen["real"], BLUE), ("arm A generated (no scale input)", gen["gen_A"], BLUE),
        ("arm B generated (+ scale coordinate)", gen["gen_B"], VERM)]
top = gs[0, :].subgridspec(1, 3, wspace=0.08)
for c, (name, fields, _) in enumerate(rows):
    a = fig.add_subplot(top[0, c])
    a.imshow(norm(fields[5]), cmap="RdBu_r", vmin=-3, vmax=3)
    a.set_title(name, fontsize=9.5); a.set_xticks([]); a.set_yticks([])
# bottom: Var(detail|coarse) at the extrapolated octave (1) — the break & repair
axc = fig.add_subplot(gs[1, :])
octJ = 1
mr = octave_conditional_moments([norm(t) for t in gen["real"]], octJ, n_bins=10)
ma = octave_conditional_moments([norm(t) for t in gen["gen_A"]], octJ, n_bins=10)
mb = octave_conditional_moments([norm(t) for t in gen["gen_B"]], octJ, n_bins=10)
axc.plot(mr["c_center"], mr["var"], "-o", color=K, lw=2.5, ms=6, label="real (target)")
axc.plot(ma["c_center"], ma["var"], "-s", color=BLUE, lw=2, ms=6, label="arm A — no scale input (BREAKS: too flat)")
axc.plot(mb["c_center"], mb["var"], "-^", color=VERM, lw=2, ms=6, label="arm B — + running-coupling coord (REPAIRS)")
axc.set_xlabel("coarse field value (std units)")
axc.set_ylabel("Var( detail | coarse )")
axc.set_title("The break & the repair, at the EXTRAPOLATED (untrained) octave — as physical curves\n"
              "arm A can't tilt the variance with the coarse field; arm B, given the scale coordinate, recovers the real slope",
              fontsize=10.5)
axc.grid(alpha=0.25); axc.legend(fontsize=9, loc="upper left")
fig.suptitle("The experiment: train the generator at small scale, generate the finer (untrained) octave, compare to real.\n"
             "All three maps look like plausible cosmology — the difference is in the conditional statistics (below).",
             fontsize=11)
fig.savefig(os.path.join(RES, "intuition_3_breakrepair.png"), dpi=130, bbox_inches="tight")
plt.close(fig)
print("wrote intuition_1_objects.png, intuition_2_fanout.png, intuition_3_breakrepair.png")
