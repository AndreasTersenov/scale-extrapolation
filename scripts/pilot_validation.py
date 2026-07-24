#!/usr/bin/env python
"""Step-2 inverted validation pilot: compute all three components and the figure.

Component 1 (slide-the-edge): reads results/scores/arms_aug_score.json (edge 1, trained 2-4,
extrapolated oct 1) and results/scores/arms_edge2_score.json (edge 2, trained 3-4,
extrapolated oct 2); compares arm-wise relative var_slope errors at each edge's first
extrapolated octave (P-edge: within factor 2).
Component 2 (self-consistency): per octave, z of |measured-on-generated coupling minus
reference|; SC-a reference = the conditioning curve (deployable), SC-b = held-out real
(truth). Includes SC-a's calibration check on the real fields themselves.
Component 3 (held-out statistics): wavelet-L1 per octave and kymatio scattering
order-2 summary, generated vs real (bootstrap over 64 fields).
"""
import json
import os
try:
    os.sched_setaffinity(0, set(range(4)))
except Exception:
    pass
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pilotstats import (scattering_logmeans, scattering_summary, wavelet_l1,
                        z_stack)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K, BLUE, GREEN, ORANGE, VERM = "#000000", "#0072B2", "#009E73", "#E69F00", "#D55E00"

coords = {int(j): v for j, v in
          json.load(open(os.path.join(REPO, "data_cache",
                                      "running_couplings.json")))["gowerstreet"].items()}
s1 = json.load(open(os.path.join(REPO, "results", "scores", "arms_aug_score.json")))
s2 = json.load(open(os.path.join(REPO, "results", "scores", "arms_edge2_score.json")))
out = {}

# ---------------- component 1: slide the edge ----------------
edges = {}
for name, s, je in (("edge1 (train 2-4)", s1, 1), ("edge2 (train 3-4)", s2, 2)):
    real = s["real"][str(je)]["var_slope"]
    edges[name] = {}
    for arm in "AB":
        g = s[f"arm{arm}"][str(je)]["var_slope"]
        edges[name][arm] = {"rel_err": abs(g - real) / abs(real),
                            "gen": g, "real": real, "oct": je}
out["edge"] = edges
r1B, r2B = edges["edge1 (train 2-4)"]["B"]["rel_err"], edges["edge2 (train 3-4)"]["B"]["rel_err"]
ratio = max(r1B, r2B) / max(min(r1B, r2B), 1e-9)
out["edge"]["P_edge_ratio_B"] = ratio
out["edge"]["P_edge_pass"] = bool(ratio <= 2.0)
print(f"[edge] arm B rel err: edge1 {r1B:.1%}, edge2 {r2B:.1%}, ratio {ratio:.2f} "
      f"-> P-edge {'PASS' if ratio <= 2 else 'FAIL'}")

# ---------------- component 2: self-consistency ----------------
sc = {}
for run, s, trained, je in (("edge1", s1, (2, 3, 4), 1), ("edge2", s2, (3, 4), 2)):
    sc[run] = {}
    for arm in ("A", "B", "real"):
        row = {}
        src = s["real"] if arm == "real" else s[f"arm{arm}"]
        for j in sorted(int(k) for k in src.keys()):
            vs, se = src[str(j)]["var_slope"], src[str(j)]["var_slope_se"]
            za = abs(vs - coords[j][0]) / max(se, 1e-9)            # SC-a: cond curve
            zb = abs(vs - s["real"][str(j)]["var_slope"]) / max(
                np.hypot(se, s["real"][str(j)]["var_slope_se"]), 1e-9)  # SC-b: truth
            row[j] = {"sc_a": float(za), "sc_b": float(zb)}
        sc[run][arm] = row
out["selfcons"] = sc
print("[selfcons] edge1 arm B SC-b by octave:",
      {j: round(v["sc_b"], 1) for j, v in sc["edge1"]["B"].items()})
print("[selfcons] SC-a calibration on REAL fields (edge1):",
      {j: round(v["sc_a"], 1) for j, v in sc["edge1"]["real"].items()})

# ---------------- component 3: held-out statistics ----------------
d = np.load(os.path.join(REPO, "results", "npz", "arms_aug.npz"))
real_f, gA, gB = d["real"], d["gen_A"], d["gen_B"]
ho = {"l1": {}, "scatter": {}}
for arm, g in (("A", gA), ("B", gB)):
    ho["l1"][arm] = {j: z_stack(wavelet_l1(g, j), wavelet_l1(real_f, j))
                     for j in (1, 2, 3, 4)}
    zc = z_stack(scattering_logmeans(g), scattering_logmeans(real_f))
    ho["scatter"][arm] = scattering_summary(zc)
    ho["scatter"][arm]["z_channels"] = np.asarray(zc).tolist()
out["heldout"] = {"l1": {a: {j: float(v) for j, v in ho["l1"][a].items()}
                         for a in "AB"},
                  "scatter": {a: {k: v for k, v in ho["scatter"][a].items()
                                  if k != "z_channels"} for a in "AB"}}
print("[heldout] L1 z by octave:", out["heldout"]["l1"])
print("[heldout] scattering:", out["heldout"]["scatter"])

json.dump(out, open(os.path.join(REPO, "results", "scores", "pilot_validation.json"), "w"),
          indent=1)

# ---------------- figure ----------------
fig, ax = plt.subplots(1, 3, figsize=(14.4, 4.8))
# panel 1: edges
labels, vals, cols = [], [], []
for name in ("edge1 (train 2-4)", "edge2 (train 3-4)"):
    for arm in "AB":
        labels.append(f"{name.split()[0]}\narm {arm}")
        vals.append(100 * edges[name][arm]["rel_err"])
        cols.append(BLUE if arm == "A" else ORANGE)
ax[0].bar(range(4), vals, color=cols)
band = 100 * r1B
ax[0].axhspan(band / 2, band * 2, color="#88888822")
ax[0].axhline(band, color=K, ls="--", lw=1.4)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(labels, fontsize=8.5)
ax[0].set_ylabel("relative var_slope error at the first extrapolated octave  [%]")
ax[0].set_title(f"Slide-the-edge: same operator error at both edges?\n(grey band = "
                f"factor 2 around edge-1 arm B; ratio {ratio:.2f} -> "
                f"{'PASS' if ratio <= 2 else 'FAIL'})", fontsize=10)
# panel 2: self-consistency (edge1)
js = sorted(sc["edge1"]["B"].keys())
for arm, col, mk in (("B", ORANGE, "o"), ("A", BLUE, "s")):
    ax[1].plot(js, [sc["edge1"][arm][j]["sc_b"] for j in js], f"-{mk}", color=col,
               label=f"arm {arm} (vs truth, SC-b)")
ax[1].plot(js, [sc["edge1"]["real"][j]["sc_a"] for j in js], ":^", color="#666666",
           label="REAL fields vs curve (SC-a calibration)")
ax[1].axhline(3, color=VERM, ls="--", lw=1.5)
ax[1].text(3.4, 3.15, "flag threshold (3σ)", color=VERM, fontsize=8.5)
ax[1].axvspan(0.6, 1.5, color="#D55E0018")
ax[1].set_xticks(js); ax[1].invert_xaxis()
ax[1].set_xlabel("octave (shaded = extrapolated)")
ax[1].set_ylabel("self-consistency deviation z")
ax[1].set_yscale("log")
ax[1].set_title("Self-consistency: generated couplings vs the curve\n"
                "(no ground truth needed — flags the failure?)", fontsize=10)
ax[1].legend(fontsize=8)
# panel 3: held-out stats
x = np.arange(4)
for arm, col, off in (("A", BLUE, -0.17), ("B", ORANGE, 0.17)):
    ax[2].bar(x + off, [abs(ho["l1"][arm][j]) for j in (1, 2, 3, 4)], 0.32,
              color=col, label=f"wavelet-L1 |z|, arm {arm}")
ax[2].axhline(3, color=VERM, ls="--", lw=1.5)
ax[2].set_xticks(x); ax[2].set_xticklabels(["oct 1\n(extrap)", "oct 2", "oct 3", "oct 4"])
ax[2].set_ylabel("|z| generated vs real")
sA, sB = ho["scatter"]["A"], ho["scatter"]["B"]
ax[2].set_title(f"Held-out statistics (never used in design)\nscattering order-2: "
                f"{sA['frac_flagged']:.0%} of channels flag (A), "
                f"{sB['frac_flagged']:.0%} (B)", fontsize=10)
ax[2].legend(fontsize=8)
for a in ax:
    a.grid(alpha=0.25)
fig.suptitle("Step 2 — the INVERTED validation pilot on the frozen failing generator: does the protocol catch what we know is there?",
             fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(REPO, "results", "figures", "stage0", "pilot_validation.png"), dpi=130,
            bbox_inches="tight")
print("wrote results/scores+figures/stage0/pilot_validation.{json,png}")
