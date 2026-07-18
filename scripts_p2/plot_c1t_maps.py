"""C1-t results ON THE MAPS (Andreas's request; SPEC-gallery honesty rules).

Rules applied: held-out TEST index 0 everywhere (first field, no cherry-picking;
C1 stacks sliced [32:] — alignment vs the C1-t real stack verified before this
script was written); arm A displayed with arm B's numbers quoted where they
differ; shared color scale per row anchored to the REAL map (1-99 pct); all
caption statistics are verbatim from committed jsons (c1_verdict_sandbox,
c1t_verdict_sandbox, c1t_gow_descriptive, c1t_peaks_gowerstreet); the only
on-figure computation is the peak circles on the DISPLAYED field itself.

FIG 1  c1t_maps.png       — real | C1 | C1-t arm A | C1-t arm B, both fields.
FIG 2  c1t_maps_peaks.png — where the peaks go now (gowerstreet, >=2.5 sigma).
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")

T_S = np.load(os.path.join(RES, "arms_c1t_sandbox.npz"))
C_S = np.load(os.path.join(RES, "arms_c1_sandbox.npz"))
T_G = np.load(os.path.join(RES, "arms_c1t_gowerstreet.npz"))
C_G = np.load(os.path.join(RES, "arms_c1_gowerstreet.npz"))
V_S1 = json.load(open(os.path.join(RES, "c1_verdict_sandbox.json")))
V_ST = json.load(open(os.path.join(RES, "c1t_verdict_sandbox.json")))
GOW = json.load(open(os.path.join(RES, "c1t_gow_descriptive.json")))
PKS = json.load(open(os.path.join(RES, "c1t_peaks_gowerstreet.json")))
TRUTH = json.load(open(os.path.join(RES, "sandbox_truth_normconv.json")))["truth"]

k = lambda d: d["kurtosis"]["value"] if "value" in d.get("kurtosis", {}) else None

# ------------------------------------------------------------- FIG 1: gallery
c1s_k = V_S1["levels"]["A"]["end-to-end"]["2"]["kurtosis"]["value"]
c1s_kB = V_S1["levels"]["B"]["end-to-end"]["2"]["kurtosis"]["value"]
cts_k = V_ST["levels"]["A"]["end-to-end"]["2"]["kurtosis"]["value"]
cts_kB = V_ST["levels"]["B"]["end-to-end"]["2"]["kurtosis"]["value"]
rows = [
    ("SANDBOX (exact truth exists)",
     [np.asarray(T_S["real"][0]), np.asarray(C_S["gen_A"][32]),
      np.asarray(T_S["gen_A"][0]), np.asarray(T_S["gen_B"][0])],
     [f"REAL test field — truth oct-2 kurt {TRUTH['2']['kurtosis']:.2f}",
      f"C1 (Gaussian base, 20k)\ne2e oct-2 kurt {c1s_k:.2f} (arm B {c1s_kB:.2f})",
      f"C1-t arm A (t base, pick @7500)\ne2e oct-2 kurt {cts_k:.2f} — ALL BARS PASS",
      f"C1-t arm B (pick @2500)\ne2e oct-2 kurt {cts_kB:.2f} — ALL BARS PASS"]),
    ("GOWERSTREET (real field, descriptive)",
     [np.asarray(T_G["real"][0]), np.asarray(C_G["gen_A"][32]),
      np.asarray(T_G["gen_A"][0]), np.asarray(T_G["gen_B"][0])],
     [f"REAL test field — oct-2 kurt {GOW['A']['2']['kurtosis']['real']:.2f}",
      f"C1 — kurt {GOW['A']['2']['kurtosis']['c1_gen']:.2f} "
      f"({GOW['A']['2']['kurtosis']['c1_rel_deficit']:+.0%})",
      f"C1-t arm A (pick @16000) — kurt {GOW['A']['2']['kurtosis']['gen']:.2f} "
      f"({GOW['A']['2']['kurtosis']['rel_deficit']:+.0%})",
      f"C1-t arm B (pick @18500) — kurt {GOW['B']['2']['kurtosis']['gen']:.2f} "
      f"({GOW['B']['2']['kurtosis']['rel_deficit']:+.0%})"])]

fig, axes = plt.subplots(2, 4, figsize=(15.5, 9.4))
for r, (rowtitle, maps, caps) in enumerate(rows):
    vmin, vmax = np.percentile(maps[0], [1, 99])
    for cidx, (ax, m, cap) in enumerate(zip(axes[r], maps, caps)):
        ax.imshow(m, cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(cap, fontsize=8.6)
        ax.set_xticks([]); ax.set_yticks([])
        if cidx == 0:
            ax.set_ylabel(rowtitle, fontsize=10)
fig.suptitle("One held-out TEST field per row, same octave-4 coarse start, three generators — "
             "color scale fixed to the REAL map (1–99 pct). The t base + caged early stop puts the "
             "missing tail weight back into the maps: bright extremes reappear where C1's were tame.",
             fontsize=10.5)
fig.subplots_adjust(top=0.87, bottom=0.03, hspace=0.24, wspace=0.06)
fig.savefig(os.path.join(RES, "c1t_maps.png"), dpi=150, bbox_inches="tight")
print("wrote c1t_maps.png")

# ------------------------------------------------------------- FIG 2: peaks
def peaks(field, nu):
    f = (field - field.mean()) / field.std()
    c = f[1:-1, 1:-1]
    is_max = np.ones(c.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            is_max &= c > f[1 + dy:f.shape[0] - 1 + dy, 1 + dx:f.shape[1] - 1 + dx]
    ys, xs = np.nonzero(is_max & (c > nu))
    return ys + 1, xs + 1


def cap(tag, name):
    m = PKS[tag]
    r = PKS["real"]
    def pct(nu):
        return (m[nu]["mean"] - r[nu]["mean"]) / r[nu]["mean"]
    return (f"{name}\n32-field means: {m['1.0']['mean']:.0f} (ν=1, {pct('1.0'):+.0%}) · "
            f"{m['2.5']['mean']:.0f} (ν=2.5, {pct('2.5'):+.0%}) · "
            f"{m['3.0']['mean']:.0f} (ν=3, {pct('3.0'):+.0%})")


panels = [
    (np.asarray(T_G["real"][0]),
     f"REAL\n32-field means: {PKS['real']['1.0']['mean']:.0f} (ν=1) · "
     f"{PKS['real']['2.5']['mean']:.0f} (ν=2.5) · {PKS['real']['3.0']['mean']:.0f} (ν=3)"),
    (np.asarray(C_G["gen_A"][32]), cap("c1_A", "C1 (Gaussian base)")),
    (np.asarray(T_G["gen_A"][0]), cap("c1t_A", "C1-t (t base + early stop)")),
]
vmin, vmax = np.percentile(panels[0][0], [1, 99])
fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2))
for ax, (m, caption) in zip(axes, panels):
    ax.imshow(m, cmap="gray", vmin=vmin, vmax=vmax)
    ys, xs = peaks(m, 2.5)
    for y, x in zip(ys, xs):
        ax.add_patch(Circle((x, y), 2.6, fill=False, color="red", lw=0.9))
    ax.set_title(f"{len(ys)} peaks ≥2.5σ on THIS map", fontsize=10)
    ax.set_xlabel(caption, fontsize=8.2)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Where the peaks go, after the fix — red circles: local maxima ≥2.5σ on the displayed (test index-0) "
             "gowerstreet field; captions: 32-field means vs real (c1t_peaks_gowerstreet.json). C1's uniform up-tilt "
             "(+24…+26%) SHRINKS under C1-t (+14…+19%) but does not close: pooled tails are calibrated, their spatial "
             "placement is the remaining audit frontier (arm B similar: +18/+11/+12%).",
             fontsize=9.5)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(os.path.join(RES, "c1t_maps_peaks.png"), dpi=150, bbox_inches="tight")
print("wrote c1t_maps_peaks.png")
