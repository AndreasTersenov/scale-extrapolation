"""Phase A' (R31 order 1): descriptive parity-mechanism localization from
COMMITTED artifacts only. CPU, env.sh stack, zero training.

Measurements per stack (gen vs its own real fields):
  M1  peak-parity profiles at block levels 1/2/3 (output side; localizes the
      biased synthesis scale) — K count-matched per phase-A conventions.
  M2  coefficient D4-null statistics per octave 1..4 (9 components: channel
      means, sign rates, cross-correlations) — nonzero => the bias LIVES IN
      THE COEFFICIENTS of that octave/channel.
  M3  corner-argmax profiles per octave 1..4 (top-decile |a| blocks) — the
      deterministic coefficient->corner link at each synthesis scale.
  M4  hybrid resynthesis (index-paired gen/real): real field with gen octave-j
      details transplanted, j=1..4 -> level-1 peak parity; plus the
      complement (gen field with real octave-1 details) — does removing the
      octave-1 coefficients REMOVE the output bias?

Stacks: c1t sandbox 32 + repl64; c1t gowerstreet; C1 gowerstreet; Stage-D
edge; taxonomy (4a NLL-head arms_aug, mu-only forensic). Writes
results_p2/parity_localization.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")
sys.path.insert(0, os.path.join(REPO, "scripts_p2"))

from parity_localization import (  # noqa: E402
    COEF_STAT_NAMES, coef_stats_profile, corner_argmax_profile,
    hybrid_with_gen_octave, peak_parity_profile,
)
from placement_instruments import stack_profiles, tstat  # noqa: E402

K_SANDBOX, K_REAL = 160, 174     # phase-A frozen conventions
OCTAVES = (1, 2, 3, 4)
LEVELS = (1, 2, 3)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def T_of(gen_fields, real_fields, fn):
    a = [fn(np.asarray(f, np.float64)) for f in gen_fields]
    b = [fn(np.asarray(f, np.float64)) for f in real_fields]
    ma, sa = stack_profiles(a)
    mb, sb = stack_profiles(b)
    T, z = tstat(ma, sa, mb, sb)
    return {"T": T, "z": [float(x) for x in z],
            "gen_mean": [float(x) for x in ma], "real_mean": [float(x) for x in mb]}


def score_stack(gen, real, K):
    out = {"M1_peak_parity": {}, "M2_coef_stats": {}, "M3_corner_argmax": {}}
    for lv in LEVELS:
        out["M1_peak_parity"][str(lv)] = T_of(
            gen, real, lambda f, lv=lv: peak_parity_profile(f, k=K, level=lv))
    for j in OCTAVES:
        out["M2_coef_stats"][str(j)] = T_of(
            gen, real, lambda f, j=j: coef_stats_profile(f, octave=j))
        out["M3_corner_argmax"][str(j)] = T_of(
            gen, real, lambda f, j=j: corner_argmax_profile(f, octave=j))
    return out


def main():
    stacks = {}
    def add(name, gen, real, K):
        stacks[name] = (np.asarray(gen, np.float64), np.asarray(real, np.float64), K)

    d = np.load(os.path.join(RES, "arms_c1t_sandbox.npz"), allow_pickle=True)
    add("sandbox32_A", d["gen_A"], d["real"], K_SANDBOX)
    add("sandbox32_B", d["gen_B"], d["real"], K_SANDBOX)
    sandbox32 = d
    g = np.load(os.path.join(RES, "c1t_repl64_gen.npz"), allow_pickle=True)
    repl_real = np.load(os.path.join(REPO, "data_cache",
                                     "tiles_sandbox_repl.npz"))["sandbox"]
    add("repl64_A", g["gen_A"], repl_real, K_SANDBOX)
    add("repl64_B", g["gen_B"], repl_real, K_SANDBOX)
    repl64 = g
    d = np.load(os.path.join(RES, "arms_c1t_gowerstreet.npz"), allow_pickle=True)
    add("gow_c1t_A", d["gen_A"], d["real"], K_REAL)
    add("gow_c1t_B", d["gen_B"], d["real"], K_REAL)
    gow_c1t = d
    d = np.load(os.path.join(RES, "arms_c1_gowerstreet.npz"), allow_pickle=True)
    add("gow_c1_A", d["gen_A"], d["real"], K_REAL)
    d = np.load(os.path.join(RES, "arms_c1_sandbox.npz"), allow_pickle=True)
    add("sandbox_c1_A", d["gen_A"], d["real"], K_SANDBOX)
    add("sandbox_c1_B", d["gen_B"], d["real"], K_SANDBOX)
    d = np.load(os.path.join(RES, "arms_stageD.npz"), allow_pickle=True)
    add("stageD_A", d["gen_A"], d["real"], K_REAL)
    add("stageD_B", d["gen_B"], d["real"], K_REAL)
    aug = np.load(os.path.join(REPO, "results", "npz", "arms_aug.npz"),
                  allow_pickle=True)
    add("tax_4a_nll_A", aug["gen_A"], aug["real"], K_REAL)
    forensic = np.load(os.path.join(RES, "forensic_nllnoise.npz"),
                       allow_pickle=True)
    add("tax_4a_muonly_A", forensic["gen_A"], forensic["real"], K_REAL)

    out = {"convention": {"K_sandbox": K_SANDBOX, "K_real": K_REAL,
                          "coef_stat_names": COEF_STAT_NAMES,
                          "corner_order": ["(0,0)", "(0,1)", "(1,0)", "(1,1)"]},
           "stacks": {}}
    for name, (gen, real, K) in stacks.items():
        t0 = time.time()
        out["stacks"][name] = score_stack(gen, real, K)
        m1 = out["stacks"][name]["M1_peak_parity"]
        log(f"{name}: M1 T(level1/2/3)="
            + "/".join(f"{m1[str(l)]['T']:.1f}" for l in LEVELS)
            + f"  [{time.time()-t0:.0f}s]")

    # ---- M4: hybrid transplants (index-paired) ---------------------------------
    out["M4_hybrids"] = {}
    for name, d, real_arr, K in (
            ("sandbox32_A", sandbox32, np.asarray(sandbox32["real"], np.float64),
             K_SANDBOX),
            ("repl64_A", repl64, np.asarray(repl_real, np.float64), K_SANDBOX),
            ("gow_c1t_A", gow_c1t, np.asarray(gow_c1t["real"], np.float64),
             K_REAL)):
        gen = np.asarray(d["gen_A"], np.float64)
        entry = {}
        for j in OCTAVES:
            hyb = [hybrid_with_gen_octave(real_arr[i], gen[i], octave=j)
                   for i in range(len(gen))]
            entry[f"real_with_gen_oct{j}"] = T_of(
                hyb, real_arr, lambda f: peak_parity_profile(f, k=K, level=1))
        degen = [hybrid_with_gen_octave(gen[i], real_arr[i], octave=1)
                 for i in range(len(gen))]
        entry["gen_with_real_oct1"] = T_of(
            degen, real_arr, lambda f: peak_parity_profile(f, k=K, level=1))
        out["M4_hybrids"][name] = entry
        log(f"M4 {name}: " + "  ".join(
            f"+oct{j}: T={entry[f'real_with_gen_oct{j}']['T']:.1f}"
            for j in OCTAVES)
            + f"  gen-minus-oct1: T={entry['gen_with_real_oct1']['T']:.1f}")

    path = os.path.join(RES, "parity_localization.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
