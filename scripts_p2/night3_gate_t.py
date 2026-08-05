"""GATE-T scoring (env.sh; prereg 2026-08-05-night3 + A-N3-1; mechanical).
Rule: FIRES iff |A_or(real)| − |A_or(gen)| ≥ 3σ (combined bootstrap SE) on
the PRIMARY trained leg (l1pp adj pooled 96 vs gow real 32). Blind leg
descriptive corroboration. Signs reported. Writes
results_p2/night3_gate_t.json."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")
sys.path.insert(0, os.path.join(REPO, "scripts_p2"))

from orientation_stat import stack_orientation

L = np.load(os.path.join(RES, "l1pp_main_gen.npz"))
trained_gen = [np.asarray(L[k][i], np.float64)
               for k in ("adj1", "adj2", "adj3") for i in range(L[k].shape[0])]
gow_real = [np.asarray(f, np.float64) for f in
            np.load(os.path.join(RES, "arms_c1t_gowerstreet.npz"),
                    allow_pickle=True)["real"]]
B = np.load(os.path.join(RES, "stage3_blind_final.npz"))
blind_gen = [np.asarray(B[k][i], np.float64)
             for k in ("final1", "final2", "final3")
             for i in range(B[k].shape[0])]
sd_real = [np.asarray(f, np.float64) for f in
           np.load(os.path.join(RES, "arms_stageD.npz"),
                   allow_pickle=True)["real"]]

out = {}
for leg, gen, real, s in (("trained_PRIMARY", trained_gen, gow_real, 7400),
                          ("blind_descriptive", blind_gen, sd_real, 7410)):
    mg, sg = stack_orientation(gen, seed=s)
    mr, sr = stack_orientation(real, seed=s + 1)
    dec = abs(mr) - abs(mg)
    sigma = float(np.hypot(sg, sr))
    out[leg] = {"A_gen": mg, "SE_gen": sg, "A_real": mr, "SE_real": sr,
                "decoherence_absdiff": dec, "sigma": sigma,
                "z": dec / sigma}
    print(f"{leg}: A_real={mr:+.4f}±{sr:.4f} A_gen={mg:+.4f}±{sg:.4f} "
          f"|real|-|gen|={dec:+.4f} z={dec / sigma:+.2f}")
fires = bool(out["trained_PRIMARY"]["z"] >= 3.0)
out["GATE_T"] = {"rule": "|A_or(real)| - |A_or(gen)| >= 3 sigma, primary",
                 "fires": fires}
with open(os.path.join(RES, "night3_gate_t.json"), "w") as f:
    json.dump(out, f, indent=1)
print(f"GATE-T: {'FIRES -> TIDAL trains' if fires else 'NOT FIRED -> TIDAL descoped'}")
