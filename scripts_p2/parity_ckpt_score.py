"""Phase A' M5, SCORE phase (env.sh stack, pywt). Scores the checkpoint-curve
generations (results_p2/parity_ckpt_gen.npz) against the frozen 32 test tiles:
M1 level-1 peak parity T and the odd-odd share; M2 octave-1 and octave-2
coefficient stats (T + the three channel-mean z's). Writes
results_p2/parity_ckpt_curve.json.
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

from parity_localization import coef_stats_profile, peak_parity_profile
from placement_instruments import stack_profiles, tstat

K = 160          # phase-A frozen sandbox convention


def T_of(gen, real, fn):
    a = [fn(np.asarray(f, np.float64)) for f in gen]
    b = [fn(np.asarray(f, np.float64)) for f in real]
    ma, sa = stack_profiles(a)
    mb, sb = stack_profiles(b)
    T, z = tstat(ma, sa, mb, sb)
    return T, z, ma


def main():
    gen = np.load(os.path.join(RES, "parity_ckpt_gen.npz"), allow_pickle=True)
    real = np.load(os.path.join(REPO, "data_cache",
                                "tiles_sandbox.npz"))["sandbox"][-32:]
    out = {"K": K, "curve": {}}
    for key in sorted(gen.files, key=lambda s: (s.split("_s")[0],
                                                int(s.split("_s")[1]))):
        arm, step = key.split("_s")
        Tp, zp, mp = T_of(gen[key], real,
                          lambda f: peak_parity_profile(f, k=K, level=1))
        entry = {"parity_T": Tp, "oddodd_share": float(mp[3]),
                 "parity_gen_mean": [float(x) for x in mp]}
        for j in (1, 2):
            Tc, zc, _ = T_of(gen[key], real,
                             lambda f, j=j: coef_stats_profile(f, octave=j))
            entry[f"coef_oct{j}_T"] = Tc
            entry[f"coef_oct{j}_mean_z"] = [float(zc[i]) for i in range(3)]
        out["curve"][key] = entry
        print(f"{key}: parity T={Tp:.1f} oddodd={mp[3]:.3f} "
              f"coef1 T={entry['coef_oct1_T']:.1f} "
              f"mean-z={np.round(entry['coef_oct1_mean_z'],1)}")
    path = os.path.join(RES, "parity_ckpt_curve.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
