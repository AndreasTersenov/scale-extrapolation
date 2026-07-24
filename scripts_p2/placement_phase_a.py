"""Phase A of the placement experiment (log/2026-07-24-prereg-placement.md, R30 GO).

CPU only; no model runs. Steps, in prereg order:
 1. Truth reference: Stage-A exact ensemble (scratch), 256 parents x redraws 0..7,
    profiles aggregated per parent FIRST (SE over 256 parents — ledger #10).
    K (count-matching) frozen = round(mean # local maxima above NU=2.5) on the
    reference; NN decile edges frozen from the reference's pooled NN distances.
 2. Gates per instrument: truth-null (parents 0..127 vs 128..255, T<3) and
    surrogate detection (rank-remapped phase randomization of redraw 8, one per
    parent, T>=5).
 3. Primary-instrument rule (mechanical): argmax T_surr among instruments passing
    both gates — committed here BEFORE any model scoring below.
 4. Measured references: committed 1x-regime artifacts (arms_c1t_sandbox 32,
    c1t_repl64 64; arms A and B) scored against the truth reference.
    GATE-S: T_primary(sandbox arm A) >= 3 required (else the gate FIRES).
 5. Real-field echo (descriptive, transfer clause): committed gowerstreet C1-t and
    Stage-D arms vs their own real test tiles; field-level SEs + per-parent
    profile deltas over the verified parent blocks [(0,10),(10,21),(21,32)]
    (K_real frozen from the real tiles).

Writes results_p2/placement_phase_a.json. Deterministic (seeded surrogates).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")
SCRATCH = os.path.expanduser("~/links/scratch/scale-extrap-p2")

import sys
sys.path.insert(0, os.path.join(REPO, "scripts_p2"))
from placement_instruments import (  # noqa: E402
    _local_maxima, effect_l2, env_rate_profile, nn_profile, parity_profile,
    pk2pt_profile, stack_profiles, surrogate, tstat,
)

NU = 2.5
N_REDRAWS_REF = 8
SURR_REDRAW = 8
PARENT_BLOCKS = [(0, 10), (10, 21), (21, 32)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def instruments(k, edges):
    return {
        "env_rate": lambda f: env_rate_profile(f, k=k, lvl=2),
        "nn": lambda f: nn_profile(f, k=k, edges=edges),
        "pk2pt": lambda f: pk2pt_profile(f, k=k),
        "parity": lambda f: parity_profile(f, k=k),
    }


def profiles_by_parent(ens, redraws, fn):
    """Per-parent mean profile (exchangeable unit = parent; ledger #10)."""
    out = []
    for p in range(ens.shape[0]):
        profs = [fn(np.asarray(ens[p, r], np.float64)) for r in redraws]
        profs = [q for q in profs if q is not None]
        if profs:
            out.append(np.mean(profs, axis=0))
    return out


def score_set(fields, ref_mean, ref_se, fn):
    profs = [fn(np.asarray(f, np.float64)) for f in fields]
    m, se = stack_profiles(profs)
    T, z = tstat(m, se, ref_mean, ref_se)
    return {"T": T, "z": [float(v) for v in z], "l2": effect_l2(m, ref_mean),
            "mean": [float(v) for v in m], "se": [float(v) for v in se],
            "n": int(len([p for p in profs if p is not None]))}


def main():
    ens = np.load(os.path.join(SCRATCH, "sandbox_ens_f32.npy"), mmap_mode="r")
    P = ens.shape[0]
    log(f"ensemble {ens.shape}")

    # ---- freeze K and NN edges from the truth reference ------------------------
    counts, nn_pool = [], []
    for p in range(P):
        for r in range(N_REDRAWS_REF):
            f = np.asarray(ens[p, r], np.float64)
            ys, xs, h = _local_maxima(f)
            counts.append(int((h > NU).sum()))
    K = int(round(float(np.mean(counts))))
    log(f"K frozen = {K} (mean count above nu={NU}: {np.mean(counts):.2f})")

    from placement_instruments import topk_peaks_xy
    rng = np.random.default_rng(0)
    for p in range(0, P, 4):                       # 64 parents suffice for edges
        f = np.asarray(ens[p, 0], np.float64)
        ys, xs, _ = topk_peaks_xy(f, K)
        pts = np.stack([ys, xs], 1).astype(float)
        d2 = ((pts[:, None] - pts[None]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        nn_pool.extend(np.sqrt(d2.min(1)))
    edges = np.quantile(np.array(nn_pool), np.arange(1, 10) / 10.0)
    edges = np.unique(np.round(edges, 6))   # grid distances are discrete ->
    # duplicate quantiles create structurally-empty classes (validation finding)
    log(f"NN edges frozen (deduped deciles): {np.round(edges, 3)}")

    INS = instruments(K, edges)
    out = {"convention": {"nu_for_K": NU, "K": K,
                          "nn_edges": [float(e) for e in edges],
                          "ref": f"parents x redraws 0..{N_REDRAWS_REF-1}, "
                                 "parent-aggregated (ledger #10)"},
           "gates": {}, "references": {}, "echo": {}}

    # ---- truth reference + gates ----------------------------------------------
    ref, null_T, surr_T = {}, {}, {}
    for name, fn in INS.items():
        t0 = time.time()
        by_parent = profiles_by_parent(ens, range(N_REDRAWS_REF), fn)
        m, se = stack_profiles(by_parent)
        ref[name] = (m, se, by_parent)
        h = len(by_parent) // 2
        Tn, _ = tstat(*stack_profiles(by_parent[:h]),
                      *stack_profiles(by_parent[h:]))
        surr = [fn(surrogate(np.asarray(ens[p, SURR_REDRAW], np.float64),
                             np.random.default_rng(1000 + p))) for p in range(P)]
        ms, ses = stack_profiles(surr)
        Ts, _ = tstat(ms, ses, m, se)
        null_T[name], surr_T[name] = Tn, Ts
        # parity is the seam/confound instrument (prereg: "a confound gate, not
        # a primary candidate"): it is uniform for truth AND surrogate by
        # symmetry, so the surrogate gate cannot apply to it — it validates on
        # the truth-null alone.
        if name == "parity":
            passed, role = bool(Tn < 3.0), "confound"
        else:
            passed, role = bool(Tn < 3.0 and Ts >= 5.0), "primary-candidate"
        out["gates"][name] = {"T_null": Tn, "T_surrogate": Ts, "pass": passed,
                              "role": role}
        log(f"{name}: T_null={Tn:.2f} T_surr={Ts:.1f} "
            f"{'PASS' if passed else 'FAIL'} [{role}] ({time.time()-t0:.0f}s)")

    validated = [n for n in INS if out["gates"][n]["pass"]]
    candidates = [n for n in validated
                  if out["gates"][n]["role"] == "primary-candidate"]
    primary = max(candidates, key=lambda n: surr_T[n]) if candidates else None
    out["primary"] = primary
    out["primary_rule"] = ("argmax T_surrogate among primary-candidate "
                           "instruments passing both gates")
    if primary is None:
        out["gate_i"] = {"fires": True}
        with open(os.path.join(RES, "placement_phase_a.json"), "w") as f:
            json.dump(out, f, indent=1)
        log("GATE-I FIRES: no primary-candidate instrument passed — STOP")
        return
    out["gate_i"] = {"fires": False}
    log(f"PRIMARY = {primary} (committed before any model scoring)")

    # ---- measured references: committed 1x artifacts ---------------------------
    for label, path, arms in (
            ("sandbox32", os.path.join(RES, "arms_c1t_sandbox.npz"), ("A", "B")),
            ("repl64", os.path.join(RES, "c1t_repl64_gen.npz"), ("A", "B"))):
        d = np.load(path, allow_pickle=True)
        for arm in arms:
            key = f"{label}_{arm}"
            out["references"][key] = {}
            for name in validated:
                m, se, _ = ref[name]
                out["references"][key][name] = score_set(d[f"gen_{arm}"], m, se,
                                                         INS[name])
            log(f"ref {key}: " + "  ".join(
                f"{n}: T={out['references'][key][n]['T']:.1f}" for n in validated))

    gate_s_T = out["references"]["sandbox32_A"][primary]["T"]
    out["gate_s"] = {"T_primary_sandbox_A": gate_s_T, "fires": bool(gate_s_T < 3.0)}
    log(f"GATE-S: T_primary(sandbox A) = {gate_s_T:.1f} -> "
        f"{'FIRES (STOP)' if out['gate_s']['fires'] else 'clear'}")

    # ---- real-field echo (descriptive; K_real from the real tiles) -------------
    for label, path in (("gowerstreet_c1t", os.path.join(RES, "arms_c1t_gowerstreet.npz")),
                        ("stageD", os.path.join(RES, "arms_stageD.npz"))):
        d = np.load(path, allow_pickle=True)
        real = [np.asarray(f, np.float64) for f in d["real"]]
        counts_r = [int((_local_maxima(f)[2] > NU).sum()) for f in real]
        K_real = int(round(float(np.mean(counts_r))))
        INS_R = instruments(K_real, edges)
        rp = {n: [INS_R[n](f) for f in real] for n in validated}
        out["echo"][label] = {"K_real": K_real}
        for arm in ("A", "B"):
            entry = {}
            for name in validated:
                m, se = stack_profiles(rp[name])
                entry[name] = score_set(d[f"gen_{arm}"], m, se, INS_R[name])
                # per-parent profile deltas (descriptive, blocks verified)
                deltas = []
                for a, b in PARENT_BLOCKS:
                    mg, _ = stack_profiles(
                        [INS_R[name](np.asarray(f, np.float64))
                         for f in d[f"gen_{arm}"][a:b]])
                    mr, _ = stack_profiles(rp[name][a:b])
                    deltas.append(effect_l2(mg, mr))
                entry[name]["per_parent_l2"] = deltas
            out["echo"][label][arm] = entry
            log(f"echo {label} {arm}: " + "  ".join(
                f"{n}: T={entry[n]['T']:.1f}" for n in validated))

    path = os.path.join(RES, "placement_phase_a.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
