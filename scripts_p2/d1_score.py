"""D1 corrected-selection re-ship scorer (env.sh; prereg 2026-08-06-d1,
R48 order 1). Full battery at the CORRECTED pick + the tier-0 empty-beam
violation instrument + ledger-#17 in-artifact reference assertion, with
the BUGGED-pick numbers reported ALONGSIDE. A5 order: gates -> C/P-T ->
entries. This is a RE-STATEMENT, not a branch adjudication.
Usage: d1_score.py <tag> <leg oracle|seed> <lo> <ckpt> <bugged_final_npz>

Reconvene lines re-stated per artifact:
  declared_peak_pass  (E2: smoothed 0.5px nu 2.5/3.0, ci95 incl 0 AND
                       per-parent not sign-consistent) -> line "all three
                       corrected seeds pass declared-res peak rule" (60)
  mf_dev_declared_le_3p5 -> "corrected-oracle MF_dev <= 3.5 finals" (35)
  native_peaks_persist_10 (both nu native excess > +10%) -> "peaks
                       persist +10%+ on all three" (85)
Writes results_p2/d1_<tag>_verdict.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results_p2")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts_p2"))

from audit_peak_ci import bootstrap_excess, per_parent_excess
from coloring_index import stack_coloring
from constraint_tier0 import adjudicate as tier0_adjudicate
from measure_generated import couplings
from minkowski_judge import judge_T
from parity_localization import peak_parity_profile
from placement_instruments import nn_profile, stack_profiles, tstat

A3 = {"2.5": 0.15290, "3.0": 0.12474}
BARS = {"var_slope": 0.10, "kurtosis": 0.15}
BLOCKS = [(0, 10), (10, 21), (21, 32)]
BAR_MF, BAND_MF = 3.5, 0.25
K_REAL = 174
S0 = 20261460

tag, leg, lo, ckpt, bugged_npz = (sys.argv[1], sys.argv[2], int(sys.argv[3]),
                                  sys.argv[4], sys.argv[5])

# ---- ledger #17: assert reference identity IN-ARTIFACT ----------------------
ref = json.load(open(os.path.join(RES, "gowerstreet_val_ref.json")))["truth"]
recheck = json.load(open(os.path.join(RES, "night3_cage_recheck.json")))
ref_kurt = ref["2"]["kurtosis"]
ref_ok = abs(recheck["reference"]["kurtosis"] - ref_kurt) < 1e-6
out = {"tag": tag, "leg": leg, "ckpt": ckpt,
       "reference_assertion": {"gowerstreet_val_ref_oct2_kurtosis": ref_kurt,
                               "recheck_reference_kurtosis":
                                   recheck["reference"]["kurtosis"],
                               "SANDBOX_kurtosis_bug_value": 4.9171,
                               "identity_ok": bool(ref_ok)}}
assert ref_ok, "ledger #17: reference identity mismatch"

GEN = np.load(os.path.join(RES, f"stage3_{tag}_final.npz"))
streams = [[np.asarray(f, np.float64) for f in GEN[k]]
           for k in ("final1", "final2", "final3")]
pooled = [f for s in streams for f in s]
real = [np.asarray(f, np.float64) for f in
        np.load(os.path.join(RES, "arms_c1t_gowerstreet.npz"),
                allow_pickle=True)["real"]]

# ---- gates ------------------------------------------------------------------
W = json.load(open(os.path.join(RES, f"stage3_{tag}_white.json")))
PT = json.load(open(os.path.join(RES, f"stage3_{tag}_pt.json")))
gates_ok = (W["determinism_gate_maxabs"] == 0.0
            and PT["cal_history"][-1] <= 0.03)
out["gates"] = {"determinism": W["determinism_gate_maxabs"],
                "fit": PT["cal_history"], "ok": bool(gates_ok)}

# ---- C + P-T (watched) ------------------------------------------------------
c_pool, se_pool = stack_coloring(pooled, 1, seed=S0)
pt_lands = bool(PT["interval"][0] <= c_pool <= PT["interval"][1])
out["coloring"] = {"C_pooled": c_pool, "SE": se_pool, "pt": PT,
                   "PT_LANDS": pt_lands}
print(f"[{tag}] C={c_pool:.4f}±{se_pool:.4f} -> "
      f"{'LANDS' if pt_lands else 'FAILS'}")


def mf_and_peaks(pool, tag_seed):
    gs = [gaussian_filter(f, 0.5) for f in pool]
    rs = [gaussian_filter(f, 0.5) for f in real]
    t_dec, _ = judge_T(gs, rs, seed=tag_seed)
    t_nat, _ = judge_T(pool, real, seed=tag_seed + 1)
    # E2 declared-resolution peak rule
    e2 = {nu: bootstrap_excess(gs, rs, float(nu), seed=tag_seed + 2 + i)
          for i, nu in enumerate(("2.5", "3.0"))}
    strm = [pool[i:i + 32] for i in range(0, len(pool), 32)] \
        if len(pool) >= 32 else [pool]
    pps = [per_parent_excess([gaussian_filter(f, 0.5) for f in s], rs,
                             BLOCKS) for s in strm]
    panel = {nu: [float(np.mean([p[nu][b] for p in pps])) for b in range(3)]
             for nu in ("2.5", "3.0")}
    sign_cons = {nu: bool(all(v > 0 for v in panel[nu]) or
                          all(v < 0 for v in panel[nu])) for nu in panel}
    ci_ok = all(e2[nu]["ci95"][0] <= 0 <= e2[nu]["ci95"][1]
                for nu in e2)
    declared_peak_pass = bool(ci_ok and not any(sign_cons.values()))
    # native peaks (persist line)
    nat = {nu: bootstrap_excess(pool, real, float(nu), seed=tag_seed + 6 + i)
           for i, nu in enumerate(("2.5", "3.0"))}
    return {"MF_declared": float(t_dec), "MF_native": float(t_nat),
            "declared_peaks": {nu: e2[nu] for nu in e2},
            "declared_panel": panel, "declared_sign_consistent": sign_cons,
            "declared_peak_pass": declared_peak_pass,
            "native_peaks": {nu: {k: nat[nu][k]
                                  for k in ("excess", "se", "ci95")}
                             for nu in nat}}


cur = mf_and_peaks(pooled, S0 + 10)
mf_cat = ("pass" if cur["MF_declared"] < BAR_MF - BAND_MF else
          ("fail" if cur["MF_declared"] > BAR_MF + BAND_MF else "at-bar"))
out["MF_dev"] = {"declared": cur["MF_declared"], "category": mf_cat,
                 "native_desc": cur["MF_native"]}
out["declared_peaks"] = {"excess": cur["declared_peaks"],
                         "panel": cur["declared_panel"],
                         "sign_consistent": cur["declared_sign_consistent"],
                         "pass": cur["declared_peak_pass"]}
out["native_peaks"] = cur["native_peaks"]
peaks_persist = all(cur["native_peaks"][nu]["excess"] > 0.10
                    for nu in ("2.5", "3.0"))
out["reconvene_lines"] = {
    "declared_peak_pass": cur["declared_peak_pass"],
    "mf_dev_declared_le_3p5": bool(cur["MF_declared"] <= BAR_MF),
    "native_peaks_persist_10": bool(peaks_persist)}
print(f"[{tag}] MF(dev) {cur['MF_declared']:.2f} ({mf_cat}); "
      f"declared-peak-pass={cur['declared_peak_pass']}; "
      f"peaks {cur['native_peaks']['2.5']['excess']:+.1%}/"
      f"{cur['native_peaks']['3.0']['excess']:+.1%}")

# ---- tier-0 empty-beam violation instrument (new battery row) ---------------
out["tier0"] = tier0_adjudicate(pooled, real, seed=S0 + 20)
print(f"[{tag}] tier0: floor={out['tier0']['floor']:.3f} "
      f"v_gen={out['tier0']['v_gen']:.2e} v_null={out['tier0']['v_null']:.2e} "
      f"z={out['tier0']['z']:.2f} violation={out['tier0']['violation']}")

# ---- marginals: e2e octaves 1-4 + hc from d1_<tag>_hc.json ------------------
real_ref = couplings(real, [1, 2, 3, 4], n_boot=50, seed=0)
e2e = couplings(pooled, [1, 2, 3, 4], n_boot=200, seed=0)
hc = json.load(open(os.path.join(RES, f"d1_{tag}_hc.json")))["final"]
rows, marg_ok = {}, True
for level, src_by_j in (("head-conditional",
                         {j: hc[str(j)] for j in (1, 2, 3, 4)}),
                        ("end-to-end", {j: e2e[j] for j in (1, 2, 3, 4)})):
    for j in (1, 2, 3, 4):
        for metric in ("var_slope", "kurtosis"):
            t, tse = real_ref[j][metric], real_ref[j][metric + "_se"]
            v, se = src_by_j[j][metric], src_by_j[j][metric + "_se"]
            rel = abs(v - t) / abs(t)
            bar = max(BARS[metric], 3 * float(np.hypot(se, tse) / abs(t)))
            ok = bool(rel <= bar)
            rows.setdefault(level, {}).setdefault(str(j), {})[metric] = \
                {"rel_err": rel, "bar": bar, "pass": ok}
            marg_ok &= ok


def T_of(gen, fn):
    a = [fn(np.asarray(f, np.float64)) for f in gen]
    b = [fn(f) for f in real]
    T, _ = tstat(*stack_profiles(a), *stack_profiles(b))
    return float(T)


parity = T_of(pooled, lambda f: peak_parity_profile(f, k=K_REAL, level=1))
NN_EDGES = np.array(json.load(open(os.path.join(
    RES, "placement_phase_a.json")))["convention"]["nn_edges"])
nns = [T_of(s, lambda f: nn_profile(f, k=K_REAL, edges=NN_EDGES))
       for s in streams]
st_ok = None
p = os.path.join(RES, f"starlet_l1_d1_{tag}.json")
if os.path.exists(p):
    st_ok = bool(json.load(open(p))["checks"]["gen_A"]["all_scored_pass"])
out["battery"] = {"marginals": rows, "marg_ok": bool(marg_ok),
                  "parity_T": parity, "starlet": st_ok,
                  "nn_mean_sd": [float(np.mean(nns)),
                                 float(np.std(nns, ddof=1))]}

# ---- bugged-pick ALONGSIDE (same statistics, the R47/R43 record) ------------
BG = np.load(os.path.join(RES, bugged_npz))
bg_pool = [np.asarray(f, np.float64)
           for k in ("final1", "final2", "final3") for f in BG[k]]
bug = mf_and_peaks(bg_pool, S0 + 30)
out["bugged_alongside"] = {
    "npz": bugged_npz,
    "MF_declared": bug["MF_declared"], "MF_native": bug["MF_native"],
    "declared_peak_pass": bug["declared_peak_pass"],
    "native_peaks": {nu: bug["native_peaks"][nu]["excess"]
                     for nu in ("2.5", "3.0")},
    "tier0": tier0_adjudicate(bg_pool, real, seed=S0 + 40)}
print(f"[{tag}] BUGGED alongside: MF {bug['MF_declared']:.2f} "
      f"declared-peak-pass={bug['declared_peak_pass']} "
      f"peaks {bug['native_peaks']['2.5']['excess']:+.1%}/"
      f"{bug['native_peaks']['3.0']['excess']:+.1%} "
      f"tier0-viol={out['bugged_alongside']['tier0']['violation']}")

with open(os.path.join(RES, f"d1_{tag}_verdict.json"), "w") as f:
    json.dump(out, f, indent=1)
print(f"[{tag}] marginals={marg_ok} parity={parity:.2f} starlet={st_ok} "
      f"nn={np.mean(nns):.2f}±{np.std(nns, ddof=1):.2f} | LINES "
      f"declared-peak={out['reconvene_lines']['declared_peak_pass']} "
      f"mf<=3.5={out['reconvene_lines']['mf_dev_declared_le_3p5']} "
      f"persist={out['reconvene_lines']['native_peaks_persist_10']}")
