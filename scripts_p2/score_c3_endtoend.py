"""C3 end-to-end scoring + mechanical bar adjudication (CPU, env.sh stack).

Scores results_p2/arms_c3_sandbox.npz with the FROZEN production scorer
(scripts/measure_generated.couplings) and adjudicates the pre-registered C3 bars
(log/2026-07-17-prereg-c3-sandbox.md, approved R10) against the normalized-
convention exact truth, together with the head-conditional results
(results_p2/c3_conditional_sandbox.json). KURTOSIS PRIMARY; dispersion
must-not-regress is a bar. The 99.9th-percentile |coefficient| readout (R10
condition 1) is DESCRIPTIVE — printed and stored, never adjudicating. Writes
results_p2/c3_verdict_sandbox.json. All rules are the prereg's, applied verbatim.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from measure_generated import couplings  # frozen production path
from sandbox.haar import octave_wc_pooled
from sandbox.truth_stats import tail_q999

TRUTH = json.load(open(os.path.join(
    REPO, "results_p2", "sandbox_truth_normconv.json")))["truth"]
TRUTH_Q999 = json.load(open(os.path.join(
    REPO, "results_p2", "sandbox_truth_q999.json")))["truth"]
COND = json.load(open(os.path.join(REPO, "results_p2",
                                   "c3_conditional_sandbox.json")))
ARMS = np.load(os.path.join(REPO, "results_p2", "arms_c3_sandbox.npz"),
               allow_pickle=True)

BARS = {"var_slope": 0.10, "kurtosis": 0.15}
TRAINED = [2, 3, 4]


def check(metric, val, se, j):
    t = TRUTH[str(j)][metric]
    tse = TRUTH[str(j)][metric + "_se"]
    rel = abs(val - t) / abs(t)
    se_rel = float(np.hypot(se, tse) / abs(t))
    bar = max(BARS[metric], 3 * se_rel)
    return {"value": val, "se": se, "truth": t, "rel_err": rel, "bar": bar,
            "pass": bool(rel <= bar)}


def e2e_q999(fields, j, n_boot=200, seed=0):
    """Pooled-standardized q999 over generated fields at octave j + bootstrap SE."""
    ws = [octave_wc_pooled(f, j)[0].astype(np.float32) for f in fields]
    point = tail_q999(np.concatenate(ws))
    rng = np.random.default_rng(seed)
    n = len(ws)
    boot = [tail_q999(np.concatenate([ws[i] for i in rng.integers(0, n, n)]))
            for _ in range(n_boot)]
    return point, float(np.nanstd(np.asarray(boot), ddof=1))


out = {"levels": {}, "curve": {}, "amplitude": {}, "q999": {}}

# ---- end-to-end (frozen scorer) ---------------------------------------------------
e2e = {}
nan_flag = {}
for arm in ("A", "B"):
    fields = [np.asarray(f, dtype=np.float64) for f in ARMS[f"gen_{arm}"]]
    nan_flag[arm] = bool(any(not np.all(np.isfinite(f)) for f in fields))
    e2e[arm] = couplings(fields, [1, 2, 3, 4], n_boot=200, seed=0)
    if "real_ref" not in out:
        real_fields = [np.asarray(f, dtype=np.float64) for f in ARMS["real"]]
        out["real_ref"] = couplings(real_fields, [1, 2, 3, 4], n_boot=50, seed=0)
    amp = {}
    for j in TRAINED:
        ds = e2e[arm][j]["detail_std"]
        tds = out["real_ref"][j]["detail_std"]
        amp[j] = {"gen": ds, "real": tds, "rel": abs(ds - tds) / tds}
    out["amplitude"][arm] = amp
    # R10 condition 1: descriptive extreme-tail readout, end-to-end level
    q = {}
    for j in [1, 2, 3, 4]:
        pt, se = e2e_q999(fields, j)
        tq = TRUTH_Q999[str(j)]["q999"]
        q[str(j)] = {"gen": pt, "gen_se": se, "truth": tq,
                     "ratio": pt / tq}
    out["q999"][arm] = {"end_to_end": q}

print("=== C3 SANDBOX VERDICT TABLE (prereg rules verbatim; KURTOSIS PRIMARY) ===\n")
rows = []
for arm in ("A", "B"):
    for level, src in (("head-conditional", COND[arm]["final"]),
                       ("end-to-end", e2e[arm])):
        for j in TRAINED:
            s = src[str(j)] if isinstance(src, dict) and str(j) in src else src[j]
            for metric in ("var_slope", "kurtosis"):
                r = check(metric, s[metric], s[metric + "_se"], j)
                rows.append((arm, level, j, metric, r))
                out["levels"].setdefault(arm, {}).setdefault(level, {}).setdefault(
                    str(j), {})[metric] = r

hdr = (f"{'arm':>3} {'level':>16} {'oct':>3} {'metric':>9} | {'gen':>8} "
       f"{'truth':>8} {'rel':>6} {'bar':>6} | P/F")
print(hdr)
for arm, level, j, metric, r in rows:
    print(f"{arm:>3} {level:>16} {j:>3} {metric:>9} | {r['value']:8.3f} "
          f"{r['truth']:8.3f} {r['rel_err']:6.1%} {r['bar']:6.1%} | "
          f"{'PASS' if r['pass'] else 'FAIL'}")

# ---- DESCRIPTIVE: q999 tail readout (R10 condition 1) -----------------------------
print("\nDESCRIPTIVE q999 (|coefficient| 99.9th pct, pooled standardized; "
      "ratio = gen/truth):")
for arm in ("A", "B"):
    hcq = {j: COND[arm]["final"][str(j)] for j in ("2", "3", "4", "1")
           if str(j) in COND[arm]["final"]}
    out["q999"][arm]["head_conditional"] = {
        j: {"gen": hcq[j]["q999"], "gen_se": hcq[j]["q999_se"],
            "truth": TRUTH_Q999[j]["q999"],
            "ratio": hcq[j]["q999"] / TRUTH_Q999[j]["q999"]} for j in hcq}
    hc_str = " ".join(f"oct{j}:{out['q999'][arm]['head_conditional'][j]['ratio']:.3f}"
                      for j in ("2", "3", "4"))
    e2_str = " ".join(f"oct{j}:{out['q999'][arm]['end_to_end'][j]['ratio']:.3f}"
                      for j in ("2", "3", "4", "1"))
    print(f"  arm {arm}: head-cond {hc_str} | end-to-end {e2_str}")

# ---- DESCRIPTIVE: octave 1 (extrapolated) ------------------------------------------
print("\nDESCRIPTIVE octave 1 (extrapolated):")
for arm in ("A", "B"):
    hc = COND[arm]["final"].get("1")
    t1 = TRUTH["1"]
    e1 = e2e[arm][1]
    out.setdefault("oct1_descriptive", {})[arm] = {
        "head_conditional": hc,
        "end_to_end": {k: e1[k] for k in ("var_slope", "var_slope_se",
                                          "kurtosis", "kurtosis_se", "detail_std")}}
    print(f"  arm {arm}: head-cond vs={hc['var_slope']:.3f} kurt={hc['kurtosis']:.2f}"
          f" | end-to-end vs={e1['var_slope']:.3f} kurt={e1['kurtosis']:.2f}"
          f" | truth vs={t1['var_slope']:.3f} kurt={t1['kurtosis']:.2f}")

# ---- checkpoint curves: collapse rule on var_slope; kurtosis curve DESCRIPTIVE -----
print("\ncheckpoint curves (oct-2 head-conditional):")
collapse = {}
for arm in ("A", "B"):
    curve = {int(k): v["var_slope"] for k, v in COND[arm]["curve_oct2"].items()}
    curve[20000] = COND[arm]["final"]["2"]["var_slope"]
    kcurve = {int(k): v["kurtosis"] for k, v in COND[arm]["curve_oct2"].items()}
    kcurve[20000] = COND[arm]["final"]["2"]["kurtosis"]
    steps = sorted(curve)
    peak = -np.inf
    fired = False
    for s_ in steps:
        peak = max(peak, curve[s_])
        if peak - curve[s_] >= 0.10:
            fired = True
    collapse[arm] = {"curve": {str(s): curve[s] for s in steps},
                     "kurtosis_curve": {str(s): kcurve[s] for s in steps},
                     "collapse_signature": fired}
    print(f"  arm {arm} vs:   " + " ".join(f"{s//1000}k:{curve[s]:.3f}"
                                           for s in steps)
          + f"  -> collapse={'FIRES' if fired else 'no'}")
    print(f"  arm {arm} kurt: " + " ".join(f"{s//1000}k:{kcurve[s]:.2f}"
                                           for s in steps) + "  (descriptive)")
out["curve"] = collapse

# ---- branch adjudication (prereg precedence top->bottom) ---------------------------
def level_pass(arm, level, metric):
    return all(out["levels"][arm][level][str(j)][metric]["pass"] for j in TRAINED)

branch = {}
for arm in ("A", "B"):
    deg = (any(out["amplitude"][arm][j]["rel"] > 0.25 for j in TRAINED)
           or nan_flag[arm])
    disp_all = (level_pass(arm, "head-conditional", "var_slope")
                and level_pass(arm, "end-to-end", "var_slope"))
    kurt_hc = level_pass(arm, "head-conditional", "kurtosis")
    kurt_e2e = level_pass(arm, "end-to-end", "kurtosis")
    if deg:
        b = "C3-DEG"
    elif not disp_all:
        b = "C3-DISP-REGRESS"
    elif not kurt_hc:
        b = "C3-TAILS-FAIL"
    elif kurt_hc and not kurt_e2e:
        b = "C3-TAILS-HC-ONLY"
    elif disp_all and kurt_hc and kurt_e2e:
        b = "C3-CAL"
    else:
        b = "other"
    branch[arm] = b
    print(f"\narm {arm} branch: {b}")
out["branch"] = branch
out["nan_flag"] = nan_flag

# gowerstreet-leg trigger (reported; tonight's orders STOP at the sandbox readout)
out["leg2_trigger"] = bool(
    all(level_pass(a, "head-conditional", "kurtosis")
        and level_pass(a, "head-conditional", "var_slope")
        and level_pass(a, "end-to-end", "var_slope") for a in ("A", "B")))
print(f"\ngowerstreet-leg trigger (kurtosis PRIMARY head-conditional both arms AND "
      f"dispersion both levels both arms): "
      f"{'PASS' if out['leg2_trigger'] else 'FAIL'}")

with open(os.path.join(REPO, "results_p2", "c3_verdict_sandbox.json"), "w") as f:
    json.dump(out, f, indent=1)
print("\nwrote results_p2/c3_verdict_sandbox.json")
