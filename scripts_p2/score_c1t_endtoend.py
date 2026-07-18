"""C1-t end-to-end scoring + mechanical bar adjudication (CPU, env.sh stack).

Scores results_p2/arms_c1t_sandbox.npz (TEST-half fields, generated from the
SELECTED checkpoint) with the FROZEN production scorer and adjudicates the
C3-frame bars (log/2026-07-18-prereg-c1t-sandbox.md; unchanged per R17: kurtosis
PRIMARY max(15%, 3*SE_rel) octaves 2-4 both levels; dispersion must-not-regress
max(10%, 3*SE_rel) both levels) against the normalized-convention exact truth,
together with the selected-checkpoint head-conditional numbers
(results_p2/c1t_selection_sandbox.json). q999 DESCRIPTIVE (R10 condition 1).
Writes results_p2/c1t_verdict_sandbox.json. All rules the prereg's, verbatim.
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
SEL = json.load(open(os.path.join(REPO, "results_p2",
                                  "c1t_selection_sandbox.json")))
ARMS = np.load(os.path.join(REPO, "results_p2", "arms_c1t_sandbox.npz"),
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
    ws = [octave_wc_pooled(f, j)[0].astype(np.float32) for f in fields]
    point = tail_q999(np.concatenate(ws))
    rng = np.random.default_rng(seed)
    n = len(ws)
    boot = [tail_q999(np.concatenate([ws[i] for i in rng.integers(0, n, n)]))
            for _ in range(n_boot)]
    return point, float(np.nanstd(np.asarray(boot), ddof=1))


out = {"levels": {}, "amplitude": {}, "q999": {},
       "selected_step": {a: SEL[a]["selected_step"] for a in ("A", "B")}}
print(f"selected checkpoints: A @{out['selected_step']['A']} "
      f"B @{out['selected_step']['B']}\n")

e2e, nan_flag = {}, {}
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
    q = {}
    for j in [1, 2, 3, 4]:
        pt, se = e2e_q999(fields, j)
        q[str(j)] = {"gen": pt, "gen_se": se, "truth": TRUTH_Q999[str(j)]["q999"],
                     "ratio": pt / TRUTH_Q999[str(j)]["q999"]}
    out["q999"][arm] = {"end_to_end": q}

print("=== C1-t SANDBOX VERDICT TABLE (prereg rules verbatim; KURTOSIS PRIMARY; "
      "TEST fields, selected ckpt) ===\n")
rows = []
for arm in ("A", "B"):
    for level, src in (("head-conditional", SEL[arm]["final"]),
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

print("\nDESCRIPTIVE q999 (ratio gen/truth):")
for arm in ("A", "B"):
    hcq = {j: SEL[arm]["final"][j] for j in ("2", "3", "4", "1")
           if j in SEL[arm]["final"]}
    out["q999"][arm]["head_conditional"] = {
        j: {"gen": hcq[j]["q999"], "truth": TRUTH_Q999[j]["q999"],
            "ratio": hcq[j]["q999"] / TRUTH_Q999[j]["q999"]} for j in hcq}
    hc = " ".join(f"oct{j}:{out['q999'][arm]['head_conditional'][j]['ratio']:.3f}"
                  for j in ("2", "3", "4"))
    e2 = " ".join(f"oct{j}:{out['q999'][arm]['end_to_end'][j]['ratio']:.3f}"
                  for j in ("2", "3", "4", "1"))
    print(f"  arm {arm}: head-cond {hc} | end-to-end {e2}")

print("\nDESCRIPTIVE octave 1 + selection-vs-20k (head-conditional oct2):")
for arm in ("A", "B"):
    hc1 = SEL[arm]["final"].get("1")
    s20 = SEL[arm]["final_at_20k"]["2"]
    ssel = SEL[arm]["final"]["2"]
    print(f"  arm {arm}: oct1 hc vs={hc1['var_slope']:.3f} kurt={hc1['kurtosis']:.2f}"
          f" | oct2 kurt selected={ssel['kurtosis']:.2f} vs at-20k={s20['kurtosis']:.2f}"
          f" (truth {TRUTH['2']['kurtosis']:.2f})")

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
        b = "C1T-DEG"
    elif not disp_all:
        b = "C1T-DISP-REGRESS"
    elif not kurt_hc:
        b = "C1T-TAILS-FAIL"
    elif kurt_hc and not kurt_e2e:
        b = "C1T-TAILS-HC-ONLY"
    else:
        b = "C1T-CAL"
    branch[arm] = b
    print(f"\narm {arm} branch: {b}")
out["branch"] = branch
out["nan_flag"] = nan_flag

out["gowerstreet_trigger"] = bool(
    all(level_pass(a, "head-conditional", "kurtosis")
        and level_pass(a, "head-conditional", "var_slope")
        and level_pass(a, "end-to-end", "var_slope") for a in ("A", "B")))
print(f"\ngowerstreet-leg trigger: "
      f"{'PASS' if out['gowerstreet_trigger'] else 'FAIL'}")

with open(os.path.join(REPO, "results_p2", "c1t_verdict_sandbox.json"), "w") as f:
    json.dump(out, f, indent=1)
print("\nwrote results_p2/c1t_verdict_sandbox.json")
