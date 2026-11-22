#!/usr/bin/env python3
"""What the class-E record does not hold, computed on the record's OWN resamples with the paper's own code imported
read-only (nothing of it is modified):
  a14/classE/classE.py   sample_rows (conf149 / all179: tables.mbb(179, 4), seed 0, the 30 seen columns removed and each
                         resample rescaled), load_refs (v3's reference tables and curves in the order of the pass, with its
                         identity asserts), tables_from_k, pi_table, REFS_ISSUED, REFS_MOVED, THR_ALL, THR_REP
  a13/warn16/score16.py  persist_at (a reference at the number of people warned by a Foehn system; the routine of the record)
  analyses/hidden/tables.py  Tab, metrics, parts, tri (pooled sums, percentile intervals)
Computed for the samples conf149 (the tables) and all179 (the identity test), leads 1 and 3, thresholds 15, 37.5, 75:
  1. perfect_40km as issued: truth only, the '40 km cell above' bit of the truth codes of v3's T1 tables;
  2. foehn_mean as issued: the ensemble mean of the FOUR members of the evaluation (v3 T1, system foehn_mean);
  3. every reference as issued (REFS_ISSUED) with its own point and interval;
  4. every moved reference (REFS_MOVED) at the number of people warned by Foehn with M = 8 ('at least 4 of 8', exact from the
     sixteen members of the pass), 16 and 4 (the stored 'at least 2 of 4' table of the pass): point, interval, factor, and the
     paired difference recomputed as the record does (checked against the record by build_warn8.py);
  5. regions (country groups G:), 37.5 ug m-3: the CAMS forecast as issued with its interval;
  6. the decision-threshold sweep of cams_qm and cams_bcsd (v3's tables of the systems at the factors of version 2).
Intervals: percentile intervals of the resamples (plain bootstrap; no member draw enters them).
    python -B complete_warn8.py [--smoke]      (FOEHN_DESIGN=ar for the other design) -> out_<label>/complements.json
--smoke: one block (conf149, lead 1, 37.5), printed, nothing written."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_warn8 as B  # noqa: E402  (design, output folder, safe_write)

sys.dont_write_bytecode = True
sys.path.insert(0, str(B.P / "a14" / "classE"))
import classE as CE  # noqa: E402  (read-only import: functions and constants; nothing runs at import)

s16, v3, bc = CE.s16, CE.v3, CE.bc
Tab, metrics, parts, tri = CE.Tab, CE.metrics, CE.parts, s16.tri
SMOKE = "--smoke" in sys.argv
SAMPLES = ("conf149",) if SMOKE else ("conf149", "all179")
LEADS = (1,) if SMOKE else (1, 3)
THRS = (37.5,) if SMOKE else tuple(CE.THR_REP)
MS = ("M8_liberal", "M16_liberal_measured", "M4_stored")
SWEEP = ("cams_qm", "cams_bcsd")
SWEEPM = ("warned_M", "right_M", "false_M", "reach_pct", "correct_pct", "hidden_warned_M", "hidden_warned_pct", "false_in_silent40_M")
assert CE.DESIGN == B.DESIGN


def clean(v):
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        v = v.item()
    return None if isinstance(v, float) and not np.isfinite(v) else v


def ci_all(tab):
    with np.errstate(all="ignore"):
        bs = tab.bs
        return {k: [float(tab.pt[k])] + [float(x) for x in np.nanpercentile(bs[k], [2.5, 97.5])] if np.isfinite(bs[k]).any()
                else [float(tab.pt[k]), None, None] for k in tab.pt}


def finfo(p):
    p = Path(p); st = p.stat()
    return dict(path=str(p), bytes=st.st_size, mtime_ns=st.st_mtime_ns)


def main():
    t0 = time.time()
    hist = CE.DATA / "hist_all2022_v2.npz"
    z = np.load(hist)
    CE.S["Z"] = {k: z[k] for k in ("keys", "HK", "T", "Hh")}
    CE.load_refs()                                                     # asserts: persist / stored-4 tables and persist curves = v3
    Z, RT, RH = CE.S["Z"], CE.S["RT"], CE.S["RH"]
    K = Z["keys"]
    v3f = v3.DATA / "hidden_v3_pop.npz"
    zv = np.load(v3f)
    Kv = zv["keys"]; rows_v = [int(np.flatnonzero((Kv[:, 0] == d) & (Kv[:, 1] == h))[0]) for d, h, _ in K]
    assert np.array_equal(Kv[rows_v], K)
    s1 = [str(s) for s in zv["sys1"]]
    T1 = zv["T1"][rows_v]                                              # (358, regions, thresholds, systems, 16), order of the pass
    FM = T1[..., s1.index("foehn_mean"), :]
    TR = T1[..., s1.index("foehn_p50"), :]
    N8 = TR[..., :8] + TR[..., 8:]                                     # residents by truth code (the same for every system)
    C40 = (np.arange(8) & 2) > 0                                       # code bit 2: the mean of the 40 km cell is above
    PF = np.concatenate([N8 * ~C40, N8 * C40], -1)                     # the perfect 40 km forecast: warned where the cell is above
    bcf = (v3.HERE.parent / "v2" / f"data{CE.SFX}" if CE.AR else v3.DATA) / "hidden_bc_pop.npz"   # as classE.load_refs
    zb = np.load(bcf); sb = [str(s) for s in zb["sys1"]]
    assert np.array_equal(zb["keys"], Kv)
    TB = zb["T1"][rows_v]
    mus = sorted({1.0} | {float(s.split("@")[1]) for s in sb if s.startswith("cams_qm@")})
    checks = {"truth codes of the pass = those of v3 (max abs, residents)":
              float(np.abs((Z["T"][..., 0, :8] + Z["T"][..., 0, 8:]) - N8).max())}
    sysf = B.N / "analyses" / "hidden" / f"data{CE.SFX}" / "hidden_systems_pop.npz"   # v3's source of perfect_40km (check only)
    if sysf.exists():
        zo = np.load(sysf); so = [str(s) for s in zo["sys1"]]; Ko = zo["keys"]
        ro = [int(np.flatnonzero((Ko[:, 0] == d) & (Ko[:, 1] == h))[0]) for d, h, _ in K]
        checks["perfect_40km from the truth codes = v3's perfect_40km table (max abs, residents)"] = float(
            np.abs(zo["T1"][ro][..., so.index("perfect_40km"), :] - PF).max())
        checks["foehn_mean of hidden_v3_pop = that of hidden_systems_pop (max abs)"] = float(
            np.abs(zo["T1"][ro][..., so.index("foehn_mean"), :] - FM).max())
        del zo
    else:
        checks["perfect_40km check"] = f"skipped: {sysf} absent"
    print("loaded", round(time.time() - t0, 1), "s;", checks, flush=True)
    REG = list(bc.REG); assert REG[0] == "World"
    W, R, days = {}, {}, {}
    for sname in SAMPLES:
        for h in LEADS:
            rows, cnt, n, how = CE.sample_rows(K, h, sname)
            days[f"{sname}/lead{h}"] = [int(x) for x in K[rows, 0]]
            for thr in THRS:
                j = CE.THR_ALL.index(thr)
                e = W.setdefault(sname, {}).setdefault(f"thr{thr:g}", {}).setdefault(f"lead{h}", {})
                e["n_start_days"], e["bootstrap"] = int(n), how
                iss = {"perfect_40km": PF[rows, 0, j], "foehn_mean": FM[rows, 0, j]}
                iss.update({r: RT[rows, 0, j, q] for q, r in enumerate(CE.REFS_ISSUED)})
                e["issued"] = {s: ci_all(Tab(a, cnt)) for s, a in iss.items()}
                HK = Z["HK"][rows, 0, j]
                fa = {"M8_liberal": CE.tables_from_k(HK, CE.pi_table(CE.NF, 8, "liberal")),
                      "M16_liberal_measured": CE.tables_from_k(HK, CE.pi_table(CE.NF, 16, "liberal")),
                      "M4_stored": Z["T"][rows, 0, j, 0]}
                e["foehn"], e["moved"] = {}, {}
                for M in MS:
                    tab = Tab(fa[M], cnt)
                    e["foehn"][M] = {k: float(v) for k, v in tab.pt.items()}
                    for q, r in enumerate(CE.REFS_MOVED):
                        with np.errstate(all="ignore"):
                            res = s16.persist_at(tab, RH[rows, j, q], cnt)
                            if res is None:
                                e["moved"].setdefault(M, {})[r] = "outside the factors 0.4-2"; continue
                            tb = tab.bs
                            own = {m: [float(res["pt"][m])] + [float(x) for x in np.nanpercentile(res["bs"][m], [2.5, 97.5])] for m in CE.MQ}
                            dif = {m: [float(tab.pt[m] - res["pt"][m])] + [float(x) for x in np.nanpercentile(tb[m] - res["bs"][m], [2.5, 97.5])]
                                   for m in ("reach_pct", "correct_pct", "hidden_warned_M")}
                        e["moved"].setdefault(M, {})[r] = dict(factor=float(res["mu"]), crossings=int(res["crossings"]),
                                                               resamples_without_a_crossing=int(res["resamples_without_a_crossing"]),
                                                               own=own, difference_as_in_the_record=dif)
                if h == 1:
                    e["sweep"] = {s: {f"{mu:g}": {m: float(v) for m, v in metrics(parts(TB[rows, 0, j, sb.index(s if mu == 1.0 else f"{s}@{mu:g}")].sum(0)), n).items()
                                                  if m in SWEEPM} for mu in mus} for s in SWEEP}
                if thr == 37.5:
                    R.setdefault(sname, {}).setdefault(f"lead{h}", {}).update(
                        {rg: {"cams": ci_all(Tab(RT[rows, k, j, CE.REFS_ISSUED.index("cams")], cnt))} for k, rg in enumerate(REG) if rg.startswith("G:")})
                print(f"{sname} lead {h} > {thr:g}: done ({round(time.time() - t0, 1)} s)", flush=True)
    out = clean(dict(
        what="complements of the class-E record for build_warn8.py (module docstring of complete_warn8.py)", design=B.LABEL,
        inputs=[finfo(p) for p in (hist, v3f, bcf) + ((sysf,) if sysf.exists() else ())],
        code=[finfo(p) for p in (Path(CE.__file__), Path(s16.__file__), Path(sys.modules["tables"].__file__), Path(__file__))],
        checks=checks, start_days=days, world=W, regions=R, sweep_factors=mus, seconds=round(time.time() - t0, 1)))
    if SMOKE:
        b = W["conf149"]["thr37.5"]["lead1"]
        print(json.dumps({s: {m: b["issued"][s][m] for m in ("warned_M", "reach_pct", "hidden_warned_pct")} for s in b["issued"]}, indent=0))
        print(json.dumps({M: {r: (v["factor"], v["own"]["reach_pct"], v["difference_as_in_the_record"]["reach_pct"]) for r, v in b["moved"][M].items()}
                          for M in b["moved"]}, indent=0))
        print("smoke: nothing written", round(time.time() - t0, 1), "s")
        return
    B.OUT.mkdir(exist_ok=True)
    B.safe_write(B.OUT / "complements.json", json.dumps(out, indent=1, allow_nan=False) + "\n")
    print("wrote", B.OUT / "complements.json", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
# end of complete_warn8.py
