#!/usr/bin/env python3
"""Model selection on 2021 against GHAP only (stations never enter): fair CRPS and RMSE of the ensemble mean over the 1 km
land pixels of the evaluation regions, by lead, for several evaluations on identical (start day, lead) forecasts, and the
change relative to the first (reference). The selection criterion is the CRPS summed over leads 1-7, all regions.
    python select_2021.py A=<run>:eval2021_sel e02=<run>:eval2021_sel_e02 ...   (first = reference)"""
import glob
import sys

import numpy as np

res = []
for arg in sys.argv[1:]:
    lab, spec = arg.split("=")
    run, ev = spec.split(":")
    P = [np.load(p) for p in sorted(glob.glob(f"runs/{run}/{ev}/part_*.npz"))]
    if not P:
        print(f"{lab}: no parts in runs/{run}/{ev}"); continue
    res.append((lab, np.concatenate([p["recs"] for p in P]), list(P[0]["rec_cols"]), [str(r) for r in P[0]["regions"]]))
common = set.intersection(*[{(int(a), int(b)) for a, b in r[1][:, :2]} for r in res])
print(f"common forecasts (start day, lead): {len(common)}; models: {[r[0] for r in res]}")
total = {}
for scope in ["all regions", "East Asia", "South Asia", "Europe", "North America", "Africa"]:
    print(f"\n{scope}: CRPS / RMSE (ug m-3) against GHAP 2021")
    print("lead " + "".join(f"{lab:>16s}" for lab, *_ in res))
    for h in sorted({int(l) for l in res[0][1][:, 1]}):
        row = []
        for lab, rec, cols, regs in res:
            c = {k: i for i, k in enumerate(cols)}
            m = (rec[:, 1] == h) & np.array([(int(a), int(b)) in common for a, b in rec[:, :2]])
            if scope != "all regions":
                if scope not in regs:
                    row.append((np.nan, np.nan)); continue
                m &= rec[:, 2] == regs.index(scope)
            n = rec[m, c["n"]].sum()
            row.append((rec[m, c["crps"]].sum() / n, np.sqrt(rec[m, c["sse"]].sum() / n)))
            if scope == "all regions":
                total[lab] = total.get(lab, 0.0) + rec[m, c["crps"]].sum() / n
        ref = row[0]
        print(f"J+{h:<3d}" + "".join(f"{cr:7.3f} /{rm:6.2f}" for cr, rm in row)
              + "   | CRPS vs " + res[0][0] + ": " + " ".join(f"{100 * (cr / ref[0] - 1):+.1f}%" for cr, _ in row))
best = min(total, key=total.get)
print("\nCRPS summed over leads (all regions): " + ", ".join(f"{k} {v:.3f}" for k, v in total.items()) + f"  -> best: {best}")
