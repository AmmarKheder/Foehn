#!/usr/bin/env python3
"""The systems recomputed by figdata_hidden_bc.py (foehn_p50, foehn_mean, foehn_40km, cams) must equal those of
../data/hidden_systems_pop.npz (job 23864988) on every forecast, region, threshold and code: the proof that the new systems
are counted on the same pixels, days and valid residents.   python check_identity.py data/hidden_bc_pop.npz"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
a = np.load(HERE / sys.argv[1]); b = np.load(HERE.parent / "data" / "hidden_systems_pop.npz")
K = b["keys"]; sa, sb = list(a["sys1"]), list(b["sys1"])
assert list(a["regions"]) == list(b["regions"]) and np.array_equal(a["thr"], b["thr"]) and float(a["margin"]) == float(b["margin"])
rows = [int(np.flatnonzero((K[:, 0] == d) & (K[:, 1] == h))[0]) for d, h, t in a["keys"]]
assert np.array_equal(a["keys"], K[rows])
for s in ("foehn_p50", "foehn_mean", "foehn_40km", "cams"):
    x, y = a["T1"][:, :, :, sa.index(s)], b["T1"][rows][:, :, :, sb.index(s)]
    print(f"CHECK {s:11s}: max abs deviation {float(np.abs(x - y).max()):.3g} residents on {len(rows)} forecasts; identical: {bool(np.array_equal(x, y))}")
d = a["diag"]
print("valid residents a forecast (M): min", d[:, 0].min() / 1e6, "max", d[:, 0].max() / 1e6, "| identical to the original:",
      bool(np.array_equal(d[:, 0], b["diag"][rows][:, 0])))
print("valid residents without a cell map (M), max over forecasts:", d[:, 1].max() / 1e6, "| outside the mask (no pattern):", d[:, 2].max() / 1e6)
print("valid residents whose cell value was extrapolated below / above the fitted range (M), mean:", d[:, 3].mean() / 1e6, d[:, 4].mean() / 1e6)
print("max deviation of the vectorised quantile map from fit_bcsd_ghap.qmap:", d[:, 5].max())
print("coverage:", "; ".join(str(v) for v in a["coverage"])); print("fit:", "; ".join(str(v) for v in a["fit"]))
for j, s in enumerate(a["diag2_systems"]):
    q = a["diag2"][:, j].sum(0)
    print(f"  {str(s):10s} residents with a value {q[0] / len(d) / 1e6:8.1f} M | population-weighted mean {q[1] / q[0]:6.2f} against {q[2] / q[0]:6.2f} "
          f"(bias {(q[1] - q[2]) / q[0]:+6.2f}) | RMSE {np.sqrt(q[3] / q[0]):6.2f} | MAE {q[4] / q[0]:6.2f} ug m-3")
