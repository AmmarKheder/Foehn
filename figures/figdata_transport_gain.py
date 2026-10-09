#!/usr/bin/env python3
"""What the FLEXGREEN transport adds, by lead time (Fig. 4e): fair CRPS and RMSE against GHAP over each region's land pixels,
2022, of the base model with the transport inputs (A) and of the same network trained identically without them (C), on
their common forecasts (start day, lead); gain = 1 - CRPS_A / CRPS_C, with 95% intervals from a 7-day block bootstrap over
start days (10,000 resamples). "All regions": sums over the nine regions of eval_global_green.py.
    python figdata_transport_gain.py   -> data/transport_gain.npz"""
import glob
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = Path("/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km/runs")
EV = {"A": RUNS / "green_flow_shift25_global_L7_noant_whres_ghres" / "eval_final",
      "C": RUNS / "green_flow_shift25_global_L7_noant_whres_notransport" / "eval_final"}


def load(ev):
    R, cols, regions = [], None, None
    for p in sorted(glob.glob(str(ev / "part_*.npz"))):
        z = np.load(p); R.append(z["recs"]); cols = list(z["rec_cols"]); regions = [str(r) for r in z["regions"]]
    R = np.concatenate(R)
    out = {}
    for r in R:
        out[(int(r[0]), int(r[1]), int(r[2]))] = (r[cols.index("n")], r[cols.index("crps")], r[cols.index("sse")])
    return out, regions


def main():
    (a, regions), (c, _) = load(EV["A"]), load(EV["C"])
    keys = sorted(set(a) & set(c))
    days = sorted({k[0] for k in keys}); leads = range(1, 8)
    names = regions + ["All regions"]
    # per (region, lead, day): n, crps sum, sse sum for A and C
    T = np.zeros((2, len(names), 7, len(days), 3))
    for (d, h, g) in keys:
        for s, src in enumerate((a, c)):
            T[s, g, h - 1, days.index(d)] += src[(d, h, g)]
            T[s, -1, h - 1, days.index(d)] += src[(d, h, g)]
    rng = np.random.default_rng(0); B = 7; nd = len(days)
    idx = (rng.integers(0, nd - B + 1, size=(10000, -(-nd // B)))[:, :, None] + np.arange(B)).reshape(10000, -1)[:, :nd]
    gain = np.full((len(names), 7, 3), np.nan); rm = np.full((2, len(names), 7), np.nan); cr = np.full((2, len(names), 7), np.nan)
    for g in range(len(names)):
        for h in range(7):
            A_, C_ = T[0, g, h], T[1, g, h]
            if A_[:, 0].sum() == 0:
                continue
            pt = 1 - (A_[:, 1].sum() / A_[:, 0].sum()) / (C_[:, 1].sum() / C_[:, 0].sum())
            bs = 1 - (A_[idx, 1].sum(1) / A_[idx, 0].sum(1)) / (C_[idx, 1].sum(1) / C_[idx, 0].sum(1))
            gain[g, h] = [100 * pt, *np.percentile(100 * bs, [2.5, 97.5])]
            for s, X in enumerate((A_, C_)):
                rm[s, g, h] = np.sqrt(X[:, 2].sum() / X[:, 0].sum()); cr[s, g, h] = X[:, 1].sum() / X[:, 0].sum()
    with np.errstate(all="ignore"):                       # per start day (for the violins): 1 - CRPS_A / CRPS_C of that forecast
        gday = 100 * (1 - (T[0, :, :, :, 1] / T[0, :, :, :, 0]) / (T[1, :, :, :, 1] / T[1, :, :, :, 0]))   # (regions, leads, days)
    np.savez(HERE / "data" / "transport_gain.npz", gain=gain, rmse=rm, crps=cr, regions=np.array(names), days=np.array(days), gain_day=gday)
    for g, nm in enumerate(names):
        print(f"{nm:16s} CRPS gain (%) by lead: " + " ".join(f"{gain[g, h, 0]:+.1f}[{gain[g, h, 1]:+.1f},{gain[g, h, 2]:+.1f}]" for h in range(7)))


if __name__ == "__main__":
    main()
