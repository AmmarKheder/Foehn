#!/usr/bin/env python3
"""Reliability of the exceedance probabilities at the stations: for a daily threshold T, the forecast probability is
the fraction of ensemble members above T (at the station's 1 km pixel); the observed frequency is the fraction of those
cases where the station's daily mean exceeded T. Perfect reliability: observed frequency = forecast probability.
Also the Brier score, its skill against the climatological base rate, and the spread / error ratio.
    python reliability_stations.py runs/<run>/eval_global [--thresholds 15 25 37.5 50 75] [--leads 1 3 7]"""
import argparse
import glob
from pathlib import Path

import numpy as np

G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
ap = argparse.ArgumentParser()
ap.add_argument("dir"); ap.add_argument("--thresholds", type=float, nargs="*", default=[15, 25, 37.5, 50, 75])
ap.add_argument("--leads", type=int, nargs="*", default=[1, 3, 7])
a = ap.parse_args()
P = [np.load(p) for p in sorted(glob.glob(str(Path(a.dir) / "part_*.npz")))]
keys = np.concatenate([p["keys"] for p in P]); ens = np.concatenate([p["st_ens"] for p in P]).astype(np.float32)
for f in ("stations_2022.npz", "stations_2022_v2.npz"):
    st = np.load(G / f)
    if len(st["lat"]) == ens.shape[1]:
        break
obs = st["daily"]; M = ens.shape[2]
for h in a.leads:
    k = np.where(keys[:, 1] == h)[0]
    E = ens[k]                                                    # (forecasts, stations, members)
    o = obs[:, keys[k, 2]].T                                      # (forecasts, stations)
    ok = np.isfinite(o) & np.all(np.isfinite(E), -1)
    E, o = E[ok], o[ok]
    mean = E.mean(-1)
    spread = np.sqrt(np.mean(E.var(-1, ddof=1)) * (M + 1) / M)
    rmse = np.sqrt(np.mean((mean - o) ** 2))
    print(f"\n=== échéance {h} j: {len(o):,} station-jours | dispersion corrigée / erreur = {spread / rmse:.2f} (1 = bien calibré)")
    for T in a.thresholds:
        p = (E > T).mean(-1); y = (o > T).astype(np.float32)
        base = y.mean()
        bs = np.mean((p - y) ** 2); bss = 1 - bs / (base * (1 - base))
        cells = []
        for q in np.unique(p):
            m = p == q
            cells.append(f"{q:.2f}→{y[m].mean():.2f} (n={int(m.sum()):,})")
        print(f"  seuil {T:5.1f} µg/m³ | fréquence de base {base:.3f} | Brier {bs:.4f} | skill vs climatologie {bss:+.2f} | "
              f"probabilité annoncée → fréquence observée: " + "  ".join(cells))
