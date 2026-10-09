#!/usr/bin/env python3
"""Station comparison on identical station-days (2022): our model (ensemble mean at the station's 1 km pixel) vs the
CAMS global forecast (00 UTC run of day d, 0.4 deg, bilinear), Aurora (0.4 deg air-pollution model initialised from the
CAMS analysis of day d, bilinear; benchmark/aurora/out_global), persistence of the station (its value on day d) and GHAP
of the target day (a product, not a forecast: the ceiling of a model trained on GHAP). Leads 1-4 (CAMS / Aurora horizon).
RMSE, bias, correlation, CSI of daily exceedance of 35 ug/m3, by network.
    python compare_baselines.py runs/<run>/eval_global [--no-aurora]"""
import glob
import sys
from pathlib import Path

import numpy as np
import zarr

G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
AUR = Path("/scratch/project_462001140/ammar/eccv/cran-pm/experiments/benchmark/aurora/out_global")
NETS = {0: "EEA", 1: "CNEMC", 3: "EPA", 4: "AirKorea", 5: "Japan", 6: "India", 2: "OpenAQ", None: "All"}
run = Path(sys.argv[1]); use_aur = "--no-aurora" not in sys.argv
P = [np.load(p) for p in sorted(glob.glob(str(run / "part_*.npz")))]
keys = np.concatenate([p["keys"] for p in P]); ens = np.concatenate([p["st_ens"] for p in P]).astype(np.float32)
gh = np.concatenate([p["st_ghap"] for p in P]).astype(np.float32)
NS = ens.shape[1]
for f in ("stations_2022.npz", "stations_2022_v2.npz"):
    st = np.load(G / f)
    if len(st["lat"]) == NS:
        break
assert len(st["lat"]) == NS, "station file of the evaluation not found"
lat, lon, src, obs = st["lat"], st["lon"], st["source"], st["daily"]


def bil(field, la, lo, lat0, dlat, lon0, dlon, nlat, nlon):
    fi = (lat0 - la) / dlat; fj = ((lo - lon0) % 360.0) / dlon
    i0 = np.clip(np.floor(fi).astype(int), 0, nlat - 2); j0 = np.floor(fj).astype(int) % nlon; j1 = (j0 + 1) % nlon
    wi, wj = np.clip(fi - i0, 0, 1), fj - np.floor(fj)
    return (1 - wi) * ((1 - wj) * field[i0, j0] + wj * field[i0, j1]) + wi * ((1 - wj) * field[i0 + 1, j0] + wj * field[i0 + 1, j1])


cams = zarr.open("/scratch/project_462001140/ammar/eccv/data/world/cams_forecast_daily/2022.zarr", mode="r")
aur = {}
if use_aur:
    for p in glob.glob(str(AUR / "part_*.npz")):
        z = np.load(p)
        alat, alon = z["lat"], z["lon"]
        for d, pr in zip(z["days"], z["pred"]):
            aur[int(d)] = pr
    print(f"Aurora: {len(aur)} start days (lat {alat[0]:.1f}..{alat[-1]:.1f}, lon {alon[0]:.1f}..{alon[-1]:.1f})")
rows = {}
for n, (d, h, t) in enumerate(keys):
    if h > 4 or (use_aur and int(d) not in aur):
        continue
    o = obs[:, t]
    f = ens[n].mean(-1)
    c = bil(np.asarray(cams[int(d), 0, int(h)], np.float32), lat, lon, 90.0, 0.4, 0.0, 0.4, 451, 900)
    a = bil(aur[int(d)][h - 1].astype(np.float32), lat, lon, float(alat[0]), float(alat[0] - alat[1]), float(alon[0]),
            float(alon[1] - alon[0]), len(alat), len(alon)) if use_aur else np.full_like(f, np.nan)
    p = obs[:, d]
    rows.setdefault(int(h), []).append(np.stack([o, f, c, a, p, gh[n]]))
labels = ["ours", "CAMS", "Aurora", "persist.", "GHAP (same day)"]
for h in sorted(rows):
    X = np.concatenate(rows[h], 1)                                         # (6, stations * forecasts)
    S = np.tile(src, len(rows[h]))
    print(f"\n=== lead {h} day(s): {len(rows[h])} forecasts | RMSE / bias / corr / CSI35 on identical station-days ===")
    print(f"{'network':9s} {'n':>8s} " + " ".join(f"{l:>26s}" for l in labels))
    for s, nm in NETS.items():
        sel = np.ones(X.shape[1], bool) if s is None else S == s
        m = sel & np.all(np.isfinite(X[:5 if use_aur else 3]), 0) & np.isfinite(X[4]) & np.isfinite(X[5])
        if m.sum() < 100:
            continue
        o = X[0, m]; cells = []
        for k in range(1, 6):
            x = X[k, m]
            if not np.isfinite(x).all():
                cells.append(f"{'-':>26s}"); continue
            e, g = o > 35, x > 35
            csi = (e & g).sum() / max((e | g).sum(), 1)
            cells.append(f"{np.sqrt(np.mean((x - o) ** 2)):6.2f} {np.mean(x - o):+6.2f} {np.corrcoef(x, o)[0, 1]:5.2f} {csi:5.2f}  ")
        print(f"{nm:9s} {int(m.sum()):8d} " + " ".join(cells))
