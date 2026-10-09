#!/usr/bin/env python3
"""CRPS test at the official stations, one day to one week (Fig. 5c): per lead 1-7, pooled over identical station-days,
the fair CRPS of Foehn's four members (Ferro 2014) and the absolute error of the deterministic references (the CRPS of a
single forecast). Conventions exactly as compare_fair.py: equal issuance (CAMS run of 00 UTC day d+1 at lead day h-1;
Aurora initialised 00 UTC day d+1, daily mean 0.25/0.5/0.25 of its 12-h steps), the persistence of the station's own
measurement of day d, and the climatology of GHAP at the station pixel (2017-2020, same calendar day +-7). CAMS and
Aurora exist to lead 4; leads 5-7 pool Foehn, persistence and climatology on their own common station-days.
    python figdata_station_crps.py   -> data/station_crps_<label>.npz"""
import glob
from pathlib import Path

import numpy as np
import zarr

import final_model as FM

HERE = Path(__file__).resolve().parent
G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
AURD = HERE.parent / "benchmark" / "aurora"
SYS = ["Foehn (fair CRPS, 4 members)", "Foehn (ensemble mean, MAE)", "CAMS (run d+1, equal issuance)",
       "Aurora (00 UTC d+1, equal issuance)", "persistence (station, day d)", "climatology (GHAP 2017-2020)"]


def bil(m, lat0, dlat, lon0, dlon):
    fi = (lat0 - LAT) / dlat; fj = ((LON - lon0) % 360.0) / dlon
    i0 = np.clip(np.floor(fi).astype(int), 0, m.shape[0] - 2); j0 = np.floor(fj).astype(int) % m.shape[1]
    j1 = (j0 + 1) % m.shape[1]; wi = np.clip(fi - i0, 0, 1); wj = fj - np.floor(fj)
    return ((1 - wi) * ((1 - wj) * m[i0, j0] + wj * m[i0, j1]) + wi * ((1 - wj) * m[i0 + 1, j0] + wj * m[i0 + 1, j1]))


st = np.load(G / "stations_2022.npz")
LAT, LON = st["lat"], st["lon"]
official = st["source"] != 2
obs = st["daily"].astype(np.float32)
CLIM = np.load(G / "clim" / "st_clim_2022.npy").astype(np.float32)
cams = zarr.open("/scratch/project_462001140/ammar/eccv/data/world/cams_forecast_daily/2022.zarr", mode="r")

aur_e = {}
for p in glob.glob(str(AURD / "out_12h_all" / "part_*.npz")):
    z = np.load(p); la_, lo_ = z["lat"], z["lon"]
    for d, pr in zip(z["days"], z["pred"]):
        aur_e[int(d)] = np.stack([bil(pr[k].astype(np.float32), float(la_[0]), float(la_[0] - la_[1]),
                                      float(lo_[0]), float(lo_[1] - lo_[0])) for k in range(pr.shape[0])])
an0 = {}


def analysis00(d):
    if d not in an0:
        z = np.load(AURD / "raw" / f"{d:03d}.npz")
        an0[d] = bil(z["surf"][0, 5] * 1e9, float(z["lat"][0]), float(z["lat"][0] - z["lat"][1]),
                     float(z["lon"][0]), float(z["lon"][1] - z["lon"][0]))
    return an0[d]


def main():
    acc = np.zeros((7, len(SYS), 2))                                     # per lead and system: [sum of CRPS or |e|, n]
    cams_cache = {}
    nfc = 0
    for p in sorted(glob.glob(str(FM.EVAL_DIR / "part_*.npz"))):
        z = np.load(p)
        for n, (d, h, t) in enumerate(z["keys"]):
            d, h, t = int(d), int(h), int(t)
            if t >= 365 or d + 1 > 364:
                continue
            M = z["st_ens"][n].astype(np.float32)                        # (8110, 4)
            y = obs[:, t]
            vals = {"pers": obs[:, d], "clim": CLIM[t]}
            if h <= 4:
                if (d + 1) not in aur_e:
                    continue
                key = (d + 1, h - 1)
                if key not in cams_cache:
                    if len(cams_cache) > 8:
                        cams_cache.pop(next(iter(cams_cache)))
                    cams_cache[key] = bil(np.asarray(cams[d + 1, 0, h - 1], np.float32), 90.0, 0.4, 0.0, 0.4)
                vals["cams"] = cams_cache[key]
                A = aur_e[d + 1]; k0 = 2 * (h - 1)
                first = analysis00(d + 1) if h == 1 else A[k0 - 1]
                vals["aur"] = 0.25 * first + 0.5 * A[k0] + 0.25 * A[k0 + 1]
            ok = official & np.isfinite(y) & np.all(np.isfinite(M), 1) & np.isfinite(vals["pers"]) & np.isfinite(vals["clim"])
            if h <= 4:
                ok &= np.isfinite(vals["cams"]) & np.isfinite(vals["aur"])
            if ok.sum() == 0:
                continue
            Mo, yo = M[ok], y[ok]
            pair = sum(np.abs(Mo[:, i] - Mo[:, j]) for i in range(4) for j in range(i + 1, 4)) / (4 * 3)
            crps = np.abs(Mo - yo[:, None]).mean(1) - pair
            k = h - 1
            acc[k, 0] += [crps.sum(), ok.sum()]
            acc[k, 1] += [np.abs(Mo.mean(1) - yo).sum(), ok.sum()]
            if h <= 4:
                acc[k, 2] += [np.abs(vals["cams"][ok] - yo).sum(), ok.sum()]
                acc[k, 3] += [np.abs(vals["aur"][ok] - yo).sum(), ok.sum()]
            acc[k, 4] += [np.abs(vals["pers"][ok] - yo).sum(), ok.sum()]
            acc[k, 5] += [np.abs(vals["clim"][ok] - yo).sum(), ok.sum()]
            nfc += 1
    with np.errstate(all="ignore"):
        mean = acc[:, :, 0] / acc[:, :, 1]
    np.savez(HERE / "data" / f"station_crps_{FM.LABEL}.npz", mean=mean, n=acc[:, :, 1], systems=np.array(SYS), leads=np.arange(1, 8))
    print(f"{nfc} forecasts pooled; station-days per lead:", acc[:, 0, 1].astype(int))
    for k in range(7):
        print(f"J+{k + 1}: " + " | ".join(f"{s.split(' (')[0]} {mean[k, j]:.2f}" for j, s in enumerate(SYS) if np.isfinite(mean[k, j])))


if __name__ == "__main__":
    main()
