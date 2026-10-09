#!/usr/bin/env python3
"""Data of the variance panels: (1) radially averaged power spectra of daily PM2.5 at 1 km over all 512 x 512 km land
tiles with complete GHAP coverage, for GHAP (target day), one ensemble member, the ensemble mean and CAMS (0.4 deg,
bilinear to 1 km), averaged over the start days whose 1 km maps were saved (lead 1); (2) quantiles of daily PM2.5 over
all station-days (identical station-days) for the observations, the ensemble mean, one member, CAMS and Aurora.
    source world/env.sh; python figdata_stats.py <run>   -> data/stats_<run>.npz"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import zarr

RUN = sys.argv[1]
E = Path("/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km/runs") / RUN / "eval_global"
G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
AUR = Path("/scratch/project_462001140/ammar/eccv/cran-pm/experiments/benchmark/aurora/out_global")
cams = zarr.open("/scratch/project_462001140/ammar/eccv/data/world/cams_forecast_daily/2022.zarr", mode="r")


def tiles(name, k):
    with open(G / name, "rb") as f:
        np.lib.format.read_magic(f); np.lib.format.read_array_header_1_0(f); hdr = f.tell()
    n = 29 * 71 * 512 * 512; fd = os.open(G / name, os.O_RDONLY)
    a = np.frombuffer(os.pread(fd, n * 2, hdr + k * n * 2), np.float16).reshape(29, 71, 512, 512); os.close(fd)
    return a.transpose(0, 2, 1, 3).reshape(29 * 512, 71 * 512)[:14400, :36000]


T = 512
k = np.fft.fftfreq(T); kr = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
bins = np.unique(np.round(np.logspace(np.log10(1.0 / T), np.log10(0.5), 40), 6))
idx = np.digitize(kr, bins)
win = np.outer(np.hanning(T), np.hanning(T))
def spectrum(x):
    x = x - x.mean()
    p = np.abs(np.fft.fft2(x * win)) ** 2
    return np.array([p[idx == b].mean() if (idx == b).any() else np.nan for b in range(1, len(bins))])
land = tiles("static1km_tiles.npy", 1) > 0
days = sorted({int(f.split("_d")[1].split("_")[0]) for f in glob.glob(str(E / "map1km_d*_h1.npy"))})
SP = {k_: [] for k_ in ("ghap", "member", "mean", "cams")}
for d in days:
    g = tiles("ghap1km_tiles.npy", 1826 + d + 1).astype(np.float32)
    mn = np.load(E / f"map1km_d{d}_h1.npy", mmap_mode="r"); m0 = np.load(E / f"map1km_d{d}_h1_member0.npy", mmap_mode="r")
    c = np.asarray(cams[d, 0, 1], np.float32)
    ntile = 0
    for r in range(0, 14400 - T, T):
        for cc in range(0, 36000 - T, T):
            gt = g[r:r + T, cc:cc + T]
            if (gt > 0).mean() < 0.995 or land[r:r + T, cc:cc + T].mean() < 0.995:
                continue
            fill = lambda a: np.where(np.isfinite(a) & (gt > 0), a, np.nanmean(np.where(gt > 0, a, np.nan)))
            la = 84 - 0.01 * (r + np.arange(T) + 0.5); lo = -180 + 0.01 * (cc + np.arange(T) + 0.5)
            fi = (90 - la) / 0.4; fj = (lo % 360) / 0.4
            i0 = np.clip(np.floor(fi).astype(int), 0, 449); j0 = np.floor(fj).astype(int) % 900; j1 = (j0 + 1) % 900
            wi, wj = (fi - i0)[:, None], (fj - np.floor(fj))[None]
            ct = (1 - wi) * ((1 - wj) * c[i0][:, j0] + wj * c[i0][:, j1]) + wi * ((1 - wj) * c[i0 + 1][:, j0] + wj * c[i0 + 1][:, j1])
            SP["ghap"].append(spectrum(fill(gt))); SP["member"].append(spectrum(fill(np.asarray(m0[r:r + T, cc:cc + T], np.float32))))
            SP["mean"].append(spectrum(fill(np.asarray(mn[r:r + T, cc:cc + T], np.float32)))); SP["cams"].append(spectrum(ct.astype(np.float32)))
            ntile += 1
    print(f"start day {d}: {ntile} complete land tiles", flush=True)
wl = 1.0 / np.sqrt(bins[:-1] * bins[1:])                                 # wavelength (km) at bin centres
# quantiles at the stations (lead 1 and 3, identical station-days)
P = [np.load(p) for p in sorted(glob.glob(str(E / "part_*.npz")))]
keys = np.concatenate([p["keys"] for p in P]); ens = np.concatenate([p["st_ens"] for p in P])
st = np.load(G / "stations_2022.npz"); lat, lon, obs = st["lat"], st["lon"], st["daily"]
aur = {}
for p in glob.glob(str(AUR / "part_*.npz")):
    z = np.load(p); alat, alon = z["lat"], z["lon"]
    for d, pr in zip(z["days"], z["pred"]):
        aur[int(d)] = pr
def bil(field, lat0, dlat, lon0, dlon):
    nlat, nlon = field.shape; fi = (lat0 - lat) / dlat; fj = ((lon - lon0) % 360.0) / dlon
    i0 = np.clip(np.floor(fi).astype(int), 0, nlat - 2); j0 = np.floor(fj).astype(int) % nlon; j1 = (j0 + 1) % nlon
    wi, wj = np.clip(fi - i0, 0, 1), fj - np.floor(fj)
    return (1 - wi) * ((1 - wj) * field[i0, j0] + wj * field[i0, j1]) + wi * ((1 - wj) * field[i0 + 1, j0] + wj * field[i0 + 1, j1])
Q = np.array([1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 99.5, 99.8, 99.9])
QS = {}
for h in (1, 3):
    vals = {k_: [] for k_ in ("obs", "mean", "member", "cams", "aurora")}
    for n in np.where(keys[:, 1] == h)[0]:
        d, t = int(keys[n, 0]), int(keys[n, 2])
        if d not in aur:
            continue
        X = [obs[:, t], ens[n].astype(np.float32).mean(-1), ens[n, :, 0].astype(np.float32),
             bil(np.asarray(cams[d, 0, h], np.float32), 90.0, 0.4, 0.0, 0.4),
             bil(aur[d][h - 1].astype(np.float32), float(alat[0]), float(alat[0] - alat[1]), float(alon[0]), float(alon[1] - alon[0]))]
        m = np.all([np.isfinite(x) for x in X], 0)
        for k_, x in zip(vals, X):
            vals[k_].append(x[m])
    for k_ in vals:
        v = np.concatenate(vals[k_]); QS[f"{k_}_{h}"] = np.percentile(v, Q); QS[f"var_{k_}_{h}"] = np.var(v)
    print(f"lead {h}: {len(v):,} station-days; 99.9th percentiles:", {k_: round(float(QS[f'{k_}_{h}'][-1]), 1) for k_ in vals}, flush=True)
out = Path(__file__).resolve().parent / "data" / f"stats_{RUN}.npz"
np.savez(out, wl=wl, Q=Q, **{f"sp_{k_}": np.nanmean(np.array(v), 0) for k_, v in SP.items()}, ntiles=len(SP["ghap"]), **QS)
print("saved", out)
