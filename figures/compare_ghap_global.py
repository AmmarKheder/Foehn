#!/usr/bin/env python3
"""Global scores against GHAP (2022) at 0.25 deg, every system on the same land cells and days: Foehn runs (ensemble
mean, 25 x 25-pixel block means of the 1 km forecast, map25 of the evaluation), CAMS (00 UTC run of day d, 0.4 deg, leads
1-4) and Aurora (0.4 deg, leads 1-4) interpolated bilinearly to the block centres, persistence (GHAP block means of day
d) and climatology (GHAP 2017-2020 within +-7 days of the calendar day, pm.npy moved to the block centres). Truth: the
GHAP block means of the target day (map25_ghap). Leads 1-4: every system; 5-7: without CAMS and Aurora.
    python compare_ghap_global.py <label>=<run>[:<eval dir>] ...   -> data/compare_ghap_global_<first label>.npz"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import zarr

HERE = Path(__file__).resolve().parent
RUNS = Path("/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km/runs")
G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
AUR = HERE.parent / "benchmark" / "aurora" / "out_global"
I2022 = 1826
NR, NC = 576, 1440
LATC = 84.0 - 0.25 * np.arange(NR) - 0.125                      # block centres of map25
LONC = -180.0 + 0.25 * np.arange(NC) + 0.125
LA2, LO2 = np.meshgrid(LATC, LONC, indexing="ij")
REG = {"Global (84N-60S)": None, "East Asia": (54, 18, 97, 146), "South Asia": (37, 5, 60, 97), "Europe": (72, 35, -25, 45),
       "North America": (72, 15, -170, -50), "Africa": (37, -35, -20, 52)}
RMASK = {k: (np.ones((NR, NC), bool) if b is None else (LA2 <= b[0]) & (LA2 > b[1]) & (LO2 >= b[2]) & (LO2 < b[3])) for k, b in REG.items()}

runs = []
for arg in sys.argv[1:]:
    label, spec = arg.split("=")
    run, ev = (spec.split(":") + ["eval_global"])[:2]
    idx = {}
    for p in sorted(glob.glob(str(RUNS / run / ev / "part_*.npz"))):
        for n, (d, h, t) in enumerate(np.load(p)["keys"]):
            idx[(int(d), int(h))] = (p, n)
    runs.append((label, idx))
    print(label, run, ev, len(idx), "forecasts", flush=True)
common = sorted(set.intersection(*[set(r[1]) for r in runs]))
print("common forecasts:", len(common), flush=True)

cache = {}


def arrays(path):
    if path not in cache:
        if len(cache) > 12:
            cache.pop(next(iter(cache)))
        z = np.load(path)
        cache[path] = (z["map25"], z["map25_ghap"])
    return cache[path]


def npy_read(path, i):
    with open(path, "rb") as f:
        np.lib.format.read_magic(f); sh, _, dt = np.lib.format.read_array_header_1_0(f); hdr = f.tell()
    n = int(np.prod(sh[1:])); fd = os.open(path, os.O_RDONLY)
    b = os.pread(fd, n * dt.itemsize, hdr + i * n * dt.itemsize); os.close(fd)
    return np.frombuffer(b, dt).reshape(sh[1:])


def to_centres(a):                                                # pm.npy points (84 - 0.25 k) -> block centres
    b = np.concatenate([a, a[-1:]], 0); b = 0.5 * (b[:-1] + b[1:])
    c = np.concatenate([b, b[:, :1]], 1); return 0.5 * (c[:, :-1] + c[:, 1:])


def bil(field, lat0, dlat, lon0, dlon):
    nlat, nlon = field.shape
    fi = (lat0 - LA2) / dlat; fj = ((LO2 - lon0) % 360.0) / dlon
    i0 = np.clip(np.floor(fi).astype(int), 0, nlat - 2); j0 = np.floor(fj).astype(int) % nlon; j1 = (j0 + 1) % nlon
    wi, wj = np.clip(fi - i0, 0, 1), fj - np.floor(fj)
    return ((1 - wi) * ((1 - wj) * field[i0, j0] + wj * field[i0, j1]) + wi * ((1 - wj) * field[i0 + 1, j0] + wj * field[i0 + 1, j1])).astype(np.float32)


# climatology of every target day (GHAP 2017-2020, +-7 days), on the block centres
meta = np.load(G / "meta.npz"); hdoy = meta["doy"][:1461]
need = sorted({d + h for d, h in common})
HIST = np.stack([npy_read(G / "pm.npy", i).astype(np.float16) for i in range(1461)])     # 2017-2020, read once
clim = {}
for t in need:
    dd = np.abs(hdoy - t); dd = np.minimum(dd, 365 - dd)
    x = HIST[np.where(dd <= 7)[0]].astype(np.float32); x[~(x > 0)] = np.nan
    with np.errstate(all="ignore"):
        clim[t] = to_centres(np.nanmean(x, 0)).astype(np.float32)
del HIST
print("climatology ready", flush=True)
cams = zarr.open("/scratch/project_462001140/ammar/eccv/data/world/cams_forecast_daily/2022.zarr", mode="r")
aur = {}
for p in glob.glob(str(AUR / "part_*.npz")):
    z = np.load(p); alat, alon = z["lat"], z["lon"]
    for d, pr in zip(z["days"], z["pred"]):
        aur[int(d)] = pr
pers = {}                                                         # GHAP block means of every day from the first run
for (d, h), (p, n) in runs[0][1].items():
    pers.setdefault(d + h, (p, n))

SYS = [r[0] for r in runs] + ["CAMS", "Aurora", "persistence", "climatology"]
S = np.zeros((len(REG), 7, len(SYS), 12))          # n, se, se2, sae, sx, sy, sx2, sy2, sxy, hits35, union35, hits75 ... (+union75 below)
U75 = np.zeros((len(REG), 7, len(SYS)))
for c_, (d, h) in enumerate(common):
    t = d + h
    if h <= 4 and d not in aur:
        continue
    X = []
    for label, idx in runs:
        p, n = idx[(d, h)]; m25, g25 = arrays(p); X.append(m25[n].astype(np.float32))
    p, n = runs[0][1][(d, h)]; y = arrays(p)[1][n].astype(np.float32)
    nr = len(runs)
    X += [bil(np.asarray(cams[d, 0, h], np.float32), 90.0, 0.4, 0.0, 0.4) if h <= 4 else np.full_like(y, np.nan),
          bil(aur[d][h - 1].astype(np.float32), float(alat[0]), float(alat[0] - alat[1]), float(alon[0]), float(alon[1] - alon[0]))
          if h <= 4 else np.full_like(y, np.nan)]
    if d in pers:
        pp, pn = pers[d]; X.append(arrays(pp)[1][pn].astype(np.float32))
    else:
        X.append(np.full_like(y, np.nan))
    X.append(clim[t])
    use = list(range(nr)) + ([nr, nr + 1] if h <= 4 else []) + [nr + 2, nr + 3]
    ok = np.isfinite(y) & (y > 0) & np.all([np.isfinite(X[k]) for k in use], 0)
    for g, (rn, rm) in enumerate(RMASK.items()):
        m = ok & rm
        if not m.any():
            continue
        yy = y[m]
        for k in use:
            x = X[k][m]; e = x - yy
            S[g, h - 1, k] += [m.sum(), e.sum(), (e * e).sum(), np.abs(e).sum(), x.sum(), yy.sum(), (x * x).sum(), (yy * yy).sum(),
                               (x * yy).sum(), ((x > 35) & (yy > 35)).sum(), ((x > 35) | (yy > 35)).sum(), ((x > 75) & (yy > 75)).sum()]
            U75[g, h - 1, k] += ((x > 75) | (yy > 75)).sum()
    if c_ % 200 == 0:
        print(c_, "/", len(common), flush=True)


def sc(v, u75):
    n = v[0]
    if n == 0:
        return None
    mx, my = v[4] / n, v[5] / n
    r = (v[8] / n - mx * my) / np.sqrt(max(v[6] / n - mx * mx, 1e-9) * max(v[7] / n - my * my, 1e-9))
    return dict(rmse=np.sqrt(v[2] / n), mae=v[3] / n, bias=v[1] / n, r=r, csi35=v[9] / max(v[10], 1), csi75=v[11] / max(u75, 1))


for g, rn in enumerate(REG):
    print(f"\n=== {rn}: against GHAP 2022 at 0.25 deg (RMSE / MAE / bias / r / CSI35 / CSI75)")
    for h in range(1, 8):
        print(f"J+{h} (cells x days {int(S[g, h - 1, 0, 0])})")
        for k, nm in enumerate(SYS):
            s = sc(S[g, h - 1, k], U75[g, h - 1, k])
            if s:
                print(f"   {nm:14s} {s['rmse']:6.2f} {s['mae']:6.2f} {s['bias']:+6.2f} {s['r']:5.2f} {s['csi35']:5.2f} {s['csi75']:5.2f}")
np.savez(HERE / "data" / f"compare_ghap_global_{runs[0][0]}.npz", S=S, U75=U75, systems=np.array(SYS), regions=np.array(list(REG)))
print("saved")
