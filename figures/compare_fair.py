#!/usr/bin/env python3
"""Fairness checks of the station comparison (official networks, 2022, stations only evaluate), on identical station-days:
  * equal issuance: Foehn's forecast for day d+h uses the observations of day d, so it can be issued at the end of day d
    (00 UTC of day d+1). The baselines issued at that same time: the CAMS run of 00 UTC d+1 (its lead h-1 day; lead 0 =
    the day of the run) and Aurora initialised at 00 UTC d+1 from the CAMS analyses of 12 UTC d and 00 UTC d+1 (12-hour
    model, ../benchmark/aurora/out_12h_all: daily mean 0.25 x 00 UTC + 0.5 x 12 UTC + 0.25 x 00 UTC of the next day, the
    first 00 UTC of lead 1 being the analysis). Also the comparison as in compare_baselines.py (CAMS run of 00 UTC d,
    Aurora from the analyses of day d), for reference.
  * equal resolution: Foehn's ensemble mean aggregated to 0.25 deg (the evaluation's block means) and to 0.5 deg (2 x 2),
    read at the stations bilinearly as the baselines are, instead of its 1 km pixel.
RMSE, bias and correlation of the ensemble mean / deterministic forecast; leads 1-4; 7-day block bootstrap of RMSE ratios.
Two persistences: the station's own measurement of day d (information Foehn never sees) and GHAP of day d at the station
pixel (the information Foehn has; from the GHAP values stored with the first run's forecasts, target day d).
    python compare_fair.py <label>=<run>[:<eval dir>] ...   -> data/compare_fair_<first label>.npz"""
import glob
import sys
from pathlib import Path

import numpy as np
import zarr

HERE = Path(__file__).resolve().parent
RUNS = Path("/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km/runs")
G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
AURD = HERE.parent / "benchmark" / "aurora"
NETS = {0: "Europe (EEA)", 1: "China (CNEMC)", 3: "USA (EPA)", 4: "Korea (AirKorea)", 5: "Japan (NIES)", 6: "India (CPCB)"}

st = np.load(G / "stations_2022.npz")
lat, lon, src, obs = st["lat"], st["lon"], st["source"], st["daily"].astype(np.float32)
NS = len(lat)
official = np.isin(src, list(NETS))
key = lambda la, lo: np.round(la, 4) * 1e6 + np.round(lo, 4)
ref = {v: i for i, v in enumerate(key(lat, lon))}
sets = {}
for f in sorted(G.glob("stations_2022*.npz")):
    z = np.load(f); sets[len(z["lat"])] = np.array([ref.get(v, -1) for v in key(z["lat"], z["lon"])])


def bil(field, lat0, dlat, lon0, dlon):
    nlat, nlon = field.shape
    fi = (lat0 - lat) / dlat; fj = ((lon - lon0) % 360.0) / dlon
    i0 = np.clip(np.floor(fi).astype(int), 0, nlat - 2); j0 = np.floor(fj).astype(int) % nlon; j1 = (j0 + 1) % nlon
    wi, wj = np.clip(fi - i0, 0, 1), fj - np.floor(fj)
    return ((1 - wi) * ((1 - wj) * field[i0, j0] + wj * field[i0, j1]) + wi * ((1 - wj) * field[i0 + 1, j0] + wj * field[i0 + 1, j1])).astype(np.float32)


CLIM = np.load(G / "clim" / "st_clim_2022.npy").astype(np.float32)   # (365, stations): GHAP 2017-2020 at the station pixel
runs = []
GH = {}                                                                  # GHAP at the station pixels, per day (first run)
for arg in sys.argv[1:]:
    label, spec = arg.split("=")
    run, ev = (spec.split(":") + ["eval_global"])[:2]
    parts = sorted(glob.glob(str(RUNS / run / ev / "part_*.npz")))
    index, ens = {}, {}
    for p in parts:
        z = np.load(p)
        e = z["st_ens"]; idx = sets[e.shape[1]]
        for n, (d, h, t) in enumerate(z["keys"]):
            index[(int(d), int(h))] = (p, n)
            full = np.full(NS, np.nan, np.float32); full[idx[idx >= 0]] = e[n, idx >= 0].astype(np.float32).mean(-1)
            ens[(int(d), int(h))] = full
            if not runs and int(t) not in GH:
                g_ = np.full(NS, np.nan, np.float32); g_[idx[idx >= 0]] = z["st_ghap"][n, idx >= 0].astype(np.float32); GH[int(t)] = g_
    runs.append((label, index, ens))
    print(f"{label}: {run}/{ev}, {len(index)} forecasts", flush=True)
common = sorted(set.intersection(*[set(r[1]) for r in runs]))
cams = zarr.open("/scratch/project_462001140/ammar/eccv/data/world/cams_forecast_daily/2022.zarr", mode="r")
aur_d, aur_e = {}, {}
for p in glob.glob(str(AURD / "out_global" / "part_*.npz")):              # as compare_baselines.py (analyses of day d)
    z = np.load(p); la_, lo_ = z["lat"], z["lon"]
    for d, pr in zip(z["days"], z["pred"]):
        aur_d[int(d)] = np.stack([bil(pr[h].astype(np.float32), float(la_[0]), float(la_[0] - la_[1]), float(lo_[0]), float(lo_[1] - lo_[0]))
                                  for h in range(pr.shape[0])])                                                     # (4, NS)
for p in glob.glob(str(AURD / "out_12h_all" / "part_*.npz")):             # initialised 00 UTC of day d0
    z = np.load(p); la_, lo_ = z["lat"], z["lon"]
    for d, pr in zip(z["days"], z["pred"]):
        aur_e[int(d)] = np.stack([bil(pr[k].astype(np.float32), float(la_[0]), float(la_[0] - la_[1]), float(lo_[0]), float(lo_[1] - lo_[0]))
                                  for k in range(pr.shape[0])])                                                     # (10, NS)
an0 = {}


def analysis00(d):
    """CAMS analysis PM2.5 at 00 UTC of day d at the stations (Aurora's initial condition), ug m-3"""
    if d not in an0:
        z = np.load(AURD / "raw" / f"{d:03d}.npz")
        an0[d] = bil(z["surf"][0, 5] * 1e9, float(z["lat"][0]), float(z["lat"][0] - z["lat"][1]), float(z["lon"][0]), float(z["lon"][1] - z["lon"][0]))
    return an0[d]


cache = {}


def coarse(label_i, d, h, res):
    """Foehn ensemble mean aggregated to 0.25 (block means of the evaluation) or 0.5 deg, at the stations"""
    p, n = runs[label_i][1][(d, h)]
    if p not in cache:
        if len(cache) > 6:
            cache.pop(next(iter(cache)))
        cache[p] = np.load(p)["map25"]
    m = cache[p][n].astype(np.float32)                                    # (576, 1440), centres 84 - 0.25 k - 0.125
    if res == 0.5:
        with np.errstate(all="ignore"):
            m = np.nanmean(m.reshape(288, 2, 720, 2), (1, 3))
    lat0, dl = (84.0 - 0.125, 0.25) if res == 0.25 else (84.0 - 0.25, 0.5)
    ok = np.isfinite(m).astype(np.float32)                                # bilinear over the valid (land) neighbours only
    with np.errstate(all="ignore"):
        return bil(np.nan_to_num(m) * ok, lat0, dl, -180.0 + dl / 2, dl) / bil(ok, lat0, dl, -180.0 + dl / 2, dl), m


SYS = [f"{r[0]} (1 km)" for r in runs] + [f"{runs[0][0]} at 0.25 deg", f"{runs[0][0]} at 0.5 deg", "CAMS (run d)", "CAMS (run d+1, equal issuance)",
                                         "Aurora (analyses d)", "Aurora (00 UTC d+1, equal issuance)", "persistence (day d)",
                                         "GHAP persistence (day d)", "climatology (GHAP 2017-2020)"]
NG = len(NETS) + 1
gid = np.array([list(NETS).index(s) if s in NETS else -1 for s in src])
S = np.zeros((NG, 4, len(SYS), 6))                                       # n, sum e, sum e2, sum x, sum y, sum xy  (+ per-day)
days = sorted({d for d, h in common})
PD = np.full((len(SYS), len(days), 4, 2), np.nan)
PDN = np.zeros((len(SYS), NG, len(days), 4, 2))                         # per network and day: sum of squared errors, count
SST = np.zeros((len(SYS), 4, NS, 2))                                     # per station: sum of squared errors, count
for c, (d, h) in enumerate(common):
    if h > 4 or d + h >= 365 or d + 1 > 364 or d not in aur_d or (d + 1) not in aur_e or d not in GH:
        continue
    t = d + h
    y = obs[:, t]
    X = [r[2][(d, h)] for r in runs]
    X += [coarse(0, d, h, 0.25)[0], coarse(0, d, h, 0.5)[0]]
    X.append(bil(np.asarray(cams[d, 0, h], np.float32), 90.0, 0.4, 0.0, 0.4))
    X.append(bil(np.asarray(cams[d + 1, 0, h - 1], np.float32), 90.0, 0.4, 0.0, 0.4))
    X.append(aur_d[d][h - 1])
    A = aur_e[d + 1]                                                     # leads 12, 24, ..., 120 h after 00 UTC of d + 1
    k0 = 2 * (h - 1)                                                     # index of 24 (h - 1) + 12 h
    first = analysis00(d + 1) if h == 1 else A[k0 - 1]
    X.append(0.25 * first + 0.5 * A[k0] + 0.25 * A[k0 + 1])
    X.append(obs[:, d])
    X.append(GH[d])
    X.append(CLIM[t])
    X = np.stack(X)
    ok = official & np.isfinite(y) & np.all(np.isfinite(X), 0)
    for g in range(NG):
        m = ok & ((gid == g) if g < len(NETS) else True)
        for k in range(len(SYS)):
            e = X[k, m] - y[m]
            S[g, h - 1, k] += [m.sum(), e.sum(), (e * e).sum(), X[k, m].sum(), y[m].sum(), (X[k, m] * y[m]).sum()]
    for k in range(len(SYS)):
        e = X[k, ok] - y[ok]
        PD[k, days.index(d), h - 1] = [(e * e).sum(), ok.sum()]
        for g in range(NG):
            mg = ok & ((gid == g) if g < len(NETS) else True)
            eg = X[k, mg] - y[mg]
            PDN[k, g, days.index(d), h - 1] = [(eg * eg).sum(), mg.sum()]
        SST[k, h - 1, ok, 0] += e * e; SST[k, h - 1, ok, 1] += 1
    if c % 200 == 0:
        print(c, "/", len(common), flush=True)
np.savez(HERE / "data" / f"compare_fair_{runs[0][0]}.npz", S=S, PD=PD, PDN=PDN, days=np.array(days), SST=SST, lat=lat, lon=lon, src=src, systems=np.array(SYS),
         nets=np.array(list(NETS.values()) + ["all"]))
for g, nm in reversed(list(enumerate(list(NETS.values()) + ["All official networks"]))):
    print(f"\n=== {nm}: RMSE / bias (ug m-3), identical station-days")
    for k, sn in enumerate(SYS):
        print(f"{sn:38s}" + "".join(f"  J+{h}: {np.sqrt(S[g, h - 1, k, 2] / S[g, h - 1, k, 0]):5.1f} {S[g, h - 1, k, 1] / S[g, h - 1, k, 0]:+5.1f}"
                                     for h in range(1, 5)))
    print(f"{'station-days':38s}" + "".join(f"  J+{h}: {int(S[g, h - 1, 0, 0]):>11d}" for h in range(1, 5)))
rng = np.random.default_rng(0)
ok_d = np.isfinite(PD[0, :, 0, 0])
nd, B = int(ok_d.sum()), 7
idx = (rng.integers(0, nd - B + 1, size=(10000, -(-nd // B)))[:, :, None] + np.arange(B)).reshape(10000, -1)[:, :nd]
for j in range(1, len(SYS)):
    out = []
    for h in range(4):
        a, b = PD[0, ok_d, h], PD[j, ok_d, h]
        r = lambda ix: np.sqrt(np.nansum(a[ix, 0], 1) / np.nansum(a[ix, 1], 1)) / np.sqrt(np.nansum(b[ix, 0], 1) / np.nansum(b[ix, 1], 1)) - 1
        bs = r(idx); pt = np.sqrt(np.nansum(a[:, 0]) / np.nansum(a[:, 1])) / np.sqrt(np.nansum(b[:, 0]) / np.nansum(b[:, 1])) - 1
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        out.append(f"J+{h + 1} {100 * pt:+.0f}% [{100 * lo:+.0f},{100 * hi:+.0f}]")
    print(f"{SYS[0]} vs {SYS[j]}: " + " ".join(out))
print("saved")
