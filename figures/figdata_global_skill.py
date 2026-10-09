#!/usr/bin/env python3
"""Data of the global skill figure: station scores of this work, CAMS (00 UTC run of day d, 0.4 deg), Aurora (0.4 deg,
initialised from the CAMS analysis of day d), climatology (2017-2020 mean of GHAP at the station pixel over the same
calendar days +-7, as in the Europe study) and GHAP (target day, a product).
At every lead, ALL systems shown are scored on the same station-days (leads 1-4: the five systems; 5-7: ours,
climatology, GHAP). Per network and lead: sums for RMSE, bias, correlation, CSI (35 ug/m3); per station (leads 1-4
pooled, same station-days): squared errors -> maps of RMSE and RMSE differences.
    source world/env.sh; python figdata_global_skill.py <run>   -> data/global_skill_<run>.npz"""
import glob
import sys
from pathlib import Path

import numpy as np
import zarr

RUN = sys.argv[1]
E = Path("/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km/runs") / RUN / "eval_global"
G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
AUR = Path("/scratch/project_462001140/ammar/eccv/cran-pm/experiments/benchmark/aurora/out_global")
P = [np.load(p) for p in sorted(glob.glob(str(E / "part_*.npz")))]
keys = np.concatenate([p["keys"] for p in P])
ens = np.concatenate([p["st_ens"] for p in P])
gh = np.concatenate([p["st_ghap"] for p in P]).astype(np.float32)
for f in ("stations_2022.npz", "stations_2022_v2.npz"):
    st = np.load(G / f)
    if len(st["lat"]) == ens.shape[1]:
        break
lat, lon, src, obs = st["lat"], st["lon"], st["source"], st["daily"]
NS, NETS = len(lat), [0, 1, 3, 4, 5, 6, 2]
cams = zarr.open("/scratch/project_462001140/ammar/eccv/data/world/cams_forecast_daily/2022.zarr", mode="r")
aur = {}
for p in glob.glob(str(AUR / "part_*.npz")):
    z = np.load(p); alat, alon = z["lat"], z["lon"]
    for d, pr in zip(z["days"], z["pred"]):
        aur[int(d)] = pr


def bil(field, lat0, dlat, lon0, dlon):
    nlat, nlon = field.shape
    fi = (lat0 - lat) / dlat; fj = ((lon - lon0) % 360.0) / dlon
    i0 = np.clip(np.floor(fi).astype(int), 0, nlat - 2); j0 = np.floor(fj).astype(int) % nlon; j1 = (j0 + 1) % nlon
    wi, wj = np.clip(fi - i0, 0, 1), fj - np.floor(fj)
    return ((1 - wi) * ((1 - wj) * field[i0, j0] + wj * field[i0, j1]) + wi * ((1 - wj) * field[i0 + 1, j0] + wj * field[i0 + 1, j1])).astype(np.float32)


MODELS = ["ours", "cams", "aurora", "climatology", "ghap"]
CL = np.load(G / "clim" / "st_clim_2022.npy")                                     # (365, stations)
assert CL.shape[1] == NS
# probabilistic climatology: the ensemble of the 2017-2020 GHAP values at the station over the same calendar days +-7
# (about 60 members); its fair CRPS against the observation of each 2022 day (independent of the forecast)
HIST = np.load(G / "clim" / "st_ghap_2017_2020.npy").astype(np.float32)          # (1461, stations)
hdoy = np.load(G / "meta.npz")["doy"][:1461]
CRC_T = np.full((365, NS), np.nan, np.float32)
for t in range(365):
    dd = np.abs(hdoy - t); dd = np.minimum(dd, 365 - dd)
    Xs = np.sort(np.where(HIST[dd <= 7] > 0, HIST[dd <= 7], np.nan).T, 1)           # (stations, members), NaN last
    Mc = np.isfinite(Xs).sum(1)
    kk = np.arange(1, Xs.shape[1] + 1)[None, :]
    wgt = np.where(kk <= Mc[:, None], 2 * kk - Mc[:, None] - 1, 0)                   # sum_{i<j} |xi - xj| on sorted values
    pair = np.sum(np.nan_to_num(Xs) * wgt, 1) / np.maximum(Mc * (Mc - 1), 1)
    with np.errstate(all="ignore"):
        CRC_T[t] = np.where(Mc >= 2, np.nanmean(np.abs(Xs - obs[:, t][:, None]), 1) - pair, np.nan)
L = 7
S = np.zeros((len(NETS) + 1, L, len(MODELS), 9))   # n, sum e, sum e2, sum x, sum y, sum x2, sum y2, sum xy, sum |e|
CR = np.zeros((len(NETS) + 1, L))                   # sum of the fair CRPS of this work (CRPS of the others = |e|)
CRC = np.zeros((len(NETS) + 1, L))                  # sum of the fair CRPS of the climatological ensemble
CSI = np.zeros((len(NETS) + 1, L, len(MODELS), 2))  # hits, union at 35 ug/m3
SE = np.zeros((NS, len(MODELS))); NN = np.zeros(NS)  # per station, leads 1-4, same station-days
net_idx = np.array([NETS.index(s) for s in src])
for n, (d, h, t) in enumerate(keys):
    if h > L:
        continue
    o = obs[:, t]
    X = np.full((len(MODELS), NS), np.nan, np.float32)
    X[0] = ens[n].astype(np.float32).mean(-1)
    if h <= 4:
        X[1] = bil(np.asarray(cams[int(d), 0, int(h)], np.float32), 90.0, 0.4, 0.0, 0.4)
        if int(d) in aur:
            X[2] = bil(aur[int(d)][h - 1].astype(np.float32), float(alat[0]), float(alat[0] - alat[1]), float(alon[0]), float(alon[1] - alon[0]))
    X[3] = CL[t]; X[4] = gh[n]
    use = [0, 3, 4] + ([1, 2] if h <= 4 else [])
    m = np.isfinite(o) & np.all(np.isfinite(X[use]), 0) & np.isfinite(CRC_T[t])
    if h <= 4 and int(d) not in aur:
        continue                                                   # leads 1-4 need the five systems on the same days
    for k in use:
        x, y = X[k, m], o[m]
        e = x - y
        for g in (net_idx[m], np.full(m.sum(), len(NETS))):
            np.add.at(S[:, h - 1, k], g, np.stack([np.ones_like(e), e, e * e, x, y, x * x, y * y, x * y, np.abs(e)], 1))
            np.add.at(CSI[:, h - 1, k, 0], g, ((x > 35) & (y > 35)).astype(float))
            np.add.at(CSI[:, h - 1, k, 1], g, ((x > 35) | (y > 35)).astype(float))
        if h <= 4:
            SE[m, k] += e * e
    Em = ens[n].astype(np.float32)[m]; M_ = Em.shape[1]; om = o[m]                   # fair CRPS of the ensemble
    pair = sum(np.abs(Em[:, a] - Em[:, b]) for a in range(M_) for b in range(a + 1, M_)) / (M_ * (M_ - 1))
    crps = np.abs(Em - om[:, None]).mean(1) - pair
    for g in (net_idx[m], np.full(m.sum(), len(NETS))):
        np.add.at(CR[:, h - 1], g, crps)
        np.add.at(CRC[:, h - 1], g, CRC_T[t][m])
    if h <= 4:
        NN[m] += 1
    if n % 400 == 0:
        print(n, "/", len(keys), flush=True)
out = Path(__file__).resolve().parent / "data" / f"global_skill_{RUN}.npz"
np.savez(out, S=S, CR=CR, CRC=CRC, CSI=CSI, SE=SE, NN=NN, models=np.array(MODELS), nets=np.array(NETS + [-1]), lat=lat, lon=lon, src=src)
print("saved", out)
