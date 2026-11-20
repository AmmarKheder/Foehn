#!/usr/bin/env python3
"""COPY of extremes_stations_own_eqiss.py (itself a copy of experiments/figures/extremes_stations_own.py) with ONE more kind
of threshold, 'matched': the events of kind 'ghap' (measurement above the local percentile of GHAP 2017-2020 at the station
pixel for the calendar day), and for every system a threshold rescaled at each station by one factor so that the system
warns exactly as often as the event occurs at that station over the scored station-days of the lead (yes on the n_e days
with the largest forecast / threshold ratio). It is the station analogue of the kind 'matched' of land_freqmatch.py.
The first three kinds must reproduce ../data/extremes_stations_own_eqiss_<label>.npz (checked at the end).
    python extremes_stations_matched.py <label>=<run>[:<eval dir>]   -> ../data/extremes_stations_kinds4_<label>.npz

Docstring of the parent script:
COPY of experiments/figures/extremes_stations_own.py, extended; the definitions are unchanged.
Detection of local extremes at the stations of the six official networks (2022; the stations only score), Foehn's
ensemble mean, the CAMS forecast of 00 UTC d+1 and Aurora initialised at 00 UTC d+1 (issued at the same time as Foehn;
this script already used the equal issuance), and the station's own measurement of day d (persistence), as yes/no forecasts
scored by the SEDI with three kinds of thresholds:
  'ghap':    GHAP 2017-2020 local percentiles at the station pixel (+-15 days), the same for every system (as over land);
  'station': the station's own 2022 percentiles, the same for every system (the measured threshold);
  'own':     the observations and every system have their own percentiles at each station over the same station-days of the
             lead time, so that every system warns as often as the event occurs (frequency-matched).
Changes with respect to the original:
  * lead times 1-7 for Foehn and 1-5 for the CAMS forecast and Aurora (the run of 00 UTC d+1 covers five target days; the
    original stopped at 4). At leads 1-4 the station-days and the counts are those of the original file (checked by
    check_reproduction.py). At leads 6-7 only Foehn and the persistence are scored.
  * the counts are also kept per start day (CD), with the thresholds fixed, for the moving-block bootstrap.
    python extremes_stations_own_eqiss.py <label>=<run>[:<eval dir>]   -> ../data/extremes_stations_own_eqiss_<label>.npz"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import zarr

FIG = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/figures")
RUNS = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km/runs")
G = Path("/scratch/project_462001804/ammar/eccv/data/world/global1km")
AURD = FIG.parent / "benchmark" / "aurora"
OUT = Path(__file__).resolve().parent.parent / "data"
NETS = {0: "EEA", 1: "CNEMC", 3: "EPA", 4: "AirKorea", 5: "NIES", 6: "CPCB"}
NETL = list(NETS) + [-1]                                                     # -1: all official stations
PCTS, WINDOW, MIN_DAYS = (90, 95, 99), 15, 100
SYS = ["Foehn (mean)", "CAMS", "Aurora", "persistence"]
HMAX, HB = 7, 5                                                              # last lead of Foehn, of the baselines

st = np.load(G / os.environ.get("STATIONS", "stations_2022.npz"))
lat, lon, src, obs = st["lat"], st["lon"], st["source"], st["daily"].astype(np.float32)
NS = len(lat)
official = np.isin(src, list(NETS))
key = lambda la, lo: np.round(la, 4) * 1e6 + np.round(lo, 4)
ref = {v: i for i, v in enumerate(key(lat, lon))}
sets = {}
for f in sorted(G.glob("stations_2022*.npz")):
    z = np.load(f); sets[len(z["lat"])] = np.array([ref.get(v, -1) for v in key(z["lat"], z["lon"])])


def nanpct(x, qs):
    """np.nanpercentile(x, qs, axis=0), vectorised (as extremes_stations_mean.py)"""
    xs = np.sort(x, axis=0); n = np.isfinite(x).sum(0); out = []
    for q in qs:
        pos = q / 100.0 * np.maximum(n - 1, 0); lo = np.floor(pos).astype(np.int64); hi = np.minimum(lo + 1, np.maximum(n - 1, 0))
        w = (pos - lo).astype(np.float32)
        out.append(np.take_along_axis(xs, lo[None], 0)[0] * (1 - w) + np.take_along_axis(xs, hi[None], 0)[0] * w)
    res = np.stack(out); res[:, n == 0] = np.nan
    return res


def bil(field, lat0, dlat, lon0, dlon):
    nlat, nlon = field.shape
    fi = (lat0 - lat) / dlat; fj = ((lon - lon0) % 360.0) / dlon
    i0 = np.clip(np.floor(fi).astype(int), 0, nlat - 2); j0 = np.floor(fj).astype(int) % nlon; j1 = (j0 + 1) % nlon
    wi, wj = np.clip(fi - i0, 0, 1), fj - np.floor(fj)
    return ((1 - wi) * ((1 - wj) * field[i0, j0] + wj * field[i0, j1]) + wi * ((1 - wj) * field[i0 + 1, j0] + wj * field[i0 + 1, j1])).astype(np.float32)


# ---- the thresholds of the paper (same for every system)
HIST = np.load(G / "clim" / "st_ghap_2017_2020.npy").astype(np.float32)          # (1461, NS) GHAP at the station pixel
hdoy = np.load(G / "meta.npz")["doy"][:1461]
THR_G = np.full((365, 3, NS), np.nan, np.float32)
for t in range(365):
    dd = np.abs(hdoy - t); dd = np.minimum(dd, 365 - dd)
    x = HIST[dd <= WINDOW]; x = np.where(x > 0, x, np.nan)
    q = nanpct(x, PCTS); q[:, np.isfinite(x).sum(0) < 60] = np.nan
    THR_G[t] = q
THR_S = nanpct(np.where(obs.T > 0, obs.T, np.nan), PCTS)                          # (3, NS) annual, station record
THR_S[:, np.isfinite(obs).sum(1) < 200] = np.nan

# ---- forecasts at the stations (as extremes_stations_mean.py)
label, spec = sys.argv[1].split("=")
run, ev = (spec.split(":") + ["eval_global"])[:2]
ens = {}
for p in sorted(glob.glob(str(RUNS / run / ev / "part_*.npz"))):
    z = np.load(p); e = z["st_ens"]; idx = sets[e.shape[1]]
    for n, (d, h, t) in enumerate(z["keys"]):
        full = np.full((NS, e.shape[2]), np.nan, np.float32); full[idx[idx >= 0]] = e[n, idx >= 0].astype(np.float32)
        ens[(int(d), int(h))] = full
print(label, len(ens), "forecasts", flush=True)
cams = zarr.open("/scratch/project_462001804/ammar/eccv/data/world/cams_forecast_daily/2022.zarr", mode="r")
aur_e = {}
for p in glob.glob(str(AURD / "out_12h_all" / "part_*.npz")):
    z = np.load(p); la_, lo_ = z["lat"], z["lon"]
    for d, pr in zip(z["days"], z["pred"]):
        aur_e[int(d)] = np.stack([bil(pr[k].astype(np.float32), float(la_[0]), float(la_[0] - la_[1]), float(lo_[0]), float(lo_[1] - lo_[0]))
                                  for k in range(pr.shape[0])])
an0 = {}


def analysis00(d):
    if d not in an0:
        z = np.load(AURD / "raw" / f"{d:03d}.npz")
        an0[d] = bil(z["surf"][0, 5] * 1e9, float(z["lat"][0]), float(z["lat"][0] - z["lat"][1]), float(z["lon"][0]), float(z["lon"][1] - z["lon"][0]))
    return an0[d]


# rows per lead: (start day, target day, [obs, Foehn mean, CAMS, Aurora, persistence]), NaN outside the common station-days
DAYS = sorted({d for d, h in ens}); di = {d: i for i, d in enumerate(DAYS)}
rows = {h: [] for h in range(1, HMAX + 1)}
for (d, h) in sorted(ens):
    if h > HMAX or d + h >= 365 or (d + 1) not in aur_e:
        continue
    t = d + h; y = obs[:, t]
    fo = ens[(d, h)]
    ok = official & np.isfinite(y) & np.isfinite(obs[:, d]) & np.isfinite(fo).all(1)
    if h <= HB:
        A = aur_e[d + 1]; k0 = 2 * (h - 1)
        aur = 0.25 * (analysis00(d + 1) if h == 1 else A[k0 - 1]) + 0.5 * A[k0] + 0.25 * A[k0 + 1]
        cam = bil(np.asarray(cams[d + 1, 0, h - 1], np.float32), 90.0, 0.4, 0.0, 0.4)
        ok &= np.isfinite(aur) & np.isfinite(cam)
    else:
        aur = cam = np.full(NS, np.nan, np.float32)
    X = np.stack([y, fo.mean(1), cam, aur, obs[:, d]]).astype(np.float32)
    X[:, ~ok] = np.nan
    rows[h].append((d, t, X))
print("forecasts per lead:", {h: len(v) for h, v in rows.items()}, flush=True)

# ---- counts[kind, tier, lead, system, network, (hits, misses, false alarms, correct negatives)] and the same per start day
C = np.zeros((4, 3, HMAX, len(SYS), len(NETL), 4))
CD = np.zeros((len(DAYS),) + C.shape)
NST = np.zeros((4, HMAX, len(NETL)), np.int64)                               # stations that enter, per kind, lead and network
netmask = [src == n if n >= 0 else official for n in NETL]
for h in range(1, HMAX + 1):
    D_ = np.array([di[d] for d, _, _ in rows[h]]); T = np.array([t for _, t, _ in rows[h]]); X = np.stack([x for _, _, x in rows[h]])   # (days, 5, NS)
    Y = X[:, 0]
    ownY = nanpct(Y, PCTS)                                                   # (3, NS): per station, over the same station-days
    ownF = [nanpct(X[:, 1 + s], PCTS) for s in range(len(SYS))]
    few = np.isfinite(Y).sum(0) < MIN_DAYS
    ownY[:, few] = np.nan
    for kind in range(4):
        for tier in range(3):
            for s in range(len(SYS)):
                if s in (1, 2) and h > HB:
                    continue
                F = X[:, 1 + s]
                if kind == 3:                                                # 'matched': the events of 'ghap', frequency bias 1 at every station
                    ty = THR_G[T, tier]
                    m = np.isfinite(Y) & np.isfinite(F) & np.isfinite(ty)
                    e_ = (Y > ty) & m; n_e = e_.sum(0)
                    with np.errstate(all="ignore"):
                        ratio = np.where(m, F / ty, -np.inf)
                    order = np.argsort(-ratio, axis=0, kind="stable"); rank = np.empty_like(order)
                    np.put_along_axis(rank, order, np.arange(len(Y))[:, None], 0)
                    g_ = (rank < n_e[None]) & m
                    for k, nm in enumerate(netmask):
                        mm = m & nm[None]
                        v = np.stack([(e_ & g_ & mm).sum(1), (e_ & ~g_ & mm).sum(1), (~e_ & g_ & mm).sum(1), (~e_ & ~g_ & mm).sum(1)], -1)
                        CD[D_, kind, tier, h - 1, s, k] = v
                        C[kind, tier, h - 1, s, k] = v.sum(0)
                        if s == 0 and tier == 2:
                            NST[kind, h - 1, k] = mm.any(0).sum()
                    continue
                if kind == 0:
                    ty = tf = THR_G[T, tier]                                 # (days, NS)
                elif kind == 1:
                    ty = tf = np.broadcast_to(THR_S[tier], Y.shape)
                else:
                    ty, tf = np.broadcast_to(ownY[tier], Y.shape), np.broadcast_to(ownF[s][tier], Y.shape)
                m = np.isfinite(Y) & np.isfinite(F) & np.isfinite(ty) & np.isfinite(tf)
                e_, g_ = (Y > ty) & m, (F > tf) & m
                for k, nm in enumerate(netmask):
                    mm = m & nm[None]
                    v = np.stack([(e_ & g_ & mm).sum(1), (e_ & ~g_ & mm).sum(1), (~e_ & g_ & mm).sum(1), (~e_ & ~g_ & mm).sum(1)], -1)  # (days, 4)
                    CD[D_, kind, tier, h - 1, s, k] = v
                    C[kind, tier, h - 1, s, k] = v.sum(0)
                    if s == 0 and tier == 2:
                        NST[kind, h - 1, k] = mm.any(0).sum()
f = OUT / f"extremes_stations_kinds4_{label}.npz"
np.savez(f, C=C, CD=CD, NST=NST, days=np.array(DAYS), systems=np.array(SYS), tiers=np.array(["P90", "P95", "P99"]),
         kinds=np.array(["ghap", "station", "own", "matched"]), nets=np.array([NETS.get(n, "all") for n in NETL]))
print("saved", f)


def sedi(c):
    h, m, f_, n = np.moveaxis(np.asarray(c, np.float64), -1, 0)
    with np.errstate(all="ignore"):
        H = np.clip(h / (h + m), 1e-6, 1 - 1e-6); F = np.clip(f_ / (f_ + n), 1e-6, 1 - 1e-6)
        return (np.log(F) - np.log(H) - np.log(1 - F) + np.log(1 - H)) / (np.log(F) + np.log(H) + np.log(1 - F) + np.log(1 - H))


old = np.load(OUT / f"extremes_stations_own_eqiss_{label}.npz")
print("first three kinds identical to extremes_stations_own_eqiss:", np.array_equal(old["C"], C[:3]), np.array_equal(old["CD"], CD[:, :3]))
for kind, kn in enumerate(("GHAP 2017-2020 local percentiles (same for all)", "station's own 2022 percentiles (same for all)",
                           "each system's own percentiles (frequency-matched)", "GHAP 2017-2020 events, thresholds matched in frequency at each station")):
    for tier in (1, 2):
        print(f"\n=== {kn} | above P{PCTS[tier]}: SEDI, all official stations, leads 1-{HMAX}")
        for s, nm in enumerate(SYS):
            print(f"  {nm:14s} " + " ".join(f"{sedi(C[kind, tier, h, s, -1]):.3f}" if C[kind, tier, h, s, -1].sum() else "  -  " for h in range(HMAX)))
