#!/usr/bin/env python3
"""(Copy of extremes_stations.py that adds the ensemble MEAN of each Foehn run as a yes/no forecast, for the SEDI of the
Methods, as over land in tail_events.py; output extremes_stations_mean_<label>.npz, the original file is untouched.)
Probabilistic forecasts of local pollution extremes at the stations (official networks, 2022; stations only evaluate).
Events after the TailWeather labelling (pm25_labeling_global.py, as tail_events.py): local percentile thresholds per
location and calendar day (+-15 days), tiers moderate / severe / extreme = above the local P90 / P95 / P99.
  thresholds 'ghap': GHAP 2017-2020 at the station pixel (the labelling's climatology, training years only);
  thresholds 'station': the station's own 2022 record (annual percentiles; robustness, no multi-year station history).
Forecasts: Foehn runs (probability = share of members above the threshold), CAMS and Aurora issued at the same time
(CAMS run of 00 UTC d+1, lead h-1; Aurora initialised 00 UTC d+1, out_12h_all; yes/no), persistence of day d (yes/no).
Scores per tier and lead 1-4, identical station-days: Brier score, Brier skill score against the base rate, ROC area,
hit rate and false-alarm ratio at probability >= 0.5.
    python extremes_stations.py <label>=<run>[:<eval dir>] ...   -> data/extremes_stations_<first label>.npz"""
import glob
import sys
from pathlib import Path

import numpy as np
import zarr

HERE = Path(__file__).resolve().parent
RUNS = Path("/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km/runs")
G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
AURD = HERE.parent / "benchmark" / "aurora"
NETS = {0: "EEA", 1: "CNEMC", 3: "EPA", 4: "AirKorea", 5: "NIES", 6: "CPCB"}
PCTS, WINDOW = (90, 95, 99), 15

import os  # noqa: E402
STATIONS = os.environ.get("STATIONS", "stations_2022.npz"); TAG = os.environ.get("TAG", "")   # UTC-day means: stations_2022_utc.npz
st = np.load(G / STATIONS)
lat, lon, src, obs = st["lat"], st["lon"], st["source"], st["daily"].astype(np.float32)
NS = len(lat)
official = np.isin(src, list(NETS))
key = lambda la, lo: np.round(la, 4) * 1e6 + np.round(lo, 4)
ref = {v: i for i, v in enumerate(key(lat, lon))}
sets = {}
for f in sorted(G.glob("stations_2022*.npz")):
    z = np.load(f); sets[len(z["lat"])] = np.array([ref.get(v, -1) for v in key(z["lat"], z["lon"])])


def nanpct(x, qs):
    """np.nanpercentile(x, qs, axis=0), vectorised (as tail_events.py)"""
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


# ---- thresholds
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
print("thresholds ready", flush=True)

# ---- forecasts
runs = []
for arg in sys.argv[1:]:
    label, spec = arg.split("=")
    run, ev = (spec.split(":") + ["eval_global"])[:2]
    ens = {}
    for p in sorted(glob.glob(str(RUNS / run / ev / "part_*.npz"))):
        z = np.load(p); e = z["st_ens"]; idx = sets[e.shape[1]]
        for n, (d, h, t) in enumerate(z["keys"]):
            full = np.full((NS, e.shape[2]), np.nan, np.float32); full[idx[idx >= 0]] = e[n, idx >= 0].astype(np.float32)
            ens[(int(d), int(h))] = full
    runs.append((label, ens)); print(label, len(ens), "forecasts", flush=True)
common = sorted(set.intersection(*[set(r[1]) for r in runs]))
cams = zarr.open("/scratch/project_462001140/ammar/eccv/data/world/cams_forecast_daily/2022.zarr", mode="r")
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


SYS = [r[0] + " (mean)" for r in runs] + [r[0] for r in runs] + ["CAMS", "Aurora", "persistence"]
NB = 41                                                                      # probability bins of 1/40: exact for 4 and 8 members
# counts[thr kind, tier, lead, system, bin, event(0/1)]
C = np.zeros((2, 3, 4, len(SYS), NB, 2))
for c, (d, h) in enumerate(common):
    if h > 4 or d + h >= 365 or (d + 1) not in aur_e:
        continue
    t = d + h
    y = obs[:, t]
    A = aur_e[d + 1]; k0 = 2 * (h - 1)
    aur = 0.25 * (analysis00(d + 1) if h == 1 else A[k0 - 1]) + 0.5 * A[k0] + 0.25 * A[k0 + 1]
    cam = bil(np.asarray(cams[d + 1, 0, h - 1], np.float32), 90.0, 0.4, 0.0, 0.4)
    ok = official & np.isfinite(y) & np.isfinite(aur) & np.isfinite(cam) & np.isfinite(obs[:, d])
    for r in runs:
        ok &= np.isfinite(r[1][(d, h)]).all(1)
    for kind, T in enumerate((THR_G[t], THR_S)):
        for tier in range(3):
            thr = T[tier]; m = ok & np.isfinite(thr)
            ev = (y[m] > thr[m]).astype(int)
            probs = [(r[1][(d, h)][m].mean(1) > thr[m]).astype(float) for r in runs]
            probs += [(r[1][(d, h)][m] > thr[m][:, None]).mean(1) for r in runs]
            probs += [(cam[m] > thr[m]).astype(float), (aur[m] > thr[m]).astype(float), (obs[m, d] > thr[m]).astype(float)]
            for s, p in enumerate(probs):
                b = np.clip(np.round(p * (NB - 1)).astype(int), 0, NB - 1)
                np.add.at(C[kind, tier, h - 1, s], (b, ev), 1)
    if c % 200 == 0:
        print(c, "/", len(common), flush=True)


def scores(cnt):
    """cnt (NB, 2) counts by probability bin and outcome -> Brier, BSS, ROC area, hit rate and FAR at p >= 0.5"""
    p = np.linspace(0, 1, NB)
    n = cnt.sum(); o = cnt[:, 1].sum() / n
    bs = ((cnt[:, 1] * (1 - p) ** 2) + (cnt[:, 0] * p ** 2)).sum() / n
    bss = 1 - bs / (o * (1 - o))
    H = np.concatenate([[1.0], np.cumsum(cnt[::-1, 1])[::-1][1:] / cnt[:, 1].sum(), [0.0]])   # hit rate for p >= bin
    F = np.concatenate([[1.0], np.cumsum(cnt[::-1, 0])[::-1][1:] / cnt[:, 0].sum(), [0.0]])
    auc = -np.trapezoid(H, F)
    hi = p >= 0.5
    pod = cnt[hi, 1].sum() / cnt[:, 1].sum(); far = cnt[hi, 0].sum() / max(cnt[hi].sum(), 1)
    return o, bs, bss, auc, pod, far


np.savez(HERE / "data" / f"extremes_stations_mean_{runs[0][0]}{TAG}.npz", C=C, systems=np.array(SYS), tiers=np.array(["P90", "P95", "P99"]))
for kind, kn in enumerate(("GHAP 2017-2020 local climatology (TailWeather labelling)", "station's own 2022 record")):
    for tier, tn in enumerate(("moderate > P90", "severe > P95", "extreme > P99")):
        print(f"\n=== thresholds: {kn} | {tn}: Brier skill score / ROC area / hit rate / false-alarm ratio (p >= 0.5)")
        for h in range(1, 5):
            o = scores(C[kind, tier, h - 1, 0])[0]
            row = []
            for s, sn in enumerate(SYS):
                _, bs, bss, auc, pod, far = scores(C[kind, tier, h - 1, s])
                row.append(f"{sn}: {bss:+.2f} / {auc:.2f} / {pod:.2f} / {far:.2f}")
            print(f"  J+{h} (event rate {o:.3f})  " + " | ".join(row))
print("saved")
