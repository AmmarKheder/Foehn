#!/usr/bin/env python3
"""Station-day values behind the station comparison of the paper (test year 2022, final model crps_e01; the stations only
score, nothing here is used to train, select or correct a model).

This is the logic of experiments/figures/compare_fair.py (read-only source, not modified), with one change: instead of
sums per network it keeps every value, so that the scores can be split by station class and resampled consistently.

For each start day d (every second day of 2022), lead h = 1..4 days and station s of stations_2022.npz it stores
  foehn      Foehn ensemble mean (4 members) at the 1 km pixel of the station      (eval_final_crps_e01/part_*.npz: st_ens)
  foehn025   the same forecast averaged to 0.25 deg, read bilinearly at the station (map25)
  foehn05    the same forecast averaged to 0.5 deg (2 x 2 cells of map25), bilinear
  cams1      CAMS forecast, run of 00 UTC d+1 (equal issuance), lead day h-1, bilinear from the 0.4 deg grid
  cams0      CAMS forecast, run of 00 UTC d, lead day h
  aur1       Aurora initialised 00 UTC d+1 (equal issuance), daily mean 0.25/0.5/0.25 of the 12-hourly fields
  aur0       Aurora from the analyses of day d (daily means of out_global)
  clim       climatology of the 1 km estimates 2017-2020 at the station pixel, target day (+-7 days)
  ghap0      1 km estimate (GHAP) of the START day d at the station pixel: what the start-day input holds there
  ghapt      1 km estimate (GHAP) of the TARGET day d+h at the station pixel
  obs0       measurement of the start day d
  obs        measurement of the target day d+h
and `paper` = the mask of compare_fair.py (official network, every one of its ten systems finite), which reproduces the
sums of experiments/figures/data/compare_fair_crps_e01.npz exactly (checked at the end).

    python extract_station_day.py     -> ../data/station_day.npz
"""
import glob
from pathlib import Path

import numpy as np
import zarr

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data"
EXP = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments")
RUN = EXP / "global1km/runs/green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4/eval_final_crps_e01"
G = Path("/scratch/project_462001804/ammar/eccv/data/world/global1km")
AURD = EXP / "benchmark" / "aurora"
REF = EXP / "figures/data/compare_fair_crps_e01.npz"
NETS = {0: "Europe (EEA)", 1: "China (CNEMC)", 3: "USA (EPA)", 4: "Korea (AirKorea)", 5: "Japan (NIES)", 6: "India (CPCB)"}

st = np.load(G / "stations_2022.npz")
lat, lon, src, obs = st["lat"], st["lon"], st["source"], st["daily"].astype(np.float32)
NS = len(lat)
official = np.isin(src, list(NETS))
# compare_fair.py places the forecasts through a coordinate key, round(lat, 4) * 1e6 + round(lon, 4), computed in float32
# (resolution 4 at these magnitudes). Stations that share a key (monitors at the same coordinates, and some distinct
# stations at the same latitude) collapse onto the last of them; the others stay empty and drop out of the paper's scores.
# The kept station keeps its own forecast and its own measurement. `kept` reproduces that rule; `paper_all` ignores it.
key = lambda la, lo: np.round(la, 4) * 1e6 + np.round(lo, 4)
ref_ = {v: i for i, v in enumerate(key(lat, lon))}
sets = {}
for f in sorted(G.glob("stations_2022*.npz")):
    z_ = np.load(f); sets[len(z_["lat"])] = np.array([ref_.get(v, -1) for v in key(z_["lat"], z_["lon"])])
kept = np.zeros(NS, bool); kept[sets[NS][sets[NS] >= 0]] = True
assert np.all(sets[NS][kept] == np.flatnonzero(kept))                    # a kept station maps onto itself
print(f"official stations with data: {int((official & np.isfinite(obs).any(1)).sum())}; scored by compare_fair.py (kept by its "
      f"coordinate key): {int((official & kept & np.isfinite(obs).any(1)).sum())}", flush=True)


def bil(field, lat0, dlat, lon0, dlon):
    """bilinear read of a (lat decreasing, lon increasing) field at the stations: as compare_fair.bil"""
    nlat, nlon = field.shape
    fi = (lat0 - lat) / dlat; fj = ((lon - lon0) % 360.0) / dlon
    i0 = np.clip(np.floor(fi).astype(int), 0, nlat - 2); j0 = np.floor(fj).astype(int) % nlon; j1 = (j0 + 1) % nlon
    wi, wj = np.clip(fi - i0, 0, 1), fj - np.floor(fj)
    return ((1 - wi) * ((1 - wj) * field[i0, j0] + wj * field[i0, j1]) + wi * ((1 - wj) * field[i0 + 1, j0] + wj * field[i0 + 1, j1])).astype(np.float32)


def coarse(m, res):
    """Foehn ensemble mean aggregated to 0.25 deg (block means of the evaluation) or 0.5 deg, at the stations"""
    m = m.astype(np.float32)                                              # (576, 1440), centres 84 - 0.25 k - 0.125
    if res == 0.5:
        with np.errstate(all="ignore"):
            m = np.nanmean(m.reshape(288, 2, 720, 2), (1, 3))
    lat0, dl = (84.0 - 0.125, 0.25) if res == 0.25 else (84.0 - 0.25, 0.5)
    ok = np.isfinite(m).astype(np.float32)                                # bilinear over the valid (land) neighbours only
    with np.errstate(all="ignore"):
        return bil(np.nan_to_num(m) * ok, lat0, dl, -180.0 + dl / 2, dl) / bil(ok, lat0, dl, -180.0 + dl / 2, dl)


CLIM = np.load(G / "clim" / "st_clim_2022.npy").astype(np.float32)        # (365, stations)
parts = sorted(glob.glob(str(RUN / "part_*.npz")))
assert len(parts) == 64, len(parts)

# ---- pass 1: where each forecast is, and GHAP at the station pixels per target day (as compare_fair: first occurrence)
index, GH = {}, {}
for p in parts:
    z = np.load(p)
    keys, sg = z["keys"], z["st_ghap"]
    assert z["st_ens"].shape[1] == NS
    for n, (d, h, t) in enumerate(keys):
        index[(int(d), int(h))] = (p, n)
        if int(t) not in GH:
            GH[int(t)] = sg[n].astype(np.float32)
print(len(index), "forecasts;", len(GH), "target days with GHAP at the station pixels", flush=True)

cams = zarr.open("/scratch/project_462001804/ammar/eccv/data/world/cams_forecast_daily/2022.zarr", mode="r")
aur_d, aur_e = {}, {}
for p in glob.glob(str(AURD / "out_global" / "part_*.npz")):
    z = np.load(p); la_, lo_ = z["lat"], z["lon"]
    for d, pr in zip(z["days"], z["pred"]):
        aur_d[int(d)] = np.stack([bil(pr[h].astype(np.float32), float(la_[0]), float(la_[0] - la_[1]), float(lo_[0]), float(lo_[1] - lo_[0]))
                                  for h in range(pr.shape[0])])
for p in glob.glob(str(AURD / "out_12h_all" / "part_*.npz")):
    z = np.load(p); la_, lo_ = z["lat"], z["lon"]
    for d, pr in zip(z["days"], z["pred"]):
        aur_e[int(d)] = np.stack([bil(pr[k].astype(np.float32), float(la_[0]), float(la_[0] - la_[1]), float(lo_[0]), float(lo_[1] - lo_[0]))
                                  for k in range(pr.shape[0])])
print("Aurora:", len(aur_d), "days (analyses d),", len(aur_e), "days (00 UTC)", flush=True)
an0 = {}


def analysis00(d):
    """CAMS analysis PM2.5 at 00 UTC of day d at the stations (Aurora's initial condition), ug m-3"""
    if d not in an0:
        z = np.load(AURD / "raw" / f"{d:03d}.npz")
        an0[d] = bil(z["surf"][0, 5] * 1e9, float(z["lat"][0]), float(z["lat"][0] - z["lat"][1]), float(z["lon"][0]), float(z["lon"][1] - z["lon"][0]))
    return an0[d]


days = sorted({d for d, h in index})
NAMES = ["foehn", "foehn025", "foehn05", "cams0", "cams1", "aur0", "aur1", "obs0", "ghap0", "clim", "ghapt", "obs"]
A = {k: np.full((len(days), 4, NS), np.nan, np.float32) for k in NAMES}
paper = np.zeros((len(days), 4, NS), bool)
paper_all = np.zeros((len(days), 4, NS), bool)
used = np.zeros((len(days), 4), bool)
by_part = {}
for (d, h), (p, n) in index.items():
    by_part.setdefault(p, []).append((d, h, n))
for c, p in enumerate(parts):
    z = np.load(p)
    ens, m25 = z["st_ens"], z["map25"]
    for d, h, n in sorted(by_part[p]):
        if h > 4 or d + h >= 365 or d + 1 > 364 or d not in aur_d or (d + 1) not in aur_e or d not in GH:
            continue                                                       # the conditions of compare_fair.py
        t, i = d + h, days.index(d)
        X = {"foehn": ens[n].astype(np.float32).mean(-1),
             "foehn025": coarse(m25[n], 0.25), "foehn05": coarse(m25[n], 0.5),
             "cams0": bil(np.asarray(cams[d, 0, h], np.float32), 90.0, 0.4, 0.0, 0.4),
             "cams1": bil(np.asarray(cams[d + 1, 0, h - 1], np.float32), 90.0, 0.4, 0.0, 0.4),
             "aur0": aur_d[d][h - 1]}
        E = aur_e[d + 1]                                                   # leads 12, 24, ..., 120 h after 00 UTC of d + 1
        k0 = 2 * (h - 1)
        first = analysis00(d + 1) if h == 1 else E[k0 - 1]
        X["aur1"] = 0.25 * first + 0.5 * E[k0] + 0.25 * E[k0 + 1]
        X["obs0"], X["ghap0"], X["clim"] = obs[:, d], GH[d], CLIM[t]
        y = obs[:, t]
        paper_all[i, h - 1] = official & np.isfinite(y) & np.all(np.isfinite(np.stack([X[k] for k in NAMES[:10]])), 0)
        paper[i, h - 1] = paper_all[i, h - 1] & kept
        X["ghapt"], X["obs"] = GH.get(t, np.full(NS, np.nan, np.float32)), y
        for k in NAMES:
            A[k][i, h - 1] = X[k]
        used[i, h - 1] = True
    print(f"part {c + 1}/{len(parts)}", flush=True)

OUT.mkdir(exist_ok=True)
np.savez_compressed(OUT / "station_day.npz", days=np.array(days), leads=np.arange(1, 5), used=used, paper=paper, paper_all=paper_all,
                    kept=kept, lat=lat, lon=lon, src=src, **A)

# ---- check against the paper's file: counts and sums of squared errors per network and lead
ref = np.load(REF)
S, sysn = ref["S"], list(ref["systems"])
col = {"foehn": 0, "foehn025": 1, "foehn05": 2, "cams0": 3, "cams1": 4, "aur0": 5, "aur1": 6, "obs0": 7, "ghap0": 8, "clim": 9}
gid = np.array([list(NETS).index(s) if s in NETS else -1 for s in src])
worst = 0.0
for g in range(7):
    m = paper & ((gid == g) if g < 6 else True)
    for h in range(4):
        assert int(m[:, h].sum()) == int(S[g, h, 0, 0]), (g, h, m[:, h].sum(), S[g, h, 0, 0])
        for k, j in col.items():
            e = (A[k][:, h] - A["obs"][:, h])[m[:, h]].astype(np.float64)
            worst = max(worst, abs((e * e).sum() / S[g, h, j, 2] - 1))
print(f"check against {REF.name}: identical station-day counts in the 7 groups x 4 leads; largest relative difference of a "
      f"sum of squared errors over 10 systems: {worst:.2e}")
print(f"days of the comparison: {int(ref['days'].shape[0])} in the paper's file, {int(used.any(1).sum())} here; "
      f"station-days at lead 1: {int(paper[:, 0].sum())}")
print("saved", OUT / "station_day.npz")
