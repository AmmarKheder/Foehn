#!/usr/bin/env python3
"""Station daily means on UTC days (00-24 UTC), the day used by GHAP ("all daily PM2.5 measurements were adjusted to the UTC
time zone", Wei et al. 2023) and therefore by Foehn, and the day of the CAMS and Aurora daily means. build_stations.py
averaged the hourly values over the local solar day (UTC + longitude / 15 h), and the EEA and EPA daily files over the
reporting day (UTC+1 and local standard time). This script recomputes every network on UTC days for exactly the stations of
stations_2022.npz (same order, coordinates and sources; NaN where a station has no UTC value), with the same rules
otherwise:
  CNEMC, OpenAQ   hourly values (UTC timestamps) -> UTC-day means, >= 12 hourly values, 0-1000 ug/m3;
  EEA             hourly values of the EEA download service (raw_parquet; Start in UTC+1, shifted by -1 h), all valid
                  hours of the day as in the official daily file (eea_pm25_daily.zarr);
  EPA AQS         hourly files (Date GMT), instruments of a site averaged per hour, >= 12 hours;
  AirKorea        hourly values (UTC mid-hour), >= 12 hours;
  Japan (NIES)    hourly TJ files (JST hour ending -> UTC mid-hour), stations and coordinates of japan_2022_daily.npz;
  India (CPCB)    15-min values -> hourly (>= 2 of 4), stuck sensors removed, as build_india.py, UTC days.
Finally the evaluation file stations_2022.npz is written: these UTC means for every network except the EEA, whose official
daily means (reporting day UTC+1) agree better with GHAP than UTC means of its hourly values and are kept; the local-day
file of build_stations.py is kept as stations_2022_localday.npz. 285 EPA sites with filter samples only have no hourly
values and are left out (NaN).
    python build_stations_utc.py   -> data/world/global1km/stations_2022_utc.npz, stations_2022.npz"""
import glob
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

S = Path("/scratch/project_462001140/ammar/eccv/data/world/stations")
G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
EEA = Path("/scratch/project_462001140/ammar/eccv/data/zarr/eea_stations")
T0 = np.datetime64("2022-01-01T00:00:00").astype("datetime64[s]").astype(np.int64)
LOC = G / "stations_2022_localday.npz"                                         # the local-day file of build_stations.py
if not LOC.exists():
    shutil.copy(G / "stations_2022.npz", LOC)
ref = np.load(LOC)
LAT, LON, SRC, LOCAL = ref["lat"], ref["lon"], ref["source"], ref["daily"]
OUT = np.full(LOCAL.shape, np.nan, np.float32)
rk = lambda la, lo: (round(float(la), 4), round(float(lo), 4))


def assign(code, lat, lon, daily, name, order=None):
    """coordinates -> every reference station of that source at them (duplicated sites share the series); order: the
    reference stations of that source in the order of `daily` (EEA: sampling points can share coordinates)"""
    idx = np.where(SRC == code)[0]
    hit = 0
    if order is not None:
        assert len(order) == len(idx), (len(order), len(idx))
        OUT[idx] = daily[order]; hit = len(idx)
    else:
        key = {}
        for i in idx:
            key.setdefault(rk(LAT[i], LON[i]), []).append(i)
        for la, lo, d in zip(lat, lon, daily):
            for i in key.get(rk(np.float32(la), np.float32(lo)), []):
                OUT[i] = d; hit += 1
    loc, utc = LOCAL[idx], OUT[idx]
    both = np.isfinite(loc) & np.isfinite(utc)
    r = np.corrcoef(loc[both], utc[both])[0, 1] if both.sum() > 10 else np.nan
    print(f"{name:14s} matched {hit:5d} of {len(idx):5d} stations | station-days local {int(np.isfinite(loc).sum()):8d} "
          f"UTC {int(np.isfinite(utc).sum()):8d} | mean local {np.nanmean(loc):5.1f} UTC {np.nanmean(utc):5.1f} | r {r:.3f}", flush=True)


def daily_from(day, station, value, nst, min_hours=12):
    ok = np.isfinite(value) & (value >= 0) & (value <= 1000) & (day >= 0) & (day < 365)
    key = station[ok].astype(np.int64) * 365 + day[ok].astype(np.int64)
    n = np.bincount(key, minlength=nst * 365).reshape(nst, 365)
    s = np.bincount(key, weights=value[ok], minlength=nst * 365).reshape(nst, 365)
    return np.where(n >= min_hours, s / np.maximum(n, 1), np.nan).astype(np.float32)


# ---- CNEMC and OpenAQ: as build_stations.hourly_to_daily, UTC day
for p, code, box, name in ((S / "cnemc/cnemc_2022_pm25.npz", 1, None, "CNEMC"),
                           (S / "openaq/openaq_2022_pm25.npz", 2, (30.0, 72.0, -25.0, 45.0), "OpenAQ")):
    z = np.load(p, allow_pickle=True)
    lat, lon, v, t, loc = z["lats"], z["lons"], z["values"].astype(np.float64), z["timestamps"].astype(np.float64), z["location_ids"]
    ok = np.isfinite(v) & (v >= 0) & (v <= 1000) & np.isfinite(lat) & np.isfinite(lon)
    if box is not None:
        ok &= ~((lat >= box[0]) & (lat <= box[1]) & (lon >= box[2]) & (lon <= box[3]))
    lat, lon, v, t, loc = lat[ok], lon[ok], v[ok], t[ok], loc[ok]
    ids, inv = np.unique(loc, return_inverse=True)
    slat = np.bincount(inv, weights=lat) / np.bincount(inv); slon = np.bincount(inv, weights=lon) / np.bincount(inv)
    d = daily_from(np.floor((t - T0) / 86400.0), inv, v, len(ids))
    assign(code, slat, slon, d, name)

# ---- EEA: hourly parquet, Start in UTC+1 -> UTC, all hours of the day (as the official daily file)
ez = zarr.open(str(EEA / "eea_pm25_daily.zarr"), mode="r")
ids = [str(s) for s in np.asarray(ez["station_ids"])]; pos = {s: k for k, s in enumerate(ids)}; xy = np.asarray(ez["station_coords"])
acc_s = np.zeros((len(ids), 365)); acc_n = np.zeros((len(ids), 365))
t_lo, t_hi = pd.Timestamp("2021-12-31 00:00"), pd.Timestamp("2023-01-02 00:00")
for f in glob.glob(str(EEA / "raw_parquet" / "**" / "*.parquet"), recursive=True):
    df = pd.read_parquet(f, columns=["Samplingpoint", "Start", "Value"])
    df["Start"] = pd.to_datetime(df["Start"], errors="coerce")
    df = df[(df["Start"] >= t_lo) & (df["Start"] < t_hi)]
    if df.empty:
        continue
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df[df["Value"].notna() & (df["Value"] >= 0) & (df["Value"] < 1000)]
    utc = df["Start"] - pd.Timedelta(hours=1)
    day = ((utc - pd.Timestamp("2022-01-01")).dt.total_seconds() // 86400).astype(int).values
    sp = df["Samplingpoint"].str.replace(r"^[A-Z]{2}/", "", regex=True).map(pos).values
    ok = (day >= 0) & (day < 365) & pd.notna(sp)
    if not ok.any():
        continue
    k = sp[ok].astype(np.int64) * 365 + day[ok]
    acc_s += np.bincount(k, weights=df["Value"].values[ok].astype(np.float64), minlength=len(ids) * 365).reshape(len(ids), 365)
    acc_n += np.bincount(k, minlength=len(ids) * 365).reshape(len(ids), 365)
eea = np.where(acc_n > 0, acc_s / np.maximum(acc_n, 1), np.nan).astype(np.float32)
off = np.asarray(ez["daily/2022"], np.float32).T[:, :365]                       # the official file, as build_stations.py
has = np.where((np.isfinite(off) & (off >= 0)).sum(1) >= 30)[0]                # its EEA stations, in its order
assign(0, xy[:, 0], xy[:, 1], eea, "EEA", order=has)

# ---- EPA AQS: hourly files (Date GMT), instruments of a site averaged per hour
cols = ["State Code", "County Code", "Site Num", "Latitude", "Longitude", "Date GMT", "Time GMT", "Sample Measurement"]
epa = pd.concat([pd.read_csv(S / "epa_aqs" / "hourly" / f"hourly_{c}_2022.zip", usecols=cols, dtype=str) for c in (88101, 88502)])
epa["v"] = pd.to_numeric(epa["Sample Measurement"], errors="coerce")
epa = epa[(epa["v"] >= 0) & (epa["v"] <= 1000)]
epa["site"] = epa["State Code"] + "-" + epa["County Code"] + "-" + epa["Site Num"]
epa["hour"] = ((pd.to_datetime(epa["Date GMT"] + " " + epa["Time GMT"]) - pd.Timestamp("2022-01-01")).dt.total_seconds() // 3600).astype(int)
hv = epa.groupby(["site", "hour"])["v"].mean().reset_index()                    # instruments of a site averaged per hour
sites = sorted(hv["site"].unique()); si = {s: k for k, s in enumerate(sites)}
d = daily_from((hv["hour"].values // 24).astype(float), hv["site"].map(si).values, hv["v"].values.astype(np.float64), len(sites))
epa["lat"] = pd.to_numeric(epa["Latitude"]); epa["lon"] = pd.to_numeric(epa["Longitude"])
pos_ = epa.groupby("site")[["lat", "lon"]].mean().loc[sites]
# the daily file gave each site the mean of its rows' coordinates: match on those, recomputed from the daily file
dly = pd.concat([pd.read_csv(S / f"epa_aqs/daily_{c}_2022.csv", usecols=["State Code", "County Code", "Site Num", "Latitude",
                 "Longitude"], dtype=str) for c in (88101, 88502)])
dly["site"] = dly["State Code"] + "-" + dly["County Code"] + "-" + dly["Site Num"]
dly["lat"] = pd.to_numeric(dly["Latitude"]); dly["lon"] = pd.to_numeric(dly["Longitude"])
dpos = dly.groupby("site")[["lat", "lon"]].mean()
la_ = np.array([dpos.loc[s, "lat"] if s in dpos.index else pos_.loc[s, "lat"] for s in sites])
lo_ = np.array([dpos.loc[s, "lon"] if s in dpos.index else pos_.loc[s, "lon"] for s in sites])
assign(3, la_, lo_, d, "EPA AQS")

# ---- AirKorea: hourly, UTC mid-hour seconds from 2022-01-01
k = np.load(S / "airkorea/korea_2022_hourly.npz"); kc = np.load(S / "airkorea/korea_coords.npz")
kxy = np.stack([np.where(kc["how"] > 0, kc["lat"], np.nan), np.where(kc["how"] > 0, kc["lon"], np.nan)], 1)
d = daily_from(np.floor(k["time"] / 86400.0), k["station"], k["pm25"].astype(np.float64), len(kxy))
assign(4, kxy[:, 0], kxy[:, 1], d, "AirKorea")

# ---- Japan (NIES): TJ files of fiscal 2021-2022, JST hour ending -> UTC
E = np.datetime64("1970-01-01T00:00:00")
JD = S / "japan_nies"


def read_fy(fy):
    codes, ends, vals = [], [], []
    for p in range(1, 48):
        zf = zipfile.ZipFile(JD / "tj" / f"j{p:02d}_{fy}.zip")
        name = [n for n in zf.namelist() if n.endswith("_12.txt")]
        if not name:
            continue
        for ln in zf.read(name[0]).decode("shift_jis", "replace").splitlines()[1:]:
            c = ln.split(",")
            if len(c) < 31:
                continue
            y = int(c[0]) + (1 if int(c[5]) <= 3 else 0)
            try:
                day = np.datetime64(f"{y:04d}-{int(c[5]):02d}-{int(c[6]):02d}T00:00:00")
            except ValueError:
                continue
            v = np.array([int(x) if x.strip() else 9999 for x in c[7:31]], np.float32)
            v[(v >= 9990) | (v > 1000)] = np.nan
            v[v < 0] = 0.0
            codes.append(np.full(24, int(c[1]))); ends.append(((day - E) / np.timedelta64(1, "s")) + 3600.0 * np.arange(1, 25) - 9 * 3600.0)
            vals.append(v)
    return np.concatenate(codes), np.concatenate(ends), np.concatenate(vals)


jz = np.load(JD / "japan_2022_daily.npz")
c1, e1, v1 = read_fy(2021); c2, e2, v2 = read_fy(2022)
codes, ends, vals = np.concatenate([c1, c2]), np.concatenate([e1, e2]), np.concatenate([v1, v2])
T0J = (np.datetime64("2022-01-01T00:00:00") - E) / np.timedelta64(1, "s")
jix = {int(c): k for k, c in enumerate(jz["codes"])}
m = np.isin(codes, jz["codes"])
d = daily_from(np.floor((ends[m] - 1800.0 - T0J) / 86400.0), np.array([jix[int(c)] for c in codes[m]]),
               vals[m].astype(np.float64), len(jz["codes"]))
assign(5, jz["lat"], jz["lon"], d, "Japan (NIES)")

# ---- India (CPCB): as build_india.py, UTC day
iz = zarr.open(zarr.storage.ZipStore(str(S / "india_caaqms/all_in_one.zarr.zip"), mode="r"), mode="r")
t = np.datetime64("2023-01-01T00:00") + iz["Timestamp"][:].astype("timedelta64[m]")
i0, i1 = np.searchsorted(t, np.datetime64("2021-12-31T00:00")), np.searchsorted(t, np.datetime64("2023-01-02T00:00"))
t = t[i0:i1] - np.timedelta64(330, "m")
pm = iz["PM2.5"][i0:i1].astype(np.float32); ilat, ilon = iz["latitude"][:], iz["longitude"][:]
pm[(pm < 0) | (pm > 1000)] = np.nan
n4 = pm.shape[0] // 4 * 4
q = pm[:n4].reshape(-1, 4, pm.shape[1]); hour_t = t[:n4:4]
with np.errstate(all="ignore"):
    hv_ = np.where(np.isfinite(q).sum(1) >= 2, np.nanmean(q, 1), np.nan)
stuck = np.zeros_like(hv_, bool); same = np.isfinite(hv_[1:]) & (hv_[1:] == hv_[:-1]); run = np.zeros(hv_.shape[1], int)
for kk in range(1, hv_.shape[0]):
    run = np.where(same[kk - 1], run + 1, 0); hit = run >= 5
    if hit.any():
        stuck[kk - 5:kk + 1, hit] = True
hv_[stuck] = np.nan
sec = (hour_t + np.timedelta64(30, "m") - np.datetime64("2022-01-01T00:00")) / np.timedelta64(1, "s")
day = np.repeat(np.floor(sec / 86400.0)[:, None], hv_.shape[1], 1)
stn = np.repeat(np.arange(hv_.shape[1])[None], hv_.shape[0], 0)
d = daily_from(day.ravel(), stn.ravel(), hv_.ravel().astype(np.float64), hv_.shape[1])
assign(6, ilat, ilon, d, "India (CPCB)")

np.savez(G / "stations_2022_utc.npz", lat=LAT, lon=LON, source=SRC, daily=OUT)
print("saved", G / "stations_2022_utc.npz", "| official networks: stations with a UTC value",
      int((np.isfinite(OUT).sum(1)[SRC != 2] > 0).sum()), "of", int((SRC != 2).sum()))
# ---- the evaluation file: UTC days everywhere but the EEA, which keeps its official daily means
final = np.where((SRC == 0)[:, None], LOCAL, OUT).astype(np.float32)
np.savez(G / "stations_2022.npz", lat=LAT, lon=LON, source=SRC, daily=final)
print("saved", G / "stations_2022.npz", "| stations with >= 30 valid days, six official networks:",
      int(((np.isfinite(final).sum(1) >= 30) & (SRC != 2)).sum()))
