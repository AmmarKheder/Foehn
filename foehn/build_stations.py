#!/usr/bin/env python3
"""Validation stations of the global test year 2022, daily means (never used in training):
  EEA (Europe, official daily values), CNEMC (China), US EPA AQS (official daily means, FRM/FEM and other methods,
  instruments of a site averaged), AirKorea (South Korea, final confirmed hourly data, geocoded addresses; build_korea.py),
  the Japanese national network (NIES hourly data, coordinates matched on OpenAQ; build_japan.py), India CPCB CAAQMS
  (build_india.py) and OpenAQ (rest of the world; stations inside the EEA domain are left to EEA, and an OpenAQ station
  within 2 km of an official station is dropped). Hourly values -> local-day means
  (UTC + longitude / 15 h) with at least 12 hourly values, values 0-1000 ug/m3; stations with >= 30 valid days.
Output: data/world/global1km/stations_2022.npz  lat, lon, source (0 EEA, 1 CNEMC, 2 OpenAQ, 3 EPA, 4 AirKorea, 5 Japan,
  6 India), daily (N, 365)."""
from pathlib import Path

import numpy as np
import zarr

S = Path("/scratch/project_462001140/ammar/eccv/data/world/stations")
OUT = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km/stations_2022.npz")
T0 = np.datetime64("2022-01-01T00:00:00").astype("datetime64[s]").astype(np.int64)


def hourly_to_daily(p, exclude_box=None, min_hours=12):
    z = np.load(p, allow_pickle=True)
    lat, lon, v, t, loc = z["lats"], z["lons"], z["values"].astype(np.float64), z["timestamps"].astype(np.float64), z["location_ids"]
    ok = np.isfinite(v) & (v >= 0) & (v <= 1000) & np.isfinite(lat) & np.isfinite(lon)
    if exclude_box is not None:
        la0, la1, lo0, lo1 = exclude_box
        ok &= ~((lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1))
    lat, lon, v, t, loc = lat[ok], lon[ok], v[ok], t[ok], loc[ok]
    day = np.floor((t + lon / 15.0 * 3600.0 - T0) / 86400.0).astype(np.int64)
    keep = (day >= 0) & (day < 365)
    lat, lon, v, day, loc = lat[keep], lon[keep], v[keep], day[keep], loc[keep]
    ids, inv = np.unique(loc, return_inverse=True)
    key = inv.astype(np.int64) * 365 + day
    n = np.bincount(key, minlength=len(ids) * 365).reshape(len(ids), 365)
    s = np.bincount(key, weights=v, minlength=len(ids) * 365).reshape(len(ids), 365)
    daily = np.where(n >= min_hours, s / np.maximum(n, 1), np.nan).astype(np.float32)
    slat = np.bincount(inv, weights=lat) / np.bincount(inv); slon = np.bincount(inv, weights=lon) / np.bincount(inv)
    has = np.isfinite(daily).sum(1) >= 30
    print(p.name, "stations with >= 30 valid days:", int(has.sum()), "of", len(ids))
    return slat[has], slon[has], daily[has]


eea = zarr.open("/scratch/project_462001140/ammar/eccv/data/zarr/eea_stations/eea_pm25_daily.zarr", mode="r")
xy = np.asarray(eea["station_coords"]); d = np.asarray(eea["daily/2022"], np.float32).T[:, :365]
d = np.where(np.isfinite(d) & (d >= 0), d, np.nan)
has = np.isfinite(d).sum(1) >= 30
parts = [(xy[has, 0], xy[has, 1], d[has], 0)]
print("EEA stations with >= 30 valid days:", int(has.sum()))
la, lo, dd = hourly_to_daily(S / "cnemc/cnemc_2022_pm25.npz"); parts.append((la, lo, dd, 1))
la, lo, dd = hourly_to_daily(S / "openaq/openaq_2022_pm25.npz", exclude_box=(30.0, 72.0, -25.0, 45.0)); parts.append((la, lo, dd, 2))

import pandas as pd  # noqa: E402
epa = pd.concat([pd.read_csv(S / f"epa_aqs/daily_{c}_2022.csv", usecols=["State Code", "County Code", "Site Num", "Latitude",
                 "Longitude", "Date Local", "Arithmetic Mean"]) for c in (88101, 88502)])
epa["site"] = epa["State Code"].astype(str) + "-" + epa["County Code"].astype(str) + "-" + epa["Site Num"].astype(str)
epa["day"] = (pd.to_datetime(epa["Date Local"]) - pd.Timestamp("2022-01-01")).dt.days
epa = epa[(epa["day"] >= 0) & (epa["day"] < 365) & (epa["Arithmetic Mean"] >= 0) & (epa["Arithmetic Mean"] <= 1000)]
g = epa.groupby(["site", "day"])["Arithmetic Mean"].mean().unstack()                     # instruments of a site averaged
g = g.reindex(columns=range(365))
pos = epa.groupby("site")[["Latitude", "Longitude"]].mean().loc[g.index]
dd = g.values.astype(np.float32); keep = np.isfinite(dd).sum(1) >= 30
parts.append((pos["Latitude"].values[keep], pos["Longitude"].values[keep], dd[keep], 3))
print("EPA AQS sites with >= 30 valid days:", int(keep.sum()), "of", len(keep))

import json  # noqa: E402
k = np.load(S / "airkorea/korea_2022_hourly.npz")                           # AirKorea: hourly (UTC mid-hour, s from 2022)
kc = np.load(S / "airkorea/korea_coords.npz")                             # official (OpenAQ) or geocoded coordinates
assert np.array_equal(kc["codes"], k["codes"])
kxy = np.stack([np.where(kc["how"] > 0, kc["lat"], np.nan), np.where(kc["how"] > 0, kc["lon"], np.nan)], 1)
st_, v_ = k["station"], k["pm25"].astype(np.float64)
ok_ = np.isfinite(v_) & (v_ >= 0) & (v_ <= 1000)
kday = np.floor((k["time"] + kxy[st_, 1] / 15.0 * 3600.0) / 86400.0)
ok_ &= np.isfinite(kday) & (kday >= 0) & (kday < 365)
key = st_[ok_].astype(np.int64) * 365 + kday[ok_].astype(np.int64)
n_ = np.bincount(key, minlength=len(kxy) * 365).reshape(-1, 365); s_ = np.bincount(key, v_[ok_], minlength=len(kxy) * 365).reshape(-1, 365)
kd = np.where(n_ >= 12, s_ / np.maximum(n_, 1), np.nan).astype(np.float32)
keep = (np.isfinite(kd).sum(1) >= 30) & np.isfinite(kxy[:, 0])
parts.append((kxy[keep, 0], kxy[keep, 1], kd[keep], 4))
print("AirKorea stations with >= 30 valid days and coordinates:", int(keep.sum()), "of", len(keep))
for f, code, nm in (("japan_nies/japan_2022_daily.npz", 5, "Japan (NIES)"), ("india_caaqms/india_2022_daily.npz", 6, "India (CAAQMS)")):
    z_ = np.load(S / f)
    parts.append((z_["lat"], z_["lon"], z_["daily"].astype(np.float32), code))
    print(nm, "stations:", len(z_["lat"]))
lat = np.concatenate([p[0] for p in parts]); lon = np.concatenate([p[1] for p in parts])
daily = np.concatenate([p[2] for p in parts]); source = np.concatenate([np.full(len(p[0]), p[3]) for p in parts])
off = np.where(source != 2)[0]
dup = np.zeros(len(lat), bool)
for i in np.where(source == 2)[0]:                                          # OpenAQ within 2 km of an official station
    near = off[(np.abs(lat[off] - lat[i]) < 0.03) & (np.abs(lon[off] - lon[i]) < 0.03 / max(np.cos(np.deg2rad(lat[i])), 0.1))]
    if len(near):
        dkm = 111.2 * np.hypot(lat[near] - lat[i], (lon[near] - lon[i]) * np.cos(np.deg2rad(lat[i])))
        dup[i] = dkm.min() < 2.0
print("OpenAQ stations dropped as duplicates of an official station (within 2 km):", int(dup.sum()))
inside = (lat <= 84) & (lat > -60) & ~dup
np.savez(OUT, lat=lat[inside].astype(np.float32), lon=lon[inside].astype(np.float32), source=source[inside], daily=daily[inside])
for k, name in enumerate(("EEA", "CNEMC", "OpenAQ", "EPA AQS", "AirKorea", "Japan", "India")):
    m = inside & (source == k)
    print(f"{name:8s} {int(m.sum()):6d} stations, {int(np.isfinite(daily[m]).sum()):9d} station-days, median {np.nanmedian(daily[m]):.1f} ug/m3")
