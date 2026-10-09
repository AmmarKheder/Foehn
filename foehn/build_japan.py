#!/usr/bin/env python3
"""Japan, 2022: hourly PM2.5 of the national monitoring network (大気汚染常時監視, NIES 環境展望台 TJ files, fiscal
years 2021-2022; JST, values hour-ending 01h..24h; 9997/9998/9999 (any value >= 9990 or > 1000) = missing; small negative values set to 0).
Coordinates: the NIES files have none, so every station is matched to an OpenAQ location (Soramame, official
coordinates) by identical hourly values in January-March 2024 (NIES fiscal year 2023 vs OpenAQ archive 2024):
best location with >= 300 common hours and >= 80 % of them equal within 0.5 ug/m3. Unmatched stations are dropped.
Daily means: local day (UTC + longitude / 15 h), >= 12 hours; stations with >= 30 valid days.
  japan_nies/japan_2022_daily.npz   codes, lat, lon, match (fraction of equal hours), daily (N, 365)
    source world/env.sh; python build_japan.py"""
import io
import zipfile
from pathlib import Path

import numpy as np

D = Path("/scratch/project_462001140/ammar/eccv/data/world/stations/japan_nies")
OQ = Path("/scratch/project_462001140/ammar/eccv/data/world/stations/openaq/openaq_2024_pm25.npz")
E = np.datetime64("1970-01-01T00:00:00")


def read_fy(fy):
    """(station code, UTC end-of-hour seconds, value) of all prefectures for a fiscal year."""
    codes, ends, vals = [], [], []
    for p in range(1, 48):
        f = D / "tj" / f"j{p:02d}_{fy}.zip"
        z = zipfile.ZipFile(f)
        name = [n for n in z.namelist() if n.endswith("_12.txt")]
        if not name:
            print("no PM2.5 file in", f.name); continue
        lines = z.read(name[0]).decode("shift_jis", "replace").splitlines()[1:]
        for ln in lines:
            c = ln.split(",")
            if len(c) < 31:
                continue
            y = int(c[0]) + (1 if int(c[5]) <= 3 else 0)
            try:
                day = np.datetime64(f"{y:04d}-{int(c[5]):02d}-{int(c[6]):02d}T00:00:00")
            except ValueError:                             # placeholder rows with impossible dates (e.g. 31 April)
                continue
            v = np.array([int(x) if x.strip() else 9999 for x in c[7:31]], np.float32)
            v[(v >= 9990) | (v > 1000)] = np.nan                  # 9997 / 9998 / 9999 = missing or invalid codes
            v[v < 0] = 0.0
            e = ((day - E) / np.timedelta64(1, "s")) + 3600.0 * np.arange(1, 25) - 9 * 3600.0
            codes.append(np.full(24, int(c[1]))); ends.append(e); vals.append(v)
    return np.concatenate(codes), np.concatenate(ends), np.concatenate(vals)


def to_hourly(codes, ends, vals, t0, t1):
    ids = np.unique(codes)
    H = int((t1 - t0) // 3600)
    a = np.full((len(ids), H), np.nan, np.float32)
    k = ((ends - t0) // 3600).astype(int)
    ok = (k >= 0) & (k < H) & np.isfinite(vals)
    a[np.searchsorted(ids, codes[ok]), k[ok]] = vals[ok]
    return ids, a


# 1. matching period: January-March 2024 (end-of-hour times)
t0, t1 = [(np.datetime64(s) - E) / np.timedelta64(1, "s") for s in ("2024-01-01T01:00:00", "2024-04-01T01:00:00")]
nies_ids, nies = to_hourly(*read_fy(2023), t0 - 3600, t1)
z = np.load(OQ, allow_pickle=True)
lat, lon, loc, ts, vv = z["lats"], z["lons"], z["location_ids"], z["timestamps"].astype(np.float64), z["values"].astype(np.float32)
jp = (lat > 24) & (lat < 46) & (lon > 122) & (lon < 146) & (ts >= t0 - 3600) & (ts < t1)
oq_ids, oq = to_hourly(loc[jp], ts[jp], vv[jp], t0 - 3600, t1)
first = {int(i): k for k, i in enumerate(loc[jp])}
oq_lat = np.array([lat[jp][first[int(i)]] for i in oq_ids]); oq_lon = np.array([lon[jp][first[int(i)]] for i in oq_ids])
print(f"matching: {len(nies_ids)} NIES stations vs {len(oq_ids)} OpenAQ locations, {nies.shape[1]} hours", flush=True)
best, frac, nover = np.full(len(nies_ids), -1), np.zeros(len(nies_ids)), np.zeros(len(nies_ids), int)
fo = np.isfinite(oq)
for a in range(0, len(nies_ids), 25):
    blk = nies[a:a + 25]
    both = np.isfinite(blk)[:, None] & fo[None]
    eq = both & (np.abs(blk[:, None] - oq[None]) < 0.5)
    n = both.sum(2); r = eq.sum(2) / np.maximum(n, 1)
    r[n < 300] = 0
    j = r.argmax(1)
    best[a:a + 25], frac[a:a + 25], nover[a:a + 25] = j, r[np.arange(len(j)), j], n[np.arange(len(j)), j]
matched = frac >= 0.8
dup = np.bincount(best[matched], minlength=len(oq_ids))
print(f"matched {matched.sum()} / {len(nies_ids)} (median equal fraction {np.median(frac[matched]):.3f}); "
      f"OpenAQ locations used twice: {(dup > 1).sum()}", flush=True)
coord = {int(c): (float(oq_lat[best[k]]), float(oq_lon[best[k]]), float(frac[k])) for k, c in enumerate(nies_ids) if matched[k]}

# 2. calendar 2022: January-March from fiscal 2021, April-December from fiscal 2022 (+ margins for local days)
c1, e1, v1 = read_fy(2021); c2, e2, v2 = read_fy(2022)
codes, ends, vals = np.concatenate([c1, c2]), np.concatenate([e1, e2]), np.concatenate([v1, v2])
T0 = (np.datetime64("2022-01-01T00:00:00") - E) / np.timedelta64(1, "s")
out_c, out_lat, out_lon, out_m, out_d = [], [], [], [], []
ids = np.unique(codes)
for c in ids:
    if int(c) not in coord:
        continue
    la, lo, m = coord[int(c)]
    s = codes == c
    mid = ends[s] - 1800.0
    day = np.floor((mid + lo / 15.0 * 3600.0 - T0) / 86400.0).astype(int)
    ok = (day >= 0) & (day < 365) & np.isfinite(vals[s])
    n = np.bincount(day[ok], minlength=365); tot = np.bincount(day[ok], vals[s][ok], minlength=365)
    d = np.where(n >= 12, tot / np.maximum(n, 1), np.nan).astype(np.float32)
    if np.isfinite(d).sum() >= 30:
        out_c.append(int(c)); out_lat.append(la); out_lon.append(lo); out_m.append(m); out_d.append(d)
print(f"2022: {len(out_c)} stations with >= 30 valid days (of {len(ids)} NIES PM2.5 stations; {len(coord)} with coordinates); "
      f"mean {np.nanmean(out_d):.1f} ug/m3", flush=True)
np.savez(D / "japan_2022_daily.npz", codes=np.array(out_c), lat=np.array(out_lat), lon=np.array(out_lon),
         match=np.array(out_m), daily=np.array(out_d))
