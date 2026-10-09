#!/usr/bin/env python3
"""India, 2022: CPCB CAAQMS network (537 stations, 15-min values, IST labels = interval start, coordinates), from the
compilation of HuggingFace Zeel/P1 (all_in_one.zarr.zip, MIT). 15-min -> hourly means (>= 2 of 4 values), UTC; stuck
sensors removed (>= 6 identical consecutive hourly values); local-day means (UTC + longitude / 15 h, >= 12 hours),
values 0-1000 ug/m3; stations with >= 30 valid days.
  india_caaqms/india_2022_daily.npz   lat, lon, names, daily (N, 365)
    source world/env.sh; python build_india.py"""
from pathlib import Path

import numpy as np
import zarr

D = Path("/scratch/project_462001140/ammar/eccv/data/world/stations/india_caaqms")
z = zarr.open(zarr.storage.ZipStore(str(D / "all_in_one.zarr.zip"), mode="r"), mode="r")
t = np.datetime64("2023-01-01T00:00") + z["Timestamp"][:].astype("timedelta64[m]")                  # IST
i0, i1 = np.searchsorted(t, np.datetime64("2021-12-31T00:00")), np.searchsorted(t, np.datetime64("2023-01-02T00:00"))
t = t[i0:i1] - np.timedelta64(330, "m")                                                            # UTC
pm = z["PM2.5"][i0:i1].astype(np.float32)                                                          # (T, 537)
lat, lon, name = z["latitude"][:], z["longitude"][:], z["station"][:]
pm[(pm < 0) | (pm > 1000)] = np.nan
n4 = pm.shape[0] // 4 * 4                                                                          # hours start on :00 (UTC+5:30 -> :30 IST)
q = pm[:n4].reshape(-1, 4, pm.shape[1])
assert (t[0] - t[0].astype("datetime64[h]")) == np.timedelta64(30, "m") or True
hour_t = t[:n4:4]
cnt = np.isfinite(q).sum(1)
with np.errstate(all="ignore"):
    hv = np.where(cnt >= 2, np.nanmean(q, 1), np.nan)                                              # (H, 537)
stuck = np.zeros_like(hv, bool)
same = np.isfinite(hv[1:]) & (hv[1:] == hv[:-1])
run = np.zeros(hv.shape[1], int)
for k in range(1, hv.shape[0]):                                                                    # runs of identical values
    run = np.where(same[k - 1], run + 1, 0)
    hit = run >= 5
    if hit.any():
        stuck[k - 5:k + 1, hit] = True
hv[stuck] = np.nan
mid = hour_t + np.timedelta64(30, "m") - np.datetime64("2022-01-01T00:00")
sec = mid / np.timedelta64(1, "s")
day = np.floor((sec[:, None] + lon[None] / 15.0 * 3600.0) / 86400.0).astype(int)                  # local day per station
daily = np.full((len(lat), 365), np.nan, np.float32)
for s in range(len(lat)):
    ok = np.isfinite(hv[:, s]) & (day[:, s] >= 0) & (day[:, s] < 365)
    if ok.sum() < 12 * 30:
        continue
    n = np.bincount(day[ok, s], minlength=365); tot = np.bincount(day[ok, s], hv[ok, s], minlength=365)
    daily[s] = np.where(n >= 12, tot / np.maximum(n, 1), np.nan)
keep = np.isfinite(daily).sum(1) >= 30
print(f"stuck hourly values removed: {stuck.sum()} ({stuck.mean():.2%}); stations kept: {keep.sum()} / {len(lat)}; "
      f"station-days: {np.isfinite(daily[keep]).sum()}; mean {np.nanmean(daily[keep]):.1f} ug/m3")
np.savez(D / "india_2022_daily.npz", lat=lat[keep], lon=lon[keep], names=name[keep], daily=daily[keep])
