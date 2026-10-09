#!/usr/bin/env python3
"""Forecast-period winds for the realistic footprint test: ECMWF IFS HRES (WeatherBench2, 1.5 deg), init 00 UTC of
each 2022 day d, u/v at 10 m and 925 hPa for leads 0..117 h every 3 h (6-hourly leads, odd 3-h steps linearly
interpolated) = 40 steps covering days d..d+4 exactly like data/wind3h.npy, on its 0.5 deg grid (84 x 140), plus the
rain of day d (sum of the 6-h accumulations of leads 6..24 h). BLH is not in WeatherBench2.
Output: data/hres_wind/<month>.npz  wind[days, 40, 4, 84, 140] float16 (u10 v10 u925 v925), rain_d0[days, 84, 140]
(mm), day_of_year[days].      python dl_hres_wind.py <month>"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

URL = "https://storage.googleapis.com/weatherbench2/datasets/hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr"
OUT = Path(__file__).resolve().parent / "data" / "hres_wind"
OUT.mkdir(parents=True, exist_ok=True)
LAT = 71.875 - 0.5 * np.arange(84)
LON = -24.875 + 0.5 * np.arange(140)


def bilinear(a, lat, lon):
    fi = np.interp(LAT, lat, np.arange(len(lat))); fj = np.interp(LON, lon, np.arange(len(lon)))
    i0 = np.clip(np.floor(fi).astype(int), 0, len(lat) - 2); j0 = np.clip(np.floor(fj).astype(int), 0, len(lon) - 2)
    wi, wj = (fi - i0)[:, None], (fj - j0)[None, :]
    g = lambda di, dj: a[..., i0 + di, :][..., j0 + dj]
    return (1 - wi) * ((1 - wj) * g(0, 0) + wj * g(0, 1)) + wi * ((1 - wj) * g(1, 0) + wj * g(1, 1))


month = int(sys.argv[1])
ds = xr.open_zarr(URL, consolidated=True, chunks=None)
sel = [t for t in ds.time.values if str(t)[:7] == f"2022-{month:02d}" and str(t)[11:13] == "00"]
ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180).sortby("longitude")
ds = ds.sel(latitude=slice(28, 74), longitude=slice(-28, 48))
leads = list(range(0, 120, 6))  # 0..114
winds, rains, doys = [], [], []
for t in sel:
    x = ds.sel(time=t)
    sfc = xr.concat([x["10m_u_component_of_wind"].sel(prediction_timedelta=leads),
                     x["10m_v_component_of_wind"].sel(prediction_timedelta=leads),
                     x["u_component_of_wind"].sel(prediction_timedelta=leads, level=925).drop_vars("level"),
                     x["v_component_of_wind"].sel(prediction_timedelta=leads, level=925).drop_vars("level")],
                    "c", coords="minimal", compat="override").load()                     # (4, 20, lat, lon)
    a = bilinear(sfc.transpose("prediction_timedelta", "c", "latitude", "longitude").values, sfc.latitude.values,
                 sfc.longitude.values)                                                     # (20, 4, 84, 140)
    w = np.empty((40, 4, 84, 140), np.float32)
    w[0::2] = a
    w[1:-1:2] = 0.5 * (a[:-1] + a[1:])
    w[-1] = a[-1]                                                                          # 117 h: hold the last lead
    tp = x["total_precipitation_6hr"].sel(prediction_timedelta=[6, 12, 18, 24]).sum("prediction_timedelta").load()
    rains.append(bilinear(tp.transpose("latitude", "longitude").values, tp.latitude.values, tp.longitude.values) * 1000.0)
    winds.append(w.astype(np.float16)); doys.append(int((np.datetime64(t, "D") - np.datetime64("2022-01-01")).astype(int)))
    print(str(t)[:10], flush=True)
np.savez(OUT / f"{month:02d}.npz", wind=np.stack(winds), rain_d0=np.stack(rains).astype(np.float32), day_of_year=np.array(doys))
print("done", month)
