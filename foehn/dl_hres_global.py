#!/usr/bin/env python3
"""GLOBAL version (0.25 deg, 84 N .. 59.75 S, lon -180 .. 179.75, 576 x 1440) of ../dejavu/dl_hres.py.
Real weather forecasts for the realistic test: ECMWF IFS HRES (WeatherBench2, 0.25 deg), initialised at 00 UTC on each
day of 2022, daily means of the forecast days D+1..D+4 (leads 24h*h + 0/6/12/18 h; precipitation = sum of the four
6-h accumulations ending at 06/12/18/24 UTC), interpolated to the paper's 0.25 deg Europe grid (169 x 281).
Channels = the 30 ERA5 store channels (u10, v10, t2m, msl, sp, then t/u/v/q/z at 1000/925/850/700/500 hPa), converted
with the store constants (x - m) / s, plus total precipitation (m/day). BLH is not in WeatherBench2.
Output: data/hres/<month>.npz  block[days, 4, 31, 576, 1440] float16, day_of_year[days].
    python dl_hres.py <month 1-12>"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

URL = "https://storage.googleapis.com/weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"
OUT = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km/hres")
OUT.mkdir(parents=True, exist_ok=True)
SURF = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "mean_sea_level_pressure", "surface_pressure"]
LEVS = ["temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity", "geopotential"]
PLEV = [1000, 925, 850, 700, 500]
MEANS = np.array([0., 0., 280., 101325., 97000.] + [260.] * 5 + [0.] * 5 + [0.] * 5 + [0.003] * 5 + [50000.] * 5, np.float32)
STDS = np.array([10., 10., 20., 1500., 5000.] + [30.] * 5 + [15.] * 5 + [10.] * 5 + [0.004] * 5 + [40000.] * 5, np.float32)
LAT = 84.0 - 0.25 * np.arange(576)
LON = -180.0 + 0.25 * np.arange(1440)

def bilinear(a, lat, lon):
    """(C, nlat, nlon) on ascending lat/lon -> (C, 576, 1440) at LAT x LON (exact on the same 0.25 deg grid)."""
    fi = np.interp(LAT, lat, np.arange(len(lat))); fj = np.interp(LON, lon, np.arange(len(lon)))
    i0 = np.clip(np.floor(fi).astype(int), 0, len(lat) - 2); j0 = np.clip(np.floor(fj).astype(int), 0, len(lon) - 2)
    wi, wj = (fi - i0)[:, None], (fj - j0)[None, :]
    g = lambda di, dj: a[:, i0 + di][:, :, j0 + dj]
    return (1 - wi) * ((1 - wj) * g(0, 0) + wj * g(0, 1)) + wi * ((1 - wj) * g(1, 0) + wj * g(1, 1))


month = int(sys.argv[1])
ds = xr.open_zarr(URL, consolidated=True, chunks=None)
times = ds.time.values
sel = [t for t in times if str(t)[:7] == f"2022-{month:02d}" and str(t)[11:13] == "00"]
import os
sel = sel[:int(os.environ.get("LIMIT", len(sel)))]
ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180).sortby("longitude")
ds = ds.sel(latitude=slice(-61, 85)) if ds.latitude[0] < ds.latitude[-1] else ds.sel(latitude=slice(85, -61))
blocks, doys = [], []
for t in sel:
    per_h = []
    for h in (1, 2, 3, 4):
        leads = [24 * h + k for k in (0, 6, 12, 18)]  # hours (int coordinate)
        tp_leads = [24 * h + k for k in (6, 12, 18, 24)]
        x = ds.sel(time=t)
        chans = [x[v].sel(prediction_timedelta=leads).mean("prediction_timedelta") for v in SURF]
        for v in LEVS:
            a = x[v].sel(prediction_timedelta=leads, level=PLEV).mean("prediction_timedelta")
            chans += [a.sel(level=p).drop_vars("level") for p in PLEV]
        tp = x["total_precipitation_6hr"].sel(prediction_timedelta=tp_leads).sum("prediction_timedelta")
        arr = xr.concat([c.reset_coords(drop=True) for c in chans + [tp]], "c", coords="minimal", compat="override").load()
        arr = bilinear(arr.transpose("c", "latitude", "longitude").values, arr.latitude.values, arr.longitude.values)
        arr[:30] = (arr[:30] - MEANS[:, None, None]) / STDS[:, None, None]
        per_h.append(arr.astype(np.float16))
    blocks.append(np.stack(per_h)); doys.append(int((np.datetime64(t, "D") - np.datetime64("2022-01-01")).astype(int)))
    print(str(t)[:10], flush=True)
np.savez(OUT / f"{month:02d}.npz", block=np.stack(blocks), day_of_year=np.array(doys))
print("done month", month, len(sel), "inits")
