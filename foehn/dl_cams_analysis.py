#!/usr/bin/env python3
"""CAMS global analysis (forecast lead 0 of the 00 and 12 UTC runs, 0.4 deg: Aurora's initial conditions and ground
truth) of the pollutants that Aurora takes at the surface, Oct 2017 - Dec 2021, for the 'Foehn trained as Aurora'
experiment (2022 is in ../benchmark/aurora/raw/<doy>.npz, same variables). Output PM2.5 only; these are its inputs:
  single level: PM1, PM2.5, PM10 (ug m-3); total columns of CO, NO, NO2, O3, SO2 (kg m-2, as delivered);
  1000 hPa: CO, NO, NO2, O3, SO2 mass mixing ratios (x 1e9: ug kg-1), the near-surface gases.
One request per month (Atmosphere Data Store, ~/.adsapirc) -> data/world/cams_analysis_surface/<yyyy>_<mm>.npz:
one (days, 2, 451, 900) float32 array per variable (00 and 12 UTC), lat (90 .. -90), lon (0 .. 359.6).
    python dl_cams_analysis.py [workers 3]"""
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cdsapi
import numpy as np
import xarray as xr

OUT = Path("/scratch/project_462001140/ammar/eccv/data/world/cams_analysis_surface")
OUT.mkdir(parents=True, exist_ok=True)
ADS = dict(l.split(":", 1) for l in (Path.home() / ".adsapirc").read_text().splitlines() if ":" in l)
SL = {"particulate_matter_1um": ("pm1", 1e9), "particulate_matter_2.5um": ("pm2p5", 1e9), "particulate_matter_10um": ("pm10", 1e9),
      "total_column_carbon_monoxide": ("tcco", 1.0), "total_column_nitrogen_monoxide": ("tc_no", 1.0),
      "total_column_nitrogen_dioxide": ("tcno2", 1.0), "total_column_ozone": ("gtco3", 1.0),
      "total_column_sulphur_dioxide": ("tcso2", 1.0)}
PL = {"carbon_monoxide": ("co", 1e9), "nitrogen_monoxide": ("no", 1e9), "nitrogen_dioxide": ("no2", 1e9), "ozone": ("go3", 1e9),
      "sulphur_dioxide": ("so2", 1e9)}
NC_LOCK = threading.Lock()
MONTHS = [(y, m) for y in range(2017, 2022) for m in range(1, 13) if (y, m) >= (2017, 10)]


def fetch(ym):
    y, m = ym
    dst = OUT / f"{y}_{m:02d}.npz"
    if dst.exists():
        return f"{y}-{m:02d} exists"
    last = int(str(np.datetime64(f"{y}-{m:02d}") + np.timedelta64(1, "M") - np.timedelta64(1, "D"))[8:10])
    zp = OUT / f"{y}_{m:02d}.zip"
    print(f"{y}-{m:02d} requested", flush=True)
    for attempt in range(4):
        try:
            cdsapi.Client(url=ADS["url"].strip(), key=ADS["key"].strip(), quiet=True, progress=False).retrieve(
                "cams-global-atmospheric-composition-forecasts",
                {"type": "forecast", "leadtime_hour": "0", "date": f"{y}-{m:02d}-01/{y}-{m:02d}-{last:02d}",
                 "time": ["00:00", "12:00"], "variable": list(SL) + list(PL), "pressure_level": ["1000"],
                 "format": "netcdf_zip"}, str(zp))
            break
        except Exception as e:                                  # ADS queue hiccups
            print(f"{y}-{m:02d} retry {attempt}: {repr(e)[:200]}", flush=True)
    tmp = OUT / f"tmp_{y}_{m:02d}"; tmp.mkdir(exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        z.extractall(tmp)
    arrs = {}
    with NC_LOCK:                                               # netCDF4 / HDF5 is not thread-safe (segfault otherwise)
        for f in sorted(tmp.glob("*.nc")):
            ds = xr.open_dataset(f)
            for v in ds.data_vars:
                arrs[v] = ds[v].load()
            ds.close()
    out, lat, lon = {}, None, None
    for long, (short, scale) in {**SL, **PL}.items():
        v = arrs[short].squeeze(drop=True)
        tdim = [d for d in v.dims if d not in ("latitude", "longitude")]
        a = v.transpose(*tdim, "latitude", "longitude").values.astype(np.float32) * scale
        out[short] = a.reshape(last, 2, a.shape[-2], a.shape[-1])
        lat, lon = v["latitude"].values, v["longitude"].values
    np.savez(dst, lat=lat, lon=lon, **out)
    for f in tmp.glob("*"):
        f.unlink()
    tmp.rmdir(); zp.unlink()
    return f"{y}-{m:02d} " + " ".join(f"{k} {np.nanmean(v):.3g}" for k, v in out.items())


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    with ThreadPoolExecutor(n) as ex:
        for r in ex.map(fetch, MONTHS):
            print(r, flush=True)
