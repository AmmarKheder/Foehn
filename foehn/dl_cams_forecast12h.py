#!/usr/bin/env python3
"""CAMS global operational forecasts of surface PM2.5, 00 UTC runs, instantaneous fields every 12 h to 120 h (the steps of
Aurora's 12-hour model), May - Nov 2022, for the 'Foehn trained as Aurora' comparison, verified against the CAMS analysis
at the valid time. One request per month (Atmosphere Data Store, ~/.adsapirc) ->
data/world/cams_forecast_12h/2022_<mm>.npz: pm25 (days, 10, 451, 900) float32 in ug m-3 for lead 12, 24, ..., 120 h, lat, lon.
    python dl_cams_forecast12h.py [workers 2]"""
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cdsapi
import numpy as np
import xarray as xr

OUT = Path("/scratch/project_462001140/ammar/eccv/data/world/cams_forecast_12h")
OUT.mkdir(parents=True, exist_ok=True)
ADS = dict(l.split(":", 1) for l in (Path.home() / ".adsapirc").read_text().splitlines() if ":" in l)
LEADS = list(range(12, 121, 12))
NC_LOCK = threading.Lock()
MONTHS = [(2022, m) for m in range(5, 12)]


def fetch(ym):
    y, m = ym
    dst = OUT / f"{y}_{m:02d}.npz"
    if dst.exists():
        return f"{y}-{m:02d} exists"
    last = int(str(np.datetime64(f"{y}-{m:02d}") + np.timedelta64(1, "M") - np.timedelta64(1, "D"))[8:10])
    zp = OUT / f"{y}_{m:02d}.zip"
    for attempt in range(4):
        try:
            cdsapi.Client(url=ADS["url"].strip(), key=ADS["key"].strip(), quiet=True, progress=False).retrieve(
                "cams-global-atmospheric-composition-forecasts",
                {"type": "forecast", "leadtime_hour": [str(h) for h in LEADS], "date": f"{y}-{m:02d}-01/{y}-{m:02d}-{last:02d}",
                 "time": ["00:00"], "variable": ["particulate_matter_2.5um"], "format": "netcdf_zip"}, str(zp))
            break
        except Exception as e:                                  # ADS queue hiccups
            print(f"{y}-{m:02d} retry {attempt}: {repr(e)[:200]}", flush=True)
    tmp = OUT / f"tmp_{y}_{m:02d}"; tmp.mkdir(exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        z.extractall(tmp)
    with NC_LOCK:                                               # netCDF4 / HDF5 is not thread-safe (segfault otherwise)
        ds = xr.open_dataset(next(tmp.glob("*.nc")))
        v = ds["pm2p5"].load()
        ds.close()
    dims = [d for d in v.dims if d not in ("latitude", "longitude")]
    print(f"{y}-{m:02d} dims {dict(v.sizes)}", flush=True)
    ref = [d for d in dims if "reference" in d or d == "time"]
    per = [d for d in dims if "period" in d or "step" in d or "lead" in d]
    order = (ref + per) if (ref and per) else dims                  # (start day, lead) explicitly, whatever the file order
    a = v.transpose(*order, "latitude", "longitude").values.astype(np.float32) * 1e9                  # kg m-3 -> ug m-3
    a = a.reshape(last, len(LEADS), a.shape[-2], a.shape[-1])
    np.savez(dst, pm25=a, leads=np.array(LEADS), lat=v["latitude"].values, lon=v["longitude"].values)
    for f in tmp.glob("*"):
        f.unlink()
    tmp.rmdir(); zp.unlink()
    return f"{y}-{m:02d} {a.shape} mean {np.nanmean(a):.1f}"


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    with ThreadPoolExecutor(n) as ex:
        for r in ex.map(fetch, MONTHS):
            print(r, flush=True)
