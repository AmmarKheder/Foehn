#!/usr/bin/env python3
"""RT 1.2 data pipeline: the live inputs for a range of past days, built in the day directories (data/<date>/) that
fetch.py/build.py/flexgreen use, with the CAMS analyses and the GFAS fires requested month by month (one ADS request
per month instead of two per day) and the IFS open data per day from the AWS mirror.
    python bulk.py cams  2026-06-01 2026-07-05      monthly ADS requests -> data/<d>/cams_00.zip, cams_12.zip
    python bulk.py gfas  2026-06-01 2026-07-05      monthly ADS requests -> data/<d>/gfas.nc (GFAS of d-1)
    python bulk.py ifs   2026-06-01 2026-07-05 [--parallel 3]     fetch.py <d> --only ifs, several days at a time
    python bulk.py build 2026-06-01 2026-07-05 [--parallel 2]     build.py <d> for the days whose inputs are complete
    python bulk.py status 2026-06-01 2026-07-05"""
import argparse
import datetime as dt
import subprocess
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import common as C


def days(first, last):
    d = C.parse_date(first); e = C.parse_date(last)
    while d <= e:
        yield d
        d += dt.timedelta(days=1)


def months(first, last):
    """(first day, last day) of each month touched by [first, last]"""
    d = C.parse_date(first).replace(day=1); e = C.parse_date(last)
    while d <= e:
        n = (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        yield max(d, C.parse_date(first)), min(n - dt.timedelta(days=1), e)
        d = n


def cams(first, last):
    """00 and 12 UTC analyses of every day of the range (plus 12 UTC of the day before, for the first day)"""
    import xarray as xr
    from fetch import ads
    from dl_cams_analysis import PL, SL
    f0 = C.parse_date(first) - dt.timedelta(days=1)
    for a, b in months(str(f0), last):
        need = [d for d in days(str(a), str(b)) if not ((C.day_dir(d) / "cams_00.zip").exists() and (C.day_dir(d + dt.timedelta(days=1)) / "cams_12.zip").exists())]
        if not need:
            print(a, b, "already there", flush=True); continue
        tmp = C.RT / "verif" / f"cams_bulk_{a}_{b}.zip"; tmp.parent.mkdir(exist_ok=True)
        if not tmp.exists():
            print(f"ADS request CAMS {a}..{b}", flush=True)
            ads().retrieve("cams-global-atmospheric-composition-forecasts",
                           {"type": ["forecast"], "leadtime_hour": ["0"], "date": [f"{a}/{b}"], "time": ["00:00", "12:00"],
                            "variable": list(SL) + list(PL), "pressure_level": ["1000"], "data_format": "netcdf", "download_format": "zip"}, str(tmp))
        with tempfile.TemporaryDirectory() as td, zipfile.ZipFile(tmp) as z:
            z.extractall(td)
            dss = [xr.open_dataset(f) for f in sorted(C.Path(td).glob("*.nc"))]
            for d in days(str(a), str(b)):
                for hour, dest_day, name in ((0, d, "cams_00.zip"), (12, d + dt.timedelta(days=1), "cams_12.zip")):
                    dest = C.day_dir(dest_day) / name
                    if dest.exists():
                        continue
                    t = np.datetime64(dt.datetime(d.year, d.month, d.day, hour))
                    with tempfile.TemporaryDirectory() as t2:
                        files = []
                        for i, ds in enumerate(dss):
                            tc = "forecast_reference_time" if "forecast_reference_time" in ds.dims else [x for x in ds.dims if "time" in x][0]
                            sub = ds.sel({tc: t}, drop=True)
                            fn = C.Path(t2) / f"part{i}.nc"; sub.to_netcdf(fn); files.append(fn)
                        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zz:
                            for fn in files:
                                zz.write(fn, fn.name)
            for ds in dss:
                ds.close()
        print(f"CAMS {a}..{b}: day files written", flush=True)


def gfas(first, last):
    import xarray as xr
    from fetch import ads
    f0 = C.parse_date(first) - dt.timedelta(days=1); l0 = C.parse_date(last) - dt.timedelta(days=1)
    for a, b in months(str(f0), str(l0)):
        need = [d for d in days(str(a), str(b)) if not (C.day_dir(d + dt.timedelta(days=1)) / "gfas.nc").exists()]
        if not need:
            print(a, b, "GFAS already there", flush=True); continue
        tmp = C.RT / "verif" / f"gfas_bulk_{a}_{b}.nc"
        if not tmp.exists():
            print(f"ADS request GFAS {a}..{b}", flush=True)
            ads().retrieve("cams-global-fire-emissions-gfas", {"variable": ["wildfire_flux_of_total_particulate_matter", "wildfire_radiative_power", "injection_height"],
                                                              "date": [f"{a}/{b}"], "data_format": "netcdf"}, str(tmp))
        with xr.open_dataset(tmp) as ds:
            tc = "valid_time" if "valid_time" in ds.coords else [x for x in ds.dims if "time" in x][0]
            for d in days(str(a), str(b)):
                dest = C.day_dir(d + dt.timedelta(days=1)) / "gfas.nc"
                if dest.exists():
                    continue
                nd = d + dt.timedelta(days=1)                                          # ADS stamps the GFAS daily file of day d at d+1 00 UTC
                sub = ds.sel({tc: np.datetime64(dt.datetime(nd.year, nd.month, nd.day))}, drop=True)
                sub.to_netcdf(dest)
                C.meta_update(d + dt.timedelta(days=1), gfas_day=str(d), gfas_version="v1.2" if d.year <= 2025 else "v1.4.2", gfas_stamp=str(a))
        print(f"GFAS {a}..{b}: day files written", flush=True)


def ifs(first, last, parallel):
    todo = [d for d in days(first, last) if not (C.day_dir(d) / "hres_B.npy").exists() and not all((C.day_dir(d) / "ifs" / f"{s:03d}.grib2").exists() for s in range(0, 193, 6))]
    print(f"IFS: {len(todo)} days to fetch, {parallel} at a time", flush=True)

    def one(d):
        r = subprocess.run([sys.executable, "fetch.py", str(d), "--only", "ifs", "--every", "5"], cwd=C.RT, capture_output=True, text=True)
        return d, r.returncode
    with ThreadPoolExecutor(parallel) as ex:
        for d, rc in ex.map(one, todo):
            print(d, "ifs rc", rc, flush=True)


def build(first, last, parallel):
    todo = []
    for d in days(first, last):
        D = C.day_dir(d)
        if (D / "cams.npy").exists() and (D / "hres_B.npy").exists() and (D / "fires.npy").exists():
            continue
        if (D / "cams_00.zip").exists() and (D / "cams_12.zip").exists() and (D / "gfas.nc").exists() and all((D / "ifs" / f"{s:03d}.grib2").exists() for s in range(0, 193, 6)):
            todo.append(d)
    print(f"build: {len(todo)} days", flush=True)

    def one(d):
        r = subprocess.run([sys.executable, "build.py", str(d)], cwd=C.RT, capture_output=True, text=True)
        return d, r.returncode, r.stdout[-200:] if r.returncode else ""
    with ThreadPoolExecutor(parallel) as ex:
        for d, rc, msg in ex.map(one, todo):
            print(d, "build rc", rc, msg, flush=True)


def status(first, last):
    for d in days(first, last):
        D = C.day_dir(d)
        f = lambda n: "x" if (D / n).exists() else "."
        nifs = sum((D / "ifs" / f"{s:03d}.grib2").exists() for s in range(0, 193, 6))
        print(d, f"cams00{f('cams_00.zip')} cams12{f('cams_12.zip')} gfas{f('gfas.nc')} ifs{nifs:02d}/33 | cams.npy{f('cams.npy')} hres_B{f('hres_B.npy')} fires{f('fires.npy')} green{'x' if (D / 'green').exists() else '.'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("first"); ap.add_argument("last"); ap.add_argument("--parallel", type=int, default=3)
    a = ap.parse_args()
    {"cams": lambda: cams(a.first, a.last), "gfas": lambda: gfas(a.first, a.last), "ifs": lambda: ifs(a.first, a.last, a.parallel),
     "build": lambda: build(a.first, a.last, a.parallel), "status": lambda: status(a.first, a.last)}[a.cmd]()
