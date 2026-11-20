#!/usr/bin/env python3
"""ERA5 daily stores of 2026 (and any later year) from ARCO-ERA5, in the format of the archives the met builder reads
(build_global_coarse.met_day): backfill of the 1 km analysis inputs (user order 2026-11-02).

  daily <year>  eccv/data/zarr/era5_global_daily/<year>.zarr  (days, 30, 721, 1440) float32: the 30 channels of
                aria/scripts/download_era5_2025.py (u10, v10, t2m, msl, sp, then t/u/v/q/z at 1000/925/850/700/500 hPa),
                daily mean of 00/06/12/18 UTC, (x - ERA5_MEANS) / ERA5_STDS, lat 90 -> -90, lon 0 -> 359.75
  extra <year>  eccv/data/world/era5_extra_daily/<year>.zarr (days, 7, 721, 1440): exactly dl_era5_extra_arco.py
                (its day_stats), written day by day so that the last month can be partial
  last          the last day with finite ERA5 fields in ARCO (bisection; ERA5 lags real time by ~6 days) ->
                data/processed/global/backfill/arco_last_day.txt
  validate      3 days of 2025: 'daily' recomputed from ARCO vs the CDS-made 2025.zarr, 'extra' vs the 2025 extra store
Both stores carry attrs done_days (list of day indices written) and last_day; done_months as the pipeline's.
  python -I dl_era5_daily_arco.py daily 2026 [--months 1 2 ..] [--last-day 2026-10-27] [--threads 16]
World venv (xarray, zarr, numcodecs), partition small."""
import argparse
import calendar
import datetime as dt
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
BF = ROOT / "data/processed/global/backfill"
PIPE = Path("/scratch/project_462001804/ammar/eccv/world/data_pipeline")
sys.path.insert(0, str(PIPE))
from common import WORLD_DATA, done_months, mark_month_done, open_daily_store  # noqa: E402
from dl_era5_extra import CHANNELS as X_CHANNELS, NLAT, NLON  # noqa: E402
from dl_era5_extra_arco import ARCO, day_stats  # noqa: E402

DAILY_DIR = Path("/scratch/project_462001804/ammar/eccv/data/zarr/era5_global_daily")
EXTRA_DIR = WORLD_DATA / "era5_extra_daily"
# verbatim from aria/scripts/download_era5_2025.py (the maker of 2025.zarr; same normalisation as 2017-2024)
SURFACE_VARS = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "mean_sea_level_pressure",
                "surface_pressure"]
PRESSURE_VARS = ["temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity", "geopotential"]
PRESSURE_LEVELS = [1000, 925, 850, 700, 500]
ERA5_MEANS = np.array([0., 0., 280., 101325., 97000., *([260.] * 5 + [0.] * 5 + [0.] * 5 + [0.003] * 5 + [50000.] * 5)], np.float32)
ERA5_STDS = np.array([10., 10., 20., 1500., 5000., *([30.] * 5 + [15.] * 5 + [10.] * 5 + [0.004] * 5 + [40000.] * 5)], np.float32)
HOURS = [0, 6, 12, 18]


def retry(fn, *args, attempts=8, base_wait=15):
    for i in range(attempts):
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001  (transient HTTP errors)
            if i == attempts - 1:
                raise
            print(f"  retry {i + 1}: {e!r}", flush=True)
            time.sleep(base_wait * (i + 1))


class Arco:
    def __init__(self):
        self.ds = xr.open_zarr(ARCO, consolidated=True, chunks=None)
        ds = self.ds
        assert float(ds.latitude[0]) == 90.0 and float(ds.longitude[0]) == 0.0 and ds.sizes["longitude"] == NLON
        lev = [int(x) for x in ds.level.values]
        self.lev_idx = [lev.index(p) for p in PRESSURE_LEVELS]
        self.times = ds.time.values
        self.t0 = self.times[0]
        assert np.all(np.diff(self.times[:48]) == np.timedelta64(1, "h"))

    def tindex(self, t64):
        i = int((t64 - self.t0) / np.timedelta64(1, "h"))
        assert self.times[i] == t64, (t64, self.times[i])
        return i

    def read(self, var, ti):
        def get():
            a = np.asarray(self.ds[var].isel(time=ti).values, np.float32)
            return a[self.lev_idx] if a.ndim == 3 else a
        return retry(get)

    def finite_day(self, day):
        """all 30 daily channels finite on a coarse sample of the day's 12 UTC field"""
        ti = self.tindex(np.datetime64(f"{day}T12"))
        for v in SURFACE_VARS + PRESSURE_VARS:
            a = self.ds[v].isel(time=ti)
            a = a.isel(latitude=slice(0, NLAT, 48), longitude=slice(0, NLON, 96)).values
            if not np.isfinite(a).all():
                return False
        return True


def daily_day(arco, ex, day):
    """(30, 721, 1440) float32 of one day, as download_era5_2025.process_quarter: mean of 00/06/12/18, standardised"""
    tis = [arco.tindex(np.datetime64(f"{day}T{h:02d}")) for h in HOURS]
    tasks = [(v, ti) for ti in tis for v in SURFACE_VARS + PRESSURE_VARS]
    res = dict(zip(tasks, ex.map(lambda q: arco.read(q[0], q[1]), tasks)))
    ch = []
    for v in SURFACE_VARS:
        ch.append(np.mean([res[(v, ti)] for ti in tis], axis=0, dtype=np.float32))
    for v in PRESSURE_VARS:
        m = np.mean([res[(v, ti)] for ti in tis], axis=0, dtype=np.float32)      # (5, 721, 1440) at PRESSURE_LEVELS
        ch.extend(m[j] for j in range(len(PRESSURE_LEVELS)))
    x = np.stack(ch).astype(np.float32)
    return (x - ERA5_MEANS[:, None, None]) / ERA5_STDS[:, None, None]


def extra_day(arco, ex, day):
    t0 = arco.tindex(np.datetime64(f"{day}T00"))
    return retry(day_stats, arco.ds, t0)


def days_of(year, month, last_day):
    n = calendar.monthrange(year, month)[1]
    out = [dt.date(year, month, d) for d in range(1, n + 1)]
    return [d for d in out if last_day is None or d <= last_day]


def cmd_last(a):
    arco = Arco()
    lo, hi = dt.date(2025, 1, 1), dt.date.today() + dt.timedelta(1)
    assert arco.finite_day(lo)
    while (hi - lo).days > 1:
        mid = lo + dt.timedelta((hi - lo).days // 2)
        if arco.finite_day(mid):
            lo = mid
        else:
            hi = mid
    BF.mkdir(parents=True, exist_ok=True)
    (BF / "arco_last_day.txt").write_text(f"{lo}\n")
    print(f"ARCO-ERA5 last finite day: {lo} (checked {dt.date.today()})", flush=True)


def last_day_arg(a):
    if a.last_day:
        return dt.date.fromisoformat(a.last_day)
    p = BF / "arco_last_day.txt"
    return dt.date.fromisoformat(p.read_text().split()[0]) if p.exists() else None


def cmd_store(a):
    kind = a.cmd
    last = last_day_arg(a)
    if kind == "daily":
        path, shape, fn = DAILY_DIR / f"{a.year}.zarr", (30, NLAT, NLON), daily_day
        attrs = {"source": "ERA5 hourly via ARCO-ERA5 (gcp-public-data-arco-era5), 00/06/12/18 UTC daily mean",
                 "channels": ["u10", "v10", "t2m", "msl", "sp"] + [f"{v}{p}" for v in ("t", "u", "v", "q", "z") for p in PRESSURE_LEVELS],
                 "normalisation": "(x - ERA5_MEANS) / ERA5_STDS of aria/scripts/download_era5_2025.py",
                 "grid": "0.25 deg, lat 90 -> -90 (721), lon 0 -> 359.75 (1440)", "maker": "dl_era5_daily_arco.py"}
    else:
        path, shape, fn = EXTRA_DIR / f"{a.year}.zarr", (len(X_CHANNELS), NLAT, NLON), extra_day
        attrs = {"source": "ERA5 hourly via ARCO-ERA5 (gcp-public-data-arco-era5), daily stats over 00..23 UTC",
                 "channels": [c[0] for c in X_CHANNELS], "units": [c[5] for c in X_CHANNELS],
                 "grid": "0.25 deg, lat 90 -> -90 (721), lon 0 -> 359.75 (1440), same as era5_global_daily",
                 "maker": "dl_era5_daily_arco.py (dl_era5_extra_arco.day_stats)"}
    z = open_daily_store(path, a.year, shape, attrs)
    done = set(int(k) for k in z.attrs.get("done_days", []))
    arco = Arco()
    months = a.months or list(range(1, 13))
    t0 = time.time()
    with ThreadPoolExecutor(a.threads) as ex:
        for m in months:
            if m in done_months(z):
                print(f"{kind} {a.year}-{m:02d}: month already done", flush=True)
                continue
            days = days_of(a.year, m, last)
            if not days:
                print(f"{kind} {a.year}-{m:02d}: no day on or before {last}", flush=True)
                continue
            for d in days:
                k = (d - dt.date(a.year, 1, 1)).days
                if k in done:
                    continue
                x = fn(arco, ex, str(d))
                assert x.shape == shape and np.isfinite(x).all(), (d, x.shape)
                z[k] = x
                done.add(k)
                z.attrs["done_days"] = sorted(done)
                z.attrs["last_day"] = str(dt.date(a.year, 1, 1) + dt.timedelta(max(done)))
                print(f"{kind} {d}: written ({time.time() - t0:.0f}s)", flush=True)
            if len(days) == calendar.monthrange(a.year, m)[1]:
                mark_month_done(z, m)
                print(f"{kind} {a.year}-{m:02d}: month complete", flush=True)
    print(f"DONE {kind} {a.year}: last_day {z.attrs.get('last_day')}, {len(done)} days", flush=True)


def cmd_validate(a):
    """ARCO vs the archives on disk, 3 days of 2025: daily (CDS-made, packed int16 netCDF) and extra (ARCO-made, expected
    bit-identical). Stats in the met crop (rows 24..600 = 84 N .. 59.75 S, build_global_coarse.crop): the CDS archive
    has pole-row artefacts in u/v (2026-11-03 diagnosis) that the crop never reads."""
    arco = Arco()
    res = {}
    zd = zarr.open(str(DAILY_DIR / "2025.zarr"), mode="r")
    zx = zarr.open(str(EXTRA_DIR / "2025.zarr"), mode="r")
    R0, R1 = 24, 600
    with ThreadPoolExecutor(a.threads) as ex:
        for d in (dt.date(2025, 1, 15), dt.date(2025, 7, 1), dt.date(2025, 12, 20)):
            k = (d - dt.date(2025, 1, 1)).days
            x, r = daily_day(arco, ex, str(d)), np.asarray(zd[k], np.float32)
            ad = np.abs(x - r)
            adc = ad[:, R0:R1]
            big = adc > 0.05
            rows = np.where(big.any((0, 2)))[0] + R0
            res[f"daily_{d}"] = {"crop_max_abs_diff_per_channel_std_units": adc.max((1, 2)).round(4).tolist(),
                                 "crop_frac_within_1e-3": float((adc < 1e-3).mean()), "crop_frac_within_1e-2": float((adc < 1e-2).mean()),
                                 "crop_n_points_diff_gt_0.05": int(big.sum()), "crop_rows_with_diff_gt_0.05": [int(v) for v in rows[:20]],
                                 "crop_corr_min": float(min(np.corrcoef(x[c, R0:R1].ravel(), r[c, R0:R1].ravel())[0, 1] for c in range(30))),
                                 "full_max_abs_diff_pole_rows_0_720": float(np.abs(ad[:, [0, 720]]).max())}
            y, s = extra_day(arco, ex, str(d)), np.asarray(zx[k], np.float32)
            res[f"extra_{d}"] = {"bit_identical": bool(np.array_equal(y, s)), "max_abs_diff": float(np.abs(y - s).max())}
            print(d, json.dumps(res[f"daily_{d}"]), json.dumps(res[f"extra_{d}"]), flush=True)
    res["ok"] = bool(all(v["crop_corr_min"] > 0.995 and v["crop_frac_within_1e-2"] > 0.99 for kk, v in res.items() if kk.startswith("daily")) and
                     all(v["max_abs_diff"] < 1e-3 for kk, v in res.items() if kk.startswith("extra")))
    res["note"] = "daily: the CDS 2025 archive is a packed-int16 netCDF (differences ~1e-3 std units everywhere) with artefacts at the pole rows; extra: same ARCO code as the 2025 store"
    BF.mkdir(parents=True, exist_ok=True)
    (BF / "era5_arco_validation.json").write_text(json.dumps(res, indent=1))
    print("ok:", res["ok"], flush=True)
    if not res["ok"]:
        sys.exit("ERA5 ARCO validation FAILED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["daily", "extra", "last", "validate"])
    ap.add_argument("year", type=int, nargs="?", default=2026)
    ap.add_argument("--months", type=int, nargs="*", default=[])
    ap.add_argument("--last-day", default=None)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    {"last": cmd_last, "validate": cmd_validate}.get(a.cmd, cmd_store)(a)
