#!/usr/bin/env python3
"""cams3h: shared definitions. Evaluation data only; nothing is fitted here.

RAW FILES  <replay>/cams3h/raw/<group>.nc: the CAMS operational forecast of surface PM2.5 (dataset
cams-global-atmospheric-composition-forecasts, type forecast, 00 UTC cycle), variable pm2p5 in kg m-3 with dimensions
(forecast_period 0, 3, ..., 120 h; forecast_reference_time; latitude 90 .. -90 by 0.4; longitude 0 .. 359.6 by 0.4).

VALUES     ug m-3 = float32(value) x 1e9 (the arithmetic of the chain, realtime/fetch_camsfc.py); "f16" = that value stored
           as float16 and read back as float32, exactly what the files of the existing arm "cams" hold (cams_fc.npz).

DAY        lead day L = 1..5 of issue day d is the UTC day d + L - 1 = forecast hours 24(L-1) .. 24L.

RULES for the daily mean of lead day L (weights on the forecast hours):
  three   the existing arm "cams" (replay/README.md, rc/coupling_live.py): 1/4 x(24(L-1)) + 1/2 x(24L-12) + 1/4 x(24L)
  paper   the rule of the paper for 2022 (eccv/world/data_pipeline/dl_cams_forecast.py, to_daily_leads: "lead day =
          mean of steps 24L .. 24L+21 h (8 steps)"; read by figures/compare_fair.py from cams_forecast_daily/2022.zarr):
          the EIGHT steps 00, 03, ..., 21 UTC of the day, equal weights; the step of 24 UTC is NOT in the day
  trap    sensitivity only: the nine steps 00 .. 24 UTC with half weights on the two ends (trapezoid)
  0324    sensitivity only: the eight steps 03, 06, ..., 24 UTC, equal weights
Every daily mean is floored at 0 and read bilinearly (linear space) on the 0.4 degree grid with the function of the
existing arm (rc/coupling_live.bilinear, grid origin 90 N, 180 W)."""
import datetime as dt
import sys
from pathlib import Path

import numpy as np

P = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
RT = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km/realtime")
R = P / "realtime_1km/replay"
OUT = R / "cams3h"
RAW = OUT / "raw"
CAT = P / "realtime_1km/verif/catalogue.npz"
ZARR_2026 = Path("/scratch/project_462001804/ammar/eccv/data/world/cams_forecast_daily/2026.zarr")   # the paper's builder, cycles to 2026-11-03
LEADS = list(range(0, 121, 3))
TARGET = [dt.date(2026, 9, 1) + dt.timedelta(k) for k in range(20)] + [dt.date(2026, 10, 10) + dt.timedelta(k) for k in range(33)]
NLEAD = 5


def bilinear_fn():
    """the bilinear reading of the existing arm (imported read-only; the module's main() is guarded)"""
    sys.path.insert(0, str(P / "realtime_1km/rc"))
    from coupling_live import bilinear
    return bilinear


def catalogue():
    cat = np.load(CAT)
    return cat["id"], cat["lat"].astype(np.float64), cat["lon"].astype(np.float64)


def weights(rule, L):
    """{forecast hour: weight} of lead day L under a rule"""
    h0 = 24 * (L - 1)
    if rule == "three":
        return {h0: 0.25, h0 + 12: 0.5, h0 + 24: 0.25}
    if rule == "paper":
        return {h0 + 3 * k: 0.125 for k in range(8)}
    if rule == "0324":
        return {h0 + 3 * k: 0.125 for k in range(1, 9)}
    if rule == "trap":
        w = {h0 + 3 * k: 0.125 for k in range(9)}
        w[h0], w[h0 + 24] = 0.0625, 0.0625
        return w
    raise ValueError(rule)


def index_raw():
    """{issue day (date): (file, index of the reference time)} over every raw netCDF file"""
    import netCDF4
    out = {}
    for f in sorted(RAW.glob("*.nc")):
        with netCDF4.Dataset(str(f)) as ds:
            ds.set_auto_mask(False)
            hours = np.asarray(ds.variables["forecast_period"][:], np.float64)
            assert ds.variables["forecast_period"].units == "hours" and hours.tolist() == [float(h) for h in LEADS], (f.name, hours)
            frt = np.asarray(ds.variables["forecast_reference_time"][:], np.int64)
            assert ds.variables["forecast_reference_time"].units == "seconds since 1970-01-01", ds.variables["forecast_reference_time"].units
            assert ds.variables["pm2p5"].dimensions == ("forecast_period", "forecast_reference_time", "latitude", "longitude")
            assert ds.variables["pm2p5"].units == "kg m**-3"
            for i, s in enumerate(frt):
                assert s % 86400 == 0, f"{f.name}: reference time {s} is not 00 UTC"
                d = dt.date(1970, 1, 1) + dt.timedelta(days=int(s // 86400))
                assert d not in out, f"{d} is in two raw files"
                out[d] = (f, i)
    return out


def read_file(f):
    """a raw file -> (dates, x32 (cycles, 41, 451, 900) float32 ug m-3, lat, lon); every value checked"""
    import netCDF4
    with netCDF4.Dataset(str(f)) as ds:
        ds.set_auto_mask(False)
        v = ds.variables["pm2p5"]
        assert v.dimensions == ("forecast_period", "forecast_reference_time", "latitude", "longitude") and v.units == "kg m**-3"
        hours = np.asarray(ds.variables["forecast_period"][:], np.float64)
        assert hours.tolist() == [float(h) for h in LEADS]
        frt = np.asarray(ds.variables["forecast_reference_time"][:], np.int64)
        vt = np.asarray(ds.variables["valid_time"][:], np.float64)                   # (cycles, steps), seconds since 1970
        assert ds.variables["valid_time"].dimensions == ("forecast_reference_time", "forecast_period")
        assert np.array_equal(vt, frt[:, None] + 3600.0 * hours[None, :]), "valid_time is not reference time + lead"
        lat = np.asarray(ds.variables["latitude"][:], np.float64)
        lon = np.asarray(ds.variables["longitude"][:], np.float64)
        assert lat.size == 451 and abs(lat[0] - 90.0) < 1e-9 and np.allclose(np.diff(lat), -0.4, atol=1e-9), lat[:3]
        assert lon.size == 900 and abs(lon[0]) < 1e-9 and np.allclose(np.diff(lon), 0.4, atol=1e-9), lon[:3]
        a = np.asarray(v[:], np.float32)                                             # (41, cycles, 451, 900) kg m-3
    assert np.isfinite(a).all() and float(a.min()) >= 0.0, (f.name, float(np.nanmin(a)))
    x32 = np.ascontiguousarray(a.transpose(1, 0, 2, 3)) * 1e9                        # the chain's arithmetic: float32 x 1e9
    assert x32.dtype == np.float32
    dates = [dt.date(1970, 1, 1) + dt.timedelta(days=int(s // 86400)) for s in frt]
    return dates, x32, lat, lon


def to_f16(x32):
    """what the chain stores (float16), read back as float32"""
    return x32.astype(np.float16).astype(np.float32)


def roll180(F):
    """longitudes 0 .. 359.6 -> -180 .. 179.6 (the roll of the existing arm: 450 columns)"""
    return np.roll(F, -450, axis=-1)


def daily_three(x, L):
    """the existing arm's line, in its float32 arithmetic: x (41, 451, 900) float32 -> (451, 900) float32"""
    a_, b_, c_ = 24 * (L - 1), 24 * L - 12, 24 * L
    return np.maximum(0.25 * x[a_ // 3] + 0.5 * x[b_ // 3] + 0.25 * x[c_ // 3], 0.0)


def daily_rule(x, L, rule):
    """weighted mean of the steps of lead day L under a rule, accumulated in float64, floored at 0: (451, 900) float64"""
    acc = np.zeros(x.shape[1:], np.float64)
    for h, w in weights(rule, L).items():
        acc += w * x[h // 3].astype(np.float64)
    return np.maximum(acc, 0.0)
