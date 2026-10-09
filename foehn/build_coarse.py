#!/usr/bin/env python3
"""Global 0.25 deg conditioning of the global FOEHN-km, in the format of ../dejavu (84 N .. 59.75 S, 576 x 1440,
longitudes -180 .. 179.75, 2017-01-01 .. 2022-12-31, one row per day):
  pm.npy    (D, 576, 1440) float32   GHAP V1 25 x 25 block means (ug/m3), NaN = sea / missing
  met.npy   (D, 37, 576, 1440) float16  ERA5 store channels (30) and extra fields (7: log BLH mean/max/min,
            log(1 + mm) rain, dewpoint, solar, cloud), standardised with global 2017-2020 statistics (stats.npz)
  meta.npz  year, doy per row; elevation (standardised as in Europe); land (GHAP present on > 50 % of days)
    python build_coarse.py stats           (global train-year means / stds)
    python build_coarse.py init
    python build_coarse.py <task> <ntasks> (fills every ntasks-th day)
    python build_coarse.py meta"""
import sys
import time
from pathlib import Path

import numpy as np
import zarr

OUT = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
Z = Path("/scratch/project_462001140/ammar/eccv/data")
ERA5 = str(Z / "zarr/era5_global_daily/{}.zarr")
EXTRA = str(Z / "world/era5_extra_daily/{}.zarr")
GHAP = str(Z / "world/ghap_025deg/v1/{}.zarr")
STATIC = str(Z / "world/static/static_025deg.zarr")
YEARS = list(range(2017, 2023))
NDAYS = {y: 366 if y % 4 == 0 else 365 for y in YEARS}
DAYS = [(y, d) for y in YEARS for d in range(NDAYS[y])]
R0, R1, W = 24, 600, 1440                        # 84 N .. 59.75 S on the ERA5 0.25 deg grid
H = R1 - R0
ELEV_MEAN, ELEV_STD = 300.0, 500.0               # replaced by the Europe constants when importable
try:
    sys.path.insert(0, "/scratch/project_462001140/ammar/eccv/cran-pm/src")
    from cranpm.data.europe_dataset_v2 import ELEV_MEAN, ELEV_STD  # noqa: F401,E402
except Exception:
    pass


def crop(a):
    """(..., 721, 1440) lat 90 -> -90, lon 0 -> 359.75  ->  (..., 576, 1440) 84 N .. 59.75 S, lon -180 .. 179.75."""
    a = a[..., R0:R1, :]
    return np.concatenate([a[..., 720:], a[..., :720]], axis=-1)


def extra_raw(x):
    return np.stack([np.log(np.maximum(x[0], 10.0)), np.log(np.maximum(x[1], 10.0)), np.log(np.maximum(x[2], 10.0)),
                     np.log1p(np.maximum(x[3], 0.0) * 1000.0), x[4], np.maximum(x[5], 0.0) / 1e6, x[6]])


def create(path, shape, descr):
    with open(path, "wb") as f:
        np.lib.format.write_array_header_1_0(f, {"descr": descr, "fortran_order": False, "shape": shape})
        hdr = f.tell()
        f.truncate(hdr + int(np.prod(shape)) * np.dtype(descr).itemsize)


def day_view(path, k, shape, dtype):
    with open(path, "rb") as f:
        np.lib.format.read_magic(f); np.lib.format.read_array_header_1_0(f); hdr = f.tell()
    n = int(np.prod(shape)) * np.dtype(dtype).itemsize
    return np.memmap(path, dtype=dtype, mode="r+", offset=hdr + k * n, shape=shape)


cmd = sys.argv[1]
if cmd == "stats":
    s1 = s2 = e1 = e2 = 0.0; n = 0
    for y in (2017, 2018, 2019, 2020):
        ez, xz = zarr.open(ERA5.format(y), mode="r"), zarr.open(EXTRA.format(y), mode="r")
        for d in range(0, NDAYS[y], 7):
            e = crop(np.asarray(ez[d], np.float64)); x = crop(extra_raw(np.asarray(xz[d], np.float64)))
            s1, s2 = s1 + e.mean((1, 2)), s2 + (e ** 2).mean((1, 2))
            e1, e2 = e1 + np.nanmean(x, (1, 2)), e2 + np.nanmean(x ** 2, (1, 2))
            n += 1
        print("stats", y, n, flush=True)
    m, em = s1 / n, e1 / n
    np.savez(OUT / "stats.npz", era5_mean=m.astype(np.float32), era5_std=np.sqrt(s2 / n - m ** 2).astype(np.float32),
             extra_mean=em.astype(np.float32), extra_std=np.sqrt(e2 / n - em ** 2).astype(np.float32))
    print("wrote stats from", n, "days")
elif cmd == "init":
    create(OUT / "pm.npy", (len(DAYS), H, W), "<f4")
    create(OUT / "met.npy", (len(DAYS), 37, H, W), "<f2")
    print("created pm.npy, met.npy for", len(DAYS), "days")
elif cmd == "meta":
    pm = np.load(OUT / "pm.npy", mmap_mode="r")
    cnt = np.zeros((H, W)); n = 0
    for k in range(0, len(DAYS), 5):
        cnt += np.isfinite(np.asarray(pm[k])); n += 1
    elev = crop(np.asarray(zarr.open(STATIC, mode="r")["elev_mean"], np.float32))
    np.savez(OUT / "meta.npz", year=np.array([y for y, _ in DAYS]), doy=np.array([d for _, d in DAYS]),
             elev=((np.nan_to_num(elev) - ELEV_MEAN) / ELEV_STD).astype(np.float32), land=cnt / n > 0.5)
    print("meta: land cells", int((cnt / n > 0.5).sum()))
else:
    task, ntasks = int(cmd), int(sys.argv[2])
    st = np.load(OUT / "stats.npz")
    em, es = st["era5_mean"][:, None, None], st["era5_std"][:, None, None]
    xm, xs = st["extra_mean"][:, None, None], st["extra_std"][:, None, None]
    opened, t0 = {}, time.time()
    for k in range(task, len(DAYS), ntasks):
        y, d = DAYS[k]
        if y not in opened:
            opened[y] = (zarr.open(ERA5.format(y), mode="r"), zarr.open(EXTRA.format(y), mode="r"), zarr.open(GHAP.format(y), mode="r"))
        ez, xz, gz = opened[y]
        g = crop(np.asarray(gz[d, 0], np.float32))
        day_view(OUT / "pm.npy", k, (H, W), np.float32)[:] = np.where(np.isfinite(g) & (g > 0), g, np.nan)
        e = (crop(np.asarray(ez[d], np.float32)) - em) / es
        x = np.nan_to_num((crop(extra_raw(np.asarray(xz[d], np.float32))) - xm) / xs, nan=0.0)
        day_view(OUT / "met.npy", k, (37, H, W), np.float16)[:] = np.concatenate([e, x]).astype(np.float16)
        if (k // ntasks) % 20 == 0:
            print(f"task {task}: day {k} {time.time() - t0:.0f}s", flush=True)
    print("done", task)
