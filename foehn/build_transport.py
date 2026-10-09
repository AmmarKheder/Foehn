#!/usr/bin/env python3
"""Inputs of the transport branch of the global FOEHN-km (footprint + emissions), on the grid of build_coarse.py
(84 N .. 59.75 S, lon -180 .. 179.75 at 0.25 deg; 0.5 deg for the winds), 2017-01-01 .. 2022-12-31:
  emis_ant.npy  (72 months, 6, 576, 1440) float16  CAMS-GLOB-ANT v6.2 total anthropogenic flux (BC, OC, SO2, NOx, NH3, CO),
                area-averaged to 0.25 deg, log10(flux + 1e-14) standardised per species (stats in emis_stats.npz)
  fires.npy     (D, 3, 576, 1440) float16  CAMS GFAS daily fire emission of particulate matter, radiative power and
                injection height, area-averaged, log-transformed and standardised (used on the START day only)
  wind05.npy    (D, 8, 4, 288, 720) float16  ERA5 u10, v10, u925, v925 at 00..21 UTC every 3 h, 2 x 2 means (m/s)
    python build_transport.py ant | fires_init | fires <task> <n> | wind_init | wind <task> <n>"""
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

OUT = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
EM = Path("/scratch/project_462001140/ammar/eccv/data/world/emissions")
GFAS = "/scratch/project_462001140/ammar/eccv/data/world/gfas_daily/{}.zarr"
WIND = "/scratch/project_462001140/ammar/eccv/data/world/era5_wind3h/{}.zarr"
YEARS = list(range(2017, 2023))
NDAYS = {y: 366 if y % 4 == 0 else 365 for y in YEARS}
DAYS = [(y, d) for y in YEARS for d in range(NDAYS[y])]
EMIS_FILE = "emis_ant_fixed.npy"    # emis_ant.npy (runs of 2026-09-29) had the 180 deg shift
SPECIES = ["black-carbon", "organic-carbon", "sulphur-dioxide", "nitrogen-oxides", "ammonia", "carbon-monoxide"]
H, W = 576, 1440


def to025(a, lon0):
    """(..., 1800, 3600) 0.1 deg field, rows 89.95 -> -89.95, first column centred at lon0 + 0.05 ->
    (..., 576, 1440) area means over the 0.25 deg cells centred on 84 - 0.25 k, -180 + 0.25 j."""
    a = np.asarray(a, np.float32)
    if abs(lon0 + 180.0) > 1e-3:                        # GFAS: 0 .. 360 -> -180 .. 180 (the CAMS-GLOB-ANT first
        # longitude is -179.9499969 in float32: an exact comparison with -180 rolled it by 180 deg, fixed 2026-09-30)
        a = np.concatenate([a[..., 1800:], a[..., :1800]], axis=-1)
    f = np.repeat(np.repeat(a, 4, -2), 4, -1)          # 0.025 deg, rows from 90 N, cols from -180
    f = np.concatenate([f[..., -5:], f, f[..., :5]], axis=-1)          # lon wrap for the 5-cell offset
    r0 = int(round((90 - 84) / 0.025)) - 5                              # cell k covers rows 10 k - 5 .. 10 k + 5 from 84 N
    f = f[..., r0:r0 + 10 * H, :10 * W]
    return f.reshape(*f.shape[:-2], H, 10, W, 10).mean((-3, -1))


def create(path, shape):
    with open(path, "wb") as f:
        np.lib.format.write_array_header_1_0(f, {"descr": "<f2", "fortran_order": False, "shape": shape})
        hdr = f.tell(); f.truncate(hdr + int(np.prod(shape)) * 2)


def view(path, k, shape):
    with open(path, "rb") as f:
        np.lib.format.read_magic(f); np.lib.format.read_array_header_1_0(f); hdr = f.tell()
    return np.memmap(path, dtype=np.float16, mode="r+", offset=hdr + k * int(np.prod(shape)) * 2, shape=shape)


cmd = sys.argv[1]
if cmd == "ant":
    out = np.zeros((72, 6, H, W), np.float32)
    for s, sp in enumerate(SPECIES):
        for yi, y in enumerate(YEARS):
            ds = xr.open_dataset(EM / f"CAMS-GLOB-ANT_v6.2_{sp}_{y}.nc", decode_times=False)
            x = ds["sum"].values                                   # (12, 1800, 3600), lat ascending?
            if float(ds.lat[0]) < float(ds.lat[-1]):
                x = x[:, ::-1]
            out[12 * yi:12 * yi + 12, s] = np.log10(to025(x, float(ds.lon[0]) - 0.05) + 1e-14)
        print("species", sp, flush=True)
    m, sd = out.mean((0, 2, 3)), out.std((0, 2, 3))
    np.savez(OUT / "emis_stats.npz", mean=m, std=sd)
    np.save(OUT / EMIS_FILE, ((out - m[None, :, None, None]) / sd[None, :, None, None]).astype(np.float16))
    print("wrote emis_ant.npy", dict(zip(SPECIES, np.round(m, 2))))
elif cmd == "fires_init":
    create(OUT / "fires.npy", (len(DAYS), 3, H, W)); print("created fires.npy")
elif cmd == "fires":
    task, n = int(sys.argv[2]), int(sys.argv[3])
    op, t0 = {}, time.time()
    for k in range(task, len(DAYS), n):
        y, d = DAYS[k]
        z = op.setdefault(y, zarr.open(GFAS.format(y), mode="r"))
        g = to025(np.asarray(z[d, [0, 3, 4]], np.float32), 0.0)          # tpm, frp, injection height
        x = np.stack([np.log10(np.maximum(g[0], 0) + 1e-14), np.log10(np.maximum(g[1], 0) + 1e-6), np.log10(np.maximum(g[2], 0) + 1.0)])
        x = (x - np.array([-13.5, -5.5, 0.1])[:, None, None]) / np.array([1.0, 1.0, 0.6])[:, None, None]
        view(OUT / "fires.npy", k, (3, H, W))[:] = x.astype(np.float16)
        if (k // n) % 50 == 0:
            print(f"fires task {task}: {k} {time.time() - t0:.0f}s", flush=True)
elif cmd == "wind_init":
    create(OUT / "wind05.npy", (len(DAYS), 8, 4, H // 2, W // 2)); print("created wind05.npy")
elif cmd == "wind":
    task, n = int(sys.argv[2]), int(sys.argv[3])
    op, t0 = {}, time.time()
    for k in range(task, len(DAYS), n):
        y, d = DAYS[k]
        z = op.setdefault(y, zarr.open(WIND.format(y), mode="r"))
        a = np.asarray(z[d][:, :4, 24:600], np.float32)                # (8, 4, 576, 1440) lon 0 -> 360
        a = np.concatenate([a[..., 720:], a[..., :720]], axis=-1)
        view(OUT / "wind05.npy", k, (8, 4, H // 2, W // 2))[:] = a.reshape(8, 4, H // 2, 2, W // 2, 2).mean((3, 5)).astype(np.float16)
        if (k // n) % 50 == 0:
            print(f"wind task {task}: {k} {time.time() - t0:.0f}s", flush=True)
