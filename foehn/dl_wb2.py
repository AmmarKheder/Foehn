#!/usr/bin/env python3
"""WeatherBench2 downloads for training on forecasts and for the 3-D Lagrangian transport (one pass over the archive).
hres <year> <month>: every 00 UTC IFS HRES run of the month (WeatherBench2 2016-2022, 0.25 deg, 6-hourly steps); each
    archive chunk (one variable, one step, 13 levels) is read once and gives
    (A) the model's forecast-time inputs, identical in content and format to the 2022 test store (dl_hres_global.py +
        dl_hres_global_ext.py): daily means of the forecast days D+1..D+7 of the 30 store channels (u10, v10, t2m, msl,
        sp, then t/u/v/q/z at 1000/925/850/700/500 hPa) as (x - m) / s, plus precipitation (m/day); the footprint winds
        u10/v10/u925/v925 of D 00 UTC .. D+7 21 UTC every 3 h (0.5 deg); the precipitation of day D (m);
        -> wb2/hres_A_<year>_<month>.npz            (skipped with --no-a, e.g. for 2022 which exists)
    (B) the 3-D transport fields at 1 deg (181 x 360, 90 N..90 S, 180 W..179 E; 1 deg box means of the 0.25 deg points)
        for the steps 0..192 h every 6 h: u, v (m/s), w (Pa/s), t (deg C), q (g/kg), z (geopotential height, km) at
        1000/925/850/700/600/500/400/300 hPa, then u10, v10 (m/s), t2m (deg C), sp (hPa), precipitation of the 6 h
        ending at the step (mm)          -> wb2/hres_B_<year>_<month>.npy  (days, 33, 53, 181, 360) float16
era5 <year> <month>: the same 53 channels from ERA5 at 00/06/12/18 UTC -> wb2/era5_B_<year>_<month>.npy (days, 4, 53, 181, 360)
static: surface geopotential height (km) and land-sea mask at 1 deg -> wb2/static_1deg.npz
    python dl_wb2.py hres 2017 1 [--no-a] [--limit N]"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr

B = "https://storage.googleapis.com/weatherbench2/datasets/"
URL_HRES = B + "hres/2016-2022-0012-1440x721.zarr"
URL_ERA5 = {y: B + ("era5/1959-2022-6h-1440x721.zarr" if y <= 2021 else "era5/1959-2023_01_10-wb13-6h-1440x721.zarr")
            for y in range(2016, 2023)}
OUT = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km/wb2")
# (A) exactly as dl_hres_global.py / dl_hres_global_ext.py
SURF = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "mean_sea_level_pressure", "surface_pressure"]
LEVS = ["temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity", "geopotential"]
PLEV = [1000, 925, 850, 700, 500]
MEANS = np.array([0., 0., 280., 101325., 97000.] + [260.] * 5 + [0.] * 5 + [0.] * 5 + [0.003] * 5 + [50000.] * 5, np.float32)
STDS = np.array([10., 10., 20., 1500., 5000.] + [30.] * 5 + [15.] * 5 + [10.] * 5 + [0.004] * 5 + [40000.] * 5, np.float32)
LAT = 84.0 - 0.25 * np.arange(576)
LON = -180.0 + 0.25 * np.arange(1440)
WSTEPS = list(range(0, 193, 6))
# (B)
L8 = [1000, 925, 850, 700, 600, 500, 400, 300]
V3 = [("u_component_of_wind", 1.0, 0.0), ("v_component_of_wind", 1.0, 0.0), ("vertical_velocity", 1.0, 0.0),
      ("temperature", 1.0, -273.15), ("specific_humidity", 1000.0, 0.0), ("geopotential", 1.0 / 9806.65, 0.0)]
V2 = [("10m_u_component_of_wind", 1.0, 0.0), ("10m_v_component_of_wind", 1.0, 0.0), ("2m_temperature", 1.0, -273.15),
      ("surface_pressure", 0.01, 0.0), ("total_precipitation_6hr", 1000.0, 0.0)]
CHANNELS = [f"{v.split('_')[0]}{p}" for v, _, _ in V3 for p in L8] + ["u10", "v10", "t2m", "sp", "tp6h"]
BSTEPS = list(range(0, 193, 6))                                   # 33 steps, 0 .. 192 h (lead 7 releases during D+7)


def bilinear(a, lat, lon):
    """(C, nlat, nlon) on ascending lat/lon -> (C, 576, 1440) at LAT x LON (as in dl_hres_global.py)."""
    fi = np.interp(LAT, lat, np.arange(len(lat))); fj = np.interp(LON, lon, np.arange(len(lon)))
    i0 = np.clip(np.floor(fi).astype(int), 0, len(lat) - 2); j0 = np.clip(np.floor(fj).astype(int), 0, len(lon) - 2)
    wi, wj = (fi - i0)[:, None], (fj - j0)[None, :]
    g = lambda di, dj: a[:, i0 + di][:, :, j0 + dj]
    return (1 - wi) * ((1 - wj) * g(0, 0) + wj * g(0, 1)) + wi * ((1 - wj) * g(1, 0) + wj * g(1, 1))


TRAP = np.array([0.125, 0.25, 0.25, 0.25, 0.125], np.float32)       # 1 deg box mean of 0.25 deg point values


def to_1deg(a, lat_desc):
    """(..., 721, 1440) 0.25 deg point values, lon -180..179.75, lat descending (90..-90) -> (..., 181, 360)."""
    a = a if lat_desc else a[..., ::-1, :]
    p = np.concatenate([a[..., :, -2:], a, a[..., :, :2]], -1)                                      # periodic in lon
    a = sum(w * p[..., :, k:k + a.shape[-1]] for k, w in enumerate(TRAP))[..., :, ::4]
    p = np.concatenate([a[..., 2:0:-1, :], a, a[..., -2:-4:-1, :]], -2)                              # mirrored at the poles
    return sum(w * p[..., k:k + 721, :] for k, w in enumerate(TRAP))[..., ::4, :]


def open_ds(url):
    ds = xr.open_zarr(url, consolidated=True, chunks=None)
    return ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180).sortby("longitude")


def load_step(x, names, levels=None):
    """list of numpy arrays (lat ascending as stored), one per variable, at one step"""
    out = []
    for v in names:
        a = x[v] if levels is None or "level" not in x[v].dims else x[v].sel(level=levels)
        out.append(np.asarray(a.values, np.float32))
    return out


def hres(a):
    global ZG, ROLL, LEVIDX
    OUT.mkdir(parents=True, exist_ok=True)
    import zarr
    raw = xr.open_zarr(URL_HRES, consolidated=True, chunks=None)          # stored order (longitude 0 .. 359.75)
    ZG = zarr.open_group(URL_HRES, mode="r")
    lon0 = raw.longitude.values
    ROLL = int(np.argmin(np.abs(((lon0 + 180) % 360) - 180 - (-180.0))))   # index of 180 deg -> first column
    ROLL = (len(lon0) - ROLL) % len(lon0)
    LEVIDX = [list(raw.level.values).index(p) for p in L8]
    ds = open_ds(URL_HRES)
    assert np.allclose(np.roll(((lon0 + 180) % 360) - 180, ROLL), ds.longitude.values), "longitude order"
    assert np.allclose(raw.latitude.values, ds.latitude.values)
    lat = ds.latitude.values; desc = lat[0] > lat[-1]
    pt = ds.prediction_timedelta.values
    step_h = (pt / np.timedelta64(1, "h")).astype(int) if np.issubdtype(pt.dtype, np.timedelta64) else pt.astype(int)
    assert all(s in step_h for s in WSTEPS), step_h
    times = [t for t in ds.time.values if str(t)[:7] == f"{a.year}-{a.month:02d}" and str(t)[11:13] == "00"]
    times = times[:a.limit] if a.limit else times
    nd = len(times)
    fB = OUT / f"hres_B_{a.year}_{a.month:02d}.npy"
    Bm = np.lib.format.open_memmap(fB, mode="w+", dtype=np.float16, shape=(nd, len(BSTEPS), len(CHANNELS), 181, 360))
    if not a.no_a:
        A_block = np.zeros((nd, 7, 31, 576, 1440), np.float16); A_wind = np.zeros((nd, 64, 4, 288, 720), np.float16)
        A_rain0 = np.zeros((nd, 576, 1440), np.float32)
    doys = []
    sl = slice(-61, 85) if not desc else slice(85, -61)
    t0 = time.time()
    for n, t in enumerate(times):
        x = ds.sel(time=t)
        sumA = np.zeros((7, 30, 721, 1440), np.float32); tpA = np.zeros((7, 721, 1440), np.float32)
        w6, tp0 = [], np.zeros((721, 1440), np.float32)
        steps = BSTEPS if a.no_a else WSTEPS
        ks = [int(np.where(step_h == s)[0][0]) for s in steps]
        assert ks == list(range(ks[0], ks[0] + len(ks))), ks                   # contiguous steps: one basic slice
        ti = int(np.where(ds.time.values == t)[0][0])
        names3, names2 = [v for v, _, _ in V3], [v for v, _, _ in V2] + ["mean_sea_level_pressure"]

        def read(v):                       # zarr basic slicing (no fancy indexing), lon 0..360 -> -180..180, then levels
            q = np.asarray(ZG[v][ti, ks[0]:ks[-1] + 1], np.float32)
            q = np.roll(q, ROLL, axis=-1)
            return q[:, LEVIDX] if v in names3 else q
        with ThreadPoolExecutor(len(names3) + len(names2)) as pool:
            futs = {v: pool.submit(read, v) for v in names3 + names2}
            ALL = {v: f.result() for v, f in futs.items()}
        for si, s in enumerate(steps):
            lv = [ALL[v][si] for v in names3]                               # 6 x (8, 721, 1440)
            u10, v10, t2m, sp, tp6, msl = (ALL[v][si] for v in names2)
            if s in BSTEPS:                                                 # (B) 3-D transport fields
                ch = [lv[i][j] * sc + off for i, (_, sc, off) in enumerate(V3) for j in range(len(L8))]
                ch += [f * sc + off for f, (_, sc, off) in zip([u10, v10, t2m, sp, tp6], V2)]
                arr = to_1deg(np.stack(ch), desc)
                if s == 0:
                    arr[-1] = np.nan                                        # no accumulation at step 0
                Bm[n, BSTEPS.index(s)] = arr.astype(np.float16)
            if a.no_a:
                continue
            if s >= 24 and (s % 24) in (0, 6, 12, 18) and s // 24 <= 7:      # (A) daily means of D+1 .. D+7
                h = s // 24 - 1
                lev5 = {name: lv[[v for v, _, _ in V3].index(name)][[L8.index(p) for p in PLEV]] for name in LEVS}
                chans = [u10, v10, t2m, msl, sp] + [lev5[name][j] for name in LEVS for j in range(5)]
                sumA[h] += np.stack(chans)
            if s >= 30 and (s % 24) in (6, 12, 18, 0) and (s - 6) // 24 <= 7 and s <= 192:   # 6 h sums ending 06..24 UTC
                h = (s - 6) // 24 - 1
                if 0 <= h < 7:
                    tpA[h] += tp6
            if s in (6, 12, 18, 24):
                tp0 += tp6
            w6.append(np.stack([u10, v10, lv[0][L8.index(925)], lv[1][L8.index(925)]]))
        if not a.no_a:
            latv = lat if not desc else lat[::-1]
            fix = (lambda q: q) if not desc else (lambda q: q[..., ::-1, :])
            sel_lat = (latv >= -61) & (latv <= 85)
            la = latv[sel_lat]; lo = ds.longitude.values
            for h in range(7):
                blk = np.concatenate([fix(sumA[h] / 4.0)[:, sel_lat], fix(tpA[h])[None][:, sel_lat]])
                blk = bilinear(blk, la, lo)
                blk[:30] = (blk[:30] - MEANS[:, None, None]) / STDS[:, None, None]
                A_block[n, h] = blk.astype(np.float16)
            w6 = np.stack([bilinear(fix(w)[:, sel_lat], la, lo) for w in w6])          # (33, 4, 576, 1440)
            t3 = np.arange(0, 190, 3)
            lo_ = np.minimum(t3 // 6, len(WSTEPS) - 2); fr = ((t3 - 6 * lo_) / 6.0)[:, None, None, None]
            w3 = (1 - fr) * w6[lo_] + fr * w6[lo_ + 1]
            A_wind[n] = w3.reshape(64, 4, 288, 2, 720, 2).mean((3, 5)).astype(np.float16)
            A_rain0[n] = bilinear(fix(tp0)[None][:, sel_lat], la, lo)[0]
        doys.append(int((np.datetime64(t, "D") - np.datetime64(f"{a.year}-01-01")).astype(int)))
        print(f"{str(t)[:10]}  {time.time() - t0:.0f}s", flush=True)
    Bm.flush(); del Bm
    if not a.no_a:
        np.savez(OUT / f"hres_A_{a.year}_{a.month:02d}.npz", block=A_block, wind=A_wind, rain_d0=A_rain0, day_of_year=np.array(doys))
    np.save(OUT / f"hres_B_{a.year}_{a.month:02d}_doy.npy", np.array(doys))
    print("done", a.year, a.month, nd, "runs", f"{time.time() - t0:.0f}s")


def era5(a):
    OUT.mkdir(parents=True, exist_ok=True)
    ds = open_ds(URL_ERA5[a.year])
    desc = ds.latitude.values[0] > ds.latitude.values[-1]
    times = [t for t in ds.time.values if str(t)[:7] == f"{a.year}-{a.month:02d}"]
    days = sorted({str(t)[:10] for t in times})[:a.limit] if a.limit else sorted({str(t)[:10] for t in times})
    Bm = np.lib.format.open_memmap(OUT / f"era5_B_{a.year}_{a.month:02d}.npy", mode="w+", dtype=np.float16,
                                   shape=(len(days), 4, len(CHANNELS), 181, 360))
    t0 = time.time()
    for n, d in enumerate(days):
        xx = ds.sel(time=slice(np.datetime64(f"{d}T00:00"), np.datetime64(f"{d}T18:00")))
        assert xx.sizes["time"] == 4, (d, xx.sizes)
        names3, names2 = [v for v, _, _ in V3], [v for v, _, _ in V2]
        with ThreadPoolExecutor(len(names3) + len(names2)) as pool:   # every variable: its 4 times in one read
            futs = {v: pool.submit(lambda v=v: np.asarray((xx[v].sel(level=L8) if v in names3 else xx[v]).values, np.float32))
                    for v in names3 + names2}
            ALL = {v: f.result() for v, f in futs.items()}
        for k in range(4):
            ch = [ALL[v][k][j] * sc + off for v, sc, off in V3 for j in range(len(L8))]
            ch += [ALL[v][k] * sc + off for v, sc, off in V2]
            Bm[n, k] = to_1deg(np.stack(ch), desc).astype(np.float16)
        print(f"{d}  {time.time() - t0:.0f}s", flush=True)
    Bm.flush()
    print("done", a.year, a.month, len(days), "days")


def static(a):
    OUT.mkdir(parents=True, exist_ok=True)
    ds = open_ds(URL_ERA5[2021])
    desc = ds.latitude.values[0] > ds.latitude.values[-1]
    zs = np.asarray(ds["geopotential_at_surface"].values, np.float32) / 9806.65
    lsm = np.asarray(ds["land_sea_mask"].values, np.float32)
    np.savez(OUT / "static_1deg.npz", zs_km=to_1deg(zs, desc), lsm=to_1deg(lsm, desc), channels=np.array(CHANNELS),
             lat=90.0 - np.arange(181), lon=-180.0 + np.arange(360.0))
    print("static done", float(np.nanmax(zs)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["hres", "era5", "static"])
    ap.add_argument("year", type=int, nargs="?", default=2017)
    ap.add_argument("month", type=int, nargs="?", default=1)
    ap.add_argument("--no-a", action="store_true", help="hres: only the 3-D transport fields (B)")
    ap.add_argument("--limit", type=int, default=0, help="first N days only (tests)")
    a = ap.parse_args()
    {"hres": hres, "era5": era5, "static": static}[a.kind](a)
