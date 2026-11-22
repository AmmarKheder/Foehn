#!/usr/bin/env python3
"""GFAS injection height 'injh' (IS4FIRES) 2017-2024 and the injh version of the global fires store (the fires input of the 1 km model used in real time).

Why: GFAS v1.4.2 (the real-time product since 2026-01-01) no longer serves 'mami' (mean altitude of maximum injection,
encoder channel 40 = fires channel 2); RT already uses 'injh' in its place. A global CD-FFM trained on the injh
store sees in training the same variable as in real time.

Stores written (same layout and encoding as the mami stores, which are NOT touched):
  data/processed/global/coarse/fires_injh_2017_2022.npy  (2191, 3, 576, 1440) float16, row = date - 2017-01-01
  data/processed/global/coarse/fires_injh_2023_2024.npy  (731, 3, 576, 1440) float16,  row = date - 2023-01-01
  channels 0, 1 (tpmfire, frpfire) = byte copies of Foehn fires.npy / coarse/fires_2023_2024.npy;
  channel 2 = injh, to025 area mean of the 0.1 deg field (build_transport.to025) then (log10(max(x, 0) + 1) - 0.1) / 0.6,
  the transform of mami in the training stores and of injh in RT's realtime/build.py.
Raw downloads: data/raw/gfas_injh/<yyyy>-<mm>.nc (monthly injh files as delivered by the ADS, 2018-07 .. 2024-12).
The ADS has NO injh before 2018-07-02 (HTTP 400; a request with another variable returns that variable only): rows
2017-01-01 .. 2018-07-01 keep mami in channel 2 (they are robustness-set days only, never training/validation/test).
Day convention: the requested date is the emission day (dl_gfas.py: the ADS stamps valid_time at the following 00 UTC).

  python dl_gfas_injh.py test                 # 2 days with tpmfire + mami + injh: size, day alignment vs gfas_daily
  python dl_gfas_injh.py download [--workers 4] [--years 2017 ...]
  python dl_gfas_injh.py build [--procs 16]
  python dl_gfas_injh.py check
World venv (cdsapi, xarray, zarr): /projappl/project_462001804/envs/world/bin/python, partition small (internet)."""
import argparse
import calendar
import datetime as dt
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
OUT = ROOT / "data/processed/global"
RAW = ROOT / "data/raw/gfas_injh"
sys.path.insert(0, str(ROOT / "scripts"))
from build_global_coarse import (D2017, D2023, H, W, create, npy_header, read_row, to025,  # noqa: E402
                                 update_manifest, write_row)

G = Path("/scratch/project_462001804/ammar/eccv/data/world/global1km")
GFAS_ZARR = "/scratch/project_462001804/ammar/eccv/data/world/gfas_daily/{}.zarr"
DATASET = "cams-global-fire-emissions-gfas"
D2025 = dt.date(2025, 1, 1)
# first emission day for which the ADS serves injh (bisection 2026-10-26: 2018-07-01 -> HTTP 400, 2018-07-02 -> OK; all
# 2017 dates refused). Earlier rows (robustness days only; split A training starts 2019-07-11) keep mami in channel 2.
FIRST_INJH = dt.date(2018, 7, 2)
STORES = [(OUT / "coarse/fires_injh_2017_2022.npy", D2017, D2023, G / "fires.npy"),
          (OUT / "coarse/fires_injh_2023_2024.npy", D2023, D2025, OUT / "coarse/fires_2023_2024.npy")]


def ads():
    import cdsapi
    rc = dict(l.split(":", 1) for l in (Path.home() / ".adsapirc").read_text().splitlines() if ":" in l)
    return cdsapi.Client(url=rc["url"].strip(), key=rc["key"].strip(), quiet=True, progress=False)


def nc_paths(p, tmpdir):
    """netCDF file(s) of an ADS download (bare .nc or a zip of .nc)"""
    import zipfile
    if zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as z:
            z.extractall(tmpdir)
        return sorted(Path(tmpdir).glob("*.nc"))
    return [Path(p)]


def read_var(paths, name):
    import xarray as xr
    for p in paths:
        with xr.open_dataset(p) as ds:
            if name in ds.data_vars:
                da = ds[name]
                tdim = [d for d in da.dims if d not in ("latitude", "longitude")][0]
                t = ds[tdim].values
                return da.transpose(tdim, "latitude", "longitude").values.astype(np.float32), ds["latitude"].values, \
                    ds["longitude"].values, t
    raise KeyError(name)


def encode_inj(g):
    return ((np.log10(np.maximum(g, 0) + 1.0) - 0.1) / 0.6).astype(np.float32)


def cmd_test(_a):
    import tempfile
    import zarr
    RAW.mkdir(parents=True, exist_ok=True)
    tgt = RAW / "test_2022-07-01_02.download"
    t0 = time.time()
    ads().retrieve(DATASET, {"variable": ["wildfire_flux_of_total_particulate_matter", "mean_altitude_of_maximum_injection",
                                          "injection_height"], "date": ["2022-07-01/2022-07-02"], "data_format": "netcdf"}, str(tgt))
    size = tgt.stat().st_size
    with tempfile.TemporaryDirectory(dir=RAW) as tmp:
        ps = nc_paths(tgt, tmp)
        tpm, lat, lon, t = read_var(ps, "tpmfire")
        mami, *_ = read_var(ps, "mami")
        injh, *_ = read_var(ps, "injh")
    z = zarr.open_array(GFAS_ZARR.format(2022), mode="r")
    doy0 = (dt.date(2022, 7, 1) - dt.date(2022, 1, 1)).days
    res = {"bytes_3vars_2days": size, "seconds": round(time.time() - t0), "time_stamps": [str(x) for x in t],
           "lat": [float(lat[0]), float(lat[-1]), int(lat.size)], "lon": [float(lon[0]), float(lon[-1]), int(lon.size)]}
    for k in range(2):
        ref = np.asarray(z[doy0 + k, [0, 4]], np.float32)
        res[f"day{k}_tpm_equal_zarr"] = bool(np.allclose(np.nan_to_num(ref[0]), np.nan_to_num(tpm[k]), rtol=0, atol=0))
        res[f"day{k}_mami_equal_zarr"] = bool(np.allclose(np.nan_to_num(ref[1]), np.nan_to_num(mami[k]), rtol=0, atol=0))
        f = (tpm[k] > 0) & np.isfinite(injh[k]) & np.isfinite(mami[k])
        res[f"day{k}_fire_pixels"] = int(f.sum())
        res[f"day{k}_corr_injh_mami_fire_pixels"] = float(np.corrcoef(injh[k][f], mami[k][f])[0, 1])
        res[f"day{k}_median_injh_mami_m"] = [float(np.median(injh[k][f])), float(np.median(mami[k][f]))]
        res[f"day{k}_injh_nan_frac"] = float(np.isnan(injh[k]).mean())
    print(json.dumps(res, indent=1), flush=True)
    (RAW / "test_report.json").write_text(json.dumps(res, indent=1))
    tgt.unlink()


def month_path(y, m):
    return RAW / f"{y}-{m:02d}.nc"


def download_month(y, m):
    dst = month_path(y, m)
    if dst.exists():
        return y, m, "exists", dst.stat().st_size
    n = calendar.monthrange(y, m)[1]
    d0 = FIRST_INJH.day if (y, m) == (FIRST_INJH.year, FIRST_INJH.month) else 1
    part = RAW / f"{y}-{m:02d}.part"
    for attempt in range(6):
        try:
            ads().retrieve(DATASET, {"variable": ["injection_height"], "date": [f"{y}-{m:02d}-{d0:02d}/{y}-{m:02d}-{n:02d}"],
                                     "data_format": "netcdf"}, str(part))
            import tempfile
            with tempfile.TemporaryDirectory(dir=RAW) as tmp:
                inj, lat, lon, t = read_var(nc_paths(part, tmp), "injh")
            assert inj.shape == (n - d0 + 1, 1800, 3600), inj.shape
            os.replace(part, dst)
            return y, m, "ok", dst.stat().st_size
        except Exception as e:  # noqa: BLE001
            print(f"{y}-{m:02d}: attempt {attempt + 1} failed: {repr(e)[:300]}", flush=True)
            time.sleep(60 * (attempt + 1))
    raise RuntimeError(f"{y}-{m:02d}: download failed")


def cmd_download(a):
    RAW.mkdir(parents=True, exist_ok=True)
    months = [(y, m) for y in a.years for m in range(1, 13) if dt.date(y, m, calendar.monthrange(y, m)[1]) >= FIRST_INJH]
    t0 = time.time()
    tot = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for y, m, st, sz in ex.map(lambda ym: download_month(*ym), months):
            tot += sz
            print(f"{y}-{m:02d}: {st} {sz / 1e6:.1f} MB ({time.time() - t0:.0f}s, total {tot / 1e9:.2f} GB)", flush=True)
    print(f"download done: {len(months)} months, {tot / 1e9:.2f} GB, {time.time() - t0:.0f}s", flush=True)


_CACHE = {}


def injh_month(y, m):
    key = (y, m)
    if key not in _CACHE:
        import tempfile
        _CACHE.clear()
        with tempfile.TemporaryDirectory(dir=RAW) as tmp:
            inj, lat, lon, t = read_var(nc_paths(month_path(y, m), tmp), "injh")
        assert abs(lat[0] - 89.95) < 1e-3 and abs(lat[-1] + 89.95) < 1e-3 and lon.size == 3600, (lat[:2], lon[:2])
        _CACHE[key] = (inj, float(lon[0]))
    return _CACHE[key]


def _row(job):
    path, i, d, src = job
    x = read_row(src, i).copy()                                             # tpmfire, frpfire: byte copy of the mami store
    if d < FIRST_INJH:                                                      # no injh on the ADS: channel 2 stays mami
        write_row(Path(path), i, x)
        return i, bool(np.isfinite(x.astype(np.float32)).all())
    inj, lon0 = injh_month(d.year, d.month)
    k0 = FIRST_INJH.day if (d.year, d.month) == (FIRST_INJH.year, FIRST_INJH.month) else 1
    g = to025(inj[d.day - k0][None], lon0 - 0.05)[0]                        # same call as realtime/build.py (lon0 = 0.05 -> 0)
    x[2] = encode_inj(g).astype(np.float16)
    fin = bool(np.isfinite(x.astype(np.float32)).all())
    write_row(Path(path), i, x)
    return i, fin


def cmd_build(a):
    for path, d0, d1, src in STORES:
        days = [d0 + dt.timedelta(k) for k in range((d1 - d0).days)]
        shape = (len(days), 3, H, W)
        assert npy_header(src)[0][1:] == shape[1:] and npy_header(src)[0][0] >= len(days), (src, npy_header(src))
        if not path.exists():
            create(path, shape, np.float16)
        t0 = time.time()
        bad = []
        # month-ordered jobs so that each worker reuses its decoded month
        with mp.get_context("fork").Pool(a.procs) as pool:
            for n, (i, fin) in enumerate(pool.imap(_row, [(str(path), i, d, str(src)) for i, d in enumerate(days)], chunksize=31)):
                if not fin:
                    bad.append(str(days[i]))
                if n % 200 == 0:
                    print(f"{path.name}: {n + 1}/{len(days)} ({time.time() - t0:.0f}s)", flush=True)
        assert not bad, f"non-finite rows {bad[:5]}"
        print(f"{path.name}: {len(days)} rows in {time.time() - t0:.0f}s", flush=True)


def cmd_check(_a):
    """finite, same day range as the mami stores, channels 0-1 identical, channel 2 vs mami: correlation on fire cells"""
    res = {}
    rng = np.random.default_rng(0)
    for path, d0, d1, src in STORES:
        sh, dtp, _ = npy_header(path)
        n = (d1 - d0).days
        assert sh == (n, 3, H, W) and dtp == np.float16, (path, sh)
        r = {"shape": list(sh), "days": [str(d0), str(d1 - dt.timedelta(1)), n]}
        rows = sorted(set([0, n - 1] + list(rng.choice(n, 40, replace=False))))
        r["rows_with_mami_kept"] = int(sum(1 for i in range(n) if d0 + dt.timedelta(i) < FIRST_INJH))
        fin = True
        same01 = True
        cors, cors_enc, med = [], [], []
        for i in rows:
            a = read_row(path, i)
            b = read_row(src, i)
            fin &= bool(np.isfinite(a.astype(np.float32)).all())
            same01 &= bool(np.array_equal(a[:2].view(np.uint16), b[:2].view(np.uint16)))
            fire = a[1].astype(np.float32) > -0.5                                  # FRP > ~0.3 W m-2 (as build.py's count)
            if fire.sum() > 100 and d0 + dt.timedelta(int(i)) >= FIRST_INJH:
                ea, eb = a[2].astype(np.float32)[fire], b[2].astype(np.float32)[fire]
                cors_enc.append(float(np.corrcoef(ea, eb)[0, 1]))
                ha, hb = 10 ** (0.6 * ea + 0.1) - 1, 10 ** (0.6 * eb + 0.1) - 1
                cors.append(float(np.corrcoef(ha, hb)[0, 1]))
                med.append([float(np.median(ha)), float(np.median(hb))])
        # every row finite (cheap: channel 2 only, strided)
        allfin = True
        for i in range(n):
            allfin &= bool(np.isfinite(read_row(path, i)[2, ::4, ::4].astype(np.float32)).all())
        r.update({"rows_checked": len(rows), "finite_sampled_rows": fin, "finite_all_rows_ch2_stride4": allfin,
                  "ch01_bit_identical_to_mami_store": same01,
                  "corr_injh_vs_mami_encoded_fire_cells_median": float(np.median(cors_enc)) if cors_enc else None,
                  "corr_injh_vs_mami_metres_fire_cells_median": float(np.median(cors)) if cors else None,
                  "corr_range_encoded": [float(np.min(cors_enc)), float(np.max(cors_enc))] if cors_enc else None,
                  "median_height_m_injh_vs_mami": np.median(np.array(med), 0).tolist() if med else None})
        res[path.name] = r
        print(path.name, json.dumps(r, indent=1), flush=True)
    update_manifest("fires_injh", res)
    (RAW / "check_report.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["test", "download", "build", "check"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--years", type=int, nargs="*", default=list(range(2017, 2025)))
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    {"test": cmd_test, "download": cmd_download, "build": cmd_build, "check": cmd_check}[a.cmd](a)
