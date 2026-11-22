#!/usr/bin/env python3
"""backfill_2025_2026.py -- the inputs of the 1 km analysis from 2025-01-01 to the last complete day,
with the SAME code paths as the 2017-2024 stores (functions imported from
build_global_coarse.py and dl_gfas_injh.py), into data/processed/global/coarse/, row = date - 2025-01-01 (730 rows,
2025-01-01 .. 2026-12-31; rows beyond the last complete day are NaN):
  cams_pm25_025_2025_2026.npy  (730, 576, 1440) f16   build_global_coarse.regrid of the 00 UTC day-0 mean
                                                     (cams_forecast_daily/<y>.zarr; 2026 by 'cams-download')
  met_2025_2026.npy            (730, 37, 576, 1440) f16  build_global_coarse.met_day (era5_global_daily + era5_extra_daily;
                                                     2026 from ARCO by dl_era5_daily_arco.py)
  fires_injh_2025_2026.npy     (730, 3, 576, 1440) f16   channels 0-1 (tpmfire, frpfire): build_global_coarse.fires_day from
                                                     gfas_daily/2025.zarr up to 2025-12-03 (its last day), afterwards the
                                                     same transform of the monthly ADS files data/raw/gfas_injh/<y>-<m>.nc
                                                     (tpmfire, frpfire, injh; 'gfas-download'); channel 2: injh of those
                                                     files, dl_gfas_injh.encode_inj (as the 2018-07-02 .. 2024 rows)
  COVERAGE_2025_2026.json      last complete day per input (also T6, written by build_global_t6.py's manifest parts)
Commands (world venv, partition small):
  gfas-download [--workers 3]   monthly ADS requests 2025-01 .. this month (partial last month: <y>-<m>_to<dd>.nc)
  cams-download                 cams_forecast_daily/2026.zarr month by month, partial last month (dl_cams_forecast code)
  met | fires | cams [--procs N]   build / extend the stores (only the rows whose sources exist and are not built yet)
  coverage                      COVERAGE_2025_2026.json from the stores' state files and the T6 manifest parts
  refs                          reference rows of 3 days per year for the store test (backfill/refs.npz)"""
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
BF = OUT / "backfill"
RAW = ROOT / "data/raw/gfas_injh"
sys.path.insert(0, str(ROOT / "scripts"))
from build_global_coarse import (CAMS, D2023, GFAS, H, H25, W, W25, cams_day, create, fires_day, met_day, npy_header,  # noqa: E402
                                 read_row, regrid, regrid_weights, to025, write_row)
from dl_gfas_injh import DATASET, ads, encode_inj, nc_paths, read_var  # noqa: E402

D2025, D2027 = dt.date(2025, 1, 1), dt.date(2027, 1, 1)
DAYS = [D2025 + dt.timedelta(k) for k in range((D2027 - D2025).days)]            # 730
GFAS_ZARR_LAST = dt.date(2025, 12, 3)        # last day of gfas_daily/2025.zarr (GFAS v1.2 stopped on the ADS 2025-12-04)
GFAS_VARS = ["wildfire_flux_of_total_particulate_matter", "wildfire_radiative_power", "injection_height"]
STORES = {"cams": ("coarse/cams_pm25_025_2025_2026.npy", (len(DAYS), H25, W25)),
          "met": ("coarse/met_2025_2026.npy", (len(DAYS), 37, H, W)),
          "fires": ("coarse/fires_injh_2025_2026.npy", (len(DAYS), 3, H, W))}
STATE = BF / "state_{}.json"
WORLD = Path("/scratch/project_462001804/ammar/eccv/data/world")
NZ_PATHS = {"era5": "/scratch/project_462001804/ammar/eccv/data/zarr/era5_global_daily/{}.zarr", "extra": str(WORLD / "era5_extra_daily/{}.zarr"),
            "cams": str(WORLD / "cams_forecast_daily/{}.zarr"), "gfas": str(WORLD / "gfas_daily/{}.zarr")}


def row_of(d):
    return (d - D2025).days


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# ------------------------------------------------------------------ source availability (per day)
_ZC = {}


def zopen(kind, year):
    import zarr
    key = (kind, year)
    if key not in _ZC:
        p = NZ_PATHS[kind].format(year)
        _ZC[key] = zarr.open_array(p, mode="r") if Path(p).exists() else None
    return _ZC[key]


def zday_ok(kind, d, sel):
    """the day's chunk exists and a coarse sample of it is finite (the stores are NaN-filled)"""
    z = zopen(kind, d.year)
    if z is None:
        return False
    k = (d - dt.date(d.year, 1, 1)).days
    dd = z.attrs.get("done_days")
    if dd is not None and k not in set(int(x) for x in dd) and (d.month not in set(z.attrs.get("done_months", []))):
        return False
    x = np.asarray(z[(k,) + sel])
    return bool(np.isfinite(x).all())


def avail_met(d):
    return zday_ok("era5", d, (slice(None), slice(0, 721, 90), slice(0, 1440, 180))) and \
        zday_ok("extra", d, (slice(None), slice(0, 721, 90), slice(0, 1440, 180)))


def avail_cams(d):
    return zday_ok("cams", d, (0, 0, slice(0, 451, 50), slice(0, 900, 100)))


def gfas_month_file(y, m):
    """complete monthly file, else the latest partial one (<y>-<m>_to<dd>.nc), else None"""
    p = RAW / f"{y}-{m:02d}.nc"
    if p.exists():
        return p, calendar.monthrange(y, m)[1]
    parts = sorted(RAW.glob(f"{y}-{m:02d}_to??.nc"))
    if parts:
        return parts[-1], int(parts[-1].stem.split("_to")[1])
    return None, 0


def avail_fires(d):
    f, nd = gfas_month_file(d.year, d.month)
    if f is None or d.day > nd:
        return False
    if d <= GFAS_ZARR_LAST:
        return zday_ok("gfas", d, (slice(None), slice(0, 1800, 300), slice(0, 3600, 600)))
    return True


AVAIL = {"met": avail_met, "cams": avail_cams, "fires": avail_fires}


# ------------------------------------------------------------------ downloads (ADS)
def request_days(y, m, d0, d1, target, variables):
    ads().retrieve(DATASET, {"variable": variables, "date": [f"{y}-{m:02d}-{d0:02d}/{y}-{m:02d}-{d1:02d}"],
                             "data_format": "netcdf"}, str(target))


def download_gfas_month(y, m, today):
    n = calendar.monthrange(y, m)[1]
    last = n if dt.date(y, m, n) < today else (today - dt.timedelta(1)).day       # GFAS of day d is on the ADS on d + 1
    if dt.date(y, m, 1) >= today:
        return y, m, "future", 0
    have, nd = gfas_month_file(y, m)
    if have is not None and nd >= last:
        return y, m, "exists", have.stat().st_size
    import tempfile
    for attempt in range(6):
        try:
            part = RAW / f"{y}-{m:02d}.part"
            request_days(y, m, 1, last, part, GFAS_VARS)
            with tempfile.TemporaryDirectory(dir=RAW) as tmp:
                ps = nc_paths(part, tmp)
                tpm, lat, lon, t = read_var(ps, "tpmfire")
                frp, *_ = read_var(ps, "frpfire")
                inj, *_ = read_var(ps, "injh")
            assert tpm.shape == frp.shape == inj.shape == (last, 1800, 3600), (tpm.shape, last)
            dst = RAW / (f"{y}-{m:02d}.nc" if last == n else f"{y}-{m:02d}_to{last:02d}.nc")
            os.replace(part, dst)
            for old in RAW.glob(f"{y}-{m:02d}_to??.nc"):
                if old != dst:
                    old.unlink()
            return y, m, "ok", dst.stat().st_size
        except Exception as e:  # noqa: BLE001
            log(f"{y}-{m:02d} (days 1..{last}): attempt {attempt + 1} failed: {repr(e)[:300]}")
            if "400" in repr(e) or "no data" in repr(e).lower():
                last -= 1                                   # the newest day is not on the ADS yet
            time.sleep(60 * (attempt + 1))
    raise RuntimeError(f"GFAS {y}-{m:02d}: download failed")


def cmd_gfas_download(a):
    RAW.mkdir(parents=True, exist_ok=True)
    today = dt.date.today()
    months = [(y, m) for y in (2025, 2026) for m in range(1, 13) if dt.date(y, m, 1) < today]
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for y, m, st, sz in ex.map(lambda ym: download_gfas_month(*ym, today), months):
            log(f"GFAS {y}-{m:02d}: {st} {sz / 1e6:.1f} MB ({time.time() - t0:.0f}s)")
    log("gfas-download done")


def cmd_cams_download(a):
    """cams_forecast_daily/<year>.zarr by month with dl_cams_forecast's request / to_daily_leads; partial current month"""
    sys.path.insert(0, "/scratch/project_462001804/ammar/eccv/world/data_pipeline")
    import shutil
    import xarray as xr
    import zarr
    from common import ads_client, day_of_year_offset, done_months, mark_month_done, netcdf_files, open_daily_store, retry
    from dl_cams_forecast import DATASET as CDS, N_LEAD_DAYS, STEPS, VARIABLES, request, to_daily_leads
    year = a.year
    today = dt.date.today()
    store = WORLD / "cams_forecast_daily" / f"{year}.zarr"
    tmp_root = WORLD / "tmp" / f"cams_fc_{year}_bf"
    client = ads_client()
    z = zarr.open_array(str(store), mode="r+") if store.exists() else None
    done_days = set(int(x) for x in z.attrs.get("done_days", [])) if z is not None else set()
    for month in range(1, 13):
        if dt.date(year, month, 1) > today:
            break
        if z is not None and month in done_months(z):
            continue
        n = calendar.monthrange(year, month)[1]
        last = n if dt.date(year, month, n) < today else today.day          # the 00 UTC run of today is out by ~10 UTC
        i0 = day_of_year_offset(year, month)
        if all(i0 + k in done_days for k in range(last)):
            continue
        tmp = tmp_root / f"{month:02d}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        fields = None
        for shrink in range(3):
            try:
                fields = []
                for short, var in VARIABLES:
                    target = tmp / f"{short}.download"
                    log(f"CAMS {year}-{month:02d} days 1..{last} {var}")
                    retry(request, client, var, year, month, list(range(1, last + 1)), STEPS, target, attempts=3, base_wait=120)
                    ncs = netcdf_files(target, tmp / short)
                    da = xr.open_dataset(ncs[0])[short]
                    daily, lat, lon = to_daily_leads(da)
                    assert daily.shape[0] == last, (daily.shape, last)
                    fields.append(daily)
                break
            except Exception as e:  # noqa: BLE001
                log(f"CAMS {year}-{month:02d}: days 1..{last} failed ({repr(e)[:200]}); retrying with one day less")
                last -= 1
                fields = None
        assert fields is not None, f"CAMS {year}-{month:02d}: download failed"
        block = np.stack(fields, axis=1)                         # (last, vars, leads, lat, lon)
        if z is None:
            attrs = {"source": f"ADS {CDS}, 00 UTC run, 3-hourly steps averaged per lead day",
                     "variables": [v[0] for v in VARIABLES], "units": "ug m-3", "lead_days": list(range(N_LEAD_DAYS)),
                     "lat_first_last_n": [float(lat[0]), float(lat[-1]), int(lat.size)],
                     "lon_first_last_n": [float(lon[0]), float(lon[-1]), int(lon.size)]}
            z = open_daily_store(store, year, block.shape[1:], attrs)
        z[i0:i0 + last] = block
        done_days |= set(range(i0, i0 + last))
        z.attrs["done_days"] = sorted(done_days)
        z.attrs["last_day"] = str(dt.date(year, 1, 1) + dt.timedelta(max(done_days)))
        if last == n:
            mark_month_done(z, month)
        shutil.rmtree(tmp, ignore_errors=True)
        log(f"CAMS {year}-{month:02d}: {last} days written")
    shutil.rmtree(tmp_root, ignore_errors=True)
    log(f"cams-download done: last_day {z.attrs.get('last_day') if z is not None else None}")


# ------------------------------------------------------------------ the day functions (2025-2026)
_WTS = []
_GF = {}


def cams_src_bf(d):
    """0.4 deg day-0 mean PM2.5 of the 00 UTC run of day d (no x 8/7: that step ended 2017-09-27), as cams_src"""
    z = zopen("cams", d.year)
    at = dict(z.attrs)
    assert at["variables"][0] == "pm2p5" and at["lead_days"][0] == 0 and at["lat_first_last_n"] == [90.0, -90.0, 451], at
    assert at["lon_first_last_n"][0] == 0.0 and at["lon_first_last_n"][2] == 900, at
    src = np.asarray(z[(d - dt.date(d.year, 1, 1)).days, 0, 0], np.float32)
    assert src.shape == (451, 900) and np.isfinite(src).all() and src.min() >= 0, d
    return src


def cams_day_bf(d):
    if not _WTS:
        _WTS.append(regrid_weights())
    return regrid(cams_src_bf(d), _WTS[0]).astype(np.float16)


def gfas_month(y, m):
    key = (y, m)
    if key not in _GF:
        import tempfile
        _GF.clear()
        f, nd = gfas_month_file(y, m)
        with tempfile.TemporaryDirectory(dir=RAW) as tmp:
            ps = nc_paths(f, tmp)
            tpm, lat, lon, t = read_var(ps, "tpmfire")
            frp, *_ = read_var(ps, "frpfire")
            inj, *_ = read_var(ps, "injh")
        assert abs(lat[0] - 89.95) < 1e-3 and abs(lat[-1] + 89.95) < 1e-3 and lon.size == 3600, (lat[:2], lon[:2])
        assert tpm.shape[0] == nd, (f, tpm.shape, nd)
        _GF[key] = (tpm, frp, inj, float(lon[0]))
    return _GF[key]


def fires_from_nc(d):
    """channels 0-1 from the monthly ADS file with the transform of build_global_coarse.fires_day"""
    tpm, frp, inj, lon0 = gfas_month(d.year, d.month)
    g = to025(np.stack([tpm[d.day - 1], frp[d.day - 1]]), lon0 - 0.05)
    x = np.stack([np.log10(np.maximum(g[0], 0) + 1e-14), np.log10(np.maximum(g[1], 0) + 1e-6)])
    x = (x - np.array([-13.5, -5.5])[:, None, None]) / np.array([1.0, 1.0])[:, None, None]
    return x.astype(np.float16)


def fires_day_bf(d):
    """(3, 576, 1440) f16: tpmfire, frpfire (zarr to 2025-12-03 / monthly ADS file after), injh (monthly ADS file)"""
    x01 = fires_day(d)[:2] if d <= GFAS_ZARR_LAST else fires_from_nc(d)
    tpm, frp, inj, lon0 = gfas_month(d.year, d.month)
    g = to025(inj[d.day - 1][None], lon0 - 0.05)[0]                         # dl_gfas_injh._row
    return np.concatenate([x01, encode_inj(g).astype(np.float16)[None]])


DAY_FN = {"cams": cams_day_bf, "met": met_day, "fires": fires_day_bf}


# ------------------------------------------------------------------ build
def create_nan(path, shape):
    """preallocated .npy whose rows are all NaN (a row never built reads as NaN, never as zeros)"""
    create(path, shape, np.float16)
    nan = np.full(shape[1:], np.nan, np.float16)
    for i in range(shape[0]):
        write_row(path, i, nan)


def load_state(kind):
    p = STATE.with_name(STATE.name.format(kind))
    return json.loads(p.read_text()) if p.exists() else {"built": [], "file": STORES[kind][0]}


def save_state(kind, st):
    BF.mkdir(parents=True, exist_ok=True)
    p = STATE.with_name(STATE.name.format(kind))
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(st, indent=1))
    os.replace(tmp, p)


def _work(job):
    kind, path, d = job
    x = DAY_FN[kind](d)
    fin = bool(np.isfinite(x.astype(np.float32)).all())
    write_row(Path(path), row_of(d), x)
    return str(d), fin


def cmd_build(a):
    kind = a.cmd
    rel, shape = STORES[kind]
    path = OUT / rel
    if not path.exists():
        log(f"{kind}: creating {path} {shape} (NaN-filled)")
        create_nan(path, shape)
    assert npy_header(path)[0] == shape, (npy_header(path)[0], shape)
    st = load_state(kind)
    built = set(st["built"])
    todo = [d for d in DAYS if str(d) not in built and d < dt.date.today() + dt.timedelta(1) and AVAIL[kind](d)]
    log(f"{kind}: {len(built)} rows built before, {len(todo)} to build ({todo[0] if todo else '-'} .. {todo[-1] if todo else '-'})")
    t0 = time.time()
    bad = []
    with mp.get_context("fork").Pool(a.procs) as pool:
        for n, (ds, fin) in enumerate(pool.imap(_work, [(kind, str(path), d) for d in todo], chunksize=8)):
            if fin:
                built.add(ds)
            else:
                bad.append(ds)
            if n % 50 == 0 or n == len(todo) - 1:
                st["built"] = sorted(built)
                save_state(kind, st)
                log(f"{kind}: {n + 1}/{len(todo)} rows ({time.time() - t0:.0f}s)")
    if bad:                                                        # a non-finite row is rewritten as NaN and not counted
        for ds in bad:
            write_row(path, row_of(dt.date.fromisoformat(ds)), np.full(shape[1:], np.nan, np.float16))
        log(f"{kind}: NON-FINITE rows left NaN: {bad[:10]}")
    st["built"] = sorted(built)
    st["last_complete_day"] = last_complete(st["built"])
    st["non_finite_days"] = bad
    st["shape"] = list(shape)
    st["updated"] = time.strftime("%Y-%m-%d %H:%M")
    save_state(kind, st)
    log(f"{kind}: done, last complete day {st['last_complete_day']}, {len(built)} rows")


def last_complete(built):
    """the last day d such that every day 2025-01-01 .. d is built"""
    s = set(built)
    last = None
    for d in DAYS:
        if str(d) in s:
            last = str(d)
        else:
            break
    return last


# ------------------------------------------------------------------ coverage + references
def cmd_coverage(_a):
    cov = {"written": time.strftime("%Y-%m-%d %H:%M"), "row_convention": "row = date - 2025-01-01 (cams, met, fires); t6_<year>.npy row = doy",
           "inputs": {}}
    for kind in ("cams", "met", "fires"):
        st = load_state(kind)
        cov["inputs"][kind] = {"file": st.get("file"), "last_complete_day": st.get("last_complete_day"),
                               "n_rows_built": len(st.get("built", [])), "non_finite_days": st.get("non_finite_days", []),
                               "built_days_gaps": gaps(st.get("built", []))}
    t6 = {}
    for y in (2025, 2026):
        p = OUT / "manifest_parts" / f"t6_{y}.json"
        if p.exists():
            info = json.loads(p.read_text())
            nd = info["shape"][0]
            rows_nan = set(info.get("rows_nan", []))
            last = None
            for k in range(nd):
                if k in rows_nan:
                    break
                last = str(dt.date(y, 1, 1) + dt.timedelta(k))
            t6[str(y)] = {"file": info["file"], "last_complete_day": last, "rows_nan": sorted(rows_nan)[:5] + (["..."] if len(rows_nan) > 5 else []),
                          "n_rows_nan": len(rows_nan), "source": info.get("source"), "era5_days_3_tail": info.get("era5_days_3_tail")}
    cov["inputs"]["t6"] = t6
    lasts = [v["last_complete_day"] for k, v in cov["inputs"].items() if k != "t6" and v.get("last_complete_day")]
    lasts += [v["last_complete_day"] for v in t6.values() if v.get("last_complete_day")] if t6 else [None]
    cov["last_complete_day_all_inputs"] = min(lasts) if lasts and all(lasts) else None
    cov["sources"] = {"cams": "ADS cams-global-atmospheric-composition-forecasts 00 UTC day-0 mean (eccv/data/world/cams_forecast_daily)",
                      "met": "ERA5: era5_global_daily (2025 CDS, 2026 ARCO via dl_era5_daily_arco.py) + era5_extra_daily (ARCO)",
                      "fires": "GFAS: gfas_daily/2025.zarr (v1.2) to 2025-12-03, monthly ADS files afterwards (v1.4.2 from 2026); channel 2 = injh",
                      "t6": "FLEXGREEN ERA5 lead 1 (transport/flexgreen_pm.py, wb2 era5_B from ARCO, fires = the injh store) + CAMS of D-1",
                      "realtime_tail": "days after the last complete day: the real-time chain (realtime_1km/prep.py, IFS + FLEXGREEN per day)"}
    for st in ("openaq", "cnemc"):
        p = BF / f"stations_{st}.json"
        if p.exists():
            cov.setdefault("stations", {})[st] = json.loads(p.read_text())
    p = OUT / "COVERAGE_2025_2026.json"
    tmp = p.with_suffix(f".json.tmp{os.getpid()}")
    tmp.write_text(json.dumps(cov, indent=1))
    os.replace(tmp, p)                                       # atomic: several jobs may refresh it
    print(json.dumps(cov, indent=1), flush=True)


def gaps(built):
    s = set(built)
    out = []
    last = last_complete(built)
    if last is None:
        return out
    for d in DAYS:
        if d <= dt.date.fromisoformat(last):
            continue
        if str(d) not in s and any(str(e) in s for e in DAYS if e > d):
            out.append(str(d))
    return out[:20]


def ref_days():
    """3 days per year: two fixed and the last complete day of the store set (from the state files)"""
    sts = {k: load_state(k) for k in ("cams", "met", "fires")}
    lasts = [sts[k].get("last_complete_day") for k in sts]
    last = min(x for x in lasts if x) if all(lasts) else None
    days = [dt.date(2025, 1, 15), dt.date(2025, 7, 1), dt.date(2025, 12, 20), dt.date(2026, 2, 10), dt.date(2026, 6, 15)]
    if last:
        days.append(dt.date.fromisoformat(last))
    return [d for d in days if last and d <= dt.date.fromisoformat(last)]


def cmd_refs(_a):
    """reference rows straight from the sources (builder functions) + the T6 file rows, for test_store_backfill.py"""
    days = ref_days()
    out = {"days": np.array([str(d) for d in days])}
    for d in days:
        out[f"cams_{d}"] = cams_day_bf(d)
        out[f"met_{d}"] = met_day(d)
        out[f"fires_{d}"] = fires_day_bf(d)
        p = OUT / f"transport/t6_{d.year}.npy"
        if p.exists():
            out[f"t6_{d}"] = read_row(p, (d - dt.date(d.year, 1, 1)).days)
        log(f"refs {d}: cams/met/fires{'/t6' if p.exists() else ''}")
    BF.mkdir(parents=True, exist_ok=True)
    np.savez(BF / "refs.npz", **out)
    # a sanity check of the two fire paths on the last zarr days: ADS monthly file vs gfas_daily zarr (channels 0-1)
    chk = {}
    for d in (dt.date(2025, 11, 20), dt.date(2025, 12, 3)):
        if avail_fires(d):
            a, b = fires_day(d)[:2], fires_from_nc(d)
            chk[str(d)] = {"bit_identical_ch01_zarr_vs_adsfile": bool(np.array_equal(a.view(np.uint16), b.view(np.uint16))),
                           "max_abs_diff": float(np.abs(a.astype(np.float32) - b.astype(np.float32)).max())}
    (BF / "fires_path_check.json").write_text(json.dumps(chk, indent=1))
    log(f"refs written: {[str(d) for d in days]}; fires path check {chk}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gfas-download", "cams-download", "met", "fires", "cams", "coverage", "refs"])
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--year", type=int, default=2026)
    a = ap.parse_args()
    (OUT / "coarse").mkdir(parents=True, exist_ok=True)
    BF.mkdir(parents=True, exist_ok=True)
    {"gfas-download": cmd_gfas_download, "cams-download": cmd_cams_download, "coverage": cmd_coverage, "refs": cmd_refs}.get(a.cmd, cmd_build)(a)
