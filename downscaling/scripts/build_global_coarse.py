#!/usr/bin/env python3
"""Global coarse inputs of the CAMS -> LGHAP v2 downscaling (PLAN.md 2.1, 2.6, 2.7 items 1-3) into
data/processed/global/coarse/:

  met_2023_2024.npy    (731, 37, 576, 1440) float16  ERA5 37 channels of 2023-01-01 .. 2024-12-31, row = (date - 2023-01-01)
                       EXACT copy of the met logic of Foehn's build_coarse.py (crop, extra_raw, stats.npz 2017-2020)
  fires_2023_2024.npy  (731, 3, 576, 1440) float16   GFAS tpmfire, frpfire, mami, EXACT copy of build_transport.py 'fires'
  cams_pm25_025.npy    (2922, 576, 1440) float16     CAMS operational 00 UTC run, day-0 mean PM2.5 (ug m-3), x 8/7 before
                       2017-09-27 (applied to the 0.4 deg day-0 mean), bilinear in log1p with periodic longitude to the
                       Foehn 0.25 deg grid (same regrid code as build_smoke_data.py), row = (date - 2017-01-01)
  days.npz             per date 2017-01-01 .. 2024-12-31: split code and (store, row) of every input (see cmd_days)

  python build_global_coarse.py met|fires|cams [--procs N] | nonreg | days | check
The Foehn stores (met.npy, fires.npy, 2017-2022) are only read. Non-regression (nonreg): the met and fires rows of
2022-12-31 (and two more days) recomputed with THIS code must be bit-identical to the Foehn rows.
World venv (zarr 3, numpy), partition small: jobs/global_coarse.sh."""
import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
OUT = ROOT / "data/processed/global"
Z = Path("/scratch/project_462001804/ammar/eccv/data")
G = Z / "world/global1km"                                    # Foehn stores (read only)
ERA5 = str(Z / "zarr/era5_global_daily/{}.zarr")
EXTRA = str(Z / "world/era5_extra_daily/{}.zarr")
GFAS = str(Z / "world/gfas_daily/{}.zarr")
CAMS = str(Z / "world/cams_forecast_daily/{}.zarr")
R0, R1, W = 24, 600, 1440                                    # build_coarse.py: 84 N .. 59.75 S
H = R1 - R0
D2017, D2023, D2025 = dt.date(2017, 1, 1), dt.date(2023, 1, 1), dt.date(2025, 1, 1)
NEW_DAYS = [D2023 + dt.timedelta(k) for k in range((D2025 - D2023).days)]          # 731
ALL_DAYS = [D2017 + dt.timedelta(k) for k in range((D2025 - D2017).days)]          # 2922
CAMS_X87_BEFORE = dt.date(2017, 9, 27)


def doy(d):
    return (d - dt.date(d.year, 1, 1)).days


# ------------------------------------------------------------------ verbatim from Foehn build_coarse.py
def crop(a):
    """(..., 721, 1440) lat 90 -> -90, lon 0 -> 359.75  ->  (..., 576, 1440) 84 N .. 59.75 S, lon -180 .. 179.75."""
    a = a[..., R0:R1, :]
    return np.concatenate([a[..., 720:], a[..., :720]], axis=-1)


def extra_raw(x):
    return np.stack([np.log(np.maximum(x[0], 10.0)), np.log(np.maximum(x[1], 10.0)), np.log(np.maximum(x[2], 10.0)),
                     np.log1p(np.maximum(x[3], 0.0) * 1000.0), x[4], np.maximum(x[5], 0.0) / 1e6, x[6]])


# ------------------------------------------------------------------ verbatim from Foehn build_transport.py
def to025(a, lon0):
    """(..., 1800, 3600) 0.1 deg field, rows 89.95 -> -89.95, first column centred at lon0 + 0.05 ->
    (..., 576, 1440) area means over the 0.25 deg cells centred on 84 - 0.25 k, -180 + 0.25 j."""
    a = np.asarray(a, np.float32)
    if abs(lon0 + 180.0) > 1e-3:
        a = np.concatenate([a[..., 1800:], a[..., :1800]], axis=-1)
    f = np.repeat(np.repeat(a, 4, -2), 4, -1)
    f = np.concatenate([f[..., -5:], f, f[..., :5]], axis=-1)
    r0 = int(round((90 - 84) / 0.025)) - 5
    f = f[..., r0:r0 + 10 * H, :10 * W]
    return f.reshape(*f.shape[:-2], H, 10, W, 10).mean((-3, -1))


_ST = {}
_Z = {}


def met_day(d):
    """(37, 576, 1440) float16 of day d, exactly as build_coarse.py's day loop"""
    if not _ST:
        st = np.load(G / "stats.npz")
        _ST.update(em=st["era5_mean"][:, None, None], es=st["era5_std"][:, None, None],
                   xm=st["extra_mean"][:, None, None], xs=st["extra_std"][:, None, None])
    import zarr
    key = ("met", d.year)
    if key not in _Z:
        _Z[key] = (zarr.open(ERA5.format(d.year), mode="r"), zarr.open(EXTRA.format(d.year), mode="r"))
    ez, xz = _Z[key]
    k = doy(d)
    e = (crop(np.asarray(ez[k], np.float32)) - _ST["em"]) / _ST["es"]
    x = np.nan_to_num((crop(extra_raw(np.asarray(xz[k], np.float32))) - _ST["xm"]) / _ST["xs"], nan=0.0)
    return np.concatenate([e, x]).astype(np.float16)


def fires_day(d):
    """(3, 576, 1440) float16 of day d, exactly as build_transport.py 'fires'"""
    import zarr
    key = ("gfas", d.year)
    if key not in _Z:
        _Z[key] = zarr.open(GFAS.format(d.year), mode="r")
    z = _Z[key]
    g = to025(np.asarray(z[doy(d), [0, 3, 4]], np.float32), 0.0)
    x = np.stack([np.log10(np.maximum(g[0], 0) + 1e-14), np.log10(np.maximum(g[1], 0) + 1e-6), np.log10(np.maximum(g[2], 0) + 1.0)])
    x = (x - np.array([-13.5, -5.5, 0.1])[:, None, None]) / np.array([1.0, 1.0, 0.6])[:, None, None]
    return x.astype(np.float16)


# ------------------------------------------------------------------ CAMS regrid (verbatim from build_smoke_data.py)
H25, W25 = 576, 1440


def regrid_weights():
    lat = 84.0 - 0.25 * np.arange(H25)
    lon = -180.0 + 0.25 * np.arange(W25)
    fi = (90.0 - lat) / 0.4
    i0 = np.floor(fi).astype(np.int64)
    wi = fi - i0
    fj = (lon % 360.0) / 0.4
    jf = np.floor(fj)
    j0 = jf.astype(np.int64) % 900
    return i0, wi, j0, (j0 + 1) % 900, fj - jf


def regrid(src, wts):
    i0, wi, j0, j1, wj = wts
    L = np.log1p(np.maximum(src, 0.0).astype(np.float64))
    top = L[i0][:, j0] * (1 - wj) + L[i0][:, j1] * wj
    bot = L[i0 + 1][:, j0] * (1 - wj) + L[i0 + 1][:, j1] * wj
    return np.expm1(top * (1 - wi)[:, None] + bot * wi[:, None])


_WTS = []


def cams_src(d):
    """0.4 deg day-0 mean PM2.5 of the 00 UTC run of day d (x 8/7 before 2017-09-27), float32 (451, 900)"""
    import zarr
    key = ("cams", d.year)
    if key not in _Z:
        z = zarr.open_array(CAMS.format(d.year), mode="r")
        at = dict(z.attrs)
        assert at["variables"][0] == "pm2p5" and at["lead_days"][0] == 0 and at["lat_first_last_n"] == [90.0, -90.0, 451], at
        assert at["lon_first_last_n"][0] == 0.0 and at["lon_first_last_n"][2] == 900, at
        assert sorted(at["done_months"]) == list(range(1, 13)), (d.year, at["done_months"])
        _Z[key] = z
    src = np.asarray(_Z[key][doy(d), 0, 0], np.float32)
    assert src.shape == (451, 900) and np.isfinite(src).all() and src.min() >= 0, d
    if d < CAMS_X87_BEFORE:
        src = src * np.float32(8.0 / 7.0)
    return src


def cams_day(d):
    if not _WTS:
        _WTS.append(regrid_weights())
    return regrid(cams_src(d), _WTS[0]).astype(np.float16)


# ------------------------------------------------------------------ .npy helpers
def npy_header(path):
    with open(path, "rb") as f:
        ver = np.lib.format.read_magic(f)
        shape, fo, dtype = (np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0)(f)
        return tuple(shape), np.dtype(dtype), f.tell()


def create(path, shape, dtype):
    """preallocated .npy (sparse), so that workers can pwrite rows"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.lib.format.write_array_header_1_0(f, {"descr": np.dtype(dtype).str, "fortran_order": False, "shape": tuple(shape)})
        hdr = f.tell()
        f.truncate(hdr + int(np.prod(shape)) * np.dtype(dtype).itemsize)


def write_row(path, i, block):
    shape, dtype, hdr = npy_header(path)
    block = np.ascontiguousarray(block, dtype=dtype)
    assert block.shape == shape[1:], (path, block.shape, shape)
    mv = memoryview(block.reshape(-1).view(np.uint8))
    n = len(mv)
    fd = os.open(path, os.O_WRONLY)
    try:
        done = 0
        while done < n:
            done += os.pwrite(fd, mv[done:done + (1 << 30)], hdr + i * n + done)
    finally:
        os.close(fd)


def read_row(path, i):
    shape, dtype, hdr = npy_header(path)
    n = int(np.prod(shape[1:]))
    return np.fromfile(path, dtype=dtype, count=n, offset=hdr + i * n * dtype.itemsize).reshape(shape[1:])


def update_manifest(key, value):
    p = OUT / "manifest_parts" / f"{key}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, indent=1, default=str))
    os.replace(tmp, p)


# ------------------------------------------------------------------ commands
def _work(job):
    kind, path, i, d = job
    fn = {"met": met_day, "fires": fires_day, "cams": cams_day}[kind]
    x = fn(d)
    xf = x.astype(np.float32)
    fin = bool(np.isfinite(xf).all())
    write_row(Path(path), i, x)
    return i, fin


def cmd_build(kind, procs):
    if kind == "cams":
        path, dl, shape = OUT / "coarse/cams_pm25_025.npy", ALL_DAYS, (len(ALL_DAYS), H25, W25)
    else:
        path = OUT / f"coarse/{kind}_2023_2024.npy"
        dl = NEW_DAYS
        shape = (len(dl), 37 if kind == "met" else 3, H, W)
    if not path.exists():
        create(path, shape, np.float16)
    assert npy_header(path)[0] == shape
    t0 = time.time()
    bad = []
    with mp.get_context("fork").Pool(procs) as pool:
        for n, (i, fin) in enumerate(pool.imap_unordered(_work, [(kind, str(path), i, d) for i, d in enumerate(dl)], chunksize=4)):
            if not fin:
                bad.append(str(dl[i]))
            if n % 100 == 0:
                print(f"{kind}: {n + 1}/{len(dl)} rows, {time.time() - t0:.0f}s", flush=True)
    assert not bad, f"{kind}: non-finite rows {bad[:10]}"
    print(f"{kind}: {len(dl)} rows written in {time.time() - t0:.0f}s -> {path}", flush=True)
    update_manifest(f"build_{kind}", {"file": str(path.relative_to(OUT)), "shape": shape, "dtype": "float16",
                                      "days": [str(dl[0]), str(dl[-1]), len(dl)], "all_finite": True,
                                      "seconds": round(time.time() - t0), "built": time.strftime("%Y-%m-%d %H:%M")})


def cmd_nonreg(_a):
    """bit equality of this code's rows with the Foehn met.npy / fires.npy rows (the reviewer's open issue)"""
    res = {}
    ok = True
    for d in (dt.date(2022, 12, 31), dt.date(2017, 1, 1), dt.date(2020, 6, 15)):
        k = (d - D2017).days
        for name, fn in (("met", met_day), ("fires", fires_day)):
            ref = read_row(G / f"{name}.npy", k)
            new = fn(d)
            eq = bool(np.array_equal(ref.view(np.uint16), new.view(np.uint16)))
            nd = int((ref.view(np.uint16) != new.view(np.uint16)).sum())
            diff = float(np.nanmax(np.abs(ref.astype(np.float32) - new.astype(np.float32)))) if nd else 0.0
            res[f"{name}_{d}"] = {"foehn_row": k, "bit_identical": eq, "n_differing_values": nd, "max_abs_diff": diff,
                                  "n_values": int(ref.size)}
            ok &= eq
            print(f"non-regression {name} {d} (Foehn row {k}): bit_identical={eq}, differing {nd}/{ref.size}, max |diff| {diff}", flush=True)
    # CAMS regrid vs the smoke store (same code, built 2026-10-23): rows 2020-09-30 .. 2021-03-31
    sm = ROOT / "data/processed/smoke/coarse/cams_pm25_025_smoke.npy"
    cams_path = OUT / "coarse/cams_pm25_025.npy"
    if sm.exists() and cams_path.exists():
        n_eq, n = 0, 0
        for t in range(0, 183, 7):
            d = dt.date(2020, 9, 30) + dt.timedelta(t)
            a, b = read_row(sm, t), read_row(cams_path, (d - D2017).days)
            n_eq += bool(np.array_equal(a.view(np.uint16), b.view(np.uint16)))
            n += 1
        res["cams_vs_smoke_store"] = {"days_checked": n, "bit_identical": n_eq}
        print(f"CAMS regrid vs smoke store: {n_eq}/{n} sampled days bit-identical", flush=True)
        ok &= n_eq == n
    res["ok"] = bool(ok)
    update_manifest("nonregression", res)
    if not ok:
        sys.exit("NON-REGRESSION FAILED")


def split_of(d):
    """PROTOCOL 1.3 split A codes: 0 other, 1 train, 2 buffer, 3 validation, 4 test, 5 robustness"""
    if dt.date(2019, 7, 11) <= d <= dt.date(2022, 12, 31):
        return 1
    if dt.date(2023, 1, 1) <= d <= dt.date(2023, 1, 7):
        return 2
    if dt.date(2023, 1, 8) <= d <= dt.date(2023, 12, 31):
        return 3
    if d.year == 2024:
        return 4
    if dt.date(2017, 1, 2) <= d <= dt.date(2019, 7, 1) and (d - dt.date(2017, 1, 2)).days % 5 == 0:
        return 5
    return 0


def cmd_days(_a):
    dates = np.array([np.datetime64(d) for d in ALL_DAYS])
    split = np.array([split_of(d) for d in ALL_DAYS], np.int8)
    old = np.array([d < D2023 for d in ALL_DAYS])
    met_store = np.where(old, 0, 1).astype(np.int8)          # 0 Foehn met.npy / fires.npy, 1 coarse/*_2023_2024.npy
    met_row = np.array([(d - D2017).days if d < D2023 else (d - D2023).days for d in ALL_DAYS], np.int32)
    cams_row = np.arange(len(ALL_DAYS), dtype=np.int32)
    t6_year = np.array([d.year for d in ALL_DAYS], np.int16)
    t6_row = np.array([doy(d) for d in ALL_DAYS], np.int16)
    t6_valid = np.array([d > D2017 for d in ALL_DAYS])         # target 2017-01-01 has no FLEXGREEN start day
    np.savez(OUT / "coarse/days.npz", date=dates, split=split, split_names=np.array(["other", "train", "buffer", "val", "test", "robust"]),
             met_store=met_store, met_row=met_row, fires_store=met_store, fires_row=met_row, cams_row=cams_row,
             t6_year=t6_year, t6_row=t6_row, t6_valid=t6_valid,
             met_store_paths=np.array([str(G / "met.npy"), str(OUT / "coarse/met_2023_2024.npy")]),
             fires_store_paths=np.array([str(G / "fires.npy"), str(OUT / "coarse/fires_2023_2024.npy")]))
    cnt = {n: int((split == i).sum()) for i, n in enumerate(["other", "train", "buffer", "val", "test", "robust"])}
    print("days.npz:", cnt, flush=True)
    assert cnt["train"] == 1270 and cnt["val"] == 358 and cnt["test"] == 366 and cnt["robust"] == 183, cnt
    update_manifest("days", {"file": "coarse/days.npz", "counts": cnt})


def cmd_check(_a):
    """CAMS store checks without any target data: 8/7 step, shared-node exactness on sampled days"""
    path = OUT / "coarse/cams_pm25_025.npy"
    shape, dtype, _ = npy_header(path)
    assert shape == (len(ALL_DAYS), H25, W25) and dtype == np.float16
    rr = np.array([r for r in range(H25) if abs(((84.0 - 0.25 * r) / 2.0) - round((84.0 - 0.25 * r) / 2.0)) < 1e-9])
    cc = np.arange(0, W25, 8)
    si = np.rint((90.0 - (84.0 - 0.25 * rr)) / 0.4).astype(int)
    sj = np.rint(((-180.0 + 0.25 * cc) % 360.0) / 0.4).astype(int) % 900
    err = 0.0
    lw = np.cos(np.deg2rad(84.0 - 0.25 * np.arange(H25)))[:, None]
    gm = {}
    for t in range(0, len(ALL_DAYS), 37):
        d = ALL_DAYS[t]
        src = cams_src(d)[si][:, sj].astype(np.float64)
        reg = read_row(path, t)
        err = max(err, float((np.abs(reg[rr][:, cc].astype(np.float64) - src) / np.maximum(src, 1e-3)).max()))
        gm[str(d)] = round(float((reg.astype(np.float64) * lw).sum() / (lw.sum() * W25)), 3)
    yearly = {}
    for y in range(2017, 2025):
        v = [g for k, g in gm.items() if k.startswith(str(y))]
        yearly[y] = round(float(np.mean(v)), 3)
    res = {"shared_node_max_rel_err": err, "shared_node_ok": err < 1e-3, "area_mean_ug_m3_by_year_sampled": yearly,
           "x8_7_applied_before": str(CAMS_X87_BEFORE)}
    print(json.dumps(res, indent=1), flush=True)
    update_manifest("cams_check", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["met", "fires", "cams", "nonreg", "days", "check"])
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    (OUT / "coarse").mkdir(parents=True, exist_ok=True)
    if a.cmd in ("met", "fires", "cams"):
        cmd_build(a.cmd, a.procs)
    else:
        {"nonreg": cmd_nonreg, "days": cmd_days, "check": cmd_check}[a.cmd](a)
