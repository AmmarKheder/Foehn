#!/usr/bin/env python3
"""ERA5 3-D FLEXGREEN driving fields (PLAN.md 2.7 item 4) from ARCO-ERA5 (the WeatherBench2 ERA5 archive stops on
2023-01-10), in the exact format of Foehn's dl_wb2.py era5(): wb2/era5_B_<year>_<month>.npy (days, 4, 53, 181, 360)
float16, 00/06/12/18 UTC, channels u, v, w (Pa/s), t (deg C), q (g/kg), z (km) at 1000/925/850/700/600/500/400/300 hPa,
then u10, v10, t2m (deg C), sp (hPa), tp6h (mm, precipitation of the 6 h ending at the step = sum of the 6 ARCO hourly
accumulations ending at the step), 1 deg box means (to_1deg, verbatim) of the 0.25 deg points.

  python dl_era5_B_arco.py <year> <month> [--out DIR] [--threads 32] [--limit N]
  python dl_era5_B_arco.py validate      # rebuild 2022-12 into a temporary file and compare with the WB2 era5_B_2022_12
Output: data/processed/global/wb2/ (one file per month). World venv (xarray, zarr, gcsfs-free https access)."""
import argparse
import calendar
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
OUT = ROOT / "data/processed/global/wb2"
WB2_FOEHN = Path("/scratch/project_462001804/ammar/eccv/data/world/global1km/wb2")
ARCO = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
# verbatim from Foehn dl_wb2.py
L8 = [1000, 925, 850, 700, 600, 500, 400, 300]
V3 = [("u_component_of_wind", 1.0, 0.0), ("v_component_of_wind", 1.0, 0.0), ("vertical_velocity", 1.0, 0.0),
      ("temperature", 1.0, -273.15), ("specific_humidity", 1000.0, 0.0), ("geopotential", 1.0 / 9806.65, 0.0)]
V2 = [("10m_u_component_of_wind", 1.0, 0.0), ("10m_v_component_of_wind", 1.0, 0.0), ("2m_temperature", 1.0, -273.15),
      ("surface_pressure", 0.01, 0.0), ("total_precipitation_6hr", 1000.0, 0.0)]
CHANNELS = [f"{v.split('_')[0]}{p}" for v, _, _ in V3 for p in L8] + ["u10", "v10", "t2m", "sp", "tp6h"]
TRAP = np.array([0.125, 0.25, 0.25, 0.25, 0.125], np.float32)


def to_1deg(a, lat_desc):
    """(..., 721, 1440) 0.25 deg point values, lon -180..179.75, lat descending (90..-90) -> (..., 181, 360)."""
    a = a if lat_desc else a[..., ::-1, :]
    p = np.concatenate([a[..., :, -2:], a, a[..., :, :2]], -1)
    a = sum(w * p[..., :, k:k + a.shape[-1]] for k, w in enumerate(TRAP))[..., :, ::4]
    p = np.concatenate([a[..., 2:0:-1, :], a, a[..., -2:-4:-1, :]], -2)
    return sum(w * p[..., k:k + 721, :] for k, w in enumerate(TRAP))[..., ::4, :]


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
        assert float(ds.latitude[0]) == 90.0 and float(ds.longitude[0]) == 0.0 and ds.sizes["longitude"] == 1440
        lev = [int(x) for x in ds.level.values]
        self.lev_idx = [lev.index(p) for p in L8]
        self.times = ds.time.values
        self.t0 = self.times[0]
        assert np.all(np.diff(self.times[:48]) == np.timedelta64(1, "h"))

    def tindex(self, t64):
        i = int((t64 - self.t0) / np.timedelta64(1, "h"))
        assert self.times[i] == t64, (t64, self.times[i])
        return i

    def read(self, var, ti):
        """one hourly field: 3-D vars -> (8, 721, 1440) at L8 (the chunk holds all 37 levels), 2-D -> (721, 1440);
        float32, longitudes rolled to -180 .. 179.75"""
        def get():
            a = np.asarray(self.ds[var].isel(time=ti).values, np.float32)
            return a[self.lev_idx] if a.ndim == 3 else a
        a = retry(get)
        return np.concatenate([a[..., 720:], a[..., :720]], axis=-1)


def build_day(arco, ex, day, tp_mode="ending"):
    """(4, 53, 181, 360) float32 of one day; tp_mode 'ending': tp6h(t) = sum of hourly tp at t-5 .. t (ERA5 hourly
    accumulations end at their valid time); 'starting': t+1 .. t+6 (only for the convention check)"""
    d0 = np.datetime64(f"{day}T00")
    steps = [d0 + np.timedelta64(6 * k, "h") for k in range(4)]
    tasks = []
    for k, t in enumerate(steps):
        ti = arco.tindex(t)
        for v, _, _ in V3:
            tasks.append(("3d", k, v, ti))
        for v, _, _ in V2[:4]:
            tasks.append(("2d", k, v, ti))
        hrs = range(ti - 5, ti + 1) if tp_mode == "ending" else range(ti + 1, ti + 7)
        for h in hrs:
            tasks.append(("tp", k, "total_precipitation", h))
    res = list(ex.map(lambda q: arco.read(q[2], q[3]), tasks))
    out = np.zeros((4, len(CHANNELS), 181, 360), np.float32)
    for k in range(4):
        got = {(q[0], q[2]): r for q, r in zip(tasks, res) if q[1] == k and q[0] != "tp"}
        tp = sum(r for q, r in zip(tasks, res) if q[1] == k and q[0] == "tp")
        ch = [got[("3d", v)][j] * sc + off for v, sc, off in V3 for j in range(len(L8))]
        ch += [got[("2d", v)] * sc + off for v, sc, off in V2[:4]]
        ch.append(tp * 1000.0)
        out[k] = to_1deg(np.stack(ch), True)
    return out


def month_days(year, month, limit=0):
    n = calendar.monthrange(year, month)[1]
    days = [f"{year}-{month:02d}-{d:02d}" for d in range(1, n + 1)]
    return days[:limit] if limit else days


def cmd_month(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    days = month_days(a.year, a.month, a.limit)
    path = out / f"era5_B_{a.year}_{a.month:02d}.npy"
    done_path = out / f".era5_B_{a.year}_{a.month:02d}.done"          # removed after the merge (inodes)
    if path.exists() and not a.overwrite:
        hdr = np.lib.format.open_memmap(path, mode="r")
        if hdr.shape[0] == len(days) and np.isfinite(np.asarray(hdr[-1, -1, 0], np.float32)).all() and (out / "manifest_wb2.jsonl").exists() \
                and f'"{path.name}"' in (out / "manifest_wb2.jsonl").read_text():
            print(f"{path.name} already complete", flush=True)
            return
    tmp = path.with_name(path.name + ".part")
    Bm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(days), 4, len(CHANNELS), 181, 360))
    arco = Arco()
    t0 = time.time()
    with ThreadPoolExecutor(a.threads) as ex:
        for n, d in enumerate(days):
            x = build_day(arco, ex, d)
            assert np.isfinite(x).all(), d
            Bm[n] = x.astype(np.float16)
            print(f"{d}  {time.time() - t0:.0f}s", flush=True)
    Bm.flush()
    del Bm
    os.replace(tmp, path)
    rec = {"file": path.name, "days": len(days), "first": days[0], "last": days[-1], "seconds": round(time.time() - t0),
           "source": ARCO, "tp6h": "sum of hourly tp at t-5 .. t", "built": time.strftime("%Y-%m-%d %H:%M")}
    with open(out / "manifest_wb2.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("done", rec, flush=True)


def cmd_validate(a):
    """2022-12 from ARCO vs the WeatherBench2 file used by every FLEXGREEN run so far: per channel max |diff| and the
    fraction of bit-identical f16 values; both tp6h conventions are tried on the first days"""
    ref = np.load(WB2_FOEHN / "era5_B_2022_12.npy", mmap_mode="r")
    days = month_days(2022, 12, a.limit)
    arco = Arco()
    res = {"days": len(days), "per_channel": {}}
    diff_max = np.zeros(len(CHANNELS))
    rel_max = np.zeros(len(CHANNELS))
    eq = np.zeros(len(CHANNELS))
    nval = 0
    tp_conv = {}
    t0 = time.time()
    with ThreadPoolExecutor(a.threads) as ex:
        for n, d in enumerate(days):
            x = build_day(arco, ex, d).astype(np.float16)
            r = np.asarray(ref[n])
            for c in range(len(CHANNELS)):
                xa, ra = x[:, c].astype(np.float32), r[:, c].astype(np.float32)
                ok = np.isfinite(ra)
                dd = np.abs(xa - ra)[ok]
                diff_max[c] = max(diff_max[c], float(dd.max()))
                rel_max[c] = max(rel_max[c], float((dd / np.maximum(np.abs(ra[ok]), 1e-3)).max()))
                eq[c] += float((x[:, c].view(np.uint16) == r[:, c].view(np.uint16)).sum())
            nval += x[:, 0].size
            if n < 2:
                xs = build_day(arco, ex, d, tp_mode="starting").astype(np.float32)
                tp_conv[d] = {"ending_max_abs_diff_mm": float(np.abs(x[:, -1].astype(np.float32) - r[:, -1].astype(np.float32)).max()),
                              "starting_max_abs_diff_mm": float(np.abs(xs[:, -1] - r[:, -1].astype(np.float32)).max()),
                              "wb2_tp6h_max_mm": float(r[:, -1].astype(np.float32).max())}
            print(f"{d}  {time.time() - t0:.0f}s  max|diff| u1000 {diff_max[0]:.4g} z300 {diff_max[47]:.4g} tp {diff_max[-1]:.4g}", flush=True)
    for c, name in enumerate(CHANNELS):
        res["per_channel"][name] = {"max_abs_diff": float(diff_max[c]), "max_rel_diff": float(rel_max[c]),
                                    "frac_bit_identical_f16": float(eq[c] / nval)}
    res["tp6h_convention_check"] = tp_conv
    res["frac_bit_identical_all"] = float(eq.sum() / (nval * len(CHANNELS)))
    # f16 precision: one ulp at |x| is 2^(floor(log2|x|) - 10): require every channel within 2 ulp of its value range
    res["ok"] = bool(all(rel_max[c] < 4e-3 or diff_max[c] < 1e-3 for c in range(len(CHANNELS))))
    p = ROOT / "data/processed/global/manifest_parts/wb2_validation.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_channel"}, indent=1), flush=True)
    print("worst channels:", sorted(res["per_channel"].items(), key=lambda kv: -kv[1]["max_rel_diff"])[:5], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("year")
    ap.add_argument("month", type=int, nargs="?", default=1)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.year == "validate":
        cmd_validate(a)
    else:
        a.year = int(a.year)
        cmd_month(a)
