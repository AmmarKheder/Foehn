#!/usr/bin/env python3
"""RT 1 km, step 1 (CPU, world venv, internet): the real-time conditioning inputs of one RT forecast day d that do
not need torch, for the 5 lead days L = 1..5 (valid day d + L - 1, forecast hours 24(L-1) .. 24L after 00 UTC of d).

Written to realtime_1km/work/<d>/ (RT's own files are only READ):
  met_ifs.npy     (5, 37, 576, 1440) float16  encoder channels 1-37 as Foehn met.npy, from the IFS 00 UTC open-data
                  run of d instead of ERA5: 30 daily means of the instants 00/06/12/18 UTC of the valid day (ERA5 used
                  the same 4 instants), normalised twice exactly as met.npy (download_era5_global.py constants, then
                  stats.npz); extras: tp and ssrd = differences of the run's accumulations over the valid day, d2m and
                  tcc = means of the 3-hourly steps; channels 31-33 (log BLH mean/max/min) are NaN here: open data has
                  no BLH, channels.py fills them (substitute of blh.py / calib_blh.py).
  pm_rt025.npy  (members, 5, 576, 1440) float16  RT PM2.5 daily mean of each member (ug m-3), trapezoid of the
                  instants 24(L-1), 24L-12, 24L h (the 0 h instant = the CAMS analysis at t0, RT's own initial
                  state), bilinear in log1p with periodic longitude from RT's 0.4 deg grid to the 0.25 deg encoder
                  grid (the regrid of build_global_coarse.py, here from lat 84 instead of 90): replaces encoder ch 0.
  pm_prev025.npy  (576, 1440) float16  PM2.5 of day d - 1 (for the T6 transport of lead day 1): trapezoid of the CAMS
                  analyses 00 UTC d-1, 12 UTC d-1, 00 UTC d (RT's cams.npy of d - 1 and d).
  prep.json       sources, lead days, missing-channel substitutes, timings.
IFS fields that RT's data/<d>/ifs/<sss>.grib2 already holds are read there; the others (msl, 2d, tcc, ssrd; or
everything if RT's GRIBs were pruned) are fetched by byte ranges into work/<d>/ifs/<sss>.grib2 (Google mirror of
ECMWF open data first, then ECMWF, then AWS).
    python prep.py 2026-10-26            python prep.py --latest      (prints the newest complete RT day)"""
import argparse
import datetime as dt
import json
import os
import random
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RT = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km/realtime")
sys.path.insert(0, str(RT / "pylib"))                                  # eccodes wheels (read only)
RT_DATA, RT_OUT = RT / "data", RT / "out"
WORK = HERE / "work"
G = Path("/scratch/project_462001804/ammar/eccv/data/world/global1km")
L5 = [1000, 925, 850, 700, 500]
NLEAD = 5
BASES = ["https://storage.googleapis.com/ecmwf-open-data", "https://data.ecmwf.int/forecasts",
         "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"]
# the fixed normalisation of eccv/data/zarr/era5_global_daily (aria/scripts/download_era5_global.py), applied before
# the stats.npz standardisation of build_coarse.py: channels u10 v10 t2m msl sp, t u v q z at 1000 925 850 700 500
ERA5_MEANS = np.array([0., 0., 280., 101325., 97000., *([260.] * 5 + [0.] * 5 + [0.] * 5 + [0.003] * 5 + [50000.] * 5)],
                      np.float32)
ERA5_STDS = np.array([10., 10., 20., 1500., 5000., *([30.] * 5 + [15.] * 5 + [10.] * 5 + [0.004] * 5 + [40000.] * 5)],
                     np.float32)
SFC_BASE = ["10u", "10v", "2t", "msl", "sp"]
PL_BASE = ["t", "u", "v", "q", "z"]


def log(msg):
    print(f"{dt.datetime.utcnow():%H:%M:%S}Z {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ days
def complete(d):
    """RT day d usable: forecast, its inputs, and the previous day's CAMS analysis and IFS fields"""
    p = d - dt.timedelta(1)
    need = [RT_OUT / str(d) / "pred.npy", RT_DATA / str(d) / "cams.npy", RT_DATA / str(d) / "hres_B.npy",
            RT_DATA / str(d) / "fires.npy", RT_DATA / str(p) / "cams.npy", RT_DATA / str(p) / "hres_B.npy",
            RT_DATA / str(p) / "fires.npy"]
    return all(x.exists() for x in need)


def latest():
    days = sorted(dt.date.fromisoformat(x.name) for x in RT_OUT.iterdir() if x.is_dir() and x.name[:2] == "20")
    for d in days[::-1]:
        if complete(d):
            return d
    raise SystemExit("no complete RT day on disk")


# ------------------------------------------------------------------------------------------------ GRIB
def read_grib(p, keep=None):
    """{(shortName, level): (721, 1440) float32} on lat 90..-90, lon -180..179.75 (as RT's build.read_grib_step)"""
    import eccodes
    out = {}
    with open(p, "rb") as f:
        while True:
            h = eccodes.codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                name = eccodes.codes_get(h, "shortName")
                lev = int(eccodes.codes_get(h, "level")) if eccodes.codes_get(h, "typeOfLevel") == "isobaricInhPa" else 0
                if keep is not None and (name, lev) not in keep and not (name == "gh" and ("z", lev) in keep):
                    continue
                ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
                la0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
                lo0 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
                jpos = eccodes.codes_get(h, "jScansPositively")
                v = eccodes.codes_get_values(h).reshape(nj, ni).astype(np.float32)
            finally:
                eccodes.codes_release(h)
            if jpos:
                v = v[::-1]
                la0 = -la0
            assert (nj, ni) == (721, 1440) and abs(la0 - 90) < 1e-6, (name, nj, ni, la0)
            shift = int(round(((lo0 + 180.0) % 360.0) / 0.25))
            out[(name, lev)] = np.roll(v, shift, axis=1) if shift else v
    for k in list(out):                                       # geopotential height (gpm) -> geopotential (exact)
        if k[0] == "gh" and ("z", k[1]) not in out:
            out[("z", k[1])] = out.pop(k) * 9.80665
    return out


def http(url, rng=None, timeout=120):
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch(d, s, want, dst):
    """byte-range download of the fields `want` {(param, level)} of step s of the 00 UTC run of d into dst"""
    ymd = f"{d:%Y%m%d}"
    last = None
    for n in range(9):
        base = BASES[n % len(BASES)]
        stem = f"{base}/{ymd}/00z/ifs/0p25/oper/{ymd}000000-{s}h-oper-fc"
        try:
            recs = [json.loads(l) for l in http(stem + ".index").decode().splitlines() if l.strip()]
            sel = []
            for r in recs:
                lev = int(r.get("levelist", 0)) if r.get("levtype") == "pl" else 0
                if r.get("levtype") not in ("pl", "sfc"):
                    continue
                if (r["param"], lev) in want:
                    sel.append(r)
            have = {(r["param"], int(r.get("levelist", 0)) if r.get("levtype") == "pl" else 0) for r in sel}
            for (p, lev) in want:                                              # z missing on old dates: take gh
                if p == "z" and (p, lev) not in have:
                    sel += [r for r in recs if r.get("levtype") == "pl" and r["param"] == "gh" and int(r.get("levelist", -1)) == lev]
            got = {(("z" if r["param"] == "gh" else r["param"]), int(r.get("levelist", 0)) if r.get("levtype") == "pl" else 0) for r in sel}
            miss = set(want) - got
            assert not miss, f"step {s}: not in the index: {sorted(miss)}"
            sel.sort(key=lambda r: r["_offset"])
            spans = []
            for r in sel:
                a, b = r["_offset"], r["_offset"] + r["_length"] - 1
                if spans and a - spans[-1][1] < 2_000_000:
                    spans[-1][1] = max(spans[-1][1], b)
                else:
                    spans.append([a, b])
            tmp = Path(str(dst) + ".part")
            with open(tmp, "wb") as f:
                for a, b in spans:
                    f.write(http(stem + ".grib2", (a, b)))
            os.replace(tmp, dst)
            return base
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(120, 10 * 2 ** n) + random.uniform(0, 5))
    raise RuntimeError(f"IFS {d} step {s}: {last!r}")


def step_fields(d, s, want, work, report):
    """the fields `want` of step s: from RT's GRIB when it has them, the rest fetched into work/ifs/<sss>_<n>.grib2"""
    out = {}
    files = [RT_DATA / str(d) / "ifs" / f"{s:03d}.grib2"] + sorted((work / "ifs").glob(f"{s:03d}_*.grib2"))
    for f in files:
        miss = {k for k in want if k not in out}
        if not miss:
            break
        if f.exists():
            out.update(read_grib(f, miss))
    miss = {k for k in want if k not in out}
    if miss:
        (work / "ifs").mkdir(parents=True, exist_ok=True)
        dst = work / "ifs" / f"{s:03d}_{len(list((work / 'ifs').glob(f'{s:03d}_*.grib2')))}.grib2"
        report["fetch_source"] = fetch(d, s, miss, dst)
        out.update(read_grib(dst, miss))
        report.setdefault("fetched", {}).setdefault(str(s), []).extend(sorted(f"{p}{lv or ''}" for p, lv in miss))
    assert all(k in out for k in want), (s, sorted(set(want) - set(out)))
    if ("tcc", 0) in out and float(out[("tcc", 0)].max()) > 1.5:          # GRIB2 open data: tcc in %, ERA5: 0-1
        out[("tcc", 0)] = out[("tcc", 0)] / 100.0
        report["tcc_percent_converted"] = True
    return out


# ------------------------------------------------------------------------------------------------ met channels
def build_met(d, work, report):
    st = np.load(G / "stats.npz")
    em, es = st["era5_mean"][:, None, None], st["era5_std"][:, None, None]
    xm, xs = st["extra_mean"][:, None, None], st["extra_std"][:, None, None]
    base_keys = [(p, 0) for p in SFC_BASE] + [(p, lv) for p in PL_BASE for lv in L5]
    out = np.full((NLEAD, 37, 576, 1440), np.nan, np.float16)
    acc = {}                                                     # tp, ssrd accumulations at 0, 24, .., 120 h

    def accum(s):
        if s not in acc:
            if s == 0:
                acc[s] = {"tp": np.zeros((721, 1440), np.float32), "ssrd": np.zeros((721, 1440), np.float32)}
            else:
                f = step_fields(d, s, {("tp", 0), ("ssrd", 0)}, work, report)
                acc[s] = {"tp": f[("tp", 0)], "ssrd": f[("ssrd", 0)]}
        return acc[s]
    for L in range(1, NLEAD + 1):
        s0 = 24 * (L - 1)
        t0 = time.time()
        b = np.zeros((30, 721, 1440), np.float64)
        for s in (s0, s0 + 6, s0 + 12, s0 + 18):
            f = step_fields(d, s, set(base_keys), work, report)
            b += np.stack([f[k] for k in base_keys])
        b /= 4.0
        dm = np.zeros((721, 1440), np.float64)
        tc = np.zeros((721, 1440), np.float64)
        for s in range(s0, s0 + 24, 3):
            f = step_fields(d, s, {("2d", 0), ("tcc", 0)}, work, report)
            dm += f[("2d", 0)]
            tc += f[("tcc", 0)]
        dm /= 8.0
        tc /= 8.0
        tp = accum(s0 + 24)["tp"] - accum(s0)["tp"]
        ss = accum(s0 + 24)["ssrd"] - accum(s0)["ssrd"]
        e = (b.astype(np.float32) - ERA5_MEANS[:, None, None]) / ERA5_STDS[:, None, None]
        e = (e[:, 24:600] - em) / es                              # rows 84 N .. 59.75 S; lon already -180 first
        x4 = np.stack([np.log1p(np.maximum(tp, 0.0) * 1000.0), dm.astype(np.float32), np.maximum(ss, 0.0) / 1e6,
                       tc.astype(np.float32)])[:, 24:600]
        x4 = np.nan_to_num((x4 - xm[3:]) / xs[3:], nan=0.0)
        out[L - 1, :30] = e.astype(np.float16)
        out[L - 1, 33:] = x4.astype(np.float16)
        log(f"met lead day {L} ({d + dt.timedelta(L - 1)}): t2m {float(b[2].mean()) - 273.15:.2f} C, "
            f"tp {float(tp.mean()) * 1000:.2f} mm/d, ssrd {float(ss.mean()) / 1e6:.2f} MJ/m2, "
            f"std-channel |mean| max {float(np.abs(out[L - 1, :30].astype(np.float32).mean((1, 2))).max()):.2f} "
            f"({time.time() - t0:.0f}s)")
    np.save(work / "met_ifs.npy", out)
    report["met_channel_means_std_units"] = [[round(float(v), 3) for v in out[L, :].astype(np.float32).mean((1, 2))]
                                             for L in range(NLEAD)]


# ------------------------------------------------------------------------------------------------ PM2.5
def regrid_weights(lat0_src):
    """0.4 deg source with first row at lat0_src (descending), lon -180 first (RT) -> 0.25 deg encoder points"""
    lat = 84.0 - 0.25 * np.arange(576)
    lon = -180.0 + 0.25 * np.arange(1440)
    fi = (lat0_src - lat) / 0.4
    i0 = np.floor(fi).astype(np.int64)
    wi = fi - i0
    fj = (lon + 180.0) / 0.4
    jf = np.floor(fj)
    j0 = jf.astype(np.int64) % 900
    return i0, wi, j0, (j0 + 1) % 900, fj - jf


def regrid(src, wts):
    """build_global_coarse.regrid: bilinear in log1p, periodic lon"""
    i0, wi, j0, j1, wj = wts
    L = np.log1p(np.maximum(src, 0.0).astype(np.float64))
    i1 = np.minimum(i0 + 1, src.shape[0] - 1)
    top = L[i0][:, j0] * (1 - wj) + L[i0][:, j1] * wj
    bot = L[i1][:, j0] * (1 - wj) + L[i1][:, j1] * wj
    return np.expm1(top * (1 - wi)[:, None] + bot * wi[:, None])


def build_pm(d, work, report):
    pred = np.load(RT_OUT / str(d) / "pred.npy").astype(np.float32)          # (M, 10, 361, 900) ug m-3, 12..120 h
    cams = np.load(RT_DATA / str(d) / "cams.npy")[:, 1].astype(np.float32)   # PM2.5 at t0 - 12 h, t0 (361, 900)
    camsp = np.load(RT_DATA / str(d - dt.timedelta(1)) / "cams.npy")[1, 1].astype(np.float32)
    M = pred.shape[0]
    assert pred.shape[1:] == (10, 361, 900), pred.shape
    wts = regrid_weights(84.0)
    inst = lambda h: cams[1][None].repeat(M, 0) if h == 0 else pred[:, h // 12 - 1]
    out = np.zeros((M, NLEAD, 576, 1440), np.float16)
    for L in range(1, NLEAD + 1):
        a, b, c = 24 * (L - 1), 24 * L - 12, 24 * L
        day = np.maximum(0.25 * inst(a) + 0.5 * inst(b) + 0.25 * inst(c), 0.0)
        for m in range(M):
            out[m, L - 1] = regrid(day[m], wts).astype(np.float16)
        log(f"PM2.5 lead day {L}: RT daily mean (members) {float(day.mean()):.2f} ug m-3, member spread "
            f"{float(day.std(0).mean()):.2f}")
    np.save(work / "pm_rt025.npy", out)
    prev = np.maximum(0.25 * camsp + 0.5 * cams[0] + 0.25 * cams[1], 0.0)
    np.save(work / "pm_prev025.npy", regrid(prev, wts).astype(np.float16))
    report["members"] = M
    report["pm_daily_mean_ug"] = [round(float(out[:, L].astype(np.float32).mean()), 3) for L in range(NLEAD)]
    report["pm_prev_mean_ug"] = round(float(prev.mean()), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", nargs="?")
    ap.add_argument("--latest", action="store_true", help="print the newest complete RT day and exit")
    ap.add_argument("--keep", type=int, default=3, help="work/<day> directories kept (older ones are deleted)")
    a = ap.parse_args()
    if a.latest:
        print(latest())
        return
    d = dt.date.fromisoformat(a.date)
    assert complete(d), f"RT day {d} is incomplete (pred.npy, cams/hres_B/fires of {d} and {d - dt.timedelta(1)})"
    work = WORK / str(d)
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    om = json.loads((RT_OUT / str(d) / "meta.json").read_text())
    report = {"date": str(d), "lead_days": {L: str(d + dt.timedelta(L - 1)) for L in range(1, NLEAD + 1)},
              "lead_day_definition": "L = 1..5: valid day d + L - 1, forecast hours 24(L-1) .. 24L after 00 UTC of d",
              "rt_ckpt": om.get("ckpt"), "rt_inputs": om.get("inputs"),
              "fires": str(RT_DATA / str(d) / "fires.npy"), "fires_note": "GFAS of d - 1 (v1.4.2, injh), persisted",
              "hres_B": {"run_d": str(RT_DATA / str(d) / "hres_B.npy"),
                         "run_d-1": str(RT_DATA / str(d - dt.timedelta(1)) / "hres_B.npy")},
              "substitutes": {"ch0 CAMS day-0 mean": "RT member daily mean (trapezoid 00/12/24 UTC)",
                              "ch1-30 ERA5": "IFS open data, same 4 instants",
                              "ch31-33 log BLH": "FLEXGREEN mixing height from IFS 1 deg profiles, calibrated (blh_calib.json)",
                              "ch34 tp, ch36 ssrd": "IFS accumulations over the valid day (exact analogue)",
                              "ch35 d2m, ch37 tcc": "IFS 3-hourly means (ERA5: 24 hourly)",
                              "ch38-40 fires": "GFAS v1.4.2 of d - 1 persisted, injh in ch 40",
                              "ch41-46 T6": "FLEXGREEN lead 1 on IFS forecast winds, PM of the previous valid day from RT"}}
    build_pm(d, work, report)
    build_met(d, work, report)
    report["seconds"] = round(time.time() - t0)
    (work / "prep.json").write_text(json.dumps(report, indent=1, default=str))
    shutil.rmtree(work / "ifs", ignore_errors=True)                  # fetched GRIBs are no longer needed
    for old in sorted(p for p in WORK.iterdir() if p.is_dir())[:-a.keep]:
        shutil.rmtree(old, ignore_errors=True)
    log(f"prep {d} done in {report['seconds']} s -> {work}")


if __name__ == "__main__":
    main()
