#!/usr/bin/env python3
"""RT step 4c, the hindcast that validates the real-time inputs of 2026 (GFAS v1.4.2 with injh, ECMWF open data,
CAMS of 2026) before any publication: forecasts for a window of past start days, with and without fires, scored against
the CAMS analysis at the valid times.
  prepare <first> <last>   fetch + build every day of the window (sequential; the CAMS verification analyses of the
                           window + 5 days in one request -> verif/cams_pm25_<first>_<last>.npz)
  submit  <first> <last>   one rt_job.sh per prepared day (FLEXGREEN + forecast with fires), then predict --no-fires
  score   <first> <last>   RMSE and bias against the CAMS analysis by lead, globally and over the fire-affected cells,
                           forecast with fires vs without, and the CAMS operational forecast if fetched
    python hindcast.py prepare 2026-09-01 2026-09-20"""
import datetime as dt
import subprocess
import sys
import tempfile
import zipfile

import numpy as np

import common as C

VERIF = C.RT / "verif"


def days(first, last):
    d = C.parse_date(first); e = C.parse_date(last)
    while d <= e:
        yield d
        d += dt.timedelta(days=1)


def fetch_verif(first, last):
    """PM2.5 of the CAMS analysis (00 and 12 UTC) over [first, last + 5 days]: the truth of the Aurora protocol"""
    import xarray as xr
    from fetch import ads, nc_files
    VERIF.mkdir(exist_ok=True)
    e = C.parse_date(last) + dt.timedelta(days=5)
    dst = VERIF / f"cams_pm25_{first}_{e}.npz"
    if dst.exists():
        return dst
    tmp = VERIF / "verif.zip"
    ads().retrieve("cams-global-atmospheric-composition-forecasts",
                   {"type": ["forecast"], "leadtime_hour": ["0"], "date": [f"{first}/{e}"], "time": ["00:00", "12:00"],
                    "variable": ["particulate_matter_2.5um"], "data_format": "netcdf", "download_format": "zip"}, str(tmp))
    if True:
        f = nc_files(tmp)[0]
        with xr.open_dataset(f) as ds:
            v = ds["pm2p5"].squeeze(drop=True)
            tdim = [x for x in v.dims if x not in ("latitude", "longitude")]
            a = v.transpose(*tdim, "latitude", "longitude").values.astype(np.float32) * 1e9
            tcoord = "valid_time" if "valid_time" in ds.coords else "forecast_reference_time"
            times = np.asarray(ds[tcoord].values).reshape(-1)
            a = a.reshape(len(times), a.shape[-2], a.shape[-1])
            np.savez(dst, pm25=a.astype(np.float16), times=times.astype("datetime64[h]"), lat=ds["latitude"].values, lon=ds["longitude"].values)
    tmp.unlink()
    print(f"verification analyses: {dst.name}, {len(times)} times {str(times[0])[:13]} .. {str(times[-1])[:13]}", flush=True)
    return dst


def prepare(first, last):
    fetch_verif(first, last)
    for d in days(first, last):
        D = C.day_dir(d)
        if (D / "fires.npy").exists() and (D / "hres_B.npy").exists() and (D / "cams.npy").exists():
            print(d, "already built", flush=True); continue
        subprocess.run([sys.executable, "fetch.py", str(d), "--every", "5"], check=True, cwd=C.RT)
        subprocess.run([sys.executable, "build.py", str(d)], check=True, cwd=C.RT)


def submit(first, last):
    for d in days(first, last):
        O = C.out_dir(d); J = C.day_dir(d) / "journal.txt"
        if (O / "pred.npy").exists() and (O / "pred_nofires.npy").exists():
            continue
        if not (J.exists() and J.read_text().rstrip().endswith("build done")):      # prepare still running for this day
            print(d, "not built yet"); continue
        if subprocess.run(["squeue", "-h", "-u", C.os.environ["USER"], "-n", f"rt_{d}"], capture_output=True, text=True).stdout.strip():
            print(d, "job already queued"); continue
        r = subprocess.run(["sbatch", "--parsable", "--job-name", f"rt_{d}", "rt_job.sh", str(d), "--hindcast"], cwd=C.RT,
                           capture_output=True, text=True, check=True)
        print(d, "job", r.stdout.strip(), flush=True)


def crop_roll(a):
    """(..., 451, 900) lat 90..-90 lon 0..359.6 -> (..., 361, 900) lat 84..-60 lon -180..179.6 (build_cams_series.crop_roll)"""
    return np.roll(a[..., 15:376, :], 450, axis=-1)


def score(first, last):
    """ensemble-mean RMSE/bias against the CAMS analysis of the valid time, by lead: all cells, land cells, fire-affected
    cells; with fires vs without; and the CAMS operational forecast (leads 24/120 h) where data/<d>/cams_fc.npz exists"""
    vfs = sorted(VERIF.glob("cams_pm25_*.npz")); tidx = {}; parts = []
    for vf in vfs:                                                                  # every verification window fetched so far
        V = np.load(vf); ts = V["times"].astype("datetime64[h]"); a = V["pm25"]
        for i, t in enumerate(ts):
            if str(t) not in tidx:
                tidx[str(t)] = len(parts); parts.append(a[i])
    truth = np.stack(parts); vf = type("x", (), {"name": ", ".join(v.name for v in vfs) + f" ({len(parts)} times)"})()
    lat = 84.0 - 0.4 * np.arange(361); w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, 900))
    land = np.load(C.G / "cams_static.npz")["static"][0] > 0.5
    fm = np.zeros((361, 900), bool)                                                     # fire-affected cells over the window
    for d in days(first, last):
        f = C.day_dir(d) / "fires.npy"
        if not f.exists():
            continue
        x = np.load(f).astype(np.float32)[1]                                           # (576, 1440): 84 N.., -180..
        ii = np.clip(np.round((84.0 - lat) / 0.25).astype(int), 0, 575); jj = np.clip(np.round((-180 + 0.4 * np.arange(900) + 180) / 0.25).astype(int), 0, 1439)
        fm |= x[ii[:, None], jj[None, :]] > 0.0                                        # standardised log10 FRP > 0: FRP > 0.3 W m-2
    print(f"verification: {vf.name}; fire-affected 0.4 deg cells over the window: {fm.sum()} ({fm.mean():.2%}); land cells {land.sum()}")
    rm = lambda e, m: float(np.sqrt((w * e ** 2)[m].sum() / w[m].sum()))
    bi = lambda e, m: float((w * e)[m].sum() / w[m].sum())
    allm = np.ones_like(land)
    res, cams = {}, {}
    for d in days(first, last):
        O = C.out_dir(d); t0 = dt.datetime(d.year, d.month, d.day)
        for tag, fn in (("fires", "pred.npy"), ("no fires", "pred_nofires.npy")) + tuple((t, f"pred_{t}.npy") for t in TAGS):
            if not (O / fn).exists():
                continue
            P = np.load(O / fn).astype(np.float32).mean(0)                              # ensemble mean (10, 361, 900)
            for k in range(10):
                vt = np.datetime64(t0 + dt.timedelta(hours=12 * (k + 1)), "h")
                if str(vt) not in tidx:
                    continue
                e = P[k] - crop_roll(truth[tidx[str(vt)]].astype(np.float32))
                res.setdefault((tag, k), []).append((rm(e, allm), rm(e, land), rm(e, fm), bi(e, fm), bi(e, land)))
        t00 = np.datetime64(t0, "h")                                                     # persistence: the analysis at t0 kept for every lead
        if str(t00) in tidx:
            P0 = crop_roll(truth[tidx[str(t00)]].astype(np.float32))
            for k in range(10):
                vt = np.datetime64(t0 + dt.timedelta(hours=12 * (k + 1)), "h")
                if str(vt) in tidx:
                    e = P0 - crop_roll(truth[tidx[str(vt)]].astype(np.float32))
                    res.setdefault(("persistence", k), []).append((rm(e, allm), rm(e, land), rm(e, fm), bi(e, fm), bi(e, land)))
        cf = C.day_dir(d) / "cams_fc.npz"
        if cf.exists():
            z = np.load(cf); a = z["pm25"]; lds = list(z["leads"])
            for j, k in [(lds.index(L), int(L) // 12 - 1) for L in lds if int(L) in (24, 120)]:  # leads 24 h and 120 h (all ten when fetched)
                vt = np.datetime64(t0 + dt.timedelta(hours=12 * (k + 1)), "h")
                if str(vt) in tidx:
                    e = crop_roll(a[j]) - crop_roll(truth[tidx[str(vt)]].astype(np.float32))
                    cams.setdefault(k, []).append((rm(e, allm), rm(e, land), rm(e, fm), bi(e, fm), bi(e, land)))
    for t in TAGS:                                                                     # a fine-tuned model vs RT 1.1 (both with fires), all cells / land / fire cells
        print(f"=== {t} vs RT 1.1 (pred.npy): RMSE all / land / fire cells, bias land")
        for k in range(10):
            a = np.array(res.get(("fires", k), [])); b = np.array(res.get((t, k), []))
            if len(a) and len(b):
                print(f"{12 * (k + 1):4d} h | 1.1 {a[:, 0].mean():6.2f} {a[:, 1].mean():6.2f} {a[:, 2].mean():6.2f} {a[:, 4].mean():+6.2f} | {t} {b[:, 0].mean():6.2f} {b[:, 1].mean():6.2f} {b[:, 2].mean():6.2f} {b[:, 4].mean():+6.2f} | "
                      f"land {100 * (b[:, 1].mean() / a[:, 1].mean() - 1):+5.1f} % | n {len(b)}")
    print("lead | RMSE all cells: fires / no fires / CAMS fc | RMSE land: fires / no fires / CAMS fc | RMSE fire cells: fires / no fires / CAMS fc | "
          "bias fire cells: fires / no fires | bias land: fires / no fires | days")
    for k in range(10):
        a = np.array(res.get(("fires", k), [])); b = np.array(res.get(("no fires", k), [])); c = np.array(cams.get(k, []))
        if not (len(a) and len(b)):
            continue
        cs = (f"{c[:, 0].mean():6.2f}", f"{c[:, 1].mean():6.2f}", f"{c[:, 2].mean():6.2f}") if len(c) else ("     -",) * 3
        pz = np.array(res.get(("persistence", k), []))
        print(f"{12 * (k + 1):4d} h | {a[:, 0].mean():6.2f} / {b[:, 0].mean():6.2f} / {cs[0]} | {a[:, 1].mean():6.2f} / {b[:, 1].mean():6.2f} / {cs[1]} | "
              f"{a[:, 2].mean():6.2f} / {b[:, 2].mean():6.2f} / {cs[2]} | {a[:, 3].mean():+6.2f} / {b[:, 3].mean():+6.2f} | "
              f"{a[:, 4].mean():+6.2f} / {b[:, 4].mean():+6.2f} | {len(a)}" + (f" (CAMS fc {len(c)})" if len(c) else "") + (f" | persistence all {pz[:, 0].mean():.2f} land {pz[:, 1].mean():.2f}" if len(pz) else ""))


TAGS = []

if __name__ == "__main__":
    cmd, first, last = sys.argv[1:4]
    TAGS = [a_ for a_ in sys.argv[4:] if not a_.startswith("-")]                           # score <first> <last> [tag ...]
    {"prepare": prepare, "submit": submit, "score": score}[cmd](first, last)
