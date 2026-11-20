#!/usr/bin/env python3
"""WP1a, decision A of the coordinator (2026-11-19), second version of the fetch (camsfc_fetch.py asked one cycle per
request): the ten leads 12, 24, ..., 120 h of the CAMS operational PM2.5 forecast (00 UTC cycle) for the issue days whose
realtime/data/<d>/cams_fc.npz holds only the leads 24 and 120 h (2026-09-01..09-20 and 2026-10-10).

Why a second version: these cycles are no longer on the fast disk of the ADS; each request goes through the queue of the
MARS archive (60 running, more than 500 queued on 2026-11-19 13:18 UTC: the first one-cycle request waited more than 25
minutes before starting). 22 one-cycle requests one after the other would take most of a day. Here the account sends
TWO requests, one after the other:
  1. the check day (a cycle that has ten stored leads, 2026-10-12) alone, in the exact form of realtime/fetch_camsfc.py;
  2. ONE request for all the missing cycles: the same dataset, variable, type, time, leads and format as the chain's
     request, with the list of date ranges instead of one date (210 fields; the dataset's limit is 10,000).
A request of this account with the same content that is already queued, running or successful is RE-ATTACHED instead of
being sent again (so a restarted job never duplicates a request). No request is sent between 09:30 and 11:45 UTC (the
daily production chain uses the same ADS account then). The client is the chain's (realtime/fetch.py ads(): the user's
account; the key is never printed).

Written ONLY under <replay>/camsfc/: <d>/cams_fc.npz in the format of the chain (pm25 (10, 451, 900) float16 ug m-3,
lat (451,), lon (900,), leads (10,)) and <d>/fetch.json; check_<D>.json (re-fetched check day against the stored file);
summary.json. Each fetched cycle is also compared with its stored two-lead file (leads 24 and 120 h): the fetch.json of
the day and summary.json hold the result. Nothing is written in realtime/data; the stored files stay as they are; no lead
is interpolated. The conversion is the chain's arithmetic (float32 x 1e9, then float16), per reference time.
    python -u camsfc_fetch2.py [--check 2026-10-12] [--no-check]"""
import argparse
import datetime as dt
import json
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

RT = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km/realtime")
R = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay")
OUT = R / "camsfc"
DATASET = "cams-global-atmospheric-composition-forecasts"
LEADS = [12 * k for k in range(1, 11)]
TARGET = [dt.date(2026, 9, 1) + dt.timedelta(k) for k in range(20)] + [dt.date(2026, 10, 10) + dt.timedelta(k) for k in range(33)]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cams_points import two_lead_check  # noqa: E402  (numpy only)


def log(msg):
    print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S}Z {msg}", flush=True)


def n_leads(p):
    try:
        return int(np.load(p)["leads"].size)
    except Exception:  # noqa: BLE001  (missing or unreadable file)
        return 0


def request(dates):
    """realtime/fetch_camsfc.py's request, with a list of date ranges"""
    return {"type": ["forecast"], "leadtime_hour": [str(L) for L in LEADS], "date": list(dates), "time": ["00:00"],
            "variable": ["particulate_matter_2.5um"], "data_format": "netcdf", "download_format": "zip"}


def ranges(days):
    """sorted dates -> ['first/last', ...] of the runs of consecutive days"""
    out, run = [], []
    for d in sorted(days):
        if run and (d - run[-1]).days != 1:
            out.append(f"{run[0]}/{run[-1]}")
            run = []
        run.append(d)
    if run:
        out.append(f"{run[0]}/{run[-1]}")
    return out


def client():
    sys.path.insert(0, str(RT))
    from fetch import ads                                # the chain's client (read only); .client = the datastore client
    return ads().client


def quiet_window():
    while True:
        now = dt.datetime.now(dt.timezone.utc)
        h = now.hour + now.minute / 60.0
        if not (9.5 <= h < 11.75):
            return
        log("09:30-11:45 UTC is the production window of the ADS account: no new request, waiting")
        time.sleep(600)


def same_request(q, dates):
    want = request(dates)
    norm = lambda v: sorted(map(str, v)) if isinstance(v, (list, tuple)) else [str(v)]
    return all(norm(q.get(k)) == norm(want[k]) for k in want)


def attach_or_submit(c, dates, tag):
    """the Remote of a request with this content: an existing one of the account (accepted, running or successful), else
    a new one. Returns (remote, info)"""
    try:
        jobs = c.get_jobs(limit=40, sortby="-created").json.get("jobs", [])
    except Exception as e:  # noqa: BLE001
        log(f"{tag}: the list of the account's requests could not be read ({e!r}); a new request is sent"[:300])
        jobs = []
    for j in jobs:
        if j.get("processID") != DATASET or j.get("status") not in ("accepted", "running", "successful"):
            continue
        try:
            r = c.get_remote(j["jobID"])
            if same_request(r.request, dates):
                log(f"{tag}: re-attached to the request created {str(j.get('created'))[:19]}Z (status {j.get('status')})")
                return r, dict(reattached=True, created=str(j.get("created"))[:19] + "Z")
        except Exception:  # noqa: BLE001  (an expired request: ignore)
            continue
    quiet_window()
    r = c.submit(DATASET, request(dates))
    log(f"{tag}: request sent for {dates}")
    return r, dict(reattached=False, created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))


def split(zip_path):
    """{yyyy-mm-dd: (10, 451, 900) float32 ug m-3}, lat, lon, layout from the netCDF file(s) of a download; the chain's
    arithmetic (values float32 x 1e9), applied per forecast reference time"""
    import xarray as xr
    sys.path.insert(0, str(RT))
    from fetch import nc_files
    tdir = Path(tempfile.mkdtemp(prefix="camsfc_"))
    tempfile.tempdir = str(tdir)                         # nc_files() unzips into a mkdtemp() directory: keep it under ours
    out, lat, lon, layout = {}, None, None, []
    try:
        for f in nc_files(zip_path):
            with xr.open_dataset(f) as ds:
                v = ds["pm2p5"]
                layout.append({k: int(n) for k, n in v.sizes.items()})
                assert "forecast_period" in v.dims, v.dims
                if "forecast_reference_time" not in v.dims:
                    v = v.expand_dims(forecast_reference_time=np.atleast_1d(ds["forecast_reference_time"].values))
                v = v.transpose("forecast_reference_time", "forecast_period", "latitude", "longitude")
                hours = (np.asarray(ds["forecast_period"].values) / np.timedelta64(1, "h")).astype(float)
                assert np.allclose(hours, LEADS), f"lead coordinate {hours.tolist()}, expected {LEADS}"
                la, lo = ds["latitude"].values, ds["longitude"].values
                assert lat is None or (np.array_equal(la, lat) and np.array_equal(lo, lon))
                lat, lon = la, lo
                for i, t in enumerate(np.atleast_1d(ds["forecast_reference_time"].values)):
                    t = np.datetime64(t, "s")
                    assert t == np.datetime64(t, "D"), f"reference time {t} is not 00 UTC"
                    a = v.isel(forecast_reference_time=i).values.astype(np.float32) * 1e9
                    assert a.shape == (len(LEADS), 451, 900), a.shape
                    assert np.isfinite(a).all() and float(a.min()) >= 0.0, (str(t), float(np.nanmin(a)))
                    out[str(np.datetime64(t, "D"))] = a
    finally:
        tempfile.tempdir = None
        shutil.rmtree(tdir, ignore_errors=True)
    return out, lat, lon, layout


def save_day(D, a, lat, lon, info):
    D.mkdir(parents=True, exist_ok=True)
    part = D / "cams_fc.part.npz"
    np.savez(part, pm25=a.astype(np.float16), lat=lat, lon=lon, leads=np.array(LEADS))
    part.rename(D / "cams_fc.npz")
    (D / "fetch.json").write_text(json.dumps(info, indent=1))


def download(r, target, tag):
    target.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r.download(str(target))                              # waits for the results, then downloads; raises if the request failed
    log(f"{tag}: downloaded {target.stat().st_size / 1e6:.1f} MB after {time.time() - t0:.0f} s of waiting in this job")
    return round(time.time() - t0)


def do_check(c, d):
    ref_p = RT / "data" / str(d) / "cams_fc.npz"
    ref = np.load(ref_p)
    assert ref["leads"].tolist() == LEADS, ref["leads"]
    D = OUT / "_check" / str(d)
    if n_leads(D / "cams_fc.npz") < len(LEADS):
        r, info = attach_or_submit(c, [f"{d}/{d}"], f"check {d}")
        z = OUT / "_dl" / f"check_{d}.zip"
        info["waited_s"] = download(r, z, f"check {d}")
        got, lat, lon, layout = split(z)
        assert list(got) == [str(d)], list(got)
        save_day(D, got[str(d)], lat, lon, dict(date=str(d), request=request([f"{d}/{d}"]), dataset=DATASET, layout=layout, **info))
        z.unlink()
    new = np.load(D / "cams_fc.npz")
    a, b = ref["pm25"], new["pm25"]
    af, bf = a.astype(np.float64), b.astype(np.float64)
    res = dict(day=str(d), stored=str(ref_p), refetched=str(D / "cams_fc.npz"), shape=list(a.shape), dtype=[str(a.dtype), str(b.dtype)],
               bitwise_identical_pm25=bool(a.dtype == b.dtype and np.array_equal(a.view(np.uint16), b.view(np.uint16))),
               share_identical=float((af == bf).mean()), max_abs_diff=float(np.abs(af - bf).max()),
               mean_abs_diff=float(np.abs(af - bf).mean()), corr=float(np.corrcoef(af.ravel(), bf.ravel())[0, 1]),
               mean_stored=float(af.mean()), mean_refetched=float(bf.mean()),
               lat_identical=bool(np.array_equal(ref["lat"], new["lat"])), lon_identical=bool(np.array_equal(ref["lon"], new["lon"])),
               leads_identical=bool(np.array_equal(ref["leads"], new["leads"])),
               checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    (OUT / f"check_{d}.json").write_text(json.dumps(res, indent=1))
    log(f"check {d}: bitwise identical {res['bitwise_identical_pm25']}, share identical {res['share_identical']:.6f}, "
        f"max |diff| {res['max_abs_diff']:.4g} ug m-3, r {res['corr']:.8f}")
    return res


def do_cycles(c):
    need = [d for d in TARGET if n_leads(RT / "data" / str(d) / "cams_fc.npz") < len(LEADS) and n_leads(OUT / str(d) / "cams_fc.npz") < len(LEADS)]
    if not need:
        log("every target cycle has its ten leads")
        return []
    dates = ranges(need)
    log(f"{len(need)} cycles to fetch in one request: {dates}")
    z = OUT / "_dl" / "cycles.zip"
    if z.exists() and zipfile.is_zipfile(z):             # a complete download kept by an earlier run (a cycle differed, or the
        log("cycles: a complete download is on disk, read as it is")   # split failed): no new request
        info = dict(reattached=None, created=None, waited_s=0)
    else:
        r, info = attach_or_submit(c, dates, "cycles")
        info["waited_s"] = download(r, z, "cycles")
    got, lat, lon, layout = split(z)
    miss = [str(d) for d in need if str(d) not in got]
    assert not miss, f"cycles absent from the download: {miss}"
    rows = []
    for d in need:
        a = got[str(d)]
        D = OUT / str(d)
        meta = dict(date=str(d), leads=LEADS, dataset=DATASET, request=request(dates), one_request_for=len(need), layout=layout,
                    mean_ug_first_last=[round(float(a[0].mean()), 4), round(float(a[-1].mean()), 4)],
                    fetched_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **info)
        save_day(D, a, lat, lon, meta)
        t = two_lead_check(RT / "data" / str(d) / "cams_fc.npz", D / "cams_fc.npz")
        if t is not None:
            meta["two_lead_check"] = {k: v for k, v in t.items() if k not in ("stored", "fetched")}
            (D / "fetch.json").write_text(json.dumps(meta, indent=1))
        rows.append(dict(day=str(d), two_lead_identical=None if t is None else t["identical_after_cast"],
                         two_lead_max_abs_diff=None if t is None else t["max_abs_diff"]))
        log(f"{d}: cams_fc.npz, mean {meta['mean_ug_first_last'][0]:.2f} / {meta['mean_ug_first_last'][1]:.2f} ug m-3 (+12 h / +120 h); "
            f"leads 24 and 120 h equal to the stored two-lead file: {rows[-1]['two_lead_identical']}")
    if all(x["two_lead_identical"] in (True, None) for x in rows):
        z.unlink()                                       # kept when a cycle differs from its stored two-lead file
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", default="2026-10-12")
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    c = client()
    res, rows, errs = None, [], []
    if not a.no_check:
        try:
            res = do_check(c, dt.date.fromisoformat(a.check))
        except Exception as e:  # noqa: BLE001  (the check must not stop the fetch)
            errs.append(f"check {a.check}: {e!r}"[:600])
            log(errs[-1])
    elif (OUT / f"check_{a.check}.json").exists():       # the check already made (first version of the fetch): kept
        res = json.loads((OUT / f"check_{a.check}.json").read_text())
    for attempt in range(2):
        try:
            rows = do_cycles(c)
            break
        except Exception as e:  # noqa: BLE001
            errs.append(f"cycles, attempt {attempt + 1}: {e!r}"[:600])
            log(errs[-1])
            time.sleep(120)
    have = [str(d) for d in TARGET if n_leads(RT / "data" / str(d) / "cams_fc.npz") >= len(LEADS) or n_leads(OUT / str(d) / "cams_fc.npz") >= len(LEADS)]
    without = [str(d) for d in TARGET if str(d) not in have]
    (OUT / "summary.json").write_text(json.dumps(dict(
        target_days=len(TARGET), with_ten_leads=len(have), without=without, fetched=rows,
        all_fetched_equal_to_two_lead_files=bool(rows) and all(x["two_lead_identical"] in (True, None) for x in rows),
        check=None if res is None else {k: res[k] for k in ("day", "bitwise_identical_pm25", "share_identical", "max_abs_diff")},
        errors=errs, updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")), indent=1))
    log(f"done: {len(have)} of {len(TARGET)} target cycles have ten leads; without: {without or 'none'}; errors: {errs or 'none'}")
    return 1 if without else 0


if __name__ == "__main__":
    sys.exit(main())
