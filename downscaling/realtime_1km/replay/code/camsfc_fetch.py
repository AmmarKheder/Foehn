#!/usr/bin/env python3
"""WP1a, decision A of the coordinator (2026-11-19): the ten leads 12, 24, ..., 120 h of the CAMS operational PM2.5
forecast (00 UTC cycle) for the issue days whose realtime/data/<d>/cams_fc.npz holds only the leads 24 and 120 h
(2026-09-01..09-20 and 2026-10-10: files of the September hindcast), so that the baseline 'cams' (daily mean of lead day
L = trapezoid of the instants 24(L-1), 24L-12, 24L h) can be built for them.

Written ONLY under <replay>/camsfc/: <d>/cams_fc.npz in the format of the chain's realtime/fetch_camsfc.py
(pm25 (10, 451, 900) float16 ug m-3, lat (451,), lon (900,), leads (10,)) and <d>/fetch.json. Nothing is written in
realtime/data; the stored two-lead files stay as they are; no lead is interpolated.
The request is the chain's own (realtime/fetch_camsfc.py, same dataset, variable, type, time, format) and the client is
the chain's (realtime/fetch.py ads(): the user's ADS account, as the daily chain uses it; the key is never printed).
Requests are sent one after the other. A day whose cams_fc.npz exists is skipped (resubmission continues).
  --check D   fetch day D (which already has ten leads in realtime/data) into camsfc/_check/D/ and compare it with the
              stored file -> camsfc/check_<D>.json
    python -I camsfc_fetch.py --check 2026-10-12 --missing          (the check first, then every day with < 10 leads)
    python -I camsfc_fetch.py 2026-09-01 2026-09-02"""
import argparse
import datetime as dt
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

RT = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km/realtime")
R = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay")
OUT = R / "camsfc"
LEADS = [12 * k for k in range(1, 11)]
TARGET = [dt.date(2026, 9, 1) + dt.timedelta(k) for k in range(20)] + [dt.date(2026, 10, 10) + dt.timedelta(k) for k in range(33)]


def log(msg):
    print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S}Z {msg}", flush=True)


def quiet_window():
    """the daily production chain uses the same ADS account at about 10:00-11:20 UTC: no replay request is sent between
    09:30 and 11:45 UTC (a queued replay request must not stand in front of the day's forecast inputs)"""
    while True:
        now = dt.datetime.now(dt.timezone.utc)
        h = now.hour + now.minute / 60.0
        if not (9.5 <= h < 11.75):
            return
        log("09:30-11:45 UTC is the production window of the ADS account: waiting")
        time.sleep(600)


def n_leads(p):
    try:
        return int(np.load(p)["leads"].size)
    except Exception:  # noqa: BLE001  (missing or unreadable file)
        return 0


def fetch(d, D):
    """one request, as realtime/fetch_camsfc.py; returns the info dict written to fetch.json"""
    import xarray as xr
    sys.path.insert(0, str(RT))
    from fetch import ads, nc_files                      # the chain's client and unzip helper (read only)
    D.mkdir(parents=True, exist_ok=True)
    tmp = D / "cams_fc.zip"
    t0 = time.time()
    ads().retrieve("cams-global-atmospheric-composition-forecasts",
                   {"type": ["forecast"], "leadtime_hour": [str(L) for L in LEADS], "date": [f"{d}/{d}"], "time": ["00:00"],
                    "variable": ["particulate_matter_2.5um"], "data_format": "netcdf", "download_format": "zip"}, str(tmp))
    t_req = time.time() - t0
    tdir = Path(tempfile.mkdtemp(prefix="camsfc_"))
    tempfile.tempdir = str(tdir)                         # nc_files() unzips into a mkdtemp() directory: keep it under ours
    try:
        f = nc_files(tmp)[0]
        with xr.open_dataset(f) as ds:
            v = ds["pm2p5"].squeeze(drop=True)
            tdim = [x for x in v.dims if x not in ("latitude", "longitude")][0]
            a = v.transpose(tdim, "latitude", "longitude").values.astype(np.float32) * 1e9
            lat, lon = ds["latitude"].values, ds["longitude"].values
            tco = np.asarray(ds[tdim].values)
            if np.issubdtype(tco.dtype, np.timedelta64):
                tvals = (tco / np.timedelta64(1, "h")).astype(float).tolist()
            elif np.issubdtype(tco.dtype, np.datetime64):
                tvals = ((tco - np.datetime64(f"{d}T00:00")) / np.timedelta64(1, "h")).astype(float).tolist()
            else:
                tvals = np.asarray(tco, float).tolist()
    finally:
        tempfile.tempdir = None
        shutil.rmtree(tdir, ignore_errors=True)
    assert a.shape == (len(LEADS), 451, 900), a.shape
    # the chain trusts the order of the file; here it is also checked (hours since 00 UTC of d)
    order_ok = bool(np.allclose(tvals, LEADS))
    assert order_ok, f"{d}: lead coordinate {tdim} = {tvals}, expected {LEADS}"
    assert np.isfinite(a).all() and float(a.min()) >= 0.0, (float(np.nanmin(a)), int((~np.isfinite(a)).sum()))
    part = D / "cams_fc.part.npz"
    np.savez(part, pm25=a.astype(np.float16), lat=lat, lon=lon, leads=np.array(LEADS))
    part.rename(D / "cams_fc.npz")
    size = tmp.stat().st_size
    tmp.unlink()
    info = dict(date=str(d), leads=LEADS, lead_dim=tdim, lead_coordinate_hours=tvals, lead_order_checked=order_ok,
                dataset="cams-global-atmospheric-composition-forecasts", type="forecast", time="00:00",
                variable="particulate_matter_2.5um", request="as realtime/fetch_camsfc.py",
                zip_bytes=int(size), seconds=round(time.time() - t0, 1), request_seconds=round(t_req, 1),
                mean_ug_first_last=[round(float(a[0].mean()), 4), round(float(a[-1].mean()), 4)],
                fetched_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    (D / "fetch.json").write_text(json.dumps(info, indent=1))
    log(f"{d}: cams_fc.npz {a.shape}, mean {info['mean_ug_first_last'][0]:.2f} / {info['mean_ug_first_last'][1]:.2f} ug m-3 "
        f"(+12 h / +120 h), request {t_req:.0f} s")
    return info


def check(d):
    """fetch a day that has ten stored leads and compare with realtime/data/<d>/cams_fc.npz"""
    ref = np.load(RT / "data" / str(d) / "cams_fc.npz")
    assert ref["leads"].tolist() == LEADS, ref["leads"]
    D = OUT / "_check" / str(d)
    if not (D / "cams_fc.npz").exists():
        fetch(d, D)
    new = np.load(D / "cams_fc.npz")
    a, b = ref["pm25"], new["pm25"]
    af, bf = a.astype(np.float64), b.astype(np.float64)
    res = dict(day=str(d), stored=str(RT / "data" / str(d) / "cams_fc.npz"), refetched=str(D / "cams_fc.npz"),
               shape=list(a.shape), dtype=[str(a.dtype), str(b.dtype)],
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("days", nargs="*")
    ap.add_argument("--missing", action="store_true", help="every target day whose stored cams_fc.npz has < 10 leads")
    ap.add_argument("--check", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.check:
        try:
            check(dt.date.fromisoformat(a.check))
        except Exception as e:  # noqa: BLE001  (the check must not stop the fetches)
            log(f"check {a.check} FAILED: {e!r}"[:600])
    days = [dt.date.fromisoformat(x) for x in a.days]
    if a.missing:
        days += [d for d in TARGET if n_leads(RT / "data" / str(d) / "cams_fc.npz") < len(LEADS)]
    days = sorted(set(days))
    log(f"{len(days)} days to fetch: {' '.join(map(str, days))}")
    bad = []
    for d in days:
        D = OUT / str(d)
        if n_leads(D / "cams_fc.npz") >= len(LEADS):
            log(f"{d}: exists, skipped")
            continue
        quiet_window()
        for attempt in range(3):
            try:
                fetch(d, D)
                break
            except Exception as e:  # noqa: BLE001  (queue hiccup: two more tries, then the next day)
                log(f"{d}: attempt {attempt + 1} failed: {e!r}"[:600])
                time.sleep(60 * (attempt + 1))
        else:
            bad.append(str(d))
    have = [str(d) for d in days if n_leads(OUT / str(d) / "cams_fc.npz") >= len(LEADS)]
    (OUT / "summary.json").write_text(json.dumps(dict(asked=[str(d) for d in days], have=have, failed=bad,
                                                      updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")), indent=1))
    log(f"done: {len(have)} of {len(days)} days in {OUT}; failed: {bad or 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
