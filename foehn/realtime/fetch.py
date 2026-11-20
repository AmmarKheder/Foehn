#!/usr/bin/env python3
"""RT step 1: download the inputs of start day d (forecast issued from the 00 UTC run of d) into data/<yyyy-mm-dd>/.
  cams_00.zip   CAMS analysis 00 UTC of d    } dataset cams-global-atmospheric-composition-forecasts, type forecast,
  cams_12.zip   CAMS analysis 12 UTC of d-1  } leadtime 0, the variables and the 1000 hPa level of dl_cams_analysis.py
  gfas.nc       GFAS daily of d - 1 (tpmfire, frpfire, injh; the training used mami, which v1.4.2 no longer provides: injh,
                the IS4FIRES injection height, replaces it; the ADS stamps the day at the following 00 UTC, as dl_gfas.py notes)
  ifs/<sss>.grib2  IFS 00 UTC run of d, steps 0..192 h every 6 h (ECMWF open data, 0.25 deg): u, v, w, t, q, z at
                1000/925/850/700/600/500/400/300 hPa and 10u, 10v, 2t, sp, tp (the hres_B channels of dl_wb2.py; 33 steps
                as hres_B, so that FLEXGREEN interpolates in time exactly as in training)
Sources that are not there yet are retried (every --every minutes, until --deadline UTC); the journal of the day says
what is missing. Open data: data.ecmwf.int (recent days), then the AWS mirror of ECMWF open data (archive).
    python fetch.py 2026-10-10 [--deadline 14:00] [--every 10] [--only cams gfas ifs]"""
import argparse
import datetime as dt
import json
import os
import random
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import common as C

L8 = [1000, 925, 850, 700, 600, 500, 400, 300]
PL_PARAMS = ["u", "v", "w", "t", "q", "z"]               # order of dl_wb2.V3: u, v, vertical velocity (Pa/s), t, q, z
SFC_PARAMS = ["10u", "10v", "2t", "sp", "tp"]
STEPS = list(range(0, 193, 6))
OPEN_DATA = [os.environ["RT_IFS_SOURCE"]] if os.environ.get("RT_IFS_SOURCE") else [          # a forced source (bulk downloads)
    "https://storage.googleapis.com/ecmwf-open-data",                                    # Google mirror (archive since 2025, generous rate limits): first
    "https://data.ecmwf.int/forecasts",                                                  # ECMWF, the last ~4 days (429 "Too Many Requests" after heavy use)
    "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"]                             # AWS mirror (archive since 2025; 503 "Slow Down" under parallel reads)
GFAS_VARS = ["wildfire_flux_of_total_particulate_matter", "wildfire_radiative_power", "injection_height"]
# GFAS v1.4.2 (2026 onwards) no longer serves the plume-rise height "mami" of the training (gfas_daily channel 4);
# the IS4FIRES injection height "injh" takes its place (decision of 2026-10-10, recorded in every day's journal)


def ads():
    import cdsapi
    rc = dict(l.split(":", 1) for l in (C.Path.home() / ".adsapirc").read_text().splitlines() if ":" in l)
    return cdsapi.Client(url=rc["url"].strip(), key=rc["key"].strip(), quiet=True, progress=False)


def nc_files(path):
    """the netCDF file(s) of an ADS download: a zip of .nc files, or a bare .nc (single-variable requests)"""
    import tempfile, zipfile
    path = C.Path(path)
    if zipfile.is_zipfile(path):
        tmp = C.Path(tempfile.mkdtemp())
        with zipfile.ZipFile(path) as z:
            z.extractall(tmp)
        return sorted(tmp.glob("*.nc"))
    return [path]


def check_zip(p):
    names = [f.name for f in nc_files(p)]
    assert names, p
    return names


def fetch_cams(d, D, deadline, every):
    from dl_cams_analysis import PL, SL                    # the variables and scale factors of the training series
    for tag, day, hour in (("00", d, "00:00"), ("12", d - dt.timedelta(days=1), "12:00")):
        dst = D / f"cams_{tag}.zip"
        if dst.exists():
            continue
        tmp = D / f"cams_{tag}.part"

        def get():
            ads().retrieve("cams-global-atmospheric-composition-forecasts",
                           {"type": ["forecast"], "leadtime_hour": ["0"], "date": [f"{day}/{day}"], "time": [hour],
                            "variable": list(SL) + list(PL), "pressure_level": ["1000"], "data_format": "netcdf",
                            "download_format": "zip"}, str(tmp))
            check_zip(tmp)
            tmp.rename(dst)
        C.retry(d, f"CAMS analysis {day} {hour} UTC", get, deadline, every)
        C.log(d, f"CAMS analysis {day} {hour} UTC: {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)")


def fetch_gfas(d, D, deadline, every):
    dst = D / "gfas.nc"
    if dst.exists():
        return
    day = d - dt.timedelta(days=1)
    tmp = D / "gfas.part"

    def get():
        ads().retrieve("cams-global-fire-emissions-gfas", {"variable": GFAS_VARS, "date": [f"{day}/{day}"],
                                                          "data_format": "netcdf"}, str(tmp))
        import xarray as xr
        with xr.open_dataset(tmp) as ds:
            assert {"tpmfire", "frpfire", "injh"} <= set(ds.data_vars), list(ds.data_vars)
            stamp = str(ds["valid_time"].values.ravel()[0])[:10] if "valid_time" in ds.coords else "?"
        tmp.rename(dst)
        return stamp
    stamp = C.retry(d, f"GFAS daily {day}", get, deadline, every)
    version = "v1.2" if day.year <= 2025 else "v1.4.2"
    C.log(d, f"GFAS daily {day}: gfas.nc ({dst.stat().st_size / 1e6:.1f} MB), ADS stamp {stamp} (expected {d}), product {version}")
    C.meta_update(d, gfas_day=str(day), gfas_version=version, gfas_stamp=stamp)


def http(url, rng=None, timeout=120):
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def open_data_bases(d):
    """the open-data sources that hold the complete 00 UTC run of day d (step 192 h published), in order of preference"""
    ymd = f"{d:%Y%m%d}"; ok = []
    for b in OPEN_DATA:
        try:
            http(f"{b}/{ymd}/00z/ifs/0p25/oper/{ymd}000000-192h-oper-fc.index", timeout=60)
            ok.append(b)
        except Exception:
            continue
    if not ok:
        raise RuntimeError(f"IFS 00 UTC run of {d}: step 192 h not published on any open-data source yet")
    return ok


def open_data_base(d):
    return open_data_bases(d)[0]


def fetch_step(d, D, base, s):
    dst = D / "ifs" / f"{s:03d}.grib2"
    if dst.exists():
        return s, dst.stat().st_size, 0
    ymd = f"{d:%Y%m%d}"
    stem = f"{base}/{ymd}/00z/ifs/0p25/oper/{ymd}000000-{s}h-oper-fc"
    recs = [json.loads(l) for l in http(stem + ".index").decode().splitlines() if l.strip()]
    want = [r for r in recs if (r.get("levtype") == "pl" and r["param"] in PL_PARAMS and int(r.get("levelist", -1)) in L8)
            or (r.get("levtype") == "sfc" and r["param"] in SFC_PARAMS)]
    zlev = {int(r["levelist"]) for r in want if r["param"] == "z"}                  # before ~June 2026 the open data has gh, not z:
    want += [r for r in recs if r.get("levtype") == "pl" and r["param"] == "gh" and int(r.get("levelist", -1)) in L8 and int(r["levelist"]) not in zlev]
    need = len(PL_PARAMS) * len(L8) + len(SFC_PARAMS) - (1 if s == 0 and not any(r["param"] == "tp" for r in want) else 0)
    assert len(want) >= need, f"step {s}: {len(want)} of {need} fields in the index"
    want.sort(key=lambda r: r["_offset"])
    spans = []                                              # merge byte ranges closer than 2 MB: few, large requests
    for r in want:
        a, b = r["_offset"], r["_offset"] + r["_length"] - 1
        if spans and a - spans[-1][1] < 2_000_000:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    tmp = dst.with_suffix(".part")
    with open(tmp, "wb") as f:
        for a, b in spans:
            f.write(http(stem + ".grib2", (a, b)))
    tmp.rename(dst)
    return s, dst.stat().st_size, len(spans)


def fetch_ifs(d, D, deadline, every):
    (D / "ifs").mkdir(exist_ok=True)
    if all((D / "ifs" / f"{s:03d}.grib2").exists() for s in STEPS):
        return
    bases = C.retry(d, f"IFS 00 UTC run of {d} (open data)", lambda: open_data_bases(d), deadline, every)
    base = bases[0]; t0 = time.time()

    def one(s):                                                                   # 429 / 503 of a source: back off and try the next source
        for n in range(1, 9):
            b = bases[(n - 1) % len(bases)]
            try:
                return fetch_step(d, D, b, s)
            except Exception as e:
                wait = min(300, 30 * 2 ** (n - 1)) + random.uniform(0, 10)
                C.log(d, f"IFS step {s} h: attempt {n} failed ({repr(e)[:120]})" + ("; giving up" if n == 8 else f"; retry in {wait:.0f} s"))
                if n == 8:
                    raise
                time.sleep(wait)
    with ThreadPoolExecutor(3 if "amazonaws" in base else 6) as ex:
        res = list(ex.map(one, STEPS))
    C.log(d, f"IFS 00 UTC {d}: {len(res)} steps 0..192 h from {base.split('//')[1].split('/')[0]}, "
             f"{sum(r[1] for r in res) / 1e9:.2f} GB, {time.time() - t0:.0f}s")
    C.meta_update(d, ifs_source=base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--deadline", default=None, help="HH:MM UTC of day d after which missing sources are given up")
    ap.add_argument("--every", type=int, default=10, help="minutes between retries of a source not yet published")
    ap.add_argument("--only", nargs="*", default=["cams", "gfas", "ifs"])
    a = ap.parse_args()
    d = C.parse_date(a.date)
    D = C.day_dir(d)
    deadline = None
    if a.deadline:
        hh, mm = map(int, a.deadline.split(":"))
        deadline = dt.datetime(d.year, d.month, d.day, hh, mm)
    C.log(d, f"fetch {d} ({', '.join(a.only)})")
    C.meta_update(d, date=str(d), gi=C.gi_of(d), year=d.year, doy=C.doy_of(d))
    if "ifs" in a.only:
        fetch_ifs(d, D, deadline, 60 * a.every)
    if "gfas" in a.only:
        fetch_gfas(d, D, deadline, 60 * a.every)
    if "cams" in a.only:
        fetch_cams(d, D, deadline, 60 * a.every)
    C.log(d, "fetch done")


if __name__ == "__main__":
    main()
