#!/usr/bin/env python3
"""cams3h step 1: the CAMS operational forecast of surface PM2.5 at EVERY 3-hourly step, 0 .. 120 h, for the 53 cycles of
00 UTC that the real-time scorer uses (issue days 2026-09-01..09-20 and 2026-10-10..11-11).

Same product, grid and request as the existing arm "cams" (realtime/fetch_camsfc.py, replay/code/camsfc_fetch2.py):
dataset cams-global-atmospheric-composition-forecasts, type forecast, time 00:00, variable particulate_matter_2.5um,
netCDF in a zip; ONLY the list of lead hours changes (0, 3, ..., 120 instead of 12, 24, ..., 120) and the dates are given
as ranges. 53 x 41 = 2,173 fields in a few requests (one per group of GROUPS).

The client is the chain's (realtime/fetch.py ads(): cdsapi with the user's account); the key is never read, printed or
logged here. A request of the account with the same content that is queued, running or successful is RE-ATTACHED instead
of being sent again. No new request is sent between 09:30 and 11:45 UTC (the daily production chain uses the account
then), and at --deadline (UTC) the requests of this script that are not finished are DELETED from the queue.

Written only under <replay>/cams3h/raw/: <group>.nc (the download, kept as it is: the service answers a one-variable
request with a bare netCDF file even when a zip is asked; <group>.zip if it does send a zip), <group>.json (request, times,
size, SHA-256), requests.json (state: the request identifiers, used to re-attach).
    python -I -B -u fetch3h.py [--dry] [--deadline 2026-11-20T08:30] [--groups 2026-09 2026-10 2026-11]"""
import argparse
import datetime as dt
import json
import sys
import time
import zipfile
from pathlib import Path

RT = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km/realtime")
R = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay")
OUT = R / "cams3h"
RAW = OUT / "raw"
DATASET = "cams-global-atmospheric-composition-forecasts"
LEADS = list(range(0, 121, 3))
GROUPS = {"2026-09a": ["2026-09-01/2026-09-10"], "2026-09b": ["2026-09-11/2026-09-20"],
          "2026-10a": ["2026-10-10/2026-10-20"], "2026-10b": ["2026-10-21/2026-10-31"],
          "2026-11": ["2026-11-01/2026-11-11"]}
sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_io  # noqa: E402


def log(msg):
    print(f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%S}Z {msg}", flush=True)


def request(dates):
    """realtime/fetch_camsfc.py's request with the 41 lead hours and a list of date ranges"""
    return {"type": ["forecast"], "leadtime_hour": [str(L) for L in LEADS], "date": list(dates), "time": ["00:00"],
            "variable": ["particulate_matter_2.5um"], "data_format": "netcdf", "download_format": "zip"}


def client():
    sys.path.insert(0, str(RT))
    from fetch import ads                                # the chain's client (read only); .client = the datastore client
    return ads().client


def in_quiet_window(now=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    h = now.hour + now.minute / 60.0
    return 9.5 <= h < 11.75


def same_request(q, dates):
    want = request(dates)
    norm = lambda v: sorted(map(str, v)) if isinstance(v, (list, tuple)) else [str(v)]  # noqa: E731
    return set(q) >= set(want) and all(norm(q.get(k)) == norm(want[k]) for k in want)


def find_existing(c, dates):
    """a request of the account with this content that is accepted, running or successful, or None"""
    try:
        jobs = c.get_jobs(limit=60, sortby="-created").json.get("jobs", [])
    except Exception as e:  # noqa: BLE001
        log(f"the list of the account's requests could not be read ({type(e).__name__}); a new request is sent")
        return None
    for j in jobs:
        if j.get("processID") != DATASET or j.get("status") not in ("accepted", "running", "successful"):
            continue
        try:
            r = c.get_remote(j["jobID"])
            if same_request(r.request, dates):
                return r
        except Exception:  # noqa: BLE001  (an expired request)
            continue
    return None


def on_disk(g):
    """the finished download of group g (netCDF or zip), or None"""
    for ext in (".nc", ".zip"):
        p = RAW / f"{g}{ext}"
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def kind_of(path):
    """'zip', 'nc' (netCDF-4 / HDF5 or classic netCDF) or None, from the first bytes"""
    with open(path, "rb") as f:
        head = f.read(8)
    if head[:4] == b"PK\x03\x04":
        return "zip"
    if head == b"\x89HDF\r\n\x1a\n" or head[:3] == b"CDF":
        return "nc"
    return None


def check_download(path, n_cycles):
    """the file can be read and holds pm2p5 for n_cycles reference times x 41 lead hours on the 451 x 900 grid"""
    k = kind_of(path)
    assert k in ("zip", "nc"), "the download is neither a zip nor a netCDF file"
    if k == "zip":
        with zipfile.ZipFile(path) as zf:
            assert zf.testzip() is None, "corrupt zip member"
            names = zf.namelist()
        return k, names
    import netCDF4
    with netCDF4.Dataset(str(path)) as ds:
        v = ds.variables["pm2p5"]
        dims = dict(zip(v.dimensions, v.shape))
        assert dims.get("forecast_period") == len(LEADS) and dims.get("latitude") == 451 and dims.get("longitude") == 900, dims
        assert dims.get("forecast_reference_time", 1) == n_cycles, dims
    return k, [f"pm2p5 {dims}"]


def n_cycles_of(dates):
    n = 0
    for rg in dates:
        a, b = (dt.date.fromisoformat(x) for x in rg.split("/"))
        n += (b - a).days + 1
    return n


def qos_note(r):
    try:
        q = (r.json.get("metadata") or {}).get("qos") or {}
        return json.dumps(q)[:400] if q else ""
    except Exception:  # noqa: BLE001
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="no request: validity and cost of each request, state of the account's queue")
    ap.add_argument("--deadline", default=None, help="UTC time (ISO) after which unfinished requests of this script are deleted")
    ap.add_argument("--groups", nargs="*", default=list(GROUPS))
    ap.add_argument("--poll", type=int, default=45)
    a = ap.parse_args()
    deadline = dt.datetime.fromisoformat(a.deadline).replace(tzinfo=dt.timezone.utc) if a.deadline else None
    RAW.mkdir(parents=True, exist_ok=True)
    c = client()
    if a.dry:
        for g in a.groups:
            rq = request(GROUPS[g])
            try:
                cost = c.estimate_costs(DATASET, rq)
            except Exception as e:  # noqa: BLE001
                cost = f"estimate failed: {type(e).__name__}: {str(e)[:300]}"
            log(f"{g}: {GROUPS[g]} cost {cost}")
        try:
            jobs = c.get_jobs(limit=30, sortby="-created").json.get("jobs", [])
            for j in jobs:
                log(f"  account request: {str(j.get('created'))[:19]}Z {j.get('processID')} status {j.get('status')}")
        except Exception as e:  # noqa: BLE001
            log(f"list of requests not read: {type(e).__name__}")
        return 0

    state = json.loads((RAW / "requests.json").read_text()) if (RAW / "requests.json").exists() else {}
    remotes = {}
    for g in a.groups:
        z = on_disk(g)
        if z is not None:
            log(f"{g}: {z.name} is on disk ({z.stat().st_size / 1e6:.1f} MB), nothing to fetch")
            continue
        r = None
        if state.get(g, {}).get("request_id"):                 # the request this script sent earlier
            try:
                r0 = c.get_remote(state[g]["request_id"])
                if r0.status in ("accepted", "running", "successful") and same_request(r0.request, GROUPS[g]):
                    r = r0
            except Exception:  # noqa: BLE001  (expired or deleted)
                r = None
        if r is None:
            r = find_existing(c, GROUPS[g])
        if r is not None:
            log(f"{g}: re-attached to request {r.request_id} (status {r.status})")
            state[g] = dict(state.get(g, {}), request_id=r.request_id, reattached=True)
        else:
            if in_quiet_window():
                log(f"{g}: 09:30-11:45 UTC is the production window of the account: no new request; stopping")
                break
            r = c.submit(DATASET, request(GROUPS[g]))
            log(f"{g}: request sent for {GROUPS[g]} ({len(LEADS)} lead hours) -> {r.request_id}")
            state[g] = dict(request_id=r.request_id, reattached=False)
        state[g].update(dates=GROUPS[g], sent_or_attached_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
        remotes[g] = r
    safe_io.write_json(RAW / "requests.json", state)

    last, t0, fails = {}, time.time(), {}
    while remotes:
        for g in list(remotes):
            r = remotes[g]
            try:
                r.update()
                st = r.status
            except Exception as e:  # noqa: BLE001  (network hiccup)
                log(f"{g}: status not read ({type(e).__name__}); retry")
                continue
            if st != last.get(g):
                log(f"{g}: status {st} after {time.time() - t0:.0f} s {qos_note(r)}")
                last[g] = st
            if st == "successful":
                tmp = RAW / f"{g}.part"
                try:
                    t1 = time.time()
                    if tmp.exists():
                        tmp.unlink()
                    r.download(str(tmp))
                    kind, names = check_download(tmp, n_cycles_of(GROUPS[g]))
                    sha, n = safe_io.finalize_file(tmp, RAW / f"{g}.{kind}")
                    info = dict(group=g, dataset=DATASET, request=request(GROUPS[g]), request_id=r.request_id, file=f"{g}.{kind}", content=names, bytes=n, sha256=sha,
                                created=str(r.created_at), started=str(r.started_at), finished=str(r.finished_at),
                                download_seconds=round(time.time() - t1), downloaded_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
                    safe_io.write_json(RAW / f"{g}.json", info)
                    log(f"{g}: downloaded {n / 1e6:.1f} MB in {time.time() - t1:.0f} s (created {info['created'][:19]}, started {info['started'][:19]}, finished {info['finished'][:19]})")
                    del remotes[g]
                except Exception as e:  # noqa: BLE001
                    fails[g] = fails.get(g, 0) + 1
                    log(f"{g}: download failed ({type(e).__name__}: {str(e)[:300]}), attempt {fails[g]}")
                    if fails[g] >= 4:
                        del remotes[g]
            elif st in ("failed", "dismissed", "rejected"):
                err = ""
                try:
                    err = json.dumps(r.json.get("results") or r.json.get("message") or "")[:600]
                except Exception:  # noqa: BLE001
                    pass
                fails[g] = fails.get(g, 0) + 1
                log(f"{g}: request {st} {err}")
                if fails[g] <= 1 and not in_quiet_window() and (deadline is None or dt.datetime.now(dt.timezone.utc) < deadline):
                    r = c.submit(DATASET, request(GROUPS[g]))
                    log(f"{g}: request sent again -> {r.request_id}")
                    remotes[g], last[g] = r, None
                    state[g].update(request_id=r.request_id, resent=True)
                    safe_io.write_json(RAW / "requests.json", state)
                else:
                    del remotes[g]
        if remotes and deadline is not None and dt.datetime.now(dt.timezone.utc) >= deadline:
            for g, r in remotes.items():
                try:
                    r.delete()
                    log(f"{g}: deadline reached, request {r.request_id} deleted from the queue (status was {last.get(g)})")
                except Exception as e:  # noqa: BLE001
                    log(f"{g}: deadline reached, request {r.request_id} could NOT be deleted ({type(e).__name__})")
            break
        if remotes:
            time.sleep(a.poll)
    have = [g for g in GROUPS if on_disk(g) is not None]
    log(f"done: {len(have)} of {len(GROUPS)} groups on disk; missing {[g for g in GROUPS if g not in have] or 'none'}")
    return 0 if len(have) == len(GROUPS) else 1


if __name__ == "__main__":
    sys.exit(main())
