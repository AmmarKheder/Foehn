#!/usr/bin/env python3
"""Live feed of the monitor-fed layer (experiment M of Amendment A11): for a UTC day, the daily mean PM2.5 of every
ASSIMILATED OpenAQ monitor, from the OpenAQ v3 API, as a day file of realtime_1km/stations_rt.py (dayfile.py).

Route (found 2026-11-20, see README.md for the measurements behind it):
  1. GET /v3/locations?parameters_id=2&limit=1000&page=k&order_by=id      (paged by id: stable, about 21 requests)
       -> for the monitors of the frozen list: their PM2.5 sensor ids, recorded position, time of the last measurement
  2. GET /v3/sensors/{sensor id}/hours?datetime_from=..&datetime_to=..     (ONE request per live PM2.5 sensor)
       -> the hourly means of the 72-hour window day-1 .. day+1 (as far as it exists), with their record counts
  3. GET /v3/sensors/{sensor id}/measurements?...   (one more request, only for the sensors with 2 to 12 records per
       hour stamped on the closing time of the hour, or whose hours close on the half hour: about one sensor in
       twenty-five, the Indian network) -> their raw records, so that the archive rule is applied to the records
  The API has no multi-sensor measurement endpoint. Its only bulk one, /v3/parameters/2/latest, pages without a
  stable order (a pass misses about 20 % of the live monitors and repeats others) and holds one value per sensor:
  it is not used.
Limits of the key (free tier): 60 requests / minute and 2,000 / hour. Defaults here: 50 / minute and 1,900 / hour
(even spacing 1.9 s), the server's own counters (x-ratelimit-remaining / -reset) are obeyed, and the first HTTP 429
stops the run (exit code 3; everything fetched so far is on disk).
Restartable: every answered sensor is one line of work/<day>/hours.jsonl (append + fsync); a restart skips them.
Firewall: only monitors of list/assim_list_v1.npz are requested; that list and every day file are asserted free of
the held-out identifiers of split_v1 (dayfile.assert_no_heldout). No held-out value is ever requested or stored.
The key: read from a FILE inside this process (--key-file, or the path in $OPENAQ_API_KEY_FILE, default
~/.config/openaq/api_key); sent as the X-API-Key header; never printed, logged or written.

    python feed_openaq.py list                               # once, on LUMI: freeze the monitor list (no network)
    python feed_openaq.py fetch 2026-11-14                   # all live sensors now, one pass (a past day, a test)
    python feed_openaq.py fetch yesterday --until 09:45      # production: sensors are fetched as their day completes
    python feed_openaq.py assemble 2026-11-14 [--out days]   # hours.jsonl -> days/stations_2026-11-14.npz
    python feed_openaq.py status 2026-11-14
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import random
import signal
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dayfile  # noqa: E402
import qc  # noqa: E402

API = "https://api.openaq.org"
PM25 = 2
KEY_FILE_DEFAULT = "~/.config/openaq/api_key"
LIST = HERE / "list" / "assim_list_v1.npz"
WORK = HERE / "work"
DAYS = HERE / "days"
PER_MIN, PER_HOUR = 50, 1900
MAX_CONSECUTIVE_FAILURES = 25         # sensors in a row that the server cannot answer: the run stops (exit code 2)
UTC = dt.timezone.utc
# sources of the frozen list (LUMI project tree; only the `list` command and the tests read them)
PROJECT = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution"))
SRC_ASSIM = PROJECT / "data/processed/assim/assim_2026.npz"
SRC_SPLIT = PROJECT / "data/processed/assim/split_v1_public.json"
SRC_FLIPPED = PROJECT / "data/processed/assim/qc_flipped_coords_v1.json"
SRC_CATALOGUE = PROJECT / "realtime_1km/verif/catalogue.npz"


def log(*a):
    print(dt.datetime.now(UTC).strftime("%H:%M:%S"), *a, flush=True)


def utc_ts(s):
    """ISO string of the API ('2026-11-14T01:00:00Z') -> epoch seconds (float); None -> nan"""
    if not s:
        return float("nan")
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def iso(ts):
    return dt.datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_text(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ the frozen list
class MonitorList:
    """the assimilated OpenAQ monitors the feed may request (list/assim_list_v1.npz); refuses held-out identifiers"""

    def __init__(self, path=LIST, split=dayfile.SPLIT):
        z = np.load(path, allow_pickle=False)
        self.id = z["id"].astype(str)
        dayfile.assert_no_heldout(self.id, split, str(path))
        self.loc = z["loc"].astype(np.int64)
        self.lat, self.lon = z["lat"].astype(np.float32), z["lon"].astype(np.float32)
        self.in_catalogue = z["in_catalogue"].astype(bool)
        assert all(i == f"OpenAQ:{l}" for i, l in zip(self.id.tolist(), self.loc.tolist()))
        assert np.unique(self.loc).size == self.loc.size
        self.index = {int(l): k for k, l in enumerate(self.loc.tolist())}
        self.held_loc = frozenset(int(i.split(":")[1]) for i in dayfile.heldout_ids(split) if i.startswith("OpenAQ:"))
        assert not self.held_loc.intersection(self.index), "FIREWALL: held-out location in the list"


def cmd_list(a):
    """freeze list/assim_list_v1.npz from data/processed/assim/assim_2026.npz (assimilated monitors only, by
    construction) minus the sign-flipped coordinates of qc_flipped_coords_v1.json; copy the public split file"""
    out = LIST.parent
    out.mkdir(parents=True, exist_ok=True)
    if LIST.exists() and not a.force:
        raise SystemExit(f"{LIST} exists (the list is frozen); --force to rebuild")
    split_copy = out / "split_v1_public.json"
    data = SRC_SPLIT.read_bytes()
    if not split_copy.exists():
        atomic_write_text(split_copy, data.decode())
    assert split_copy.read_bytes() == data, "list/split_v1_public.json differs from the project's file"
    held = dayfile.heldout_ids(split_copy)
    z = np.load(SRC_ASSIM, allow_pickle=False)
    assert str(z["split_sha256"]) == json.loads(data)["split_v1_sha256"], "assim_2026.npz was built with another split"
    ids, net = z["id"].astype(str), z["network"].astype(str)                 # identifiers and positions only: no value read
    flipped = set(json.loads(SRC_FLIPPED.read_text())["ids"])
    keep = (net == "OpenAQ") & ~np.isin(ids, sorted(flipped))
    ids, lat, lon, pos = ids[keep], z["lat"][keep], z["lon"][keep], z["pos"][keep]
    assert not held.intersection(ids.tolist()), "FIREWALL: held-out id in assim_2026.npz"
    loc = np.array([int(s.split(":")[1]) for s in ids], np.int64)
    cat = np.load(SRC_CATALOGUE, allow_pickle=False)["id"].astype(str)
    in_cat = np.isin(ids, cat)
    order = np.argsort(loc)
    dayfile._atomic_savez(LIST, id=ids[order], loc=loc[order], lat=lat[order].astype(np.float32),
                          lon=lon[order].astype(np.float32), pos=pos[order].astype(np.int64), in_catalogue=in_cat[order])
    meta = dict(written=dt.datetime.now(UTC).isoformat(timespec="seconds"), n=int(ids.size), n_in_catalogue=int(in_cat.sum()),
                n_outside_1km_mask=int((pos < 0).sum()), n_flipped_removed=int(np.isin(z["id"].astype(str), sorted(flipped)).sum()),
                n_heldout_in_split=len(held), n_heldout_openaq=sum(i.startswith("OpenAQ:") for i in held),
                source=str(SRC_ASSIM), source_sha256=sha256_file(SRC_ASSIM), split=str(SRC_SPLIT),
                split_sha256=hashlib.sha256(data).hexdigest(), split_v1_sha256=json.loads(data)["split_v1_sha256"],
                flipped=str(SRC_FLIPPED), catalogue=str(SRC_CATALOGUE),
                rule="OpenAQ monitors of assim_2026.npz (the assimilation set of the 2026 station archive: every monitor "
                     "of the archive that is not held out, yearly QC of A3(b) applied) minus the sign-flipped coordinates; "
                     "monitors unknown to that archive are NOT in the list (they have no yearly QC)")
    atomic_write_text(out / "assim_list_v1.json", json.dumps(meta, indent=1))
    MonitorList()                                                            # read-back with the firewall assert
    log(f"[list] {meta['n']} assimilated OpenAQ monitors ({meta['n_in_catalogue']} in the verification catalogue, "
        f"{meta['n_outside_1km_mask']} outside the 1 km mask, {meta['n_flipped_removed']} flipped removed) -> {LIST}")


# ------------------------------------------------------------------------------------------------ rate limit, client
class RateLimited(Exception):
    pass


class ApiError(Exception):
    pass


class RateLimiter:
    """at most per_min requests in any 60 s and per_hour in any 3600 s (rolling, this process and earlier runs through
    state_file), evenly spaced; the server's own counters of the key are obeyed when they are lower"""

    def __init__(self, per_min=PER_MIN, per_hour=PER_HOUR, state_file=None, clock=time.time, sleep=time.sleep, spacing=None):
        assert 1 <= per_min <= 60 and 1 <= per_hour <= 2000, "the key allows 60 / minute and 2,000 / hour"
        self.per_min, self.per_hour = int(per_min), int(per_hour)
        # spacing None: requests evenly spread over the hour (a long fetch); a number: short bursts at that spacing
        self.interval = max(60.0 / self.per_min, 3600.0 / self.per_hour) if spacing is None else max(float(spacing), 60.0 / self.per_min)
        self.clock, self.sleep = clock, sleep
        self.t = deque()
        self.server_until = 0.0
        self.state_file = Path(state_file) if state_file else None
        self.waited = 0.0
        if self.state_file and self.state_file.exists():
            try:
                now = self.clock()
                self.t.extend(sorted(x for x in json.loads(self.state_file.read_text()) if now - 3600 < x <= now))
            except (ValueError, OSError):
                pass

    def acquire(self):
        while True:
            now = self.clock()
            while self.t and self.t[0] <= now - 3600:
                self.t.popleft()
            wait = self.server_until - now
            if self.t:
                wait = max(wait, self.t[-1] + self.interval - now)
            if len(self.t) >= self.per_hour:
                wait = max(wait, self.t[len(self.t) - self.per_hour] + 3600 - now)
            if len(self.t) >= self.per_min:
                wait = max(wait, self.t[len(self.t) - self.per_min] + 60 - now)
            if wait <= 0:
                break
            self.waited += wait
            self.sleep(wait)
        self.t.append(self.clock())

    def feedback(self, headers):
        """x-ratelimit-remaining / -reset of the answer (the whole key, every process that uses it)"""
        try:
            rem, reset = int(headers.get("x-ratelimit-remaining", "")), float(headers.get("x-ratelimit-reset", ""))
        except (TypeError, ValueError):
            return
        if rem <= 3:
            self.server_until = max(self.server_until, self.clock() + reset + 1.0)

    def save(self):
        if self.state_file:
            try:
                atomic_write_text(self.state_file, json.dumps(list(self.t)))
            except OSError:
                pass


def read_key(path=None):
    p = Path(os.path.expanduser(path or os.environ.get("OPENAQ_API_KEY_FILE") or KEY_FILE_DEFAULT))
    if not p.is_file():
        raise SystemExit(f"OpenAQ key file not found: {p} (--key-file or $OPENAQ_API_KEY_FILE = the PATH of a file "
                         "that holds the key; the key itself is never passed on a command line or in the environment)")
    if p.stat().st_mode & 0o077:
        print(f"[feed] warning: {p} is readable by group or others: chmod 600", file=sys.stderr)
    key = p.read_text().strip()
    if not key:
        raise SystemExit(f"{p} is empty")
    return key


class OpenAQ:
    """GET with the key in the X-API-Key header; the key never leaves this object (errors are scrubbed)"""

    def __init__(self, key_file=None, limiter=None, timeout=60.0):
        import requests
        self._rq = requests
        self.__key = read_key(key_file)
        self._s = requests.Session()
        self._s.headers.update({"X-API-Key": self.__key, "Accept": "application/json", "User-Agent": "openaq-day-feed/1"})
        self.lim = limiter or RateLimiter()
        self.timeout = timeout
        self.n_requests = self.n_retries = 0
        self.t_requests = 0.0
        self.rate = {}

    def _scrub(self, text):
        return str(text).replace(self.__key, "<key>")

    def get(self, path, _attempts=5, **params):
        """-> parsed JSON; None on 404; RateLimited on 429 (never retried); ApiError after _attempts failed attempts"""
        last = ""
        for attempt in range(_attempts):
            if attempt:
                time.sleep(min(90, 5 * 3 ** (attempt - 1)))
                self.n_retries += 1
            self.lim.acquire()
            t0 = time.time()
            try:
                r = self._s.get(API + path, params=params, timeout=self.timeout)
            except (self._rq.ConnectionError, self._rq.Timeout) as e:
                self.n_requests += 1
                self.t_requests += time.time() - t0
                last = type(e).__name__
                continue
            self.n_requests += 1
            self.t_requests += time.time() - t0
            self.rate = {k.lower(): v for k, v in r.headers.items() if k.lower().startswith("x-ratelimit")}
            self.lim.feedback(self.rate)
            if r.status_code == 429:
                raise RateLimited(f"HTTP 429 on {path} after {self.n_requests} requests of this run; server counters {self.rate}")
            if r.status_code == 404:
                return None
            if r.status_code in (408, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
                continue
            if r.status_code != 200:
                raise ApiError(f"HTTP {r.status_code} on {path}: {self._scrub(r.text[:200])}")
            try:
                return r.json()
            except ValueError:
                last = "invalid JSON"
        raise ApiError(f"{path}: gave up after {_attempts} attempts ({last})")


# ------------------------------------------------------------------------------------------------ the two endpoints
def snapshot_locations(api, lst):
    """/v3/locations?parameters_id=2, paged by id -> {location id: {sensors, last, lat, lon, provider, monitor}} for
    the monitors of the list; the other locations are counted, not kept"""
    out, page, n_all, n_held = {}, 1, 0, 0
    t0 = time.time()
    while True:
        j = api.get("/v3/locations", parameters_id=PM25, limit=1000, page=page, order_by="id", sort_order="asc")
        if j is None:
            raise ApiError("/v3/locations: 404")
        res = j.get("results") or []
        for x in res:
            n_all += 1
            lid = int(x["id"])
            if lid in lst.held_loc:
                n_held += 1                                           # identifier counted; nothing else is kept
                continue
            if lid not in lst.index:
                continue
            co = x.get("coordinates") or {}
            out[lid] = dict(sensors=[int(s["id"]) for s in (x.get("sensors") or []) if (s.get("parameter") or {}).get("id") == PM25],
                            last=utc_ts((x.get("datetimeLast") or {}).get("utc")), lat=co.get("latitude"), lon=co.get("longitude"),
                            provider=(x.get("provider") or {}).get("name"), monitor=x.get("isMonitor"), mobile=x.get("isMobile"))
        if len(res) < 1000:
            break
        page += 1
        assert page < 200, "/v3/locations does not end"
    assert not lst.held_loc.intersection(out)
    return dict(t=t0, t_end=time.time(), requests=page, n_locations=n_all, n_heldout_skipped=n_held, loc=out)


def fetch_sensor(api, sensor, t_from, t_to):
    """/v3/sensors/{id}/hours of the window -> rows [from, to, coverage end, records, mean, min, max] (epoch s)"""
    rows, page = [], 1
    while True:
        j = api.get(f"/v3/sensors/{int(sensor)}/hours", _attempts=3, datetime_from=iso(t_from), datetime_to=iso(t_to), limit=1000, page=page)
        if j is None:
            return None
        res = j.get("results") or []
        for x in res:
            if ((x.get("parameter") or {}).get("id")) not in (PM25, None):
                continue
            p, cov, sm = x.get("period") or {}, x.get("coverage") or {}, x.get("summary") or {}
            avg = sm.get("avg")
            if avg is None:
                avg = x.get("value")
            pf, pt = utc_ts((p.get("datetimeFrom") or {}).get("utc")), utc_ts((p.get("datetimeTo") or {}).get("utc"))
            if pf != pf or pt != pt:
                continue                                              # a row without its period cannot be placed
            cto = utc_ts((cov.get("datetimeTo") or {}).get("utc"))
            rows.append([int(pf), int(pt), None if cto != cto else round(cto, 3), cov.get("observedCount"), avg,
                         sm.get("min"), sm.get("max")])
        if len(res) < 1000:
            return rows
        page += 1
        assert page < 20


RAW_MIN, RAW_MAX, RAW_PAGES = 2, 12, 2     # records per hour for which the raw records are fetched; pages of 1,000 at most


def needs_raw(rows):
    """True for a sensor whose hourly means are NOT the location-hour means of the archive rule: a few records per hour
    (RAW_MIN .. RAW_MAX) AND either records stamped exactly when the hour closes (15-minute data at :00: that record
    changes hour, and with a gap a whole hour can be lost) or hours that close off the hour (UTC+x:30 zones). Its raw
    records are then fetched and the rule is applied to them. Not needed: hourly sensors (one record per hour), sensors
    with irregular stamps (their records never fall on the closing time: the hourly mean IS the location-hour mean),
    denser sensors (more than RAW_MAX records per hour: one record in thirteen or more is displaced)."""
    multi = [r for r in rows if r[3] and r[3] >= RAW_MIN]
    if len(multi) < 6:
        return False
    c = sorted(r[3] for r in multi)
    if c[len(c) // 2] > RAW_MAX:
        return False
    on_edge = sum(1 for r in multi if r[2] is not None and abs(r[2] - r[1]) < 1.0)
    off_hour = sum(1 for r in multi if r[1] % 3600)
    return on_edge >= 0.2 * len(multi) or off_hour >= 0.5 * len(multi)


def fetch_raw(api, sensor, t_from, t_to):
    """/v3/sensors/{id}/measurements of the window -> [[stamp = END of the record's period (epoch s), value], ...];
    None when the window holds more than RAW_PAGES pages"""
    out, page = [], 1
    while True:
        j = api.get(f"/v3/sensors/{int(sensor)}/measurements", _attempts=3, datetime_from=iso(t_from), datetime_to=iso(t_to), limit=1000, page=page)
        res = (j or {}).get("results") or []
        for x in res:
            ts = utc_ts((((x.get("period") or {}).get("datetimeTo")) or {}).get("utc"))
            if ts == ts and x.get("value") is not None:
                out.append([round(ts, 3), x["value"]])
        if len(res) < 1000:
            return out
        page += 1
        if page > RAW_PAGES:
            return None


# ------------------------------------------------------------------------------------------------ state on disk
def read_hours(path):
    """hours.jsonl -> {sensor id: line dict} (the last complete line of a sensor wins; a truncated last line is ignored)"""
    out = {}
    if not Path(path).exists():
        return out
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
                out[int(d["s"])] = d
            except (ValueError, KeyError):
                continue
    return out


def window(day):
    """request window of a day: the 72 hours day-1 .. day+1 of the stuck rule, with a 2-hour margin on each side"""
    t0 = qc.day_epoch(day)
    return t0 - 86400 - 7200, t0 + 2 * 86400 + 7200


def candidates(snap, day):
    """sensors that can still give >= 18 valid hours of `day`: PM2.5 sensors of the listed locations whose last
    measurement is not older than 17:00 UTC of the day -> {sensor: (location, last stamp)}"""
    t17 = qc.day_epoch(day) + 17 * 3600
    out = {}
    for lid, v in snap["loc"].items():
        if v["last"] == v["last"] and v["last"] >= t17:
            for s in v["sensors"]:
                out[int(s)] = (int(lid), v["last"])
    return out


def save_snapshot(wd, snap):
    atomic_write_text(wd / "locations.json", json.dumps({**snap, "loc": {str(k): v for k, v in snap["loc"].items()}}))


def load_snapshot(wd):
    j = json.loads((wd / "locations.json").read_text())
    j["loc"] = {int(k): v for k, v in j["loc"].items()}
    return j


def resolve_day(s):
    if s == "yesterday":
        return dt.datetime.now(UTC).date() - dt.timedelta(days=1)
    return dt.date.fromisoformat(s)


# ------------------------------------------------------------------------------------------------ fetch
def add_raw(api, line, t_from, t_to):
    """adds the raw records of a sensor to its line ('m'), or why they are missing ('m_err'); the hourly means stay"""
    try:
        raw = fetch_raw(api, line["s"], t_from, t_to)
        if raw is None:
            line["m_err"] = f"more than {RAW_PAGES} pages"
        else:
            line["m"] = raw
    except ApiError as e:
        line["m_err"] = str(e)[-60:]


def cmd_fetch(a, api=None, lst=None, now=time.time, sleep=time.sleep):
    day = resolve_day(a.day)
    lst = lst or MonitorList()
    wd = Path(a.work) / str(day)
    wd.mkdir(parents=True, exist_ok=True)
    api = api or OpenAQ(a.key_file, RateLimiter(a.per_min, a.per_hour, Path(a.work) / "requests_last_hour.json"))
    hours_path = wd / "hours.jsonl"
    old_lines = read_hours(hours_path)
    done = {s for s, d in old_lines.items() if not (a.retry_failed and d.get("err"))}
    # sensors answered by an earlier run that still lack their raw records (a run made with --no-raw, or before them)
    refine = [] if a.no_raw else [s for s in sorted(done) if "m" not in old_lines[s] and "m_err" not in old_lines[s] and needs_raw(old_lines[s].get("r") or [])]
    t_from, t_to_full = window(day)
    day_end = qc.day_epoch(day) + 86400
    until = None
    if a.until:
        hh, mm = map(int, a.until.split(":"))
        until = day_end + hh * 3600 + mm * 60                       # UTC time of the day after `day` (the issue day)
    t_run, n0 = now(), api.n_requests
    stats = dict(day=str(day), started=iso(t_run), resumed_sensors=len(done), waves=[], status="running")
    stop = {"sig": None}

    def on_term(sig, _frm):
        stop["sig"] = sig
    old = signal.signal(signal.SIGTERM, on_term)
    fh = open(hours_path, "a")
    n_new = n_bad = n_failed = n_raw = 0
    rc = 0
    try:
        while True:
            snap = snapshot_locations(api, lst)
            save_snapshot(wd, snap)
            cand = candidates(snap, day)
            pending = [s for s in cand if s not in done]
            # a sensor is fetched once its location has reported past the end of the day (its day is complete);
            # when the time left only just suffices for what is pending, or without --until, everything is fetched
            need = len(pending) * api.lim.interval
            final = until is None or now() + need + 120 >= until or now() >= day_end + 30 * 3600
            ready = pending if final else [s for s in pending if cand[s][1] >= day_end]
            random.Random(f"{a.seed}-{day}-{len(done)}").shuffle(ready)
            if a.max_requests:
                ready = ready[:max(0, a.max_requests - (api.n_requests - n0))]
            if a.refine_only:
                ready = []
            wave = dict(t=iso(now()), listed_live=len(cand), done_before=len(done), pending=len(pending), ready=len(ready), final=bool(final))
            log(f"[fetch {day}] wave {len(stats['waves']) + 1}: {len(snap['loc'])} listed locations seen in {snap['requests']} requests, "
                f"{len(cand)} candidate sensors, {len(done)} done, {len(pending)} pending, {len(ready)} to fetch now"
                f"{' (final pass)' if final else ''}")
            for k, s in enumerate(ready):
                if stop["sig"] or (until and now() >= until):
                    break
                tq = now()
                try:
                    rows, err = fetch_sensor(api, s, t_from, min(t_to_full, tq + 3600)), None
                    if rows is None:
                        err = "HTTP 404"
                    n_bad = 0
                except ApiError as e:                                 # one sensor the server cannot answer: noted, the day goes on
                    rows, err = None, str(e)[-60:]
                    n_bad += 1
                    n_failed += 1
                    if n_bad >= MAX_CONSECUTIVE_FAILURES:
                        raise ApiError(f"{n_bad} sensors in a row failed (last: {e}): the API is not answering")
                line = dict(s=int(s), l=cand[s][0], tq=round(tq, 1), n=-1 if rows is None else len(rows), r=rows or [])
                if err:
                    line["err"] = err
                elif not a.no_raw and needs_raw(rows):
                    add_raw(api, line, t_from, min(t_to_full, tq + 3600))
                    n_raw += 1
                fh.write(json.dumps(line, separators=(",", ":")) + "\n")
                done.add(s)
                n_new += 1
                if n_new % 25 == 0:
                    fh.flush()
                    os.fsync(fh.fileno())
                    api.lim.save()
                if n_new % 250 == 0:
                    el = now() - t_run
                    log(f"[fetch {day}] {len(done)} / {len(cand)} sensors, {api.n_requests - n0} requests in {el / 60:.1f} min "
                        f"({(api.n_requests - n0) / max(el, 1) * 3600:.0f} / h), {api.n_retries} retries, server window {api.rate.get('x-ratelimit-remaining', '?')} left")
            wave["fetched"] = n_new
            stats["waves"].append(wave)
            if refine and not stop["sig"] and not (a.max_requests and api.n_requests - n0 >= a.max_requests):
                log(f"[fetch {day}] raw records of {len(refine)} sensors answered earlier (a few records per hour)")
                while refine and not stop["sig"] and not (until and now() >= until):
                    line = dict(old_lines[refine.pop()])
                    add_raw(api, line, t_from, min(t_to_full, line["tq"] + 3600))
                    fh.write(json.dumps(line, separators=(",", ":")) + "\n")
                    n_raw += 1
            left = [s for s in cand if s not in done]
            if a.refine_only and not stop["sig"]:
                stats["status"] = f"raw records only ({len(left)} sensors not fetched)"
                break
            if stop["sig"]:
                stats["status"] = "stopped by signal"
                rc = 1
                break
            if a.max_requests and api.n_requests - n0 >= a.max_requests:
                stats["status"] = "stopped at --max-requests"
                break
            if not left:
                stats["status"] = "complete"
                break
            if until is None or now() >= until:
                stats["status"] = "complete" if until is None and not left else f"deadline: {len(left)} sensors not fetched"
                break
            sleep(max(30.0, min(a.wave_minutes * 60.0, until - now() - len(left) * api.lim.interval - 120)))
    except RateLimited as e:
        stats["status"] = f"STOPPED: {e}"
        rc = 3
    except ApiError as e:
        stats["status"] = f"FAILED: {e}"
        rc = 2
    finally:
        fh.flush()
        os.fsync(fh.fileno())
        fh.close()
        api.lim.save()
        signal.signal(signal.SIGTERM, old)
        stats.update(ended=iso(now()), wall_s=round(now() - t_run, 1), requests=api.n_requests - n0, retries=api.n_retries,
                     seconds_in_requests=round(api.t_requests, 1), seconds_waited_for_rate=round(api.lim.waited, 1),
                     sensors_fetched_this_run=n_new, sensors_failed_this_run=n_failed, sensors_done=len(done),
                     sensors_with_raw_records_this_run=n_raw,
                     per_min=api.lim.per_min, per_hour=api.lim.per_hour)
        with open(wd / "fetch_runs.jsonl", "a") as f:
            f.write(json.dumps(stats) + "\n")
        log(f"[fetch {day}] {stats['status']}: {n_new} sensors ({n_failed} not answered by the server) in {stats['requests']} requests, "
            f"wall {stats['wall_s'] / 60:.1f} min ({stats['seconds_in_requests'] / 60:.1f} min inside requests), {len(done)} sensors on disk")
    return rc


# ------------------------------------------------------------------------------------------------ assemble
def rows_arrays(lines, lst, use_raw=True):
    """hours.jsonl lines -> (loc, t_from, t_to, cov_to, count, avg) of the listed locations; a sensor with raw records
    ('m') contributes them as one-record rows (the archive rule itself) instead of its hourly means"""
    L, F, T, C, N, V = [], [], [], [], [], []
    for d in lines.values():
        lid = int(d["l"])
        assert lid in lst.index, f"FIREWALL: location {lid} of hours.jsonl is not in the assimilated list"
        if use_raw and d.get("m") is not None:
            for ts, v in d["m"]:
                L.append(lid); F.append(ts - 3600.0); T.append(ts); C.append(ts); N.append(1.0); V.append(np.nan if v is None else v)
            continue
        for r in d.get("r") or []:
            L.append(lid); F.append(r[0]); T.append(r[1]); C.append(np.nan if r[2] is None else r[2])
            N.append(np.nan if r[3] is None else r[3]); V.append(np.nan if r[4] is None else r[4])
    return (np.array(L, np.int64), np.array(F, np.float64), np.array(T, np.float64), np.array(C, np.float64),
            np.array(N, np.float64), np.array(V, np.float64))


def day_values(day, lines, snap, lst, use_raw=True):
    """-> (list index of the kept monitors, daily mean, valid hours, counts) with the rules of qc.py"""
    loc, tf, tt, ct, cnt, avg = rows_arrays(lines, lst, use_raw)
    ids, mean, nh = qc.daily_means(loc, qc.hour_index(tf, tt, ct, cnt), avg, cnt, day)
    k = np.array([lst.index[int(l)] for l in ids], np.int64)
    rec_lat = np.array([snap["loc"].get(int(l), {}).get("lat") if snap["loc"].get(int(l), {}).get("lat") is not None else np.nan for l in ids], np.float64)
    rec_lon = np.array([snap["loc"].get(int(l), {}).get("lon") if snap["loc"].get(int(l), {}).get("lon") is not None else np.nan for l in ids], np.float64)
    ok = qc.position_ok(rec_lat, rec_lon, lst.lat[k], lst.lon[k]) if ids.size else np.zeros(0, bool)
    counts = dict(sensors_answered=len(lines), sensors_with_rows=sum(1 for d in lines.values() if d.get("n", 0) > 0),
                  sensors_failed=sum(1 for d in lines.values() if d.get("err")), sensors_with_raw_records=sum(1 for d in lines.values() if d.get("m") is not None),
                  locations_with_rows=int(np.unique(loc).size), monitors_ge18h=int(ids.size), dropped_position=int((~ok).sum()))
    return k[ok], mean[ok], nh[ok], counts


def cmd_assemble(a):
    day = resolve_day(a.day)
    lst = MonitorList()
    wd = Path(a.work) / str(day)
    lines, snap = read_hours(wd / "hours.jsonl"), load_snapshot(wd)
    k, mean, nh, counts = day_values(day, lines, snap, lst)
    runs = [json.loads(x) for x in (wd / "fetch_runs.jsonl").read_text().splitlines()] if (wd / "fetch_runs.jsonl").exists() else []
    cand = candidates(snap, day)
    meta = dict(source="OpenAQ v3 API: /v3/locations?parameters_id=2 + /v3/sensors/{id}/hours", list=str(LIST.name),
                assembled=dt.datetime.now(UTC).isoformat(timespec="seconds"), candidate_sensors=len(cand),
                sensors_not_fetched=len([s for s in cand if s not in lines]), **counts,
                fetch_first=min((d["tq"] for d in lines.values()), default=None), fetch_last=max((d["tq"] for d in lines.values()), default=None),
                requests=sum(r.get("requests", 0) for r in runs), fetch_status=runs[-1]["status"] if runs else None)
    p = dayfile.write_day_file(a.out, day, lst.id[k], lst.lat[k], lst.lon[k], mean, nhours=nh, meta=meta)
    log(f"[assemble {day}] {k.size} assimilated monitors with >= {qc.MIN_HOURS} valid hours ({counts['dropped_position']} dropped for "
        f"their position, {meta['sensors_not_fetched']} candidate sensors not fetched) -> {p}")
    return 0


def cmd_status(a):
    day = resolve_day(a.day)
    wd = Path(a.work) / str(day)
    lines = read_hours(wd / "hours.jsonl")
    print(f"{day}: {len(lines)} sensors answered, {sum(1 for d in lines.values() if d.get('n', 0) > 0)} with rows")
    if (wd / "locations.json").exists():
        cand = candidates(load_snapshot(wd), day)
        print(f"  last snapshot: {len(cand)} candidate sensors, {len([s for s in cand if s not in lines])} not fetched")
    if (wd / "fetch_runs.jsonl").exists():
        for x in (wd / "fetch_runs.jsonl").read_text().splitlines():
            r = json.loads(x)
            print(f"  run {r['started']} .. {r['ended']}: {r['status']}; {r['requests']} requests, wall {r['wall_s'] / 60:.1f} min")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list"); p.add_argument("--force", action="store_true")
    p = sub.add_parser("fetch")
    p.add_argument("day", help="UTC day of the measurements (yyyy-mm-dd or 'yesterday')")
    p.add_argument("--work", default=str(WORK))
    p.add_argument("--key-file", default=None, help="file that holds the OpenAQ key (default $OPENAQ_API_KEY_FILE or ~/.config/openaq/api_key)")
    p.add_argument("--per-min", type=int, default=PER_MIN)
    p.add_argument("--per-hour", type=int, default=PER_HOUR)
    p.add_argument("--until", default=None, help="HH:MM UTC of the day AFTER the measurements: wave mode, stop at that time")
    p.add_argument("--wave-minutes", type=float, default=20.0)
    p.add_argument("--max-requests", type=int, default=0)
    p.add_argument("--retry-failed", action="store_true", help="ask again the sensors that a previous run could not get (HTTP 5xx, 404)")
    p.add_argument("--no-raw", action="store_true", help="hourly means only: no second request for the sensors with a few records per hour")
    p.add_argument("--refine-only", action="store_true", help="no new sensor: only the raw records that the answered sensors still lack")
    p.add_argument("--seed", default="0")
    for name in ("assemble", "status"):
        p = sub.add_parser(name)
        p.add_argument("day")
        p.add_argument("--work", default=str(WORK))
        if name == "assemble":
            p.add_argument("--out", default=str(DAYS))
    a = ap.parse_args(argv)
    return {"list": cmd_list, "fetch": cmd_fetch, "assemble": cmd_assemble, "status": cmd_status}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
