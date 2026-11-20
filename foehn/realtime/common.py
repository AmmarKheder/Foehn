"""RT, the real-time chain of Foehn-CAMS (foehn_cams.py, the model of the Aurora protocol, 0.4 deg): shared paths, dates,
journal and retries. One directory per start day d: data/<yyyy-mm-dd>/ (inputs in the exact training format) and
out/<yyyy-mm-dd>/ (forecast). The model, FLEXGREEN and the checkpoint are those of the paper; only the chain is new."""
import datetime as dt
import os
import json
import sys
import time
from pathlib import Path

RT = Path(__file__).resolve().parent                      # .../global1km/realtime
GLOBAL1KM = RT.parent
sys.path.insert(0, str(GLOBAL1KM))
sys.path.insert(0, str(RT / "pylib"))                     # eccodes / cfgrib (binary wheels), cdsapi lives in the world env
DATA, OUT, LOGS = RT / "data", RT / "out", RT / "logs"
G = Path("/scratch/project_462001804/ammar/eccv/data/world/global1km")
RUN = GLOBAL1KM / "runs" / "cams04_flow_shift2.6_emis"    # Foehn-CAMS of the paper (trained --emis, 35 epochs)
CKPT = RUN / "final.ckpt"                                  # the checkpoint of the paper's ED predictions (cams12h/pred.npy)
VERSION = "1.1"
if (RT / "model.json").exists():                           # the live model: {"version": "1.2", "ckpt": "runs/rt12_a/final.ckpt"} (realtime/)
    _m = json.loads((RT / "model.json").read_text()); VERSION = _m["version"]; CKPT = RT / _m["ckpt"]
DAY0 = dt.date(2017, 1, 1)                                 # day index gi of meta.npz / fires.npy: days since 1 Jan 2017
CAMS0 = dt.date(2017, 10, 1)                               # cams_series.npy: t = 2 x (days since 1 Oct 2017) + (0: 00 UTC, 1: 12 UTC)


def parse_date(s):
    return dt.date.fromisoformat(s)


def gi_of(d):
    return (d - DAY0).days


def doy_of(d):
    return (d - dt.date(d.year, 1, 1)).days


def day_dir(d):
    p = DATA / d.isoformat()
    p.mkdir(parents=True, exist_ok=True)
    return p


def out_dir(d):
    p = OUT / d.isoformat()
    p.mkdir(parents=True, exist_ok=True)
    return p


def log(d, msg):
    """print and append to the day's journal (data/<date>/journal.txt), UTC time stamped"""
    line = f"{dt.datetime.utcnow():%Y-%m-%d %H:%M:%S}Z  {msg}"
    print(line, flush=True)
    with open(day_dir(d) / "journal.txt", "a") as f:
        f.write(line + "\n")


def meta_update(d, **kw):
    p = day_dir(d) / "meta.json"
    m = json.loads(p.read_text()) if p.exists() else {}
    m.update(kw)
    p.write_text(json.dumps(m, indent=1, default=str))
    return m


def meta(d):
    p = day_dir(d) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def retry(d, what, fn, deadline_utc=None, every_s=600, attempts=None):
    """call fn() until it succeeds; between failures wait every_s; give up at deadline_utc (datetime, UTC) or after
    `attempts` tries. Every failure is written to the journal. Returns fn()'s value, or raises the last error."""
    n = 0
    while True:
        n += 1
        try:
            return fn()
        except Exception as e:                                                  # source not there yet, queue hiccup
            msg = repr(e)[:400]
            late = (deadline_utc is not None and dt.datetime.utcnow() + dt.timedelta(seconds=every_s) > deadline_utc) or \
                   (attempts is not None and n >= attempts)
            log(d, f"{what}: attempt {n} failed ({msg})" + ("; giving up" if late else f"; retry in {every_s // 60} min"))
            if late:
                raise
            time.sleep(every_s)
