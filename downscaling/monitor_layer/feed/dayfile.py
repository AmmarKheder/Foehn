"""Day files of the monitor-fed layer (experiment M of Amendment A11, Addendum 2, point 1): writer, loader, firewall.

A day file holds the UTC daily mean PM2.5 of the ASSIMILATED monitors of one UTC day, in the format that
realtime_1km/stations_rt.py defines and realtime_1km/infer.py --stations reads:
    stations_<day>.npz : id (str, "OpenAQ:<location id>"), network (str), lat, lon (float32), value (float32, ug m-3),
                         date (str, the UTC day of the MEASUREMENTS), rule (str)
    extra keys written here (ignored by infer.py): nhours (uint8, valid hours behind the mean), meta (JSON string)

Firewall (A3(a), A11 Addendum 2): a held-out monitor of split_v1 is never an input. The held-out identifiers are
read from list/split_v1_public.json (a byte copy of data/processed/assim/split_v1_public.json: identifiers only, no
value, no position). write_day_file() REFUSES to write and load_day_file() REFUSES to return a file that contains one,
so no code that goes through this loader can read a held-out monitor's value.

Lag rule: for lead day L >= 1 the network may only see monitor values of L days before the valid day or older; with the
replay convention (lead day L of issue day d is valid on d + L - 1) that is the day before the issue for every lead:
monitor_day(d) = d - 1. check_lag() asserts it.
"""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPLIT = HERE / "list" / "split_v1_public.json"
RULE = ("A11-M feed: UTC daily mean of the assimilated OpenAQ monitors (frozen list assim_list_v1), QC of the "
        "verification observations (0 < PM <= 1000, location-hour means, stuck runs >= 24 h removed, >= 18 valid hours); "
        "held-out ids of split_v1 excluded (asserted)")
_HELD = {}


def as_date(d):
    return d if isinstance(d, dt.date) else dt.date.fromisoformat(str(d))


def heldout_ids(split=SPLIT):
    """frozenset of the held-out monitor identifiers of split_v1 (identifiers only)"""
    p = str(split)
    if p not in _HELD:
        pub = json.loads(Path(split).read_text())
        ids = frozenset(pub["heldout_ids"])
        assert len(ids) == int(pub["n_heldout"]) > 0, f"{split}: {len(ids)} ids, n_heldout {pub.get('n_heldout')}"
        _HELD[p] = ids
    return _HELD[p]


def split_sha256(split=SPLIT):
    return hashlib.sha256(Path(split).read_bytes()).hexdigest()


def assert_no_heldout(ids, split=SPLIT, what="monitor set"):
    """AssertionError if any identifier belongs to the held-out set of split_v1"""
    bad = heldout_ids(split).intersection(np.asarray(ids, str).tolist())
    assert not bad, f"FIREWALL: {len(bad)} held-out monitors in {what}, e.g. {sorted(bad)[:3]}"


def monitor_day(issue_day):
    """the only monitor day a forecast issued on issue_day may read: the day before the issue (every lead)"""
    return as_date(issue_day) - dt.timedelta(days=1)


def check_lag(file_date, valid_day, lag, allow_analysis=False):
    """lag = lead day L (the valid day is L - 1 days after the issue): the monitors must be L days older than the valid
    day or more; lag 0 (monitors of the valid day itself = the same-day analysis of A3) only when allow_analysis"""
    file_date, valid_day, lag = as_date(file_date), as_date(valid_day), int(lag)
    if lag == 0:
        assert allow_analysis, "lag 0 (monitors of the valid day) is the A3 analysis, not a forecast input"
        assert file_date <= valid_day, f"monitors of {file_date} are later than the valid day {valid_day}"
        return
    assert lag >= 1, f"lag {lag}"
    assert file_date <= valid_day - dt.timedelta(days=lag), \
        f"LAG RULE: monitors of {file_date} given to valid day {valid_day} with lag {lag} (latest allowed {valid_day - dt.timedelta(days=lag)})"
    assert file_date < valid_day


def _atomic_savez(path, **arr):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}.npz")
    with open(tmp, "wb") as f:
        np.savez(f, **arr)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def write_day_file(out_dir, day, ids, lat, lon, value, nhours=None, network="OpenAQ", rule=RULE, meta=None, split=SPLIT):
    """-> <out_dir>/stations_<day>.npz ; refuses held-out identifiers, duplicates, non-finite or out-of-range values"""
    day = as_date(day)
    ids = np.asarray(ids, str)
    lat, lon, value = (np.asarray(x, np.float32) for x in (lat, lon, value))
    assert ids.ndim == 1 and ids.shape == lat.shape == lon.shape == value.shape, (ids.shape, lat.shape, lon.shape, value.shape)
    assert_no_heldout(ids, split, f"the day file of {day}")
    assert len(set(ids.tolist())) == ids.size, "duplicate monitor ids"
    assert np.isfinite(value).all() and (value > 0).all() and (value <= 1000).all(), "values outside (0, 1000]"
    assert np.isfinite(lat).all() and np.isfinite(lon).all() and (np.abs(lat) <= 90).all() and (np.abs(lon) <= 180).all()
    net = np.asarray(network, str)
    net = np.full(ids.size, str(net)) if net.ndim == 0 else net
    assert net.shape == ids.shape
    nh = np.full(ids.size, 0, np.uint8) if nhours is None else np.asarray(nhours, np.uint8)
    assert nh.shape == ids.shape
    m = dict(meta or {})
    m.setdefault("split_v1_public_sha256", split_sha256(split))
    m.setdefault("n", int(ids.size))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = _atomic_savez(out / f"stations_{day}.npz", id=ids, network=net, lat=lat, lon=lon, value=value,
                      date=np.array(str(day)), rule=np.array(str(rule)), nhours=nh, meta=np.array(json.dumps(m)))
    back = load_day_file(p, expect_date=day, split=split)                 # read-back: the file on disk is what was meant
    assert back["id"].size == ids.size and np.array_equal(back["value"], value)
    return p


def load_day_file(path, expect_date=None, split=SPLIT):
    """THE loader of this experiment: -> dict(id, network, lat, lon, value, date (datetime.date), nhours, meta (dict)).
    Asserts: no held-out identifier (checked on the identifiers, before the value array is read), the expected date,
    finite positive values. A file that fails is never returned."""
    with np.load(path, allow_pickle=False) as z:
        ids = z["id"].astype(str)
        assert_no_heldout(ids, split, str(path))                          # before any value is read
        date = dt.date.fromisoformat(str(z["date"]))
        if expect_date is not None:
            assert date == as_date(expect_date), f"{path}: holds {date}, expected {as_date(expect_date)}"
        out = dict(id=ids, network=z["network"].astype(str), lat=np.asarray(z["lat"], np.float32),
                   lon=np.asarray(z["lon"], np.float32), value=np.asarray(z["value"], np.float32), date=date,
                   rule=str(z["rule"]) if "rule" in z.files else "",
                   nhours=np.asarray(z["nhours"], np.uint8) if "nhours" in z.files else np.zeros(ids.size, np.uint8),
                   meta=json.loads(str(z["meta"])) if "meta" in z.files else {})
    assert out["value"].shape == out["lat"].shape == out["lon"].shape == ids.shape
    assert len(set(ids.tolist())) == ids.size, f"{path}: duplicate ids"
    assert np.isfinite(out["value"]).all() and (out["value"] > 0).all(), f"{path}: non-finite or non-positive values"
    return out


def load_for_issue(day_dir, issue_day, valid_day=None, lag=None, split=SPLIT):
    """day file that a forecast issued on issue_day may read (monitors of issue_day - 1), with the lag rule checked for
    the given valid day and lag when they are passed"""
    md = monitor_day(issue_day)
    f = load_day_file(Path(day_dir) / f"stations_{md}.npz", expect_date=md, split=split)
    if valid_day is not None:
        check_lag(f["date"], valid_day, lag if lag is not None else (as_date(valid_day) - md).days)
    return f
