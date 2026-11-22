"""a11/stageM/final_test/ft_lib.py -- shared definitions of the FINAL TEST of the monitor-fed layer (experiment M) on the
autumn 2026 real-time record (draft Addendum 8 to Amendment A11: ADDENDUM_draft.md). numpy only: imported by the CPU
scripts (build_inputs.py, score_final.py, test_final.py cpu) and by the GPU inference (infer_m.py).

Fixed here, before any score of this test:
  * the issue days (the 48 days scored by realtime_1km/replay/score_cams3h: 2026-09-06..09-20 and 2026-10-10..11-11);
  * the checkpoint of M and the blending radius R (from the 2021 validation only: select_2021.py, data/selection.json);
  * the place rule of Addendum 3 (2-degree cells of split_v1) and the firewall on every file that holds an observation.
"""
import datetime as dt
import hashlib
import json
import os
import struct
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

P = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution"))
SM = P / "a11/stageM"
FT = SM / "final_test"
R = P / "realtime_1km/replay"
DATA = FT / "data"
OUT = FT / "out"
RESULT = FT / "result"
FILED = FT / "FILED"

# ---- the record --------------------------------------------------------------------------------------------------------
DAYS_ALL = R / "days_all.txt"
FIRST_TEST_ISSUE = "2026-09-06"           # Addendum 4: the coarse model in service was fine-tuned on analyses to 2026-09-05
POINTS = R / "points"                      # station-free chain of the record: km (5, 8, n), clim (5, n); no observation
CAMS3H = R / "cams3h/points"               # CAMS forecast of the same cycle, mean of its eight 3-hourly steps (5, n)
WORK = R / "work"                          # encoder channels chan_L<L>.npz of every issue day (no observation)
POSITIONS = R / "positions.npz"            # catalogue monitors -> evaluated pixels (positions only)
CATALOGUE = P / "realtime_1km/verif/catalogue.npz"
RECORD_PAIRS = R / "score_cams3h/work/pairs.npz"   # the record's station-day table: the TRUTH (read by the scorer only)

# ---- experiment M ------------------------------------------------------------------------------------------------------
RUN_M = SM / "runs/pmM_base"
SEL2021 = RUN_M / "val_lag/sel2021_all.json"
CKPT_FREE = P / "runs_ghap/pmG_final/ckpt/best.ckpt"
SELECTION = DATA / "selection.json"        # written by select_2021.py: checkpoint and R (2021 validation only)
INPUTS = DATA / "inputs.npz"               # written by build_inputs.py: monitor inputs of every issue day
FORECASTS = OUT / "forecasts.npz"          # written by infer_m.py (full run)

# ---- monitors ----------------------------------------------------------------------------------------------------------
FEED_REPLAY = SM / "feed/replay"           # stations_<day>.npz: QC'd UTC daily means of the listed catalogue monitors
FEED_LIST = SM / "feed/list/assim_list_v1.npz"
FEED_AVAIL = SM / "feed/reports/availability.json"
ASSIM_2026 = P / "data/processed/assim/assim_2026.npz"
SPLIT_PUBLIC = P / "data/processed/assim/split_v1_public.json"
SPLIT_FULL = P / "data/processed/stations/split_v1.json"     # read ONCE by build_inputs.py (key heldout_blocks), before the firewall
REVIEW_CELLS = SM / "review/out/heldout_proximity.json"      # the 188 cells as reconstructed independently (cross-check)
GHAP_ROOT = P / "data/processed/ghap"
GHAP_NC, GHAP_LAT0, GHAP_LON0, GHAP_RES = 36000, 83.995, -179.995, 0.01
R_EARTH_KM = 6371.0
DIST_CLASSES = (("0-10km", 0.0, 10.0), ("10-50km", 10.0, 50.0), ("50-150km", 50.0, 150.0), (">150km", 150.0, np.inf))

# Every path that holds a monitor OBSERVATION used as a truth (or the held-out monitors' values): the inference, the input
# builder and the tests run with an audit hook that refuses to open any of them. The scorer opens RECORD_PAIRS only after
# FILED exists (decision mode); its --tests mode installs the same firewall.
TRUTH_MARKERS = ("/a11/test2026/obs/", "/hindcast/", "/realtime_1km/verif/obs/", "/realtime_1km/verif/raw/", "/data/processed/stations",
                 "heldout_sel", "/replay/score/work", "/replay/score_cams3h/work", "/replay/score_cal/", "/realtime_1km/rc/",
                 "/a11/stageM/feed/days/", "/a11/stageM/feed/work/", "/world/stations", "/test2022")
_TRACE = []
_FW = []


def issue_days():
    d = [x for x in DAYS_ALL.read_text().split() if x >= FIRST_TEST_ISSUE]
    assert len(d) == 48 and d[0] == "2026-09-06" and d[-1] == "2026-11-11", (len(d), d[:1], d[-1:])
    return d


def install_truth_firewall(trace=False):
    """audit hook (once per process): open() of a truth file raises PermissionError; trace=True also records every opened
    path (written by dump_trace)"""
    if _FW:
        return

    def hook(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            p = os.fsdecode(args[0])
            if trace:
                _TRACE.append(p)
            if any(m in p for m in TRUTH_MARKERS):
                raise PermissionError(f"final_test firewall: {p} holds observations (read only by the scorer after FILED)")

    sys.addaudithook(hook)
    _FW.append(hook)


def opened_paths():
    return sorted(set(_TRACE))


def sha256_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _atomic(path, writer, suffix):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.stem + ".", suffix=suffix, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o660)                             # the project's group permissions (mkstemp creates 0600)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def save_npz(path, **arrays):
    """np.savez (uncompressed) through a temporary file in the same directory, fsync, rename"""
    return _atomic(path, lambda f: np.savez(f, **arrays), ".npz")


def save_json(path, obj):
    return _atomic(path, lambda f: f.write((json.dumps(obj, indent=1, default=_jdef) + "\n").encode()), ".json")


def save_text(path, text):
    return _atomic(path, lambda f: f.write(text.encode()), ".txt")


def _jdef(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (dt.date, Path)):
        return str(x)
    raise TypeError(type(x))


def npz_memmap(path, key):
    """read-only memory map of member `key` of an UNCOMPRESSED .npz, without reading the member: rows can then be gathered
    one by one (the held-out rows of a file are never copied into an array of the process)"""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(key + ".npy")
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"{path}:{key} is compressed: no row access without reading the whole member")
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        h = f.read(30)
        assert h[:4] == b"PK\x03\x04", path
        n, e = struct.unpack("<HH", h[26:30])
        f.seek(info.header_offset + 30 + n + e)
        ver = np.lib.format.read_magic(f)
        rd = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = rd(f)
        off = f.tell()
    assert not dtype.hasobject, (path, key)
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=off, order="F" if fortran else "C")


def gather_rows(path, key, rows, col=None):
    """values of member `key` at the sorted unique row indices `rows` (and column `col` of a 2-D member), read through
    npz_memmap; returns a new float32 array"""
    rows = np.asarray(rows, np.int64)
    assert rows.size == 0 or (np.all(np.diff(rows) > 0) and rows[0] >= 0), "rows must be sorted and unique"
    mm = npz_memmap(path, key)
    out = np.asarray(mm[rows] if col is None else mm[rows, int(col)], np.float32).copy()
    del mm
    return out


def cell2(lat, lon):
    """2-degree cell of split_v1 (scripts/build_stations_a3.py cmd_split): floor((lat + 90) / 2) * 180 + floor((lon + 180) / 2)"""
    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    return np.floor((lat + 90.0) / 2.0).astype(np.int64) * 180 + np.floor((lon + 180.0) / 2.0).astype(np.int64)


def unit(lat, lon):
    la, lo = np.deg2rad(np.asarray(lat, np.float64)), np.deg2rad(np.asarray(lon, np.float64))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def nearest_km(qlat, qlon, mlat, mlon, chunk=2048):
    """great-circle distance (km) from each query point to its nearest point of the set (inf when the set is empty)"""
    n = len(qlat)
    out = np.full(n, np.inf)
    if len(mlat) == 0:
        return out
    mu = unit(mlat, mlon)
    qu = unit(qlat, qlon)
    for s in range(0, n, chunk):
        c = np.clip(qu[s:s + chunk] @ mu.T, -1.0, 1.0).max(1)
        out[s:s + chunk] = R_EARTH_KM * np.arccos(c)
    return out


def pixel_latlon(pos):
    """centre of the 1 km pixels of the GHAP mask order (as replay/code/infer_points.sub_store)"""
    flat = np.asarray(np.load(GHAP_ROOT / "grid/ghap_mask.npy", mmap_mode="r")[np.asarray(pos, np.int64)]).astype(np.int64)
    r, c = flat // GHAP_NC, flat % GHAP_NC
    return GHAP_LAT0 - GHAP_RES * r, GHAP_LON0 + GHAP_RES * c


def dist_class(d):
    """index of the distance class of DIST_CLASSES (lower bound exclusive, upper inclusive, as cdffm_aq_m.lagval)"""
    d = np.asarray(d, np.float64)
    k = np.full(d.shape, 3, np.int64)
    k[d <= 150.0] = 2
    k[d <= 50.0] = 1
    k[d <= 10.0] = 0
    return k


def load_selection():
    s = json.loads(SELECTION.read_text())
    assert s["checkpoint"]["step"] > 0 and s["R_km"] in (10.0, 50.0, 150.0, None), s
    return s
