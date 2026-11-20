"""cdffm_aq.store_global -- GlobalStore: the global 2017-2024 store (data/processed/global, see its manifest.json) with the
SmokeStore interface of data_aq.py, so BatchStream, PixelArrays, assemble_inputs, train_aq and eval_aq run on it
unchanged (train_aq/eval_aq only need 'global' added to --store / open_store).

Firewall (PROTOCOL.md 1.7, 2.1):
  * the training sampler sees train_dates only: split A training days 2019-07-11 .. 2022-12-31 (1,270), asserted;
  * the point files opened are lghap_points/{idx,val}_{2019b,2020,2021,2022}.npy; nothing in the sealed directory (2017, 2018,
    2019-01-01..07-10, 2024) is ever opened by this class, and points() refuses any date outside the training split;
  * hold_dates = the 358 validation days 2023-01-08 .. 12-31 (full_2023.npy rows); 2024 is not reachable;
  * no station file is opened.
  * install_firewall() (called by __init__ unless firewall=False) adds an audit hook that makes every open() of a sealed
    LGHAP file (data/processed/global/sealed/, the raw LGHAP zips) or of a station file raise PermissionError in the
    process (PROTOCOL.md 1.7-8, Appendix B-7): training and selection code cannot read 2024 or 2017..2019-07-10.
Inputs of day d (raw, as SmokeStore.raw_inputs): CAMS = coarse/cams_pm25_025.npy row (d - 2017-01-01); ERA5 = Foehn
met.npy row (d - 2017-01-01) up to 2022-12-31, coarse/met_2023_2024.npy row (d - 2023-01-01) after; T6 =
transport/t6_<year>.npy row doy(d). Fires (--fires, PROTOCOL.md amendment 2026-10-26):
  injh (default)  coarse/fires_injh_2017_2022.npy / fires_injh_2023_2024.npy: channel 2 = GFAS v1.2 injh from
                  2018-07-02 (mami before: robustness days only); channels 0-1 bit-identical to the mami stores. Used only
                  once its builder's check (data/raw/gfas_injh/check_report.json) has passed.
  mami            Foehn fires.npy (2017-2022) / coarse/fires_2023_2024.npy (channel 2 = mami), as the smoke.
sigma_k (PROTOCOL.md A1(e)): 2 = seasonal sigma_p(doy) of the K = 2 variance fit (runs before 2026-10-27); 0 = annual
sigma_p (constant term), used by the retrained baseline pmA_baseK0.
Derived split A23 (PROTOCOL.md A1(h), the final fit): split A's climatology, statics, sigma floor and CAMS/T6
normalisation (files *_A, no refit); train_dates = A's 1,270 days + the 2023 hold-out days 2023-01-08 .. 12-31 minus the
24 SEL days (hold indices 0, 15, .., 345) = 1,604 days, served from lghap_points/{idx,val}_2023.npy; hold_dates are A's
(so the SEL indices name the same days); 2024 and 2017-01-01 .. 2019-07-10 stay sealed. Split A is unchanged.
raw_inputs(d) reads only the input stores (CAMS, met, fires, T6), never LGHAP: it serves any day of 2017-2024 (the 2024
predictions of the unsealing pass are made by a model process under the firewall).
Backfill 2025-2026 (user order 2026-11-02, scripts/backfill_2025_2026.py): days >= 2025-01-01 come from
coarse/{cams_pm25_025,met,fires_injh}_2025_2026.npy (row = d - 2025-01-01) and transport/t6_2025|2026.npy (row = doy),
through Backfill2025: a day is served only up to the last complete day of EVERY input (COVERAGE_2025_2026.json), beyond
it raw_inputs raises KeyError (those days belong to the real-time chain). The 2017-2024 path is unchanged."""
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np

from . import PROJECT
from .data_aq import (FOEHN_GLOBAL1KM, LGHAP_LAT0, LGHAP_LON0, LGHAP_NLON, LGHAP_RES, MET_FIRST, Norm, NpyRows,
                      parse_date)

GLOBAL_ROOT = PROJECT / "data" / "processed" / "global"
D2023 = dt.date(2023, 1, 1)
D2025 = dt.date(2025, 1, 1)
T6_YEARS = range(2017, 2027)
BACKFILL_FILES = {"cams": "coarse/cams_pm25_025_2025_2026.npy", "met": "coarse/met_2025_2026.npy",
                  "fires": {"injh": "coarse/fires_injh_2025_2026.npy", "mami": None}}       # no mami store after 2024
COVERAGE = "COVERAGE_2025_2026.json"
TRAIN_FILES = ("2019b", "2020", "2021", "2022")
INJH_REPORT = PROJECT / "data" / "raw" / "gfas_injh" / "check_report.json"
FIRES_FILES = {"mami": (FOEHN_GLOBAL1KM / "fires.npy", "coarse/fires_2023_2024.npy"),
               "injh": ("coarse/fires_injh_2017_2022.npy", "coarse/fires_injh_2023_2024.npy")}
SEALED_MARKERS = ["/data/processed/global/sealed", "/lghap_v2", "/data/stations", "/stations/"]   # + split extras
SEL_HOLD_IDX = tuple(range(0, 346, 15))         # the 24 SEL days 2023-01-08 + 15k (PROTOCOL.md A1(c)), indices into hold_dates
# derived splits (PROTOCOL.md A1(h)): base split's files and hold-out days; extra training days taken from the hold-out
# days in [extra_train], minus exclude_hold_idx; their points from the point files of point_files
DERIVED_SPLITS = {"A23": {"base": "A", "extra_train": ("2023-01-08", "2023-12-31"), "exclude_hold_idx": SEL_HOLD_IDX,
                          "point_files": ("2023",), "n_train": 1604}}
_FIREWALL = []


def injh_status(root=GLOBAL_ROOT, report=INJH_REPORT):
    """(ok, reason): the injh fires stores exist with the expected shapes and their builder's check passed"""
    root = Path(root)
    paths = [root / f for f in FIRES_FILES["injh"]]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        return False, f"injh store not built yet: {missing}"
    if not Path(report).exists():
        return False, f"injh store not checked yet ({report} missing)"
    rep = json.loads(Path(report).read_text())
    for p in paths:
        r = rep.get(p.name)
        if r is None:
            return False, f"{p.name} absent from {report}"
        bad = [k for k in ("finite_sampled_rows", "finite_all_rows_ch2_stride4", "ch01_bit_identical_to_mami_store")
               if r.get(k) is not True]
        if bad:
            return False, f"{p.name}: check failed on {bad} ({report})"
    return True, "ok"


class Backfill2025:
    """the 2025-2026 input rows (row = d - 2025-01-01) and their coverage. lookup(d) gives the (NpyRows, row) of every
    input of day d >= 2025-01-01, or raises KeyError when d is beyond the last complete day of any input (or the
    backfill is absent): rows never built are NaN in the files, so no day is served without its coverage entry."""

    def __init__(self, root, fires, t6):
        root = Path(root)
        files = {"cams": root / BACKFILL_FILES["cams"], "met": root / BACKFILL_FILES["met"]}
        ff = BACKFILL_FILES["fires"].get(fires)
        files["fires"] = root / ff if ff else None
        self.rows = {k: NpyRows(p) for k, p in files.items() if p is not None and p.exists()}
        self.t6 = t6
        self.last = {}
        cov = root / COVERAGE
        if cov.exists():
            c = json.loads(cov.read_text()).get("inputs", {})
            for k in ("cams", "met", "fires"):
                v = c.get(k, {}).get("last_complete_day")
                self.last[k] = parse_date(v) if v else None
            self.last["t6"] = {int(y): (parse_date(v["last_complete_day"]) if v.get("last_complete_day") else None)
                               for y, v in c.get("t6", {}).items()}
        self.available = all(k in self.rows for k in ("cams", "met", "fires")) and all(self.last.get(k) for k in ("cams", "met", "fires"))
        lasts = [self.last.get(k) for k in ("cams", "met", "fires")]
        last_t6 = None                       # the last T6 day whose earlier T6 years are complete (min over years was wrong:
        for y in sorted(self.last.get("t6", {})):   # a complete 2025 capped last_day at 2025-12-31 while lookup served 2026)
            v = self.last["t6"][y]
            if v is None:
                break
            last_t6 = v
            if v < dt.date(y, 12, 31):
                break
        lasts.append(last_t6)
        self.last_day = min(lasts) if self.available and all(lasts) else None   # last day served (all inputs, lookup's rule)

    def lookup(self, d):
        if not self.available:
            raise KeyError(f"{d}: the 2025-2026 inputs are not backfilled (scripts/backfill_2025_2026.py, {COVERAGE})")
        for k in ("cams", "met", "fires"):
            if d > self.last[k]:
                raise KeyError(f"{d}: after the last complete {k} day {self.last[k]} ({COVERAGE}); the real-time chain serves it")
        t6l = self.last.get("t6", {}).get(d.year)
        if t6l is None or d > t6l or d.year not in self.t6:
            raise KeyError(f"{d}: after the last complete T6 day {t6l} ({COVERAGE}); the real-time chain serves it")
        k = (d - D2025).days
        return {"cams": (self.rows["cams"], k), "met": (self.rows["met"], k), "fires": (self.rows["fires"], k),
                "t6": (self.t6[d.year], (d - dt.date(d.year, 1, 1)).days)}


def install_firewall(extra=()):
    """audit hook (once per process): open() of a sealed LGHAP file or a station file raises PermissionError; extra:
    more path markers (e.g. the 2022 point files when the training split ends before 2022: 2022 stays clean)"""
    for m in extra:
        if m not in SEALED_MARKERS:
            SEALED_MARKERS.append(m)
    if _FIREWALL:
        return

    def hook(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            p = os.fsdecode(args[0])
            if any(m in p for m in SEALED_MARKERS):
                raise PermissionError(f"firewall (PROTOCOL.md 1.7, 2.1): {p} is sealed / evaluation-only")

    sys.addaudithook(hook)
    _FIREWALL.append(hook)


class GlobalStore:
    name = "global"

    def __init__(self, root=GLOBAL_ROOT, split="A", fires="injh", firewall=True, sigma_k=2):
        from .model_aq import GRIDS
        root = Path(root)
        self.root = root
        if int(sigma_k) not in (0, 2):
            raise ValueError(f"sigma_k={sigma_k!r}: 2 (seasonal sigma, K = 2) or 0 (annual sigma, PROTOCOL.md A1(e))")
        self.sigma_k = int(sigma_k)                 # read by data_aq.PixelArrays.mu_sigma
        man = json.loads((root / "manifest.json").read_text())
        self.manifest = man
        lg = man["lghap"]
        der = DERIVED_SPLITS.get(split)
        base = der["base"] if der else split            # files of the split (climatology, statics): the base's
        if f"split_{base}" not in lg:
            raise SystemExit(f"GlobalStore(split={split!r}): split not built (manifest lghap.split_{base} missing; "
                             f"refit climatology/statics, cell means and BCSD for it first)")
        sp = lg[f"split_{base}"]
        self.split, self.base_split = split, base
        t1_ = parse_date(sp["train"][1])
        files_ = lg["points"]["files"]
        if firewall:   # point files of training-period years outside this split are unreadable (e.g. 2022 for A21)
            install_firewall([files_[n][k] for n in TRAIN_FILES for k in ("idx", "val")
                              if parse_date(files_[n]["first"]) > t1_])
        t0, t1, nT = sp["train"]
        self.train_dates = [parse_date(t0) + dt.timedelta(k) for k in range(int(nT))]
        assert self.train_dates[-1] == parse_date(t1)
        v0, v1 = sp["val"]
        self.hold_dates = [parse_date(v0) + dt.timedelta(k) for k in range((parse_date(v1) - parse_date(v0)).days + 1)]
        assert all(d < D2023 for d in self.train_dates) and all(d.year == 2023 for d in self.hold_dates)
        self.sel_dates = [self.hold_dates[t] for t in SEL_HOLD_IDX]
        if der:                                         # A1(h): + the non-SEL 2023 hold-out days in extra_train
            e0, e1 = (parse_date(x) for x in der["extra_train"])
            excl = {self.hold_dates[t] for t in der["exclude_hold_idx"]}
            self.train_dates = self.train_dates + [d for d in self.hold_dates if e0 <= d <= e1 and d not in excl]
            assert len(self.train_dates) == der["n_train"] and not excl & set(self.train_dates)
            assert all(d.year < 2024 for d in self.train_dates) and self.train_dates == sorted(set(self.train_dates))
        self.grid_native = GRIDS["025"]
        mask = np.load(root / "grid/lghap_mask.npy").astype(np.int64)
        i, j = mask // LGHAP_NLON, mask % LGHAP_NLON
        self.lat = LGHAP_LAT0 + LGHAP_RES * i
        self.lon = LGHAP_LON0 + LGHAP_RES * j
        self.n_pix = mask.size
        self.clim_mean = np.load(root / f"lghap_clim/clim_mean_{base}.npy")
        self.clim_var = np.load(root / f"lghap_clim/clim_var_{base}.npy")
        self.static = np.load(root / f"lghap_clim/static_{base}.npy")
        self.static_dim = self.static.shape[1]
        self.sigma_floor = float(sp.get("sigma_floor", lg["static"]["sigma_floor"]))
        # training point files only (never sealed/), and only those overlapping this split's training days
        files = lg["points"]["files"]
        self._pts = []
        for name in TRAIN_FILES + (der["point_files"] if der else ()):
            f = files[name]
            if parse_date(f["first"]) > self.train_dates[-1]:
                continue
            assert not f["sealed"] and not f["idx"].startswith("sealed")
            self._pts.append((parse_date(f["first"]), parse_date(f["last"]), NpyRows(root / f["idx"]), NpyRows(root / f["val"])))
        self.n_points = self._pts[0][2].shape[1]
        assert all(idx.shape[1] == val.shape[1] == self.n_points for _, _, idx, val in self._pts)
        self.hold = NpyRows(root / "lghap_eval/full_2023.npy")
        self.cams = NpyRows(root / "coarse/cams_pm25_025.npy")
        self.met = (NpyRows(FOEHN_GLOBAL1KM / "met.npy"), NpyRows(root / "coarse/met_2023_2024.npy"))
        if fires not in FIRES_FILES:
            raise ValueError(f"fires={fires!r}: choose one of {sorted(FIRES_FILES)}")
        if fires == "injh":
            ok, why = injh_status(root)
            if not ok:
                raise SystemExit(f"GlobalStore(fires='injh'): {why}. Use fires='mami' (--fires mami) for tests only.")
        self.fires_name = fires
        self.fires = tuple(NpyRows(root / f) for f in FIRES_FILES[fires])
        self.t6 = {y: NpyRows(root / f"transport/t6_{y}.npy") for y in T6_YEARS
                   if (root / f"transport/t6_{y}.npy").exists()}
        self.bf = Backfill2025(root, fires, self.t6)
        nc = sp.get("normalisation", man["normalisation_convenience"])
        self.norm = Norm(nc["cams_log1p_mean_std"][0], nc["cams_log1p_mean_std"][1], nc["t6_channel_mean"], nc["t6_channel_std"])
        # whole LGHAP grid as the "tile" (maps in eval_aq)
        self.tile_shape = (12000, LGHAP_NLON)
        self.tile_row, self.tile_col = i, j
        self.tile_row_lat = LGHAP_LAT0 + LGHAP_RES * np.arange(12000)
        self.tile_col_lon0 = LGHAP_LON0
        self.res_deg = LGHAP_RES
        assert self.clim_mean.shape == (self.n_pix, 5) and self.static.shape[0] == self.n_pix
        assert self.hold.shape == (365, self.n_pix)
        assert self.cams.row_shape == (self.grid_native.H, self.grid_native.W)
        assert self.met[0].row_shape == self.met[1].row_shape == (37, self.grid_native.H, self.grid_native.W)
        assert self.fires[0].row_shape == self.fires[1].row_shape == (3, self.grid_native.H, self.grid_native.W)
        assert len(self.met[0]) >= (D2023 - MET_FIRST).days and len(self.fires[0]) >= (D2023 - MET_FIRST).days
        for k, r in self.bf.rows.items():
            assert r.row_shape == self.raw_shapes()[k] and len(r) == 730, (k, r.shape)
        self._train_set = set(self.train_dates)

    def raw_shapes(self):
        return {"cams": self.cams.row_shape, "met": self.met[0].row_shape, "fires": self.fires[0].row_shape,
                "t6": (6, self.grid_native.H // 2, self.grid_native.W // 2)}

    def raw_inputs(self, d, out=None):
        out = out or {}
        if d >= D2025:                                  # backfill 2025-2026 (coverage-checked)
            rows = self.bf.lookup(d)
        else:
            new = d >= D2023
            k = (d - D2023).days if new else (d - MET_FIRST).days
            rows = {"cams": (self.cams, (d - MET_FIRST).days), "met": (self.met[new], k), "fires": (self.fires[new], k),
                    "t6": (self.t6[d.year], (d - dt.date(d.year, 1, 1)).days)}
        return {key: r.read(i, out.get(key)) for key, (r, i) in rows.items()}

    def points(self, t):
        """training day t (index into train_dates): (positions k (n_points,) uint32, raw LGHAP integers int16)"""
        d = self.train_dates[t]
        assert d in self._train_set
        for f0, f1, idx, val in self._pts:
            if f0 <= d <= f1:
                r = (d - f0).days
                return idx.read(r), val.read(r)
        raise KeyError(f"{d}: not a training day")

    def holdout(self, t):
        """validation day t (index into hold_dates, 2023): raw LGHAP integers at every mask pixel"""
        d = self.hold_dates[t]
        return self.hold.read((d - D2023).days)
