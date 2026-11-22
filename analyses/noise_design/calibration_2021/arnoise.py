#!/usr/bin/env python3
"""A15 (a15/A15_draft.md, section 4): an EVOLVING noise for the frozen Foehn 1 km model (no retraining, no weight change).
Member m, start day d (0-based day of year of the start, ds.doy[i]), lead h:
    z(m, d, 1) = eps_1
    z(m, d, h) = rho^(h-1) eps_1 + sqrt(1 - rho^2) * sum_{k=2..h} rho^(h-k) eps_k
eps_k[tr, tc] = Generator(Philox(SeedSequence([1521, year, d, m, k, tr, tc]))).standard_normal((512, 512), float32): one
counter-keyed N(0, 1) tile per storage tile of the 29 x 71 grid of noise_tiles.npy, nothing stored on disk. Coefficients in
float64 cast to float32, float32 accumulation in the order k = 1..h (zero coefficients skipped), one cast to float16.
For every (d, h) z is white N(0, 1) (sum of squared coefficients = 1), members and start days use disjoint keys: the
per-lead law of the ensemble is that of the stored design; only the joint law over leads (AR(1), rho) and starts changes.

ARNoise is a duck-typed eval_global_green Npy (shape, dtype, _read on whole tiles), so the UNCHANGED tile_window and
full_map cut z(m, d, h) exactly as they cut noise_tiles.npy and every window of one (m, d, h) reads the same global field
(no seams). Identity setting (tests and smoke (a) only): base 'stored', rho = 1 returns the bytes of noise_tiles.npy.
WindowsAR = eval_global_green.Windows whose x0[m] = tile_window(ARNoise(2021, ds.doy[i], h, rho, base), m, ...).

Imported FIRST by every A15 script: installs the station / monitor audit hook (exact copy of a13/warn16/sample16.py) and
refuses any numpy other than the job container's 1.26.3 (the Philox float32 normals differ between numpy versions).
deny_2022() adds the hook of the analysis scripts (no 2022 file)."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


def _audit(event, args):                       # stations and monitors are evaluation data only: never opened here
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0]).lower()
        if p.endswith((".py", ".pyc", ".so")) or "/site-packages/" in p:     # code, not data
            return
        if "station" in p or "openaq" in p or "monitor" in p:
            raise PermissionError(f"A13 warn16: station/monitor file refused: {p}")


sys.addaudithook(_audit)

NUMPY_REQUIRED = "1.26.3"
assert np.__version__ == NUMPY_REQUIRED, f"A15 runs only in the job container (numpy {NUMPY_REQUIRED}), not {np.__version__}"

A15 = Path(__file__).resolve().parent
ROOT = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution"))
EPI = ROOT / "episodes_diag"
GLOBAL1KM = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/global1km"))
RUN = GLOBAL1KM / "runs" / "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4"
EV = RUN / "eval2021_sel_crps_e01"
sys.path.insert(0, str(GLOBAL1KM))
import torch  # noqa: E402

import eval_global_green as egg  # noqa: E402  (imported unchanged, read only)
from fullflow_global import NTC, NTR, TILE  # noqa: E402

ROOT_WORD, YEAR = 1521, 2021
B = TILE * TILE
H7 = 7
CAL = [1 + 24 * q for q in range(15)]                          # j % 3 == 0 of the 45 complete 2021 starts 1, 9, ..., 353
CONF = [1 + 8 * j for j in range(45) if j % 3 != 0]
assert TILE == 512 and (NTR, NTC) == (29, 71) and len(CAL) == 15 and len(CONF) == 30 and not set(CAL) & set(CONF)

_DENY = ("ext_2022", "eval_final", "revision_s2", "_2022")
_deny_on = False


def _audit_2022(event, args):                  # analysis scripts: no file of the 2022 test year
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0]).lower()
        if p.endswith((".py", ".pyc", ".so")) or "/site-packages/" in p:
            return
        if any(s in p for s in _DENY):
            raise PermissionError(f"A15: 2022 file refused: {p}")


def deny_2022():
    global _deny_on
    if not _deny_on:
        sys.addaudithook(_audit_2022)
        _deny_on = True


# ======================================================================== the noise
def innovation(year, doy, m, k, tr, tc):
    """eps_k of member m, start day doy, storage tile (tr, tc): a fresh Philox stream per 7-word key (fixed length)"""
    key = (ROOT_WORD, year, doy, m, k, tr, tc)
    for v in key:
        assert isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) and 0 <= int(v) < 2 ** 32, key
    ss = np.random.SeedSequence([int(v) for v in key])
    return np.random.Generator(np.random.Philox(ss)).standard_normal((TILE, TILE), dtype=np.float32)


def coeffs(h, rho):
    """a_h1 .. a_hh of z_h = sum_k a_hk eps_k: float64, returned as float32"""
    h, rho = int(h), float(rho)
    assert h >= 1 and 0.0 <= rho <= 1.0, (h, rho)
    s = np.sqrt(1.0 - rho * rho)
    return np.array([rho ** (h - 1)] + [s * rho ** (h - k) for k in range(2, h + 1)], np.float64).astype(np.float32)


class ARNoise:
    """duck-typed Npy of z(m, d, h) for one (year, start day, lead, rho): shape (members, 29, 71, 512, 512), '<f2'"""

    def __init__(self, year, doy, lead, rho, base, members=2 ** 20):
        assert base in ("philox", "stored"), base
        self.year, self.doy, self.lead, self.rho, self.base = int(year), int(doy), int(lead), float(rho), base
        assert 1 <= self.lead <= H7 and 0.0 <= self.rho <= 1.0, (lead, rho)
        self.identity = base == "stored" and self.rho == 1.0
        self.c = coeffs(self.lead, self.rho)
        self.shape = (int(members) if base == "philox" else min(int(members), 4), NTR, NTC, TILE, TILE)
        self.dtype = np.dtype("<f2")
        self._npy = None

    @property
    def npy(self):                             # noise_tiles.npy, opened lazily (egg.Npy, os.pread)
        if self._npy is None:
            self._npy = egg.Npy(egg.G / "noise_tiles.npy")
            assert self._npy.dtype == self.dtype and tuple(self._npy.shape[1:]) == (NTR, NTC, TILE, TILE), self._npy.shape
        return self._npy

    def _stored(self, m, tr, tc):
        assert 0 <= m < self.npy.shape[0], m
        return self.npy._read(((m * NTR + tr) * NTC + tc) * B, B).reshape(TILE, TILE)

    def _arith(self, m, tr, tc):
        acc = np.zeros((TILE, TILE), np.float32)
        for k in range(1, self.lead + 1):
            ck = self.c[k - 1]
            if ck == 0:
                continue
            e = self._stored(m, tr, tc).astype(np.float32) if (k == 1 and self.base == "stored") else \
                innovation(self.year, self.doy, m, k, tr, tc)
            acc += ck * e
        return acc.astype(np.float16)

    def tile(self, m, tr, tc):
        return self._stored(m, tr, tc) if self.identity else self._arith(m, tr, tc)

    def _read(self, offset, count):
        """count values from flat offset (whole tiles only; a read may cross tile rows and members)"""
        assert offset % B == 0 and count % B == 0 and count > 0, (offset, count)
        t0, n = offset // B, count // B
        out = np.empty(count, np.float16)
        for j in range(n):
            m, r = divmod(t0 + j, NTR * NTC)
            tr, tc = divmod(r, NTC)
            assert m < self.shape[0], (m, self.shape)
            out[j * B:(j + 1) * B] = self.tile(m, tr, tc).reshape(-1)
        return out

    def __getstate__(self):                    # DataLoader workers reopen their own descriptor
        d = dict(self.__dict__)
        d["_npy"] = None
        return d


class WindowsAR(egg.Windows):
    """eval_global_green.Windows with the evolving noise: jobs [(start index, lead, window)], rhos parallel to jobs"""

    def __init__(self, ds, jobs, win, members, rhos, base):
        assert len(rhos) == len(jobs) and base in ("philox", "stored"), (len(rhos), len(jobs), base)
        assert base == "philox" or members <= 4
        self.ds, self.jobs, self.win, self.members, self.lags = ds, jobs, win, members, []     # no lagged members
        self.rhos, self.base = [float(r) for r in rhos], base

    def __getitem__(self, n):
        i, h, w = self.jobs[n]
        k0, j0 = (int(v) for v in self.win[w])
        b = self.ds.item(i, h, k0, j0)
        del b["x1"], b["mask"]
        nz = ARNoise(YEAR, int(self.ds.doy[i]), h, self.rhos[n], self.base)
        b["x0"] = torch.from_numpy(np.stack([egg.tile_window(nz, m, 25 * k0, 25 * j0, egg.PH, egg.PW) for m in range(self.members)]))
        for l in self.lags:
            for k, v in self.ds.lagged(i, h, k0, j0, l).items():
                b[f"lag{l}_{k}"] = v
        return b


GOLDEN_KEYS = ((2021, 1, 0, 1, 0, 0), (2021, 353, 3, 7, 28, 70), (2021, 9, 1, 2, 14, 35), (2021, 177, 63, 4, 7, 60))


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def golden_hashes():
    out = {"numpy": np.__version__}
    for key in GOLDEN_KEYS:
        e = innovation(*key)
        nm = "eps_" + "_".join(str(v) for v in key)
        out[nm + "/f32"], out[nm + "/f16"] = sha(e), sha(e.astype(np.float16))
    out["z_2021_9_lead7_rho0.7_philox_m2_tile14_35/f16"] = sha(ARNoise(2021, 9, 7, 0.7, "philox").tile(2, 14, 35))
    return out


# ======================================================================== atomic writes, JSON
def fsync_dir(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def inside_a15(path):
    p = Path(path).resolve()
    assert str(p).startswith(str(A15) + os.sep), f"{p} is outside {A15}"
    return p


def write_bytes(path, data):
    p = inside_a15(path)
    tmp = p.with_name(p.name + ".tmp_a15")
    with open(tmp, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    with open(tmp, "rb") as f:
        assert f.read() == data, tmp
    os.replace(tmp, p)
    fsync_dir(p.parent)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    return x


def save_json(path, obj):
    write_bytes(path, (json.dumps(jsonable(obj), indent=1) + "\n").encode())


# ======================================================================== the 2021 episode data (analysis scripts)
def episodes_2021():
    """episodes_diag/data/ext_2021.npz through revise.load ('S1'), restricted to the cells complete in every one of the 45
    stored 2021 runs (2021 only), ordered by 10-degree tile exactly as revise.main_s12"""
    sys.path.insert(0, str(EPI / "code"))
    import revise  # noqa: E402  (read only)
    D = revise.load("S1", 1)
    keep = D["ok"].all(0)
    lat, lon = D["lat"][keep], D["lon"][keep]
    t = ((84.0 - lat) // 10).astype(int) * 36 + ((lon + 180.0) // 10).astype(int)
    _, tid = np.unique(t, return_inverse=True)
    order = np.argsort(tid, kind="stable")
    revise.restrict(D, keep, order)
    D["tid"] = tid[order]
    lab = [int(v) for v in D["labels"]]
    assert lab == list(range(1, 354, 8)), lab
    with np.load(EPI / "data" / "ext_2021.npz") as z:           # pm.npy index of the first truth day (2021: 1461 + 1 - 61)
        D["truth_gi0"] = int(z["truth_gi0"])
    assert D["truth_gi0"] == 1461 + 1 - 60 - 1, D["truth_gi0"]
    return D, revise


def subset_runs(D, idx):
    """the runs idx of D (run axis of Y, X, Mn, P, W, tidx, labels); cells, truth series and weights shared"""
    E = dict(D)
    for k in ("Y", "X", "Mn", "P", "W", "tidx", "labels"):
        E[k] = np.ascontiguousarray(np.asarray(D[k])[idx])
    return E


def cells_hash(cells):
    return hashlib.sha256(np.asarray(cells, "<i8").tobytes()).hexdigest()
