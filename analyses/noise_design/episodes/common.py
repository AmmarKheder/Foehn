#!/usr/bin/env python3
"""episodes_diag: shared helpers (readers by os.pread, atomic saves, run-length functions, bootstrap counts, regions).
Definitions: ../README.md (written before any computation)."""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution/episodes_diag"))
G = Path((os.environ["FOEHN_WORK"] + "/eccv/data/world/global1km"))
RUN = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/global1km/runs/"
           "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4"))
CAMS_PRED = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution/a11/stage1/runs/"
                 "cams04_flow_shift2.6_emis_dev/val_e35"))
# the 9 boxes (N, S, W, E) of eval_global_green.py REGIONS (copied; cell in box: lat <= N, lat > S, W <= lon < E)
REGIONS = {"Europe": (72, 35, -25, 45), "North America": (72, 15, -170, -50), "South America": (15, -56, -95, -30),
           "Africa": (37, -35, -20, 52), "Middle East": (42, 12, 35, 63), "South Asia": (37, 5, 60, 97),
           "East Asia": (54, 18, 97, 146), "Southeast Asia": (18, -11, 92, 142), "Oceania": (-10, -48, 110, 180)}
SEASONS = ("DJF", "MAM", "JJA", "SON")
L_HIST = 60                      # days of truth history for the episode age
EPS = 0.1                        # ug m-3 floor before logarithms
SEED, NBOOT = 20261121, 2000


class Npy:
    """contiguous reads of a .npy file by os.pread (no memmap: the login node refuses to map the large stores)"""

    def __init__(self, path):
        self.path = str(path)
        with open(self.path, "rb") as f:
            v = np.lib.format.read_magic(f)
            hdr = np.lib.format.read_array_header_1_0(f) if v == (1, 0) else np.lib.format.read_array_header_2_0(f)
            self.shape, fo, self.dtype = hdr
            assert not fo, "fortran order"
            self.hdr = f.tell()
        self.fd = os.open(self.path, os.O_RDONLY)

    def read(self, offset, count):
        """count elements from element offset"""
        n = count * self.dtype.itemsize
        buf = bytearray(n)
        mv, done = memoryview(buf), 0
        while done < n:
            k = os.preadv(self.fd, [mv[done:]], self.hdr + offset * self.dtype.itemsize + done)
            assert k > 0, (self.path, offset, count)
            done += k
        return np.frombuffer(buf, self.dtype, count)

    def plane(self, lead):
        """the trailing 2-D plane at leading index tuple `lead`"""
        Hh, Ww = self.shape[-2:]
        k = int(np.ravel_multi_index(lead, self.shape[:-2]))
        return self.read(k * Hh * Ww, Hh * Ww).reshape(Hh, Ww)

    def block(self, lead, n_planes):
        """n_planes consecutive trailing planes from leading index tuple `lead` (C order)"""
        Hh, Ww = self.shape[-2:]
        k = int(np.ravel_multi_index(lead, self.shape[:-2]))
        return self.read(k * Hh * Ww, n_planes * Hh * Ww).reshape(n_planes, Hh, Ww)


def fsync_dir(path):
    fd = os.open(str(path), os.O_RDONLY)
    os.fsync(fd); os.close(fd)


def _inside(path):
    """outputs only under episodes_diag/ (or under $EPI_SCRATCH, the scratch folder of a smoke test)"""
    p = Path(path).resolve()
    ok = [str(ROOT.resolve())] + ([str(Path(os.environ["EPI_SCRATCH"]).resolve())] if os.environ.get("EPI_SCRATCH") else [])
    assert any(str(p).startswith(o) for o in ok), f"{p} is outside {ok}"
    return p


def save_npz(path, **arrays):
    """atomic: temp file in the same folder, fsync, rename"""
    p = _inside(path)
    tmp = p.with_name(p.stem + ".tmp_w.npz")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p); fsync_dir(p.parent)


def save_text(path, text):
    p = _inside(path)
    tmp = p.with_name(p.name + ".tmp_w")
    data = text.encode()
    with open(tmp, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    with open(tmp, "rb") as f:
        assert f.read() == data
    os.replace(tmp, p); fsync_dir(p.parent)


def tolist(x):
    if isinstance(x, dict):
        return {str(k): tolist(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [tolist(v) for v in x]
    if isinstance(x, np.ndarray):
        return tolist(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def save_json(path, obj):
    save_text(path, json.dumps(tolist(obj), indent=1))


# ---------------------------------------------------------------- run-length functions (axis 0 = time)
def lead_run(ex):
    """number of leading True values along axis 0"""
    return np.cumprod(ex, axis=0, dtype=np.int8).sum(0, dtype=np.int16)


def first_true(ex):
    """1-based index of the first True along axis 0, n + 1 if none"""
    n = ex.shape[0]
    any_ = ex.any(0)
    return np.where(any_, ex.argmax(0) + 1, n + 1).astype(np.int16)


def age_back(hist, T):
    """hist (L, C): truth of days d-1, d-2, ... d-L; T (C,). Returns (count of consecutive days > T from d-1 backwards,
    censored flag: the count stopped at a NaN day or at the L-day limit instead of a day <= T)."""
    L = hist.shape[0]
    g = hist > T                                     # NaN -> False
    n = lead_run(g).astype(np.int32)
    nan = np.isnan(hist)
    stop = np.minimum(n, L - 1)
    cens = (n >= L) | nan[stop, np.arange(hist.shape[1])]
    return n, cens


# ---------------------------------------------------------------- bootstrap over runs
def boot_counts(n, block, nboot=NBOOT, seed=SEED):
    """(nboot, n) counts of each run in each draw: runs with replacement (block 1) or moving blocks of `block` runs"""
    rng = np.random.default_rng(seed)
    K = np.zeros((nboot, n), np.float64)
    if block <= 1 or n <= block:
        for b in range(nboot):
            K[b] = np.bincount(rng.integers(0, n, n), minlength=n)
        return K
    nb = int(np.ceil(n / block))
    for b in range(nboot):
        st = rng.integers(0, n - block + 1, nb)
        idx = (st[:, None] + np.arange(block)[None]).ravel()[:n]
        K[b] = np.bincount(idx, minlength=n)
    return K


def season_of_month(m):
    """month 1..12 -> 0 DJF, 1 MAM, 2 JJA, 3 SON"""
    return {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}[int(m)]


def region_masks(lat, lon):
    """(9, C) bool, cell in box: lat <= N, lat > S, W <= lon < E (as region_slices of eval_global_green.py)"""
    return np.stack([(lat <= n) & (lat > s) & (lon >= w) & (lon < e) for n, s, w, e in REGIONS.values()])
