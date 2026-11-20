#!/usr/bin/env python3
"""A7(e), scoring side: GHAP V1 2022 at the station pixels and on the rings of the official stations (reference
product, not a model). World venv, numpy only; no model code. Refuses to run before every prediction is on disk
(preds/done_sameday.npy and preds/done_chain.npy all set), as the frozen test did; the opening is logged in
test2022_official/UNSEALING_A7.md (the central logs/UNSEALING.md is not modified by this amendment).

Read: eccv/data/world/global1km/ghap1km_global.npy rows 1826..2190 (whole-day reads), native value, NaN = hole
(GHAP <= 0 or not finite), as test2022_ghap/score_2022.py cmd_truth. Check first: the same extraction on two days of
2021 equals the stored hold-out field data/processed/ghap/ghap_eval/full_2021.npy at the positions.
Output: truth/ghap_official_2022.npy (365, n_pix, 4) float32 = station pixel, ring means at 1 / 3 / 10 km over the valid
ring pixels (NaN where fewer than half of the ring pixels are valid that day: score_2022.ring_means), truth/truth.json.
    python -I truth_official.py [--procs 32]"""
import argparse
import datetime as dt
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
O = ROOT / "test2022_official"
GH = ROOT / "data/processed/ghap"
SRC = Path("/scratch/project_462001804/ammar/eccv/data/world/global1km/ghap1km_global.npy")
NR, NC, D0 = 14400, 36000, dt.date(2017, 1, 1)
Y0, ND = dt.date(2022, 1, 1), 365
RINGS = (1, 3, 10)
G_ = {}


def _hdr():
    with open(SRC, "rb") as f:
        ver = np.lib.format.read_magic(f)
        shape, _, dtype = (np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0)(f)
        assert tuple(shape) == (2191, NR, NC) and np.dtype(dtype) == np.float16
        return f.tell()


def _day(d):
    hdr = _hdr()
    a = np.empty(NR * NC, np.float16)
    mv = memoryview(a.view(np.uint8))
    fd = os.open(SRC, os.O_RDONLY)
    off = hdr + (d - D0).days * NR * NC * 2
    p = 0
    while p < len(mv):
        k = os.preadv(fd, [mv[p:]], off + p)
        assert k > 0
        p += k
    os.close(fd)
    return a


def ring_means(x, idx, dd):
    """x (n_pos,) values at the positions (NaN = invalid) -> (n_pix,) ring means, NaN if fewer than 4 d valid pixels"""
    V = np.where(idx >= 0, x[np.maximum(idx, 0)], np.nan)
    n = np.isfinite(V).sum(1)
    return np.where(n >= 4 * dd, np.nansum(V, 1) / np.maximum(n, 1), np.nan).astype(np.float32)


def _truth_day(t):
    a = _day(Y0 + dt.timedelta(t))
    v = a[G_["flat"]].astype(np.float32)
    del a
    v[~(np.isfinite(v) & (v > 0))] = np.nan
    out = np.empty((G_["mon"].size, 4), np.float32)
    out[:, 0] = v[G_["mon"]]
    for j, dd in enumerate(RINGS):
        out[:, j + 1] = ring_means(v, G_[f"ring{dd}"], dd)
    return t, out, int(np.isnan(v).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()
    ds, dc = np.load(O / "preds/done_sameday.npy"), np.load(O / "preds/done_chain.npy")
    assert ds.all() and dc.all(), (f"predictions incomplete ({int(ds.sum())}/365 same-day days, {int(dc.sum())}/214 issue "
                                   f"days): GHAP 2022 is not opened")
    z = np.load(O / "pos/st_ring_official.npz")
    pos = z["pos"].astype(np.int64)
    mask = np.load(GH / "grid/ghap_mask.npy")
    G_["flat"] = mask[pos].astype(np.int64)
    G_["mon"] = z["mon_idx"].astype(np.int64)
    for dd in RINGS:
        G_[f"ring{dd}"] = z[f"ring{dd}"].astype(np.int64)
    chk = {}
    hold = np.load(GH / "ghap_eval/full_2021.npy", mmap_mode="r")
    for d in (dt.date(2021, 1, 8), dt.date(2021, 7, 1)):
        v = _day(d)[G_["flat"]].astype(np.float32)
        ref = np.asarray(hold[(d - dt.date(2021, 1, 1)).days][pos])
        mine = np.where(np.isfinite(v) & (v > 0), np.maximum(np.rint(v), 1), -1).astype(np.int16)
        chk[str(d)] = bool(np.array_equal(mine, ref))
    print("2021 extraction check:", chk, flush=True)
    assert all(chk.values()), chk
    with open(O / "UNSEALING_A7.md", "a") as f:
        f.write(f"- {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Helsinki), job {os.environ.get('SLURM_JOB_ID', '-')}: [A7] OPENING "
                f"GHAP V1 2022 ({SRC}, rows 1826..2190, whole-day reads) at the {pos.size:,} station and ring positions of the "
                f"official stations (test2022_official/pos/st_ring_official.npz) -> test2022_official/truth/ghap_official_2022.npy "
                f"(station pixel + ring means). Reference arm of amendment A7 for the frozen runs_ghap/pmG_final; every "
                f"prediction of test2022_official/preds was complete before this opening (done flags all set). Second "
                f"opening of GHAP 2022 for this system (first: PROTOCOL.md A2(d), 2026-10-28).\n")
    t0 = time.time()
    out = np.full((ND, G_["mon"].size, 4), np.nan, np.float32)
    holes = np.zeros(ND, np.int64)
    with Pool(a.procs) as pool:
        for t, x, nh in pool.imap_unordered(_truth_day, range(ND), chunksize=1):
            out[t] = x
            holes[t] = nh
    (O / "truth").mkdir(exist_ok=True)
    np.save(O / "truth/ghap_official_2022.npy", out)
    info = dict(file=str(O / "truth/ghap_official_2022.npy"), shape=list(out.shape), seconds=time.time() - t0, check_2021=chk,
                hole_position_days=int(holes.sum()), nan_station_pixel_days=int(np.isnan(out[..., 0]).sum()),
                mean_station_pixel=float(np.nanmean(out[..., 0])), max_station_pixel=float(np.nanmax(out[..., 0])),
                nan_by_where=[int(np.isnan(out[..., j]).sum()) for j in range(4)])
    (O / "truth/truth.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1), flush=True)


if __name__ == "__main__":
    main()
