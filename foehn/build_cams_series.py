#!/usr/bin/env python3
"""CAMS analysis series for 'Foehn trained as Aurora': every 12 h (00 and 12 UTC) from 1 Oct 2017 to 31 Dec 2022, on
the native CAMS / Aurora grid (0.4 deg, no regridding), rows 84 N .. 60 S (the domain of Foehn's weather and transport;
CAMS row i = latitude 90 - 0.4 i, rows 15..375) and longitudes rolled to -180 .. 179.6 (column j = -180 + 0.4 j).
  cams_series.npy   (T, 13, 361, 900) float16, T = 2 x days, t = 2 x (day index from 1 Oct 2017) + (0: 00 UTC, 1: 12 UTC)
                    channels: pm1, pm2p5, pm10 (ug m-3); total columns tcco, tc_no, tcno2, gtco3, tcso2 (mg m-2);
                    co, no, no2, go3, so2 at 1000 hPa (ug kg-1)  -- physical units (float16-safe: clipped to [0, 65000],
                    reached only by a few extrapolated 1000 hPa values and fire-plume CO columns), normalised in the loader
  cams_series.done.npy  (days,) bool, days written
  cams_series_meta.npz  names, first day, lat, lon; per channel: eps (1% of the training-period mean) and the mean / std
                        of log(x + eps) over the training period (Oct 2017 - Apr 2022, every 5th time step)
Sources: data/world/cams_analysis_surface/<yyyy>_<mm>.npz (dl_cams_analysis.py) for Oct 2017 - Dec 2021 and the Aurora
initial conditions ../benchmark/aurora/raw/<doy>.npz for 2022 (dl_cams.py: surf kg m-3 / kg m-2, atmos kg kg-1).
Writes with pwrite (no memmap: login-node mmap limits, Lustre page faults).
    python build_cams_series.py            (months not downloaded yet are reported and skipped; rerun later)"""
import os
from pathlib import Path

import numpy as np

W = Path("/scratch/project_462001140/ammar/eccv/data/world")
SRC = W / "cams_analysis_surface"
RAW = Path(__file__).resolve().parents[1] / "benchmark" / "aurora" / "raw"
OUT = W / "global1km" / "cams_series.npy"
DONE = OUT.with_suffix(".done.npy")
META = OUT.with_name("cams_series_meta.npz")
NAMES = ["pm1", "pm2p5", "pm10", "tcco", "tc_no", "tcno2", "gtco3", "tcso2", "co", "no", "no2", "go3", "so2"]
R0, R1 = 15, 376                                     # lat 84 .. -60 at 0.4 deg
FIRST = np.datetime64("2017-10-01")
LAST = np.datetime64("2022-12-31")
ND = int((LAST - FIRST).astype(int)) + 1
SHAPE = (2 * ND, 13, R1 - R0, 900)
SURF_RAW = ("u10", "v10", "t2m", "msl", "pm1", "pm2p5", "pm10", "tcco", "tc_no", "tcno2", "gtco3", "tcso2")   # dl_cams.py order
ATMOS_RAW = ("z", "u", "v", "t", "q", "co", "no", "no2", "go3", "so2")                                      # levels ascending
COLS = np.array([1.0] * 3 + [1e6] * 5 + [1.0] * 5, np.float32)[:, None, None]   # kg m-2 -> mg m-2 for the columns
TRAIN_END = 2 * int((np.datetime64("2022-05-01") - FIRST).astype(int))        # first time step of the test period


def header():
    with open(OUT, "rb") as f:
        np.lib.format.read_magic(f); sh, _, dt = np.lib.format.read_array_header_1_0(f)
        return f.tell(), sh, dt


def create():
    with open(OUT, "wb") as f:
        np.lib.format.write_array_header_1_0(f, {"descr": np.lib.format.dtype_to_descr(np.dtype(np.float16)),
                                                 "fortran_order": False, "shape": SHAPE})
        f.truncate(f.tell() + int(np.prod(SHAPE)) * 2)


def write_rows(t0, block):
    """series[t0:t0 + len(block)] = block (pwrite, sequential)"""
    hdr, sh, dt = header()
    block = np.ascontiguousarray(np.clip(block, 0, 65000), dtype=dt); assert block.shape[1:] == tuple(sh[1:]), block.shape
    mv = memoryview(block.reshape(-1).view(np.uint8)); n = len(mv); row = int(np.prod(sh[1:])) * dt.itemsize; done = 0
    fd = os.open(OUT, os.O_WRONLY)
    try:
        while done < n:
            done += os.pwrite(fd, mv[done:done + (1 << 30)], hdr + t0 * row + done)
        os.fsync(fd)
    finally:
        os.close(fd)


def read_row(t):
    hdr, sh, dt = header(); row = int(np.prod(sh[1:])) * dt.itemsize
    fd = os.open(OUT, os.O_RDONLY)
    try:
        b = os.pread(fd, row, hdr + t * row)
    finally:
        os.close(fd)
    return np.frombuffer(b, dt).reshape(sh[1:])


def crop_roll(a, lat, lon):
    """(..., 451, 900) on lat 90..-90, lon 0..359.6 -> (..., 361, 900) on lat 84..-60, lon -180..179.6"""
    assert abs(lat[0] - 90) < 1e-3 and abs(lat[R0] - 84) < 1e-3 and abs(lat[R1 - 1] + 60) < 1e-3 and abs(lon[0]) < 1e-3, (lat[:2], lon[:2])
    return np.roll(a[..., R0:R1, :], 450, axis=-1)


def main():
    if not OUT.exists():
        create()
    done = np.load(DONE) if DONE.exists() else np.zeros(ND, bool)
    for y in range(2017, 2022):
        for m in range(1, 13):
            if (y, m) < (2017, 10):
                continue
            d0 = int((np.datetime64(f"{y}-{m:02d}-01") - FIRST).astype(int))
            f = SRC / f"{y}_{m:02d}.npz"
            if done[d0]:
                continue
            if not f.exists():
                print(f"{y}-{m:02d}: not downloaded yet", flush=True)
                continue
            z = np.load(f)
            block = np.stack([z[n] for n in NAMES], 2) * COLS                            # (days, 2, 13, 451, 900)
            nd = block.shape[0]
            write_rows(2 * d0, crop_roll(block, z["lat"], z["lon"]).reshape(2 * nd, *SHAPE[1:]))
            done[d0:d0 + nd] = True; np.save(DONE, done)
            print(f"{y}-{m:02d}: {nd} days, pm2p5 mean {block[:, :, 1].mean():.2f}", flush=True)
            del block
    i22 = int((np.datetime64("2022-01-01") - FIRST).astype(int))
    scale = np.array([1e9] * 3 + [1e6] * 5, np.float32)[None, :, None, None]                    # pm ug m-3, columns mg m-2
    for doy in range(365):
        if done[i22 + doy]:
            continue
        z = np.load(RAW / f"{doy:03d}.npz")
        surf = z["surf"][:, [SURF_RAW.index(n) for n in NAMES[:8]]] * scale
        atmos = z["atmos"][:, [ATMOS_RAW.index(n) for n in NAMES[8:]], -1] * 1e9                 # 1000 hPa, kg kg-1 -> ug kg-1
        write_rows(2 * (i22 + doy), crop_roll(np.concatenate([surf, atmos], 1), z["lat"], z["lon"]))
        done[i22 + doy] = True
        if doy % 30 == 0:
            np.save(DONE, done); print(f"2022 doy {doy}: pm2p5 mean {surf[:, 1].mean():.2f}", flush=True)
    np.save(DONE, done)
    # normalisation: eps = 1% of the channel mean, then mean / std of log(x + eps), training period, every 5th step
    ts = [t for t in range(0, TRAIN_END, 5) if done[t // 2]]
    s1 = np.zeros(13); s2 = np.zeros(13); s0 = 0
    for t in ts:
        s1 += read_row(t).astype(np.float64).mean((1, 2))
    eps = (0.01 * s1 / len(ts)).astype(np.float32)
    s1[:] = 0
    for t in ts:
        x = np.log(read_row(t).astype(np.float32).clip(0) + eps[:, None, None])
        s1 += x.mean((1, 2)); s2 += (x.astype(np.float64) ** 2).mean((1, 2)); s0 += 1
    mean = s1 / s0; std = np.sqrt(s2 / s0 - mean ** 2)
    lat = 84.0 - 0.4 * np.arange(R1 - R0); lon = -180.0 + 0.4 * np.arange(900)
    np.savez(META, names=np.array(NAMES), first=str(FIRST), lat=lat, lon=lon, eps=eps, mean=mean.astype(np.float32),
             std=std.astype(np.float32), days_done=done, n_stats=s0)
    print(f"done {done.sum()}/{ND} days | stats on {s0} steps | eps {dict(zip(NAMES, eps))}")
    print(f"log-mean {dict(zip(NAMES, np.round(mean, 2)))}\nlog-std {dict(zip(NAMES, np.round(std, 2)))}")


if __name__ == "__main__":
    main()
