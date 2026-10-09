#!/usr/bin/env python3
"""Test-time inputs of the global model (2022, forecast issued at the end of day d with the HRES run of 00 UTC day d),
converted from the monthly downloads (dl_hres_global.py, dl_hres_global_ext.py) to day-indexed files:
  hres_block.npy  (365, 7, 31, 576, 1440) float16  daily means of forecast days d+1 .. d+7 (store units, tp in m/day)
  hres_wind.npy   (365, 64, 4, 288, 720)  float16  u10, v10, u925, v925 every 3 h over days d .. d+7 (m/s)
  hres_rain0.npy  (365, 576, 1440) float32          precipitation of day d from the first 24 h of the run (m)
  noise_tiles.npy (M, 29, 71, 512, 512) float16     one global N(0, 1) field per ensemble member (shared by windows)
    python prep_test.py hres | noise [M]"""
import sys
from pathlib import Path

import numpy as np

G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")


def create(path, shape, descr):
    if path.exists():                                   # resumable: never truncate a file that is being filled
        return np.load(path, mmap_mode="r+")
    with open(path, "wb") as f:
        np.lib.format.write_array_header_1_0(f, {"descr": descr, "fortran_order": False, "shape": shape})
        hdr = f.tell(); f.truncate(hdr + int(np.prod(shape)) * np.dtype(descr).itemsize)
    return np.load(path, mmap_mode="r+")


if sys.argv[1] == "hres":
    blk = create(G / "hres_block.npy", (365, 7, 31, 576, 1440), "<f2")
    wnd = create(G / "hres_wind.npy", (365, 64, 4, 288, 720), "<f2")
    rn0 = create(G / "hres_rain0.npy", (365, 576, 1440), "<f4")
    months = [int(x) for x in sys.argv[2:]] or list(range(1, 13))
    for m in months:
        a = np.load(G / "hres" / f"{m:02d}.npz"); e = np.load(G / "hres" / f"{m:02d}_ext.npz")
        assert np.array_equal(a["day_of_year"], e["day_of_year"]), m
        for k, d in enumerate(a["day_of_year"]):
            blk[d, :4] = a["block"][k]; blk[d, 4:] = e["block57"][k]
            wnd[d] = e["wind"][k]; rn0[d] = e["rain_d0"][k]
        blk.flush(); wnd.flush(); rn0.flush()
        (G / "hres" / f"done_{m:02d}").touch()
        print("month", m, len(a["day_of_year"]), flush=True)
elif sys.argv[1] == "days":                             # a day is available when all its fields were written (non-zero)
    blk, wnd, rn0 = (np.load(G / f, mmap_mode="r") for f in ("hres_block.npy", "hres_wind.npy", "hres_rain0.npy"))
    seen = np.array([bool(np.any(blk[d, 0, 2]) and np.any(blk[d, 6, 2]) and np.any(wnd[d, 63, 0]) and np.any(rn0[d]))
                     for d in range(365)])
    np.save(G / "hres_days.npy", seen)
    print("HRES days available:", int(seen.sum()), "of 365; missing:", np.where(~seen)[0].tolist())
elif sys.argv[1] == "noise":
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    out = create(G / ("noise_tiles.npy" if M == 4 else f"noise_tiles{M}.npy"), (M, 29, 71, 512, 512), "<f2")   # member m: seed 1000 + m
    for m in range(M):
        rng = np.random.default_rng(1000 + m)
        for r in range(29):
            out[m, r] = rng.standard_normal((71, 512, 512), dtype=np.float32).astype(np.float16)
        print("member", m, flush=True)
    out.flush()
