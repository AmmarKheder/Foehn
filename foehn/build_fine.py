#!/usr/bin/env python3
"""Global 1 km GHAP V1 daily PM2.5 as one float16 file (the target and the fine input of the global FOEHN-km):
data/world/global1km/ghap1km_global.npy (D, 14400, 36000), 2017-01-01 .. 2022-12-31, rows 84 N .. 60.01 S (GHAP rows
600:15000), all longitudes -180 .. 179.99. 0 = sea / no retrieval (GHAP V1 convention).
    python build_fine.py init
    python build_fine.py <task> <ntasks>      (fills every ntasks-th day)"""
import sys
import time
from pathlib import Path

import numpy as np
import zarr

OUT = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
SRC = "/scratch/project_462001140/ammar/eccv/data/zarr/ghap_global_daily/{}.zarr"
YEARS = list(range(2017, 2023))
NDAYS = {y: 366 if y % 4 == 0 else 365 for y in YEARS}
R0, R1, W = 600, 15000, 36000
F = OUT / "ghap1km_global.npy"
DAYS = [(y, d) for y in YEARS for d in range(NDAYS[y])]

H = R1 - R0
if sys.argv[1] == "init":    # header + sparse file of the final size (no mapping of 2.3 TB)
    with open(F, "wb") as f:
        np.lib.format.write_array_header_1_0(f, {"descr": "<f2", "fortran_order": False, "shape": (len(DAYS), H, W)})
        hdr = f.tell()
        f.truncate(hdr + len(DAYS) * H * W * 2)
    print("created", F, len(DAYS), "header", hdr); sys.exit()
task, ntasks = int(sys.argv[1]), int(sys.argv[2])
with open(F, "rb") as f:
    np.lib.format.read_magic(f); np.lib.format.read_array_header_1_0(f); HDR = f.tell()


class Out:  # day k of the file, mapped one day at a time
    def __setitem__(self, k, a):
        m = np.memmap(F, dtype=np.float16, mode="r+", offset=HDR + k * H * W * 2, shape=(H, W)); m[:] = a; m.flush(); del m


out = Out()
stores = {}
t0 = time.time()
for k in range(task, len(DAYS), ntasks):
    y, d = DAYS[k]
    if y not in stores:
        stores[y] = zarr.open(SRC.format(y), mode="r")
    z = stores[y]
    if d >= z.shape[0]:
        out[k] = 0; continue
    a = np.asarray(z[d, R0:R1, :], np.float32)
    a = np.where(np.isfinite(a) & (a > 0), a, 0).astype(np.float16)
    out[k] = a
    if (k // ntasks) % 10 == 0:
        print(f"task {task}: day {k} ({y}-{d}) {time.time() - t0:.0f}s", flush=True)
print("done task", task, f"{time.time() - t0:.0f}s")
