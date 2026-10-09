#!/usr/bin/env python3
"""GHAP V1 daily PM2.5 over the paper's Europe window at 1 km, 2017-2022, one float16 memmap aligned with
../dejavu/data/pm.npy rows (2017-01-01 .. 2022-12-31): data/ghap1km.npy (D, 4192, 6992), 0 = sea / no data (as in GHAP V1).
    python build_ghap1km.py init | <year>"""
import sys
from pathlib import Path

import numpy as np
import zarr

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)
YEARS = list(range(2017, 2023))
ND = {y: 366 if y % 4 == 0 else 365 for y in YEARS}
D, H, W, R0, C0 = sum(ND.values()), 4192, 6992, 1800, 15500
F = OUT / "ghap1km.npy"
if __name__ == "__main__":
    if sys.argv[1] == "init":  # header + reserved size, without mapping 128 GB (login-node limits)
        with open(F, "wb") as f:
            np.lib.format.write_array_header_1_0(f, {"descr": "<f2", "fortran_order": False, "shape": (D, H, W)})
            hdr = f.tell(); f.truncate(hdr + D * H * W * 2)
        print("created", D, "header", hdr); sys.exit()
    y = int(sys.argv[1]); off = sum(ND[k] for k in YEARS if k < y)
    with open(F, "rb") as f:
        np.lib.format.read_magic(f); np.lib.format.read_array_header_1_0(f); hdr = f.tell()
    out = np.memmap(F, dtype=np.float16, mode="r+", offset=hdr + off * H * W * 2, shape=(ND[y], H, W))  # this year only
    z = zarr.open(f"/scratch/project_462001140/ammar/eccv/data/zarr/ghap_global_daily/{y}.zarr", mode="r")
    for d in range(min(ND[y], z.shape[0])):
        out[d] = np.asarray(z[d, R0:R0 + H, C0:C0 + W], np.float32).astype(np.float16)
        if d % 30 == 0:
            print(y, d, flush=True)
    out.flush(); print("done", y)
