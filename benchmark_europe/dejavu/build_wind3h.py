#!/usr/bin/env python3
"""3-hourly transport winds for the footprint layer, aligned with data/pm.npy rows (2017-01-01 .. 2022-12-31):
data/wind3h.npy (D, 8, 5, 84, 140) float16 = [u10, v10, u925, v925, BLH] at 00..21 UTC, physical units (m/s, m),
Europe window (ERA5 rows 72:240, lon wrap from col 1340, 168 x 280) averaged 2x2 to 0.5 deg.
    python build_wind3h.py init          (create the file)
    python build_wind3h.py <year>        (fill one year)"""
import sys
from pathlib import Path

import numpy as np
import zarr

DATA = Path(__file__).resolve().parent / "data"
SRC = "/scratch/project_462001140/ammar/eccv/data/world/era5_wind3h/{}.zarr"
YEARS = list(range(2017, 2023))
NDAYS = {y: 366 if y % 4 == 0 else 365 for y in YEARS}
D = sum(NDAYS.values())
CH = [0, 1, 2, 3, 8]  # u10, v10, u925, v925, blh
if sys.argv[1] == "init":
    np.lib.format.open_memmap(DATA / "wind3h.npy", mode="w+", dtype=np.float16, shape=(D, 8, 5, 84, 140))
    print("created", D); sys.exit()
y = int(sys.argv[1])
off = sum(NDAYS[k] for k in YEARS if k < y)
out = np.load(DATA / "wind3h.npy", mmap_mode="r+")
z = zarr.open(SRC.format(y), mode="r")
assert z.shape[0] == NDAYS[y], (z.shape, y)
for d in range(NDAYS[y]):
    a = np.asarray(z[d], np.float32)[:, CH][:, :, 72:240]                  # (8, 5, 168, 1440)
    a = np.concatenate([a[..., 1340:], a[..., :280 - 100]], axis=-1)        # (8, 5, 168, 280)
    out[off + d] = a.reshape(8, 5, 84, 2, 140, 2).mean((3, 5)).astype(np.float16)
    if d % 60 == 0:
        print(y, d, flush=True)
out.flush()
print("done", y)
