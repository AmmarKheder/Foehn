#!/usr/bin/env python3
"""Static 1 km inputs of the global FOEHN-km on the grid of ghap1km_global.npy (84 N .. 60.01 S, -180 .. 179.99):
  elev1km_global.npy (14400, 36000) float16: ETOPO-based elevation of data/world/static/static_1km.zarr, (z - 300) / 500
  land1km_global.npy (14400, 36000) bool: GHAP present (> 0) on more than half of the 2017-2020 days (every 15th)
Orientation check: elevation at a few known places."""
from pathlib import Path

import numpy as np
import zarr

OUT = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
R0, R1, W = 600, 15000, 36000
H = R1 - R0
z = zarr.open("/scratch/project_462001140/ammar/eccv/data/world/static/static_1km.zarr", mode="r")["elevation"]
elev = np.nan_to_num(np.asarray(z[R0:R1, :], np.float32))
for name, la, lo in (("Paris", 48.85, 2.35), ("Mont Blanc", 45.83, 6.86), ("Everest", 27.99, 86.93), ("Denver", 39.74, -104.99),
                     ("Dead Sea", 31.5, 35.5), ("Amazon (Manaus)", -3.1, -60.0)):
    r, c = int(round((84.0 - la) / 0.01)), int(round((lo + 180.0) / 0.01))
    print(f"{name:16s} elevation {elev[r, c]:7.0f} m")
np.save(OUT / "elev1km_global.npy", ((elev - 300.0) / 500.0).astype(np.float16))
with open(OUT / "ghap1km_global.npy", "rb") as f:
    np.lib.format.read_magic(f); np.lib.format.read_array_header_1_0(f); hdr = f.tell()
days = range(0, 1461, 15)
cnt = np.zeros((H, W), np.float32)
for d in days:
    a = np.memmap(OUT / "ghap1km_global.npy", dtype=np.float16, mode="r", offset=hdr + d * H * W * 2, shape=(H, W))
    cnt += np.asarray(a) > 0
land = (cnt / len(days)) > 0.5
np.save(OUT / "land1km_global.npy", land)
print("land fraction of the grid", float(land.mean()), "land pixels", int(land.sum()))
