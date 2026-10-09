#!/usr/bin/env python3
"""Static 1 km inputs over the paper's Europe window (4192 x 6992): data/static1km.npz
  elev: GMTED Europe (20160 x 33600, stored south -> north) area-averaged to the GHAP grid, north -> south, (z - 300) / 500
  land: fraction of 2017-2020 days (every 15th) with a valid GHAP value > 0.5"""
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data"
H, W = 4192, 6992
g = np.asarray(zarr.open("/scratch/project_462001140/ammar/eccv/data/zarr/elevation/gmted2010_europe.zarr", mode="r")[:], np.float32)
g = torch.from_numpy(np.ascontiguousarray(g[::-1]))[None, None]                     # north -> south
elev = F.adaptive_avg_pool2d(g, (H, W))[0, 0].numpy()
elev = ((elev - 300.0) / 500.0).astype(np.float16)
with open(OUT / "ghap1km.npy", "rb") as f:
    np.lib.format.read_magic(f); np.lib.format.read_array_header_1_0(f); hdr = f.tell()
days = range(0, 1461, 15)
cnt = np.zeros((H, W), np.float32)
for d in days:
    a = np.memmap(OUT / "ghap1km.npy", dtype=np.float16, mode="r", offset=hdr + d * H * W * 2, shape=(H, W))
    cnt += np.isfinite(a) & (a > 0)
land = (cnt / len(days)) > 0.5
np.savez(OUT / "static1km.npz", elev=elev, land=land)
print("elev range", float(elev.min()), float(elev.max()), "land fraction", land.mean())
