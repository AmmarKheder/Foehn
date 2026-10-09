#!/usr/bin/env python3
"""Europe crops (the paper's ERA5 window) of a global 0.25 deg daily store, one chunk per day.
era5: era5_global_daily, bit-identical to the paper's era5_europe_daily stores (checked on 2018/2021);
extra: era5_extra_daily (BLH mean/max/min, precipitation, dewpoint, solar radiation, cloud cover).
Usage: python build_era5_cache.py <era5|extra> 2017 ..."""
import sys

import numcodecs
import numpy as np
import zarr

from cranpm.data.europe_dataset_v2 import DATA, ERA5_C0, ERA5_CACHE, ERA5_H, ERA5_R0, ERA5_W, EXTRA_CACHE, WORLD

SOURCES = {"era5": (DATA / "era5_global_daily", ERA5_CACHE), "extra": (WORLD / "era5_extra_daily", EXTRA_CACHE)}
src_dir, cache_dir = SOURCES[sys.argv[1]]
for y in map(int, sys.argv[2:]):
    src = zarr.open(str(src_dir / f"{y}.zarr"), mode="r")
    dst = zarr.open(str(cache_dir / f"{y}.zarr"), mode="w", shape=(src.shape[0], src.shape[1], ERA5_H, ERA5_W),
                    chunks=(1, src.shape[1], ERA5_H, ERA5_W), dtype="f4",
                    compressor=numcodecs.Blosc(cname="zstd", clevel=3, shuffle=2))
    for d in range(src.shape[0]):
        a = np.asarray(src[d], np.float32)[:, ERA5_R0:ERA5_R0 + ERA5_H]
        dst[d] = np.concatenate([a[..., ERA5_C0:], a[..., :ERA5_W - (a.shape[-1] - ERA5_C0)]], axis=-1)
    dst.attrs.update(complete=True, channels=src.attrs.get("channels"),
                     source=f"{src_dir.name}/{y}.zarr rows {ERA5_R0}:{ERA5_R0 + ERA5_H}, lon wrap from col {ERA5_C0}")
    print("done", y, flush=True)
