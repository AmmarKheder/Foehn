#!/usr/bin/env python3
"""Assemble the forecast-time inputs of the training years from the WeatherBench2 downloads (dl_wb2.py hres, product A):
hres_block_train.npy (1826, 7, 31, 576, 1440) float16, global day index 0..1825 (2017-01-01 .. 2021-12-31), same content
and units as the 2022 test store hres_block.npy (daily means of D+1..D+7 of the 31 channels). Days without a run stay
NaN and are listed in hres_train_days.npy (bool).
    python build_hres_train.py            (CPU job; checks every month file and the value ranges)"""
import os
from pathlib import Path

import numpy as np

G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
meta = np.load(G / "meta.npz"); year, doy = meta["year"], meta["doy"]
N, SH = 1826, (7, 31, 576, 1440)
out = G / "hres_block_train.npy"
mm = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(N,) + SH)
ok = np.zeros(N, bool)
for y in range(2017, 2022):
    for m in range(1, 13):
        f = G / "wb2" / f"hres_A_{y}_{m:02d}.npz"
        if not f.exists():
            print("MISSING", f); continue
        z = np.load(f)
        blk, dd = z["block"], z["day_of_year"]
        for k, d in enumerate(dd):
            gi = int(np.where((year == y) & (doy == d))[0][0])
            mm[gi] = blk[k]; ok[gi] = True
        x = blk.astype(np.float32)
        print(f"{y}-{m:02d}: {len(dd)} runs | channel means {np.round(x.mean((0, 1, 3, 4))[[0, 2, 5, 10, 30]], 3)} | NaN {np.isnan(x).mean():.4f}", flush=True)
mm.flush()
np.save(G / "hres_train_days.npy", ok)
print("runs present:", int(ok.sum()), "of", N)
