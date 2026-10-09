#!/usr/bin/env python3
"""Test-format HRES stores of another year (validation year 2021, for the GHAP calibration), from the WeatherBench2
downloads (dl_wb2.py hres, product A): hres_block_<y>.npy (days, 7, 31, 576, 1440), hres_wind_<y>.npy (days, 64, 4, 288,
720), hres_rain0_<y>.npy (days, 576, 1440), hres_days_<y>.npy - same content and layout as the 2022 stores.
    python build_hres_year.py 2021"""
import sys
from pathlib import Path

import numpy as np

G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
y = int(sys.argv[1]); nd = 366 if y % 4 == 0 else 365
mk = lambda n, sh, dt: np.lib.format.open_memmap(G / f"{n}_{y}.npy", mode="w+", dtype=dt, shape=(nd,) + sh)
blk, wnd, rn0 = mk("hres_block", (7, 31, 576, 1440), np.float16), mk("hres_wind", (64, 4, 288, 720), np.float16), mk("hres_rain0", (576, 1440), np.float32)
ok = np.zeros(nd, bool)
for m in range(1, 13):
    z = np.load(G / "wb2" / f"hres_A_{y}_{m:02d}.npz")
    for k, d in enumerate(z["day_of_year"]):
        blk[d], wnd[d], rn0[d] = z["block"][k], z["wind"][k], z["rain_d0"][k]; ok[d] = True
    print(y, m, len(z["day_of_year"]), flush=True)
for a in (blk, wnd, rn0):
    a.flush()
np.save(G / f"hres_days_{y}.npy", ok)
print("runs", int(ok.sum()), "of", nd)
