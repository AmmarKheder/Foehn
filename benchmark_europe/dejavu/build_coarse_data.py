#!/usr/bin/env python3
"""Compact 0.25 deg Europe dataset (168 x 280) for 2017-2022, one row per calendar day (global index):
  pm.npy    (D, 168, 280) float32  GHAP V1 block means, ug/m3, NaN = sea / missing
  met.npy   (D, 37, 168, 280) float16  standardised ERA5 store channels (30) + extra fields (7: BLH x3, rain, dewpoint,
            solar, cloud), same standardisation as EuropeDatasetV2(mode="fixed", extra=True)
  meta.npz  year, doy per row; elevation (standardised); land mask."""
import numpy as np
from pathlib import Path

from cranpm.data.europe_dataset_v2 import ELEV_MEAN, ELEV_STD, ERA5_C0, ERA5_R0, EuropeDatasetV2

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)
CH, CW = 168, 280
years = list(range(2017, 2023))
ds = EuropeDatasetV2(years, mode="fixed", train=False, extra=True)
rows = [(y, d) for y in years for d in range(ds.ndays[y])]
D = len(rows)
pm = np.lib.format.open_memmap(OUT / "pm.npy", mode="w+", dtype=np.float32, shape=(D, CH, CW))
met = np.lib.format.open_memmap(OUT / "met.npy", mode="w+", dtype=np.float16, shape=(D, 37, CH, CW))
for i, (y, d) in enumerate(rows):
    g = np.asarray(ds.ghap025[y][d, 0, ERA5_R0:ERA5_R0 + CH], np.float32)
    pm[i] = np.concatenate([g[:, ERA5_C0:], g[:, :CW - (g.shape[1] - ERA5_C0)]], axis=1)
    era5, _ = ds._era5(y, d)
    met[i] = np.concatenate([era5, ds._extra(y, d)])[:, :CH, :CW].astype(np.float16)
    if i % 200 == 0:
        print(i, D, y, d, flush=True)
pm.flush(); met.flush()
land = np.isfinite(np.asarray(pm)).mean(0) > 0.5
np.savez(OUT / "meta.npz", year=np.array([r[0] for r in rows]), doy=np.array([r[1] for r in rows]),
         elev=((ds.elev_coarse[:CH, :CW] - ELEV_MEAN) / ELEV_STD).astype(np.float32), land=land)
print("done", D, "days; land cells", int(land.sum()))
