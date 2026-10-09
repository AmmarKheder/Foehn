#!/usr/bin/env python3
"""GHAP 1 km at the station pixels for every day of 2017-2020 (training years), to build the climatological forecast
of the paper (2017-2020 mean over the same calendar days +-7 days, as in the Europe study). Parallel direct reads.
    python station_ghap_history.py <task> <ntasks>   -> clim/st_ghap_hist_<task>.npy ; then: python ... merge"""
import os
import sys
from pathlib import Path

import numpy as np

G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
st = np.load(G / "stations_2022.npz")
r = np.clip(np.floor((84 - st["lat"]) / 0.01).astype(np.int64), 0, 14399); c = np.clip(np.floor((st["lon"] + 180) / 0.01).astype(np.int64), 0, 35999)
with open(G / "ghap1km_tiles.npy", "rb") as f:
    np.lib.format.read_magic(f); np.lib.format.read_array_header_1_0(f); HDR = f.tell()
OFF = ((((r // 512) * 71 + c // 512) * 512 + r % 512) * 512 + c % 512)        # offset inside a day (elements)
DAY = 29 * 71 * 512 * 512
if sys.argv[1] == "merge":
    parts = sorted((G / "clim").glob("st_ghap_hist_*.npy"))
    H = np.full((1461, len(r)), np.nan, np.float32)
    for p in parts:
        z = np.load(p, allow_pickle=True).item(); H[z["days"]] = z["vals"]
    np.save(G / "clim" / "st_ghap_2017_2020.npy", H.astype(np.float16))
    # climatology of each 2022 day of year: mean over 2017-2020 of the same calendar days +-7 (values > 0)
    meta = np.load(G / "meta.npz"); doy = meta["doy"][:1461]; year = meta["year"][:1461]
    CL = np.full((365, len(r)), np.nan, np.float32)
    for t in range(365):
        dd = np.abs(doy - t); dd = np.minimum(dd, 365 - dd)                     # circular calendar distance
        sel = dd <= 7
        v = H[sel]; v = np.where(v > 0, v, np.nan)
        with np.errstate(all="ignore"):
            CL[t] = np.nanmean(v, 0)
    np.save(G / "clim" / "st_clim_2022.npy", CL)
    print("climatology at", len(r), "stations, finite:", np.isfinite(CL).mean().round(3), "| from", len(parts), "parts")
    sys.exit()
task, n = int(sys.argv[1]), int(sys.argv[2])
fd = os.open(G / "ghap1km_tiles.npy", os.O_RDONLY)
order = np.argsort(OFF)                                                          # read in file order
days = list(range(task, 1461, n)); vals = np.zeros((len(days), len(r)), np.float32)
for k, d in enumerate(days):
    base = HDR + 2 * (d * DAY)
    v = np.empty(len(r), np.float32)
    for i in order:
        v[i] = np.frombuffer(os.pread(fd, 2, base + 2 * int(OFF[i])), np.float16)[0]
    vals[k] = v
np.save(G / "clim" / f"st_ghap_hist_{task:03d}.npy", {"days": np.array(days), "vals": vals}, allow_pickle=True)
print("task", task, "done", len(days), "days")
