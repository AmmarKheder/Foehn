#!/usr/bin/env python3
"""Annual mean daily PM2.5 of 2022 at 1 km (GHAP; mean over the days with a retrieval at each pixel), for the teaser.
One rank per process: sums of its days -> data/pm_annual/part_<rank>_{sum,n}.npy; rank 0 then merges them into
data/pm_annual_2022.npy (14400, 36000) float16 (NaN where no day), and the 40 km cell means (360, 900).
    srun -n 8 python figdata_pm_annual.py"""
import os
import time
from pathlib import Path

import numpy as np

from figdata_exposure import GH, GW, I2022, ghap_day

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "pm_annual"


def main():
    rank, world = int(os.environ.get("SLURM_PROCID", 0)), int(os.environ.get("SLURM_NTASKS", 1))
    OUT.mkdir(parents=True, exist_ok=True)
    s = np.zeros((GH, GW), np.float32); n = np.zeros((GH, GW), np.uint16)
    for d in range(rank, 365, world):
        x = ghap_day(I2022 + d).astype(np.float32); ok = x > 0
        s += np.where(ok, x, 0); n += ok
        if d % 40 == rank:
            print(f"rank {rank}: day {d}", flush=True)
    np.save(OUT / f"part_{rank}_sum.npy", s); np.save(OUT / f"part_{rank}_n.npy", n)
    (OUT / f"done_{rank}").touch()
    if rank:
        return
    while len(list(OUT.glob("done_*"))) < world:
        time.sleep(10)
    S = np.zeros((GH, GW), np.float64); N = np.zeros((GH, GW), np.int32)
    for r in range(world):
        S += np.load(OUT / f"part_{r}_sum.npy"); N += np.load(OUT / f"part_{r}_n.npy")
    with np.errstate(all="ignore"):
        mean = np.where(N > 0, S / N, np.nan).astype(np.float32)
    np.save(HERE / "data" / "pm_annual_2022.npy", mean.astype(np.float16))
    ok = np.isfinite(mean).reshape(GH // 40, 40, GW // 40, 40)
    with np.errstate(all="ignore"):
        cell = np.where(ok, np.nan_to_num(mean).reshape(GH // 40, 40, GW // 40, 40), 0).sum((1, 3)) / ok.sum((1, 3))
    np.save(HERE / "data" / "pm_annual_2022_40km.npy", cell.astype(np.float32))
    for r in range(world):
        for k in ("sum", "n"):
            (OUT / f"part_{r}_{k}.npy").unlink()
        (OUT / f"done_{r}").unlink()
    print("saved annual mean: global land mean", float(np.nanmean(mean)), flush=True)


if __name__ == "__main__":
    main()
