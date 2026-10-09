#!/usr/bin/env python3
"""Data of the predictability figure: for every 0.25 deg cell and lead time (2022), the fraction of the daily variance
predicted, SS = 1 - MSE / Var(truth) (truth = GHAP cell mean of the target day; 1 = perfect, 0 = no better than the
2022 mean of the cell), for this work (ensemble mean) and for persistence (GHAP of the start day), plus the
population of every cell. Cells need >= 200 valid days.
    source world/env.sh; python figdata_predictability.py <run>   -> data/predictability_<run>.npz"""
import glob
import sys
from pathlib import Path

import numpy as np

RUN = sys.argv[1]
E = Path("/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km/runs") / RUN / "eval_global"
G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
P = [np.load(p) for p in sorted(glob.glob(str(E / "part_*.npz")))]
keys = np.concatenate([p["keys"] for p in P])
pm = np.load(G / "pm.npy", mmap_mode="r")
# climatology (as in the Europe study): 2017-2020 mean of GHAP over the same calendar days +-7, here on the 0.25 deg cells
meta = np.load(G / "meta.npz"); doy_all = np.minimum(meta["doy"][:1461], 364)
S_ = np.zeros((365, 576, 1440), np.float32); N_ = np.zeros((365, 576, 1440), np.float32)
for k0 in range(0, 1461, 61):
    blk = np.asarray(pm[k0:min(k0 + 61, 1461)], np.float32)          # training years only (2017-2020)
    for j, x in enumerate(blk):
        ok_ = np.isfinite(x); S_[doy_all[k0 + j]] += np.where(ok_, x, 0); N_[doy_all[k0 + j]] += ok_
CLIM = np.full((365, 576, 1440), np.nan, np.float32)
for t in range(365):
    w = [(t + o) % 365 for o in range(-7, 8)]
    with np.errstate(all="ignore"):
        CLIM[t] = S_[w].sum(0) / N_[w].sum(0)
del S_, N_
print("climatology ready", flush=True)
SS, SSP, NV = np.full((7, 576, 1440), np.nan, np.float32), np.full((7, 576, 1440), np.nan, np.float32), np.zeros((7, 576, 1440), np.int16)
SSC = np.full((7, 576, 1440), np.nan, np.float32)
MEANF, MEANT = np.full((576, 1440), np.nan, np.float32), np.full((576, 1440), np.nan, np.float32)
for h in range(1, 8):
    rows = [(i, j) for i, p in enumerate(P) for j in np.where(p["keys"][:, 1] == h)[0]]
    F = np.stack([P[i]["map25"][j] for i, j in rows]).astype(np.float32)
    T = np.stack([P[i]["map25_ghap"][j] for i, j in rows]).astype(np.float32)
    d0 = np.array([P[i]["keys"][j, 0] for i, j in rows])
    Pz = np.asarray(pm[1826 + d0], np.float32)                                     # persistence: GHAP of the start day
    tday = np.array([P[i]["keys"][j, 2] for i, j in rows]); Cz = CLIM[tday]          # climatology of the target day
    ok = np.isfinite(F) & np.isfinite(T) & np.isfinite(Pz) & np.isfinite(Cz)
    n = ok.sum(0)
    with np.errstate(all="ignore"):
        mt = np.where(ok, T, 0).sum(0) / n
        var = np.where(ok, (T - mt) ** 2, 0).sum(0) / n
        mse = np.where(ok, (F - T) ** 2, 0).sum(0) / n; msep = np.where(ok, (Pz - T) ** 2, 0).sum(0) / n
        msec = np.where(ok, (Cz - T) ** 2, 0).sum(0) / n
        good = (n >= 200) & (var > 0.25)
        SS[h - 1] = np.where(good, 1 - mse / var, np.nan); SSP[h - 1] = np.where(good, 1 - msep / var, np.nan)
        SSC[h - 1] = np.where(good, 1 - msec / var, np.nan)
        if h == 1:
            MEANF = np.where(good, np.where(ok, F, 0).sum(0) / n, np.nan); MEANT = np.where(good, mt, np.nan)
    NV[h - 1] = n
    print(f"lead {h}: {len(rows)} forecasts, {np.isfinite(SS[h - 1]).sum()} cells, median SS {np.nanmedian(SS[h - 1]):.2f}, "
          f"climatology {np.nanmedian(SSC[h - 1]):.2f}, persistence {np.nanmedian(SSP[h - 1]):.2f}", flush=True)
out = Path(__file__).resolve().parent / "data" / f"predictability_{RUN}.npz"
np.savez(out, SS=SS, SSP=SSP, SSC=SSC, NV=NV, MEANF=MEANF, MEANT=MEANT, pop=np.load(G / "pop025.npy"))
print("saved", out)
