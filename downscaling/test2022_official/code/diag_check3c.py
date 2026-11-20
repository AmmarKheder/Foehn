#!/usr/bin/env python3
"""A7(g) check 3, diagnosis (3): spatial structure of the difference to the frozen test on the anomalous member-days.
Predictions only. For a member-day: relative difference r = (mine - theirs) / theirs at the shared station pixels;
sign coherence within 2 deg boxes (a smooth large-scale pattern gives coherent signs, rounding noise does not)."""
import datetime as dt
from pathlib import Path
import numpy as np
ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
O = ROOT / "test2022_official"
F = ROOT / "test2022_ghap"
Y0 = dt.date(2022, 1, 1)
meta = np.load(O / "pos/st_meta_official.npz")
print("meta keys:", meta.files)
fidx = meta["frozen_idx"]
sh = fidx >= 0
S = np.load(O / "preds/sameday_ds.npy", mmap_mode="r")
lat = meta["pix_lat"][sh]
lon = meta["pix_lon"][sh]
if lat is None:
    for k in meta.files:
        print(k, meta[k].shape, meta[k].dtype)
    raise SystemExit
for (t, m) in ((358, 4), (282, 5), (166, 0), (34, 4), (16, 7), (7, 5), (282, 7), (166, 2), (100, 0)):
    d = Y0 + dt.timedelta(t)
    y = np.load(F / "preds_st" / "cdffm" / f"{d}.npz")["members"][m, fidx[sh]].astype(np.float64)
    x = np.asarray(S[t][m, sh, 0]).astype(np.float64)
    r = (x - y) / np.maximum(y, 1e-3)
    # log-space difference in float16 steps is too coarse: use r directly
    box = (np.floor(lat / 2).astype(int) * 1000 + np.floor(lon / 2).astype(int))
    ub, inv, cnt = np.unique(box, return_inverse=True, return_counts=True)
    nz = r != 0
    coh = []
    for b in np.where(cnt >= 20)[0]:
        s = np.sign(r[(inv == b) & nz])
        if s.size >= 10:
            coh.append(abs(s.mean()))
    print(f"day {t} ({d}) member {m}: nonzero {nz.mean():.3f} | r quantiles (1, 25, 50, 75, 99 %) {np.round(np.quantile(r, [0.01, 0.25, 0.5, 0.75, 0.99]), 5)} "
          f"| mean r {r.mean():+.2e} | share r>0 among nonzero {np.mean(r[nz] > 0) if nz.any() else float('nan'):.3f} | boxes {len(coh)}, median |mean sign| {np.median(coh) if coh else float('nan'):.2f} "
          f"| |r| median among nonzero {np.median(np.abs(r[nz])) if nz.any() else float('nan'):.2e} | y median {np.median(y):.1f}")
    # by latitude band of 10 deg: share nonzero and mean sign
    for lo_, hi_ in ((-60, 0), (0, 20), (20, 30), (30, 40), (40, 50), (50, 84)):
        k = (lat >= lo_) & (lat < hi_)
        if k.sum() > 30:
            print(f"     lat [{lo_},{hi_}): n {int(k.sum())}, nonzero {nz[k].mean():.3f}, mean sign {np.sign(r[k][nz[k]]).mean() if nz[k].any() else float('nan'):+.2f}, max |r| {np.abs(r[k]).max():.4f}")
