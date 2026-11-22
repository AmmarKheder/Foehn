#!/usr/bin/env python3
"""Independent check of the member-0 estimator of A14 4a on 0.25 deg cells: at the populated pixels, where all four stored
members exist (RUN/eval_pop_crps_e01), the cell mean of each member over the populated pixels of each 0.25 deg cell;
ratio sum_c (4/3) u_m^2 / sum_c s2 for m = 0..3 (u_m = member-m cell mean minus the mean of four; 1 on average over m).
All forecasts at leads 1 and 3; cells with >= 1, >= 100 and >= 400 populated pixels (of 625). Read-only; writes j_m0.json here."""
import os  # FOEHN_WORK, FOEHN_ENVS: roots of the working trees and of the environments
import glob
import json
from pathlib import Path

import numpy as np

RUN = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/global1km/runs/green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4"))
EVP = RUN / "eval_pop_crps_e01"
OUT = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution/a14/classU_review/judge/j_m0.json"))
idx = np.load(EVP / "pop_idx.npy").astype(np.int64)
row, col = idx // 36000, idx % 36000
cell = (row // 25) * 1440 + (col // 25)
uc, ci = np.unique(cell, return_inverse=True)
npix = np.bincount(ci).astype(np.float64)
big = npix >= 100
big4 = npix >= 400
fc = []
for p in sorted(glob.glob(str(EVP / "part_*.npz")), key=lambda s: int(s.split("_")[-1].split(".")[0])):
    k = int(p.split("_")[-1].split(".")[0])
    for n, (d, h, t) in enumerate(np.load(p)["keys"]):
        if int(h) in (1, 3):
            fc.append((int(d), int(h), k, n))
fc = sorted(fc)
acc = {(h, s): dict(u2=np.zeros(4), s2=0.0, ncd=0.0) for h in (1, 3) for s in ("all", "ge100", "ge400")}
used = {1: 0, 3: 0}
for j, (d, h, k, n) in enumerate(fc):
    X = np.asarray(np.load(EVP / f"pop_{k}.npy", mmap_mode="r")[n], np.float32)
    fin = np.isfinite(X).all(1)
    w = fin.astype(np.float64)
    cnt = np.bincount(ci, weights=w, minlength=len(uc))
    cm = np.stack([np.bincount(ci, weights=np.where(fin, X[:, m], 0.0), minlength=len(uc)) for m in range(4)], 1)
    okc = cnt > 0
    cm = cm[okc] / cnt[okc, None]
    xb = cm.mean(1, keepdims=True); u = cm - xb
    s2 = (u ** 2).sum(1) / 3
    for s, sel in (("all", np.ones(okc.sum(), bool)), ("ge100", big[okc]), ("ge400", big4[okc])):
        a = acc[(h, s)]
        a["u2"] += (4 / 3 * u[sel] ** 2).sum(0); a["s2"] += s2[sel].sum(); a["ncd"] += sel.sum()
    used[h] += 1
res = {"forecasts_used": used, "ratios": {}}
for (h, s), a in acc.items():
    res["ratios"][f"lead{h}_{s}"] = dict(ratio_by_member=(a["u2"] / a["s2"]).tolist(), cell_days=a["ncd"])
print(json.dumps(res, indent=1))
OUT.write_text(json.dumps(res, indent=1))
