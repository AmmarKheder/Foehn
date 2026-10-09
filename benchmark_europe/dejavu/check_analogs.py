"""How close are the flow sources to the truth? RMSE (ug/m3, land) vs PM_{i+h} of: noise-scale reference (climatology
std), persistence PM_i, top-1 analog successor, mean of top-8 analog successors. 2021 and 2022."""
import numpy as np
from pathlib import Path
D = Path(__file__).resolve().parent / "data"
pm = np.load(D / "pm.npy", mmap_mode="r"); meta = np.load(D / "meta.npz"); an = np.load(D / "analogs.npz")
year, land, start, valid = meta["year"], meta["land"], an["start"], an["valid"]
clim = np.nanmean(np.asarray(pm)[year <= 2020], 0)
for yr in (2021, 2022):
    print(yr)
    for h in (1, 2, 3, 4):
        q = [i for i in np.where(year == yr)[0] if i + h < len(year) and year[i + h] == yr and valid[i] and valid[i + h] and start[i, h - 1, 0] >= 0]
        e = {k: [0.0, 0] for k in ("clim", "pers", "a1", "a8")}
        for i in q[::2]:
            t = pm[i + h]; m = land & np.isfinite(t)
            preds = {"clim": clim, "pers": pm[i], "a1": pm[start[i, h - 1, 0] + h],
                     "a8": np.nanmean([pm[j + h] for j in start[i, h - 1, :8]], 0)}
            for k, p in preds.items():
                mm = m & np.isfinite(p); e[k][0] += ((p[mm] - t[mm]) ** 2).sum(); e[k][1] += mm.sum()
        print(f"  T+{h}: " + "  ".join(f"{k} {np.sqrt(v[0] / v[1]):.2f}" for k, v in e.items()))
