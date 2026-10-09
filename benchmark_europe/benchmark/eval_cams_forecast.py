"""Operational CAMS global forecast (00 UTC run, lead days 1-4, 0.4 deg) scored against GHAP (0.25 deg block means),
Europe window of the paper, 2022, same valid days / land cells as the benchmark (dejavu/data)."""
import json
import numpy as np
import zarr
from pathlib import Path

D = Path("/scratch/project_462001140/ammar/eccv/cran-pm/experiments/dejavu/data")
pm = np.load(D / "pm.npy", mmap_mode="r"); meta = np.load(D / "meta.npz"); valid = np.load(D / "analogs.npz")["valid"]
year, doy = meta["year"], meta["doy"]
cams = zarr.open("/scratch/project_462001140/ammar/eccv/data/world/cams_forecast_daily/2022.zarr", mode="r")
lat = 72.0 - 0.25 * np.arange(168); lon = -25.0 + 0.25 * np.arange(280)
fi = (90.0 - lat) / 0.4; fj = (lon % 360) / 0.4
i0 = np.floor(fi).astype(int); j0 = np.floor(fj).astype(int); wi = (fi - i0)[:, None]; wj = (fj - j0)[None, :]
j1 = (j0 + 1) % 900
def interp(a):
    return (1 - wi) * ((1 - wj) * a[i0][:, j0] + wj * a[i0][:, j1]) + wi * ((1 - wj) * a[i0 + 1][:, j0] + wj * a[i0 + 1][:, j1])
acc = {h: np.zeros(6) for h in (1, 2, 3, 4)}
for i in np.where(year == 2022)[0]:
    for h in (1, 2, 3, 4):
        if i + h >= len(year) or year[i + h] != 2022 or not (valid[i] and valid[i + h]):
            continue
        f = interp(np.asarray(cams[int(doy[i]), 0, h], np.float32))      # pm2p5, lead day h
        t = pm[i + h]; m = np.isfinite(t) & (t > 0) & np.isfinite(f)
        e, g = t[m] > 25, f[m] > 25
        acc[h] += [m.sum(), ((f[m] - t[m]) ** 2).sum(), (f[m] - t[m]).sum(), (e & g).sum(), (e | g).sum(), 0]
res = {f"T+{h}": dict(rmse=float(np.sqrt(a[1] / a[0])), bias=float(a[2] / a[0]), csi25=float(a[3] / max(a[4], 1))) for h, a in acc.items()}
Path("cams_forecast_2022.json").write_text(json.dumps(res, indent=1))
for k, v in res.items():
    print(f"CAMS operational forecast {k}: RMSE {v['rmse']:.2f}  bias {v['bias']:+.2f}  CSI>25 {v['csi25']:.3f}")
