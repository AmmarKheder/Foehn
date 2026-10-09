"""dl_wb2.py hres 2022 1 --limit 2 must reproduce the 2022 test store (hres_block / hres_wind / hres_rain0, built by
dl_hres_global.py + dl_hres_global_ext.py) on its first two runs; and the 3-D fields (B) must be physical."""
import numpy as np
G = "/scratch/project_462001140/ammar/eccv/data/world/global1km"
z = np.load(f"{G}/wb2/hres_A_2022_01.npz")
def head(f, n):
    """first n records of a big .npy without memory-mapping it (login-node virtual memory limit)"""
    with open(f, "rb") as fh:
        np.lib.format.read_magic(fh); sh, _, dt = np.lib.format.read_array_header_1_0(fh)
        cnt = n * int(np.prod(sh[1:])); return np.fromfile(fh, dtype=dt, count=cnt).reshape((n,) + tuple(sh[1:]))
n = len(z["day_of_year"]); assert list(z["day_of_year"]) == list(range(n)), z["day_of_year"]
blk, wnd, rn0 = (head(f"{G}/{f}", n) for f in ("hres_block.npy", "hres_wind.npy", "hres_rain0.npy"))
for name, new, old in (("block", z["block"], blk[:n]), ("wind", z["wind"], wnd[:n]), ("rain_d0", z["rain_d0"], rn0[:n])):
    new, old = np.asarray(new, np.float32), np.asarray(old, np.float32)
    d = np.abs(new - old); ok = np.isfinite(old)
    print(f"{name:8s} shape {new.shape} | max |diff| {np.nanmax(d[ok]):.4g} | mean |diff| {np.nanmean(d[ok]):.3g} | mean |old| {np.nanmean(np.abs(old[ok])):.3g}"
          f" | NaN new {np.isnan(new).mean():.3g} old {np.isnan(old).mean():.3g}")
    if name == "block":
        per_c = [float(np.nanmax(d[:, :, c])) for c in range(31)]
        print("   max |diff| per channel:", " ".join(f"{v:.3g}" for v in per_c))
Bm = np.load(f"{G}/wb2/hres_B_2022_01.npy")
ch = [str(c) for c in np.load(f"{G}/wb2/static_1deg.npz")["channels"]] if False else None
print("B shape", Bm.shape)
x = np.asarray(Bm[0], np.float32)                       # (29, 53, 181, 360)
names = [f"{v}{p}" for v in ("u", "v", "w", "t", "q", "z") for p in (1000, 925, 850, 700, 600, 500, 400, 300)] + ["u10", "v10", "t2m", "sp", "tp6h"]
for k, nm in enumerate(names):
    a = x[:, k]
    if nm in ("u850", "w500", "t1000", "t300", "q1000", "q300", "z1000", "z500", "z300", "u10", "t2m", "sp", "tp6h", "u300"):
        print(f"   {nm:6s} min {np.nanmin(a):9.3f} mean {np.nanmean(a):9.3f} max {np.nanmax(a):9.3f} NaN {np.isnan(a).mean():.3f}")
