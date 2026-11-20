#!/usr/bin/env python3
"""Independent check of data/hidden_systems_pop.npz for one forecast (start day 64, lead 3): the people warned by the CAMS
forecast and by Aurora are recomputed here with the interpolation function of figures/compare_fair.py (copied verbatim, with
latitude and longitude as arguments) and with explicit sums instead of the code table.   python check_independent.py"""
import glob
from pathlib import Path

import numpy as np
import zarr

G = Path("/scratch/project_462001804/ammar/eccv/data/world/global1km")
RUN = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km/runs/green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4")
EVP = RUN / "eval_pop_crps_e01"
AURD = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/benchmark/aurora")
GW = 36000
D, H = 64, 3


def bil(field, lat0, dlat, lon0, dlon, lat, lon):                       # compare_fair.py:bil
    nlat, nlon = field.shape
    fi = (lat0 - lat) / dlat; fj = ((lon - lon0) % 360.0) / dlon
    i0 = np.clip(np.floor(fi).astype(int), 0, nlat - 2); j0 = np.floor(fj).astype(int) % nlon; j1 = (j0 + 1) % nlon
    wi, wj = np.clip(fi - i0, 0, 1), fj - np.floor(fj)
    return ((1 - wi) * ((1 - wj) * field[i0, j0] + wj * field[i0, j1]) + wi * ((1 - wj) * field[i0 + 1, j0] + wj * field[i0 + 1, j1])).astype(np.float32)


pz = np.load(G / "pop_pixels.npz"); idx, pop = pz["idx"], pz["pop"].astype(np.float64)
r, c = idx // GW, idx % GW
lat, lon = 84.0 - 0.01 * (r + 0.5), -180.0 + 0.01 * (c + 0.5)
for p in glob.glob(str(EVP / "part_*.npz")):
    keys = np.load(p)["keys"]
    hit = np.flatnonzero((keys[:, 0] == D) & (keys[:, 1] == H))
    if len(hit):
        k, n, t = p.split("_")[-1].split(".")[0], int(hit[0]), int(keys[hit[0], 2])
print("forecast", D, H, "target", t, "part", k, "row", n)
m = np.asarray(np.load(EVP / f"pop_{k}.npy", mmap_mode="r")[n], np.float32)
g1 = np.asarray(np.load(EVP / f"pop_ghap_{k}.npy", mmap_mode="r")[n], np.float32)
g40 = np.asarray(np.load(EVP / f"cell40_{k}.npy", mmap_mode="r")[n, 1], np.float32).ravel()[(r // 40) * (GW // 40) + c // 40]
ok = np.isfinite(g1) & np.isfinite(g40) & np.isfinite(m).all(1)
cams = np.asarray(zarr.open("/scratch/project_462001804/ammar/eccv/data/world/cams_forecast_daily/2022.zarr", mode="r")[D + 1, 0, H - 1], np.float32)
A = None
for p in glob.glob(str(AURD / "out_12h_all" / "part_*.npz")):
    z = np.load(p)
    if D + 1 in z["days"]:
        A = z["pred"][list(z["days"]).index(D + 1)].astype(np.float32); la, lo = z["lat"], z["lon"]
k0 = 2 * (H - 1)
aur = 0.25 * A[k0 - 1] + 0.5 * A[k0] + 0.25 * A[k0 + 1]
cb = bil(cams, 90.0, 0.4, 0.0, 0.4, lat, lon)
ab = bil(aur, float(la[0]), float(la[0] - la[1]), float(lo[0]), float(lo[1] - lo[0]), lat, lon)
z = np.load(Path(__file__).resolve().parent / "data" / "hidden_systems_pop.npz")
K = z["keys"]; f = int(np.flatnonzero((K[:, 0] == D) & (K[:, 1] == H))[0]); sys1 = list(z["sys1"]); thr = list(z["thr"])
c16 = np.arange(16); E, C, W = (c16 & 1) > 0, (c16 & 2) > 0, (c16 & 8) > 0
worst = 0.0
for T in (15.0, 37.5, 75.0):
    ex, hid = ok & (g1 > T), ok & (g1 > T) & ~(g40 > T)
    for name, x in (("cams", cb), ("aurora", ab), ("foehn_mean", m.mean(1)), ("foehn_p50", None)):
        w = ((m > T).sum(1) >= 2) if x is None else (x > T)
        mine = np.array([pop[ok & w].sum(), pop[ex & w].sum(), pop[hid & w].sum(), pop[ex].sum(), pop[hid].sum()])
        a = z["T1"][f, 0, thr.index(T), sys1.index(name)]
        ref = np.array([a[W].sum(), a[W & E].sum(), a[W & E & ~C].sum(), a[E].sum(), a[E & ~C].sum()])
        dev = np.abs(mine - ref).max() / ref.max(); worst = max(worst, dev)
        print(f"> {T:g} {name:10s} warned {mine[0] / 1e6:8.2f} M (file {ref[0] / 1e6:8.2f}), rightly {mine[1] / 1e6:8.2f} ({ref[1] / 1e6:8.2f}), "
              f"hidden warned {mine[2] / 1e6:6.2f} ({ref[2] / 1e6:6.2f}) | max rel. dev. {dev:.1e}")
print("worst relative deviation", worst)
