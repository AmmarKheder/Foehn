#!/usr/bin/env python3
"""World maps (Robinson) of daily PM2.5 on one day of the 2022 test year: (A) monitoring stations, (B) GHAP (the 1 km
reference product, not a forecast), (C) CAMS operational forecast issued the day before (0.4 deg), (D) our global 1 km
model, forecast issued the day before (at 1 km when the evaluation saved the 1 km map of that day, else 5 km).
C and D: RMSE, correlation and bias at the stations of the day.
Run in the pytorch container with PYTHONPATH=pylib_carto:
    python fig_world_model.py <run> <start doy> [lead]"""
import os
import sys
from pathlib import Path

import cartopy
cartopy.config["data_dir"] = str(Path(__file__).resolve().parent / "cartopy_data")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from fig_style import save

RUN, D0 = sys.argv[1], int(sys.argv[2]); H = int(sys.argv[3]) if len(sys.argv) > 3 else 1
DT = D0 + H
G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
E = Path("/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km/runs") / RUN / "eval_global"
date = (np.datetime64("2022-01-01") + np.timedelta64(DT, "D")).astype(str)
# our model at the stations (1 km pixel, ensemble mean) from the evaluation parts
f_st = None
for p in glob.glob(str(E / "part_*.npz")):
    z = np.load(p); k = np.where((z["keys"][:, 0] == D0) & (z["keys"][:, 1] == H))[0]
    if len(k):
        f_st = z["st_ens"][k[0]].astype(np.float32).mean(-1); break
assert f_st is not None, "forecast at the stations not found"
for f in ("stations_2022.npz", "stations_2022_v2.npz"):                  # the station file of that evaluation
    st = np.load(G / f)
    if len(st["lat"]) == len(f_st):
        break
assert len(st["lat"]) == len(f_st)
obs = st["daily"][:, DT]
ok = np.isfinite(obs) & np.isfinite(f_st)
lat, lon, src, o, fm = st["lat"][ok], st["lon"][ok], st["source"][ok], obs[ok], f_st[ok]
cams = np.load(G / "cams_2022_lead1_pm25.npy", mmap_mode="r")[DT - 1].astype(np.float32)          # issued DT-1, valid DT
fi = (90.0 - lat) / 0.4; fj = (lon % 360.0) / 0.4
i0 = np.clip(np.floor(fi).astype(int), 0, 449); j0 = np.floor(fj).astype(int) % 900; j1 = (j0 + 1) % 900
wi, wj = fi - i0, fj - np.floor(fj)
cm = (1 - wi) * ((1 - wj) * cams[i0, j0] + wj * cams[i0, j1]) + wi * ((1 - wj) * cams[i0 + 1, j0] + wj * cams[i0 + 1, j1])
score = lambda x: (float(np.sqrt(np.mean((x - o) ** 2))), float(np.corrcoef(x, o)[0, 1]), float(np.mean(x - o)))
sc_cams, sc_ours = score(cm), score(fm)
KM1 = (E / f"map1km_d{D0}_h{H}.npy").exists()
model_map = np.load(E / (f"map1km_d{D0}_h{H}.npy" if KM1 else f"map5km_d{D0}_h{H}.npy")).astype(np.float32)   # 1 km or 5 km
land = None
with open(G / "ghap1km_tiles.npy", "rb") as f:
    np.lib.format.read_magic(f); np.lib.format.read_array_header_1_0(f); hdr = f.tell()
n = 29 * 71 * 512 * 512
fd = os.open(G / "ghap1km_tiles.npy", os.O_RDONLY)
g = np.frombuffer(os.pread(fd, n * 2, hdr + (1826 + DT) * n * 2), np.float16).reshape(29, 71, 512, 512).transpose(0, 2, 1, 3).reshape(29 * 512, 71 * 512)[:14400, :36000].astype(np.float32)
g[g <= 0] = np.nan
if KM1:
    ghap_map = g
    with open(G / "static1km_tiles.npy", "rb") as f:
        np.lib.format.read_magic(f); np.lib.format.read_array_header_1_0(f); hdr_s = f.tell()
    st1 = np.frombuffer(os.pread(os.open(G / "static1km_tiles.npy", os.O_RDONLY), n * 2, hdr_s + 1 * n * 2), np.float16)
    land = st1.reshape(29, 71, 512, 512).transpose(0, 2, 1, 3).reshape(29 * 512, 71 * 512)[:14400, :36000] > 0
    model_map[~land] = np.nan
else:
    with np.errstate(all="ignore"):
        ghap_map = np.nanmean(g.reshape(2880, 5, 7200, 5), (1, 3))
    del g
RES = 0.01 if KM1 else 0.05
bounds = [0, 5, 10, 15, 20, 30, 40, 50, 70, 100, 150, 500]
cmap = ListedColormap(plt.get_cmap("RdYlBu_r")(np.linspace(0.02, 0.98, len(bounds) - 1))); cmap.set_over("#67001f")
norm = BoundaryNorm(bounds, cmap.N)
proj, pc = ccrs.Robinson(), ccrs.PlateCarree()
RS = 4200 if KM1 else 1800                     # output raster of the reprojection (~ the 600 dpi panel width)
fig = plt.figure(figsize=(7.2, 4.6))
titles = ("Stations (observed)", "GHAP 1 km (reference product, not a forecast)",
          "CAMS forecast, issued the day before (0.4$^\\circ$)", "Our model, issued the day before (1 km)" if KM1 else
          "Our model, issued the day before (1 km, shown at 5 km)")
names = {0: "EEA", 1: "CNEMC", 3: "EPA", 4: "AirKorea", 5: "Japan", 6: "India", 2: "OpenAQ"}
for k in range(4):
    ax = fig.add_axes([0.005 + 0.5 * (k % 2), 0.56 - 0.44 * (k // 2), 0.49, 0.38], projection=proj)
    ax.set_global()
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#f2f2f0", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#ffffff", edgecolor="none", zorder=0)
    if k == 0:
        oo = np.argsort(o)
        ax.scatter(lon[oo], lat[oo], c=o[oo], s=1.6, cmap=cmap, norm=norm, transform=pc, linewidths=0, zorder=3)
        txt = " + ".join(f"{int((src == s).sum()):,} {nm}" for s, nm in names.items()) + f"\nDaily mean PM$_{{2.5}}$, {date}"
    elif k == 1:
        ax.imshow(ghap_map, origin="upper", extent=(-180, 180, -60, 84), transform=pc, cmap=cmap, norm=norm, regrid_shape=RS, zorder=1)
        gi = np.clip(((84 - lat) / RES).astype(int), 0, ghap_map.shape[0] - 1); gj = np.clip(((lon + 180) / RES).astype(int), 0, ghap_map.shape[1] - 1)
        txt = f"at the stations: RMSE {score(np.nan_to_num(ghap_map[gi, gj], nan=np.nanmean(o)))[0]:.1f}"
    elif k == 2:
        lon2 = np.concatenate([0.4 * np.arange(450, 900) - 360.0, 0.4 * np.arange(450)])
        c2 = np.concatenate([cams[:, 450:], cams[:, :450]], axis=1)
        ax.pcolormesh(lon2, 90.0 - 0.4 * np.arange(451), c2, cmap=cmap, norm=norm, transform=pc, shading="auto", rasterized=True, zorder=1)
        txt = f"RMSE {sc_cams[0]:.1f} µg m$^{{-3}}$  |  R {sc_cams[1]:.2f}  |  bias {sc_cams[2]:+.1f}"
    else:
        ax.imshow(model_map, origin="upper", extent=(-180, 180, -60, 84), transform=pc, cmap=cmap, norm=norm, regrid_shape=RS, zorder=1)
        txt = f"RMSE {sc_ours[0]:.1f} µg m$^{{-3}}$  |  R {sc_ours[1]:.2f}  |  bias {sc_ours[2]:+.1f}"
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.3, edgecolor="#8a8a8a", zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.2, edgecolor="#b0b0b0", zorder=4)
    ax.spines["geo"].set_linewidth(0.5); ax.spines["geo"].set_edgecolor("#999999")
    ax.set_title(titles[k], fontsize=7, fontweight="bold", pad=2)
    ax.text(0.01, 0.98, "ABCD"[k], transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    ax.text(0.02, 0.03, txt, transform=ax.transAxes, fontsize=4.8, va="bottom", bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="#999999", lw=0.4))
cax = fig.add_axes([0.2, 0.06, 0.6, 0.022])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="max", ticks=bounds, spacing="uniform")
cb.set_label("PM$_{2.5}$ surface concentration (µg m$^{-3}$)", fontsize=6.5); cb.ax.tick_params(labelsize=5.8, length=2)
save(fig, f"fig_world_model_{date}_h{H}" + ("_1km" if KM1 else ""))
print(date, "stations", len(o), f"CAMS RMSE {sc_cams[0]:.1f} R {sc_cams[1]:.2f} bias {sc_cams[2]:+.1f} | ours RMSE {sc_ours[0]:.1f} R {sc_ours[1]:.2f} bias {sc_ours[2]:+.1f}")
