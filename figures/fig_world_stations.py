#!/usr/bin/env python3
"""World maps (Robinson) of daily PM2.5 on one day of the 2022 test year:
(A) ground truth at the monitoring stations (EEA, CNEMC, OpenAQ; local-day means, >= 18 h),
(B) the CAMS operational forecast issued the day before (lead 1 day, 0.4 deg), with its RMSE, correlation and bias at the
stations of the day. A third panel with the global 1 km model is added when its forecasts exist.
Run with PYTHONPATH=pylib_carto (cartopy) in the pytorch container, no torch needed:
    python fig_world_stations.py [day_of_year (0-based), default 20 = 21 Jan 2022]"""
import sys
from pathlib import Path

import cartopy
cartopy.config["data_dir"] = str(__import__("pathlib").Path(__file__).resolve().parent / "cartopy_data")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from fig_style import save

D = int(sys.argv[1]) if len(sys.argv) > 1 else 20
G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
st = np.load(G / "stations_2022.npz")
obs = st["daily"][:, D]
ok = np.isfinite(obs)
lat, lon, src, obs = st["lat"][ok], st["lon"][ok], st["source"][ok], obs[ok]
date = (np.datetime64("2022-01-01") + np.timedelta64(D, "D")).astype(str)

# CAMS forecast issued on day D-1, lead 1 day, valid on day D; grid 90 -> -90 (451), 0 -> 359.6 (900)
cams = np.load(G / "cams_2022_lead1_pm25.npy", mmap_mode="r")[D - 1].astype(np.float32)      # issued D-1, valid D
clat = 90.0 - 0.4 * np.arange(451); clon = 0.4 * np.arange(900)
fi = (90.0 - lat) / 0.4; fj = (lon % 360.0) / 0.4
i0 = np.clip(np.floor(fi).astype(int), 0, 449); j0 = np.floor(fj).astype(int) % 900; j1 = (j0 + 1) % 900
wi, wj = fi - i0, fj - np.floor(fj)
at_st = ((1 - wi) * ((1 - wj) * cams[i0, j0] + wj * cams[i0, j1]) + wi * ((1 - wj) * cams[i0 + 1, j0] + wj * cams[i0 + 1, j1]))
rmse = float(np.sqrt(np.mean((at_st - obs) ** 2))); r = float(np.corrcoef(at_st, obs)[0, 1]); bias = float(np.mean(at_st - obs))

bounds = [0, 5, 10, 15, 20, 30, 40, 50, 70, 100, 150, 500]
cmap = ListedColormap(plt.get_cmap("RdYlBu_r")(np.linspace(0.02, 0.98, len(bounds) - 1)))
cmap.set_over("#67001f")
norm = BoundaryNorm(bounds, cmap.N)
proj = ccrs.Robinson()
fig = plt.figure(figsize=(7.2, 2.55))
names = {0: "EEA", 1: "CNEMC", 3: "EPA", 2: "OpenAQ"}
for k, title in enumerate(("Ground truth: monitoring stations", "CAMS forecast (lead 1 day), 0.4$^\\circ$")):
    ax = fig.add_axes([0.005 + 0.5 * k, 0.2, 0.49, 0.72], projection=proj)
    ax.set_global()
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#f2f2f0", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#ffffff", edgecolor="none", zorder=0)
    if k == 0:
        o = np.argsort(obs)                                                   # high values on top
        ax.scatter(lon[o], lat[o], c=obs[o], s=2.2, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), linewidths=0, zorder=3)
        txt = " + ".join(f"{int((src == s).sum()):,} {n}" for s, n in names.items()) + f"\nDaily mean PM$_{{2.5}}$, {date}"
    else:
        lon2 = np.concatenate([clon[450:] - 360.0, clon[:450]]); c2 = np.concatenate([cams[:, 450:], cams[:, :450]], axis=1)
        ax.pcolormesh(lon2, clat, c2, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading="auto", rasterized=True, zorder=1)
        txt = f"RMSE {rmse:.1f} µg m$^{{-3}}$\nR {r:.2f}  |  bias {bias:+.1f}"
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.3, edgecolor="#8a8a8a", zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.2, edgecolor="#b0b0b0", zorder=4)
    ax.spines["geo"].set_linewidth(0.5); ax.spines["geo"].set_edgecolor("#999999")
    ax.set_title(title, fontsize=7.5, fontweight="bold", pad=3)
    ax.text(0.02, 0.97, "AB"[k], transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
    ax.text(0.02, 0.04, txt, transform=ax.transAxes, fontsize=5.2, va="bottom",
            bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="#999999", lw=0.4))
cax = fig.add_axes([0.2, 0.1, 0.6, 0.035])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="max",
                  ticks=bounds, spacing="uniform")
cb.set_label("PM$_{2.5}$ surface concentration (µg m$^{-3}$)", fontsize=6.5); cb.ax.tick_params(labelsize=5.8, length=2)
save(fig, f"fig_world_stations_{date}")
print(date, "stations", len(obs), {n: int((src == s).sum()) for s, n in names.items()}, f"CAMS RMSE {rmse:.1f} R {r:.2f} bias {bias:+.1f}")
