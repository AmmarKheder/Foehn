#!/usr/bin/env python3
"""Fig. 4 of the Nature paper: predictability, mean and variance. a-c, fraction of the daily variance predicted in 2022
(SS = 1 - MSE / variance of the daily truth in the cell; 1 = perfect, <= 0 = no better than the 2022 mean of the cell)
at lead times 1, 3 and 7 days. d, people living where SS > 0.5 (thick) or > 0.25 (thin): this work versus climatology
(2017-2020 mean of the same calendar days). e, mean PM2.5 of 2022 forecast one day ahead. f, power spectra of daily
PM2.5 at 1 km (all complete 512 x 512 km land tiles): GHAP, one ensemble member, the ensemble mean and CAMS. g, quantiles
of daily PM2.5 at the stations (identical station-days, lead time 1 day).
    python fig4_predictability.py <run>    (data: figdata_predictability.py)"""
import sys
from pathlib import Path

import cartopy.crs as ccrs
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from nature_style import DOUBLE, STYLE, letter, plt, pm_cmap, ryb, save, world_axes, PM_BOUNDS

RUN = sys.argv[1] if len(sys.argv) > 1 else "fp_flow_shift25_global_L7_fix"
z = np.load(Path(__file__).resolve().parent / "data" / f"predictability_{RUN}.npz")
SS, SSC, pop, MEANF = z["SS"], z["SSC"], z["pop"].astype(np.float64), z["MEANF"]
st = np.load(Path(__file__).resolve().parent / "data" / f"stats_{RUN}.npz")
lat = 84 - 0.25 * np.arange(576) - 0.125; lon = -180 + 0.25 * np.arange(1440) + 0.125

SB = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
scmap = ryb(len(SB) - 1)
scmap.set_under("#d9d0c7"); scmap.set_bad((0, 0, 0, 0))
snorm = BoundaryNorm(SB, scmap.N)

fig = plt.figure(figsize=(DOUBLE, 5.6))
for k, h in enumerate((1, 3, 7)):
    ax = world_axes(fig, [0.005 + k * 0.333, 0.64, 0.325, 0.34], bg=False)
    import cartopy.feature as cfeature
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#f2f5f8", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#e4e4e4", edgecolor="none", zorder=0)
    ax.pcolormesh(lon, lat, np.ma.masked_invalid(SS[h - 1]), cmap=scmap, norm=snorm, transform=ccrs.PlateCarree(),
                  shading="nearest", rasterized=True, zorder=1)
    share = pop[np.nan_to_num(SS[h - 1], nan=-9) > 0.5].sum() / 1e9
    ax.set_title(f"Lead time {h} day{'s' if h > 1 else ''}", fontsize=7, pad=2, fontweight="bold")
    ax.text(0.5, -0.03, f"{share:.1f} billion people where SS > 0.5", transform=ax.transAxes, ha="center", va="top", fontsize=6)
    letter(ax, "abc"[k], x=0.02, y=0.98)
cax = fig.add_axes([0.3, 0.6, 0.4, 0.011])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=snorm, cmap=scmap), cax=cax, orientation="horizontal", extend="min", spacing="uniform", ticks=SB)
cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.4, pad=1); cb.outline.set_linewidth(0.4)
cb.set_label("Fraction of the daily variance predicted (SS)   ·   grey: not predictable (SS ≤ 0)", fontsize=6, labelpad=1)

# d: people living where daily PM2.5 is predictable
ax = fig.add_axes([0.06, 0.07, 0.2, 0.42])
leads = np.arange(1, 8)
for th, lw, alpha in ((0.5, 1.0, 1.0), (0.25, 0.6, 0.5)):
    for key, arr in (("ours", SS), ("climatology", SSC)):
        sty = STYLE[key]
        y = [pop[np.nan_to_num(arr[h], nan=-9) > th].sum() / 1e9 for h in range(7)]
        ax.plot(leads, y, color=sty["color"], ls=sty["ls"], lw=lw, marker=sty["marker"] if th == 0.5 else "", ms=sty["ms"] + 0.4,
                markeredgecolor="white", markeredgewidth=0.4, alpha=alpha, zorder=sty["z"])
ax.set_xticks(leads); ax.set_xlim(0.7, 7.3); ax.set_ylim(0, 8)
ax.set_xlabel("Lead time (days)"); ax.set_ylabel("People (billions)")
ax.set_title("People where daily PM$_{2.5}$\nis predictable", fontsize=6.5, pad=3)
ax.legend([plt.Line2D([], [], color=STYLE["ours"]["color"], lw=1.0), plt.Line2D([], [], color=STYLE["climatology"]["color"], ls=STYLE["climatology"]["ls"], lw=0.9),
           plt.Line2D([], [], color="#777777", lw=1.0), plt.Line2D([], [], color="#777777", lw=0.6, alpha=0.5)],
          ["Foehn", "Climatology", "SS > 0.5", "SS > 0.25"], loc="upper right", frameon=False, fontsize=5.5)
letter(ax, "d", x=-0.25, y=1.02)

# e: mean PM2.5 of 2022 (one-day-ahead forecasts)
cmap, norm = pm_cmap()
ax = world_axes(fig, [0.285, 0.1, 0.4, 0.38], bg=False)
ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#f2f5f8", edgecolor="none", zorder=0)
ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#e4e4e4", edgecolor="none", zorder=0)
ax.pcolormesh(lon, lat, np.ma.masked_invalid(MEANF), cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading="nearest", rasterized=True, zorder=1)
ax.set_title("Mean PM$_{2.5}$ of 2022, forecast one day ahead", fontsize=6.5, pad=2)
letter(ax, "e", x=0.02, y=1.02)
cax = fig.add_axes([0.33, 0.09, 0.31, 0.01])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="max", spacing="uniform", ticks=PM_BOUNDS)
cb.ax.set_xticklabels([f"{t:g}" for t in PM_BOUNDS]); cb.ax.tick_params(labelsize=5.3, length=1.5, width=0.4, pad=1); cb.outline.set_linewidth(0.4)
cb.set_label("µg m$^{-3}$", fontsize=5.8, labelpad=1)
# f: spectra (variance by spatial scale)
ax = fig.add_axes([0.765, 0.33, 0.22, 0.2])
wl = st["wl"]
for key, lab, col, ls, lw in (("ghap", "GHAP (target)", "#111111", "-", 1.0), ("member", "One member", STYLE["member"]["color"], "-", 1.0),
                              ("mean", "Ensemble mean", STYLE["ours"]["color"], "-", 1.0), ("cams", "CAMS (40 km)", STYLE["cams"]["color"], "-", 0.8)):
    ax.plot(wl, st[f"sp_{key}"], color=col, ls=ls, lw=lw, label=lab)
ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis(); ax.set_xlim(512, 2)
ax.set_xlabel("Wavelength (km)", labelpad=1); ax.set_ylabel("Power (µg$^{2}$ m$^{-6}$)", labelpad=1)
ax.set_title("Variance by spatial scale", fontsize=6.5, pad=2); ax.legend(frameon=False, fontsize=5.2, loc="lower left")
letter(ax, "f", x=-0.27, y=1.02)
# g: quantiles at the stations
ax = fig.add_axes([0.765, 0.065, 0.22, 0.2])
Q = st["Q"]; qo = st["obs_1"]
for key, lab, col, ls in (("member", "One member", STYLE["member"]["color"], "-"), ("mean", "Ensemble mean", STYLE["ours"]["color"], "-"),
                          ("cams", "CAMS", STYLE["cams"]["color"], "-"), ("aurora", "Aurora", STYLE["aurora"]["color"], "-")):
    ax.plot(qo, st[f"{key}_1"], color=col, ls=ls, lw=0.9, marker="s" if key == "cams" else "o", ms=1.8, label=lab)
lim = [1, 300]; ax.plot(lim, lim, color="#111111", lw=0.6, ls=(0, (1, 1)))
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Observed quantile (µg m$^{-3}$)", labelpad=1); ax.set_ylabel("Forecast quantile", labelpad=1)
ax.set_title("Distribution at the stations (1-99.9%)", fontsize=6.5, pad=2); ax.legend(frameon=False, fontsize=5.2, loc="upper left")
letter(ax, "g", x=-0.27, y=1.02)
save(fig, f"fig4_predictability_{RUN}")
