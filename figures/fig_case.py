#!/usr/bin/env python3
"""Case study: the January 2022 East Asian haze, seen a week ahead (North China Plain -> Korea).
a, the target: satellite-derived PM2.5 (GHAP, 1 km) on the peak day, with the monitoring stations (circles).
b, the peak day forecast 7, 5, 3 and 1 days ahead: Foehn ensemble mean (1 km), Foehn probability of exceeding 75 ug m-3
   (16 members), CAMS and Aurora (40 km) 3 and 1 days ahead.
c, d, daily PM2.5 of January 2022 forecast 3 days ahead, mean over the monitors of Beijing-Tianjin-Hebei and Seoul:
   stations (dotted), GHAP at the stations (black), Foehn members (5-95 % and 25-75 % bands, mean), CAMS, Aurora.
e, convergence of the peak-day Beijing-Tianjin-Hebei mean with lead time: Foehn members (boxes), CAMS, Aurora, observed.
f, warnings of PM2.5 > 75 ug m-3 at 1 km, population-weighted over the domain (January 2022, same days for all systems):
   critical success index by lead time (Foehn: probability >= 0.5).
    python fig_case.py <run>       (data: figdata_case.py)"""
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch

from nature_style import DOUBLE, PM_BOUNDS, STYLE, letter, plt, pm_cmap, ryb, save

RUN = sys.argv[1] if len(sys.argv) > 1 else "fp_flow_shift25_global_L7_fix"
Z = np.load(f"data/case_eastasia_{RUN}.npz")
peak = int(Z["peak"])
day = (np.datetime64("2022-01-01") + np.timedelta64(peak, "D")).astype("datetime64[D]").astype(object)
n, s, w, e = Z["box"]
PC = ccrs.PlateCarree()
cmap, norm = pm_cmap()
PB = [0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
pcmap = ryb(len(PB) - 1); pcmap.set_bad((0, 0, 0, 0)); pnorm = BoundaryNorm(PB, pcmap.N)
asp = 1.0 / np.cos(np.deg2rad(0.5 * (n + s)))
slat, slon, sobs = Z["st_lat"], Z["st_lon"], Z["st_obs_peak"]
sm = (slat <= n) & (slat >= s) & (slon >= w) & (slon <= e) & np.isfinite(sobs)

fig = plt.figure(figsize=(DOUBLE, 7.0))


def mapax(rect, title=None):
    ax = fig.add_axes(rect, projection=PC)
    ax.set_extent([w, e, s, n], crs=PC); ax.set_aspect(asp)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#eef2f6", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.3, edgecolor="#444444", zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.25, edgecolor="#555555", zorder=4)
    ax.spines["geo"].set_linewidth(0.5)
    if title:
        ax.set_title(title, fontsize=6, pad=2)
    return ax


def coarse(field, lat0, dlat, lon0, dlon):
    lats = lat0 - dlat * np.arange(field.shape[0]); lons = (lon0 + dlon * np.arange(field.shape[1]) + 180.0) % 360.0 - 180.0
    ri = np.where((lats <= n + 0.5) & (lats >= s - 0.5))[0]; o = np.argsort(lons); ls = lons[o]
    ci = o[(ls >= w - 0.5) & (ls <= e + 0.5)]
    return field[np.ix_(ri, ci)], (lons[ci].min() - dlon / 2, lons[ci].max() + dlon / 2, lats[ri].min() - dlat / 2, lats[ri].max() + dlat / 2)


# a: target
ax = mapax([0.03, 0.70, 0.5, 0.27])
ax.imshow(np.ma.masked_invalid(Z["ghap_peak"].astype(np.float32)), origin="upper", extent=(w, e, s, n), transform=PC, cmap=cmap,
          norm=norm, interpolation="nearest", zorder=1)
ax.scatter(slon[sm], slat[sm], c=sobs[sm], cmap=cmap, norm=norm, s=5, linewidths=0.3, edgecolors="#111111", transform=PC, zorder=5)
for (b0, b1, b2, b3), lab in (((36.0, 41.0, 114.0, 119.5), "Beijing-Tianjin-Hebei"), ((37.0, 38.0, 126.5, 127.5), "Seoul")):
    ax.plot([b2, b3, b3, b2, b2], [b0, b0, b1, b1, b0], color="#111111", lw=0.7, transform=PC, zorder=6)
    ax.text(b2, b1 + 0.15, lab, fontsize=5.5, transform=PC, zorder=7, va="bottom")
ax.set_title(f"Target: satellite-derived PM$_{{2.5}}$ (GHAP, 1 km), {day:%-d %B %Y}; circles: monitoring stations", fontsize=6.5, pad=3)
letter(ax, "a", x=-0.02, y=1.02)
cax = fig.add_axes([0.56, 0.86, 0.015, 0.10])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend="max", spacing="uniform", ticks=PM_BOUNDS)
cb.ax.set_yticklabels([f"{t:g}" for t in PM_BOUNDS]); cb.ax.tick_params(labelsize=5, length=1.5, width=0.4, pad=1)
cb.outline.set_linewidth(0.4); cb.set_label("PM$_{2.5}$ (µg m$^{-3}$)", fontsize=5.5, labelpad=1)
cax = fig.add_axes([0.56, 0.71, 0.015, 0.10])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=pnorm, cmap=pcmap), cax=cax, spacing="uniform", ticks=PB)
cb.ax.set_yticklabels([f"{t:g}" for t in PB]); cb.ax.tick_params(labelsize=5, length=1.5, width=0.4, pad=1)
cb.outline.set_linewidth(0.4); cb.set_label("P(PM$_{2.5}$ > 75)", fontsize=5.5, labelpad=1)

# f: warnings CSI by lead (right of a)
ax = fig.add_axes([0.70, 0.735, 0.27, 0.22])
W = Z["warn"]
csi = lambda x: x[..., 0] / np.maximum(x.sum(-1), 1)
leads = np.arange(1, 8)
for k, (key, lab) in enumerate((("ours", "Foehn (P ≥ 0.5)"), ("cams", "CAMS"), ("aurora", "Aurora"))):
    v = csi(W[:, k]); ok = W[:, k].sum(-1) > 0
    st = STYLE[key]
    ax.plot(leads[ok], v[ok], color=st["color"], lw=st["lw"], marker=st["marker"], ms=2.6, label=lab, markeredgecolor="white", markeredgewidth=0.3)
ax.set_xlabel("Lead time (days)"); ax.set_ylabel("Critical success index"); ax.set_xticks(leads); ax.set_ylim(0, 0.8)
ax.set_title("Warnings > 75 µg m$^{-3}$, population-weighted", fontsize=6, pad=2)
ax.legend(fontsize=5.2, loc="upper right")
letter(ax, "f", x=-0.22, y=1.02)

# b: the peak day by lead
cols = (7, 5, 3, 1)
x0, wd, hd = 0.03, 0.235, 0.135
for j, h in enumerate(cols):
    ax = mapax([x0 + j * 0.243, 0.535, wd, hd], f"Foehn, {h} day{'s' if h > 1 else ''} ahead")
    ax.imshow(np.ma.masked_invalid(Z[f"foehn_mean_{h}"].astype(np.float32)), origin="upper", extent=(w, e, s, n), transform=PC,
              cmap=cmap, norm=norm, interpolation="nearest", zorder=1)
    if j == 0:
        letter(ax, "b", x=-0.03, y=1.05)
        ax.text(-0.03, 0.5, "ensemble mean", transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=5.5)
    ax = mapax([x0 + j * 0.243, 0.405, wd, hd])
    ax.imshow(np.ma.masked_invalid(Z[f"foehn_p75_{h}"].astype(np.float32)), origin="upper", extent=(w, e, s, n), transform=PC,
              cmap=pcmap, norm=pnorm, interpolation="nearest", zorder=1)
    if j == 0:
        ax.text(-0.03, 0.5, "P(> 75 µg m$^{-3}$)", transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=5.5)
al, alo = Z["aurora_lat"], Z["aurora_lon"]
for j, (sysn, h) in enumerate((("CAMS", 3), ("CAMS", 1), ("Aurora", 3), ("Aurora", 1))):
    ax = mapax([x0 + j * 0.243, 0.26, wd, hd], f"{sysn} (40 km), {h} day{'s' if h > 1 else ''} ahead")
    img, ext = (coarse(Z[f"cams_{h}"], 90.0, 0.4, 0.0, 0.4) if sysn == "CAMS" else
                coarse(Z[f"aurora_{h}"], float(al[0]), float(al[0] - al[1]), float(alo[0]), float(alo[1] - alo[0])))
    ax.imshow(np.ma.masked_invalid(img), origin="upper", extent=ext, transform=PC, cmap=cmap, norm=norm, interpolation="nearest", zorder=1)

# c, d: series of January at lead 3
dates = lambda t: [(np.datetime64("2022-01-01") + np.timedelta64(int(v), "D")).astype("datetime64[D]").astype(object) for v in t]
for k, (zk, title) in enumerate((("bth", "Beijing-Tianjin-Hebei"), ("seoul", "Seoul"))):
    R = Z[f"{zk}_series_3"]
    t = dates(R[:, 0]); M = R[:, 5:]
    ax = fig.add_axes([0.065 + k * 0.33, 0.045, 0.27, 0.15])
    ax.fill_between(t, np.percentile(M, 5, 1), np.percentile(M, 95, 1), color=STYLE["member"]["color"], alpha=0.35, lw=0)
    ax.fill_between(t, np.percentile(M, 25, 1), np.percentile(M, 75, 1), color=STYLE["member"]["color"], alpha=0.6, lw=0)
    ax.plot(t, R[:, 3], color=STYLE["cams"]["color"], lw=0.6)
    ax.plot(t, R[:, 4], color=STYLE["aurora"]["color"], lw=0.6)
    ax.plot(t, R[:, 2], color="#111111", lw=0.7)
    ax.plot(t, M.mean(1), color=STYLE["ours"]["color"], lw=0.9)
    ax.plot(t, R[:, 1], color="#111111", lw=0.9, ls=(0, (1, 1.2)))
    ax.axvline(day, color="#999999", lw=0.4)
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 8, 15, 22, 29])); ax.set_xlim(t[0], t[-1]); ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    ax.tick_params(axis="x", labelsize=5.5)
    ax.set_title(f"{title}, 3 days ahead ({int(Z[f'{zk}_nst'])} stations)", fontsize=6, pad=2)
    ax.set_ylabel("PM$_{2.5}$ (µg m$^{-3}$)") if k == 0 else None
    ax.set_ylim(0, None)
    letter(ax, "cd"[k], x=-0.16, y=1.02)
handles = [plt.Line2D([], [], color="#111111", lw=0.9, ls=(0, (1, 1.2))), plt.Line2D([], [], color="#111111", lw=0.7),
           plt.Line2D([], [], color=STYLE["ours"]["color"], lw=0.9), Patch(color=STYLE["member"]["color"], alpha=0.5, lw=0),
           plt.Line2D([], [], color=STYLE["cams"]["color"], lw=0.6), plt.Line2D([], [], color=STYLE["aurora"]["color"], lw=0.6)]
fig.legend(handles, ["Stations", "GHAP at the stations", "Foehn, mean", "Foehn, 25-75 / 5-95 %", "CAMS", "Aurora"], loc="lower left",
           bbox_to_anchor=(0.03, 0.205), ncol=6, fontsize=5.4, handlelength=1.6, columnspacing=1.0)
# e: convergence of the peak with lead time
ax = fig.add_axes([0.73, 0.045, 0.25, 0.15])
C = Z["convergence"]
bp = ax.boxplot([C[h - 1, 3:] for h in leads], positions=leads, widths=0.5, whis=(5, 95), showfliers=False, patch_artist=True,
                medianprops=dict(color=STYLE["ours"]["color"], lw=0.9), boxprops=dict(facecolor=STYLE["member"]["color"], lw=0.5),
                whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5))
ax.axhline(np.nanmean(C[:, 0]), color="#111111", lw=0.9, ls=(0, (1, 1.2)))
ax.plot(leads, C[:, 1], "s", color=STYLE["cams"]["color"], ms=2.6)
ax.plot(leads, C[:, 2], "o", color=STYLE["aurora"]["color"], ms=2.6)
ax.set_xlabel("Lead time (days)"); ax.set_ylabel("Peak-day PM$_{2.5}$ (µg m$^{-3}$)"); ax.set_xticks(leads)
ax.set_title(f"Beijing-Tianjin-Hebei, {day:%-d %b}: forecast vs lead", fontsize=6, pad=2)
letter(ax, "e", x=-0.2, y=1.02)
save(fig, f"fig_case_eastasia_{RUN}")
