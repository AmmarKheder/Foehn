#!/usr/bin/env python3
"""Fig. 1 (global teaser, after the Europe teaser): following the air worldwide.
a, Foehn forecast at 1 km of the target day (issued H days ahead) on the globe, with the back-trajectories that its
   transport layer integrates through the HRES forecast wind for air reaching cities (dots every 12 h); four zones around
   the globe at 1 km with their trajectories and the observations of the monitoring stations (circles, same colours).
b, the hero zone in the forecasts of CAMS and Aurora (40 km), Foehn (1 km) and the satellite-derived target (GHAP).
c, daily PM2.5 of 2022 in the four zones, H days ahead (mean over the zone's monitors): stations (dotted), GHAP at the
   station pixels (black), Foehn ensemble mean and range of the members, CAMS and Aurora.
    python fig1_teaser.py [run] [D0] [H]          (data: figdata_teaser.py)"""
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import ConnectionPatch, Patch, Rectangle

from nature_style import DOUBLE, PM_BOUNDS, STYLE, letter, plt, pm_cmap, save, world_axes

RUN = sys.argv[1] if len(sys.argv) > 1 else "fp_flow_shift25_global_L7_fix"
D0 = int(sys.argv[2]) if len(sys.argv) > 2 else 19
H = int(sys.argv[3]) if len(sys.argv) > 3 else 2
Z = np.load(f"data/teaser_{RUN}_d{D0}_h{H}.npz", allow_pickle=True)
SIDE = {"left": ["po", "scl"], "right": ["ncp", "igp"]}                # zones around the globe (top, bottom)
SERIES = ["po", "ncp", "igp", "scl"]
HERO = str(Z["hero"])
day = (np.datetime64("2022-01-01") + np.timedelta64(D0 + H, "D")).astype("datetime64[D]").astype(object)
cmap, norm = pm_cmap()
PC = ccrs.PlateCarree()
names, paths = [str(n) for n in Z["names"]], Z["paths"]
TRAJ = dict(color="white", lw=0.45, zorder=6, path_effects=[pe.Stroke(linewidth=1.0, foreground="#111111"), pe.Normal()],
            solid_capstyle="round")

fig = plt.figure(figsize=(DOUBLE, 6.3))


WORLD = ["Los Angeles", "Mexico City", "Santiago", "São Paulo", "Lagos", "Cairo", "Paris", "Milan", "Tehran", "Lahore", "Delhi",
         "Dhaka", "Bangkok", "Beijing", "Shanghai", "Seoul", "Jakarta", "Johannesburg"]


def draw_paths(ax, box=None, lw_scale=1.0, label=False, ms=2.4, dots=True):
    for n, name in enumerate(names):
        p = paths[n].copy()
        la0, lo0 = p[0, 1], p[0, 0]
        if box is None and name not in WORLD:
            continue
        if box is not None and not (box[1] <= la0 <= box[0] and box[2] <= lo0 <= box[3]):
            continue
        p[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(p[:, 0])))           # continuous across the date line
        ax.plot(p[:, 0], p[:, 1], transform=PC, **TRAJ)
        if dots:
            ax.plot(p[4::4, 0], p[4::4, 1], "o", ms=1.0 * lw_scale, color="#111111", transform=PC, zorder=7)
        ax.plot(lo0, la0, "o", ms=ms, mfc="white", mec="#111111", mew=0.5, transform=PC, zorder=8)
        if label:
            ax.annotate(name, (lo0, la0), xycoords=PC._as_mpl_transform(ax), xytext=(2.5, 2), textcoords="offset points",
                        fontsize=5, color="#111111", zorder=9, path_effects=[pe.withStroke(linewidth=1.2, foreground="white")])


# ---------------------------------------------------------------- a: globe
axw = world_axes(fig, [0.215, 0.595, 0.57, 0.37], bg=False)
axw.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#eef2f6", edgecolor="none", zorder=0)
axw.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#e6e6e6", edgecolor="none", zorder=0)
F4 = Z["F4"].astype(np.float32)
axw.imshow(np.ma.masked_invalid(F4), origin="upper", extent=(-180, 180, -60, 84), transform=PC, cmap=cmap, norm=norm,
           regrid_shape=(1400, 2800), zorder=1, interpolation="nearest")
draw_paths(axw, ms=1.8, dots=False)
axw.set_title(f"Foehn forecast of {day:%-d %B %Y} at 1 km, issued {H} days ahead, and the air it follows", fontsize=7, pad=3)
letter(axw, "a", x=-0.36, y=1.02)
cax = fig.add_axes([0.33, 0.578, 0.34, 0.009])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="max",
                  spacing="uniform", ticks=PM_BOUNDS)
cb.ax.set_xticklabels([f"{t:g}" for t in PM_BOUNDS]); cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.4, pad=1)
cb.outline.set_linewidth(0.4); cb.set_label("PM$_{2.5}$ (µg m$^{-3}$)", fontsize=6, labelpad=1)

lat_s, lon_s, obs_s = Z["st_lat"], Z["st_lon"], Z["st_obs_day"]


def zone_axes(rect, key):
    n, s, w, e = Z[f"z_{key}_box"]
    ax = fig.add_axes(rect, projection=PC)
    ax.set_extent([w, e, s, n], crs=PC)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(0.5 * (n + s))))
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#eef2f6", edgecolor="none", zorder=0)
    return ax, (n, s, w, e)


for side, keys in SIDE.items():
    for k, key in enumerate(keys):
        x0 = 0.004 if side == "left" else 0.801
        ax, (n, s, w, e) = zone_axes([x0, 0.785 - k * 0.205, 0.195, 0.175], key)
        ax.imshow(np.ma.masked_invalid(Z[f"z_{key}_foehn"].astype(np.float32)), origin="upper", extent=(w, e, s, n),
                  transform=PC, cmap=cmap, norm=norm, interpolation="nearest", zorder=1)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.25, edgecolor="#555555", zorder=3)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.3, edgecolor="#444444", zorder=3)
        m = (lat_s <= n) & (lat_s >= s) & (lon_s >= w) & (lon_s <= e) & np.isfinite(obs_s)
        ax.scatter(lon_s[m], lat_s[m], c=obs_s[m], cmap=cmap, norm=norm, s=4, linewidths=0.3, edgecolors="#111111",
                   transform=PC, zorder=5)
        draw_paths(ax, box=(n, s, w, e), label=True)
        ax.set_title(str(Z[f"z_{key}_title"]), fontsize=6, pad=2)
        ax.spines["geo"].set_linewidth(0.5)
        km = 100 if (e - w) < 8 else 200                                 # scale bar
        dlo = km / (111.2 * np.cos(np.deg2rad(s + 0.08 * (n - s))))
        xb, yb = w + 0.05 * (e - w), s + 0.07 * (n - s)
        ax.plot([xb, xb + dlo], [yb, yb], color="#111111", lw=1.0, transform=PC, zorder=9, solid_capstyle="butt")
        ax.text(xb + dlo / 2, yb + 0.03 * (n - s), f"{km} km", ha="center", va="bottom", fontsize=5, transform=PC, zorder=9,
                path_effects=[pe.withStroke(linewidth=1.2, foreground="white")])
        axw.add_patch(Rectangle((w, s), e - w, n - s, transform=PC, fill=False, lw=0.6, edgecolor="#111111", zorder=9))
        corner = (w, 0.5 * (n + s)) if side == "left" else (e, 0.5 * (n + s))
        xy = axw.projection.transform_point(*corner, PC)
        con = ConnectionPatch(xyA=xy, coordsA=axw.transData, xyB=(1.0 if side == "left" else 0.0, 0.5), coordsB=ax.transAxes,
                              color="#555555", lw=0.4, zorder=0)
        fig.add_artist(con)

# ---------------------------------------------------------------- b: hero zone in the four systems
n, s, w, e = Z[f"z_{HERO}_box"]
lat04 = 90.0 - 0.4 * np.arange(451); lon04 = 0.4 * np.arange(900)


def coarse_crop(field, lat0, dlat, lon0, dlon):
    lats = lat0 - dlat * np.arange(field.shape[0]); lons = (lon0 + dlon * np.arange(field.shape[1]) + 180.0) % 360.0 - 180.0
    ri = np.where((lats <= n + 0.4) & (lats >= s - 0.4))[0]; order = np.argsort(lons); lons_s = lons[order]
    ci = order[(lons_s >= w - 0.4) & (lons_s <= e + 0.4)]
    return field[np.ix_(ri, ci)], (lons[ci].min() - dlon / 2, lons[ci].max() + dlon / 2, lats[ri].min() - dlat / 2, lats[ri].max() + dlat / 2)


al, alo = Z["aurora_lat"], Z["aurora_lon"]
strip = [("CAMS (40 km)", *coarse_crop(Z["cams_day"], 90.0, 0.4, 0.0, 0.4)),
         ("Aurora (40 km)", *coarse_crop(Z["aurora_day"], float(al[0]), float(al[0] - al[1]), float(alo[0]), float(alo[1] - alo[0]))),
         ("Foehn (1 km)", Z[f"z_{HERO}_foehn"].astype(np.float32), (w, e, s, n)),
         ("Target: GHAP, satellite-derived (1 km)", Z[f"z_{HERO}_ghap"].astype(np.float32), (w, e, s, n))]
for k, (ttl, img, ext) in enumerate(strip):
    ax = fig.add_axes([0.012 + k * 0.248, 0.315, 0.235, 0.215], projection=PC)
    ax.set_extent([w, e, s, n], crs=PC); ax.set_aspect(1.0 / np.cos(np.deg2rad(0.5 * (n + s))))
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#eef2f6", edgecolor="none", zorder=0)
    ax.imshow(np.ma.masked_invalid(img), origin="upper", extent=ext, transform=PC, cmap=cmap, norm=norm, interpolation="nearest", zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.3, edgecolor="#444444", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.25, edgecolor="#555555", zorder=3)
    m = (lat_s <= n) & (lat_s >= s) & (lon_s >= w) & (lon_s <= e) & np.isfinite(obs_s)
    ax.scatter(lon_s[m], lat_s[m], c=obs_s[m], cmap=cmap, norm=norm, s=3, linewidths=0.25, edgecolors="#111111", transform=PC, zorder=5)
    ax.set_title(ttl, fontsize=6, pad=2); ax.spines["geo"].set_linewidth(0.5)
    if k == 0:
        letter(ax, "b", x=-0.02, y=1.04)
        ax.text(-0.025, 0.5, f"{Z[f'z_{HERO}_title']}, {H} days ahead", transform=ax.transAxes, rotation=90, ha="right",
                va="center", fontsize=6)

# ---------------------------------------------------------------- c: daily series, H days ahead
dates = lambda t: np.array([(np.datetime64("2022-01-01") + np.timedelta64(int(v), "D")).astype("datetime64[D]").astype(object) for v in t])
for k, key in enumerate(SERIES):
    R = Z[f"s_{key}"]
    gap = np.where(np.diff(R[:, 0]) > 3)[0]                           # break the lines over missing days
    R = np.insert(R, gap + 1, np.nan, axis=0); R[gap + 1 + np.arange(len(gap)), 0] = R[gap + np.arange(len(gap)), 0] + 1
    t = dates(R[:, 0])
    ax = fig.add_axes([0.055 + k * 0.248, 0.075, 0.205, 0.19])
    ax.fill_between(t, R[:, 4], R[:, 5], color=STYLE["member"]["color"], alpha=0.45, lw=0, zorder=2)
    ax.plot(t, R[:, 6], color=STYLE["cams"]["color"], lw=0.5, zorder=3)
    ax.plot(t, R[:, 7], color=STYLE["aurora"]["color"], lw=0.5, zorder=3)
    ax.plot(t, R[:, 2], color="#111111", lw=0.6, zorder=4)
    ax.plot(t, R[:, 3], color=STYLE["ours"]["color"], lw=0.8, zorder=5)
    ax.plot(t, R[:, 1], color="#111111", lw=0.8, ls=(0, (1, 1.2)), zorder=6)
    ax.axvline(day, color="#888888", lw=0.4, ls="-", zorder=1)
    okr = lambda c: np.isfinite(R[:, c]) & np.isfinite(R[:, 1])
    r = lambda c: np.corrcoef(R[okr(c), c], R[okr(c), 1])[0, 1]
    ax.text(0.5, 1.005, f"r: Foehn {r(3):.2f} · CAMS {r(6):.2f} · Aurora {r(7):.2f} · GHAP {r(2):.2f}",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=5, color="#333333")
    ticks = [np.datetime64(f"2022-{m:02d}-01").astype(object) for m in range(1, 13)]
    ax.set_xticks(ticks); ax.set_xticklabels([d.strftime("%b")[0] for d in ticks], fontsize=5.5)
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(0, None)
    ax.set_title(f"{Z[f'z_{key}_title']} ({int(np.nanmedian(R[:, 8]))} stations)", fontsize=6, pad=8)
    if k == 0:
        ax.set_ylabel(f"PM$_{{2.5}}$, {H} days ahead (µg m$^{{-3}}$)", fontsize=6)
        letter(ax, "c", x=-0.2, y=1.04)
handles = [plt.Line2D([], [], color="#111111", lw=0.8, ls=(0, (1, 1.2))), plt.Line2D([], [], color="#111111", lw=0.6),
           plt.Line2D([], [], color=STYLE["ours"]["color"], lw=0.8), Patch(color=STYLE["member"]["color"], alpha=0.45, lw=0),
           plt.Line2D([], [], color=STYLE["cams"]["color"], lw=0.5), plt.Line2D([], [], color=STYLE["aurora"]["color"], lw=0.5)]
fig.legend(handles, ["Stations (observed)", "GHAP at the stations (target)", "Foehn, ensemble mean", "Foehn, range of the members",
                     "CAMS (40 km)", "Aurora (40 km)"], loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.005), fontsize=5.8,
           handlelength=1.8, columnspacing=1.2)
save(fig, f"fig1_teaser_{RUN}_d{D0}_h{H}")
