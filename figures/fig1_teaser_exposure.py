#!/usr/bin/env python3
"""Fig. 1 (teaser): what a 40 km grid cannot see. Data only (GHAP 1 km daily PM2.5 of 2022 x GHS-POP 2020 residents).
a, Robinson map: residents per day, on average over 2022, who breathe air above the WHO 24-h interim target at 1 km but
live in a 40 km cell (the CAMS / Aurora grid) whose mean is below it (figdata_exposure.py), per 40 km cell.
b-g, six cities, one per region (fixed rule, teaser_cities.py), on their day with the most such residents: daily PM2.5 at
1 km (left) and the same field averaged on the 40 km cells (right).
    python fig1_teaser_exposure.py"""
import json

import cartopy.crs as ccrs
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm
from matplotlib.patches import ConnectionPatch

from fig_exposure_zoom import ghap_box, pop_box
from nature_style import DOUBLE, HERE, MM, letter, plt, save, world_axes

HID = "#C2410C"
# call-out slot (x0, y0) per city, each near its continent so that the leader lines stay short
SLOTS = {"Cairo": (0.01, 0.725), "Dhaka": (0.345, 0.725), "Seoul": (0.68, 0.725),
         "Lima": (0.01, 0.012), "Kinshasa": (0.345, 0.012), "Jakarta": (0.68, 0.012)}
W_, H_ = 0.155, 0.158                                                        # one sub-panel (figure fraction)


def city(c):
    N, S, W, E = c["box"]; d = c["doy"]
    r0, r1, c0, c1 = int(round((84 - N) * 100)), int(round((84 - S) * 100)), int(round((W + 180) * 100)), int(round((E + 180) * 100))
    x = ghap_box(1826 + d, r0, r1, c0, c1); pop = pop_box(r0, r1, c0, c1)
    ok = x > 0; nr, nc = (r1 - r0) // 40, (c1 - c0) // 40
    with np.errstate(all="ignore"):
        cell = np.where(ok, x, 0).reshape(nr, 40, nc, 40).sum((1, 3)) / ok.reshape(nr, 40, nc, 40).sum((1, 3))
    x40 = np.repeat(np.repeat(cell, 40, 0), 40, 1)
    w = np.where(ok & np.isfinite(x40), pop, 0.0)
    hidden = (w * ((x > THR) & ~(x40 > THR))).sum()
    return x, x40, ok, (W, E, S, N), hidden


def subpanel(ax, field, ok, ext, cmap, norm, title, grid_color):
    ax.imshow(np.where(ok, field, np.nan), extent=ext, origin="upper", cmap=cmap, norm=norm, interpolation="nearest")
    nc, nr = int(round((ext[1] - ext[0]) / 0.4)), int(round((ext[3] - ext[2]) / 0.4))
    for k in range(1, nc):
        ax.axvline(ext[0] + 0.4 * k, color=grid_color, lw=0.3, alpha=0.8)
    for k in range(1, nr):
        ax.axhline(ext[3] - 0.4 * k, color=grid_color, lw=0.3, alpha=0.8)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_aspect(1 / np.cos(np.deg2rad(0.5 * (ext[2] + ext[3]))))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("#E9EEF2")
    for s_ in ax.spines.values():
        s_.set_visible(True); s_.set_linewidth(0.5); s_.set_edgecolor("#333333")
    ax.text(0.04, 0.95, title, transform=ax.transAxes, fontsize=6, fontweight="bold", va="top", color="#111111",
            path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])


def main():
    global THR
    meta = json.load(open(HERE / "data" / "teaser_cities.json"))
    THR = float(meta["threshold"]); C = {c["city"]: c for c in meta["cities"]}
    ez = np.load(HERE / "data" / "exposure_2022.npz")
    kt = list(ez["thr"]).index(THR); R = [str(r) for r in ez["regions"]]
    per_day = ez["maps"][1, kt] / len(ez["days"])                                   # residents hidden, per day on average
    tot_hid = ez["res"][:, R.index("World"), 2, kt].sum() / len(ez["days"])
    tot_exp = ez["res"][:, R.index("World"), 0, kt].sum() / len(ez["days"])
    fig = plt.figure(figsize=(DOUBLE, 182 * MM))
    fig.text(0.5, 0.993, f"Every day of 2022, {tot_hid / 1e6:.0f} million people breathed air above the WHO interim target",
             ha="center", va="top", fontsize=9, fontweight="bold", color="#111111")
    fig.text(0.5, 0.972, "that no alert issued on a 40 km grid could reach — even with a perfect forecast", ha="center", va="top",
             fontsize=8, color=HID, fontweight="bold")
    # a: the world map
    ax = world_axes(fig, [0.0, 0.245, 1.0, 0.43])
    lat_e = 84.0 - 0.4 * np.arange(361); lon_e = -180.0 + 0.4 * np.arange(901)
    cmap_w = LinearSegmentedColormap.from_list("hid", ["#FFE8CC", "#FDAE61", "#F46D43", "#C2185B", "#5B0F3C"], N=256)
    pm = ax.pcolormesh(lon_e, lat_e, np.ma.masked_less(per_day, 30), cmap=cmap_w, norm=LogNorm(30, 3e5), transform=ccrs.PlateCarree(),
                       rasterized=True, zorder=3)
    letter(ax, "a", x=0.04, y=0.62)
    # colour bars below the map
    cax = fig.add_axes([0.015, 0.655, 0.22, 0.008])
    cb = fig.colorbar(pm, cax=cax, orientation="horizontal", extend="both")
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.5, length=2, top=True, labeltop=True, bottom=False, labelbottom=False)
    cb.ax.xaxis.set_label_position("top")
    cb.set_label(f"a: people per day above {THR:g} µg m$^{{-3}}$\nwith no alert at 40 km, per 40 km cell", fontsize=5.8)
    cmap = LinearSegmentedColormap.from_list("pmthr", [(0.0, "#FFF6DD"), (0.4999, "#F59E3D"), (0.5, "#D7301F"), (1.0, "#3B0A24")], N=512)
    norm = TwoSlopeNorm(vmin=5, vcenter=THR, vmax=150)
    cax2 = fig.add_axes([0.825, 0.655, 0.165, 0.008])
    cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax2, orientation="horizontal", extend="max")
    cb2.set_ticks([5, 15, THR, 75, 150]); cb2.set_ticklabels(["5", "15", f"{THR:g}", "75", "150"])
    cb2.outline.set_linewidth(0.4); cb2.ax.tick_params(labelsize=5.5, length=2, top=True, labeltop=True, bottom=False, labelbottom=False)
    cb2.ax.xaxis.set_label_position("top")
    cb2.set_label(f"b–g: daily PM$_{{2.5}}$ (µg m$^{{-3}}$)\nred: above the WHO 24-h interim target, alert", fontsize=5.8)
    # b-g: the cities
    for k, (name, (x0, y0)) in enumerate(SLOTS.items()):
        c = C[name]
        x, x40, ok, ext, hidden = city(c)
        a1 = fig.add_axes([x0, y0, W_, H_]); a2 = fig.add_axes([x0 + W_ + 0.006, y0, W_, H_])
        subpanel(a1, x, ok, ext, cmap, norm, "1 km", "white")
        subpanel(a2, x40, ok, ext, cmap, norm, "40 km", "white")
        date = (np.datetime64("2022-01-01") + np.timedelta64(int(c["doy"]), "D")).astype(object)
        top = y0 + H_ + 0.005
        fig.text(x0, top + 0.019, f"$\\bf{{{'bcdefg'[k]}}}$  {name}, {date.day} {date.strftime('%B')}", fontsize=6.8,
                 fontweight="bold", va="bottom")
        fig.text(x0 + 0.017, top, f"{hidden / 1e6:.1f} million people above the target, no alert at 40 km", fontsize=6,
                 fontweight="bold", color=HID, va="bottom")
        lon_c, lat_c = 0.5 * (ext[0] + ext[1]), 0.5 * (ext[2] + ext[3])
        xy = ax.projection.transform_point(lon_c, lat_c, ccrs.PlateCarree())
        # leader line from the corner of the call-out (facing the map) nearest to the city, horizontally
        xc = fig.transFigure.inverted().transform(ax.transData.transform(xy))[0]
        yy = 0.0 if y0 > 0.5 else 1.0
        corners = [(a1, 0.0), (a1, 1.0), (a2, 0.0), (a2, 1.0)]
        a_, xx = min(corners, key=lambda c_: abs(fig.transFigure.inverted().transform(c_[0].transAxes.transform((c_[1], yy)))[0] - xc))
        fig.add_artist(ConnectionPatch(xyA=(xx, yy), coordsA=a_.transAxes, xyB=xy, coordsB=ax.transData, color="#666666", lw=0.4,
                                       zorder=10, shrinkB=2.5))
        ax.plot(lon_c, lat_c, "o", ms=3.2, mfc="white", mec="#222222", mew=0.6, transform=ccrs.PlateCarree(), zorder=11)
    save(fig, "fig1_teaser_exposure")
    print(f"world: {tot_hid / 1e6:.0f} M residents per day hidden of {tot_exp / 1e6:.0f} M above {THR:g} ({100 * tot_hid / tot_exp:.0f} %)")


if __name__ == "__main__":
    main()
