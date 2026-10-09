#!/usr/bin/env python3
"""Main figure 'What a 40 km grid hides': a Robinson world map of the person-days of 2022 above a WHO 24-h interim target
that 40 km cells (the CAMS / Aurora grid) do not see, per 40 km cell (figdata_exposure.py), with six city call-outs at
1 km on their most telling day of 2022 (scan_exposure_cities.py): PM2.5 at 1 km, the 40 km cells drawn, in vermillion the
cells that hold residents above the target although the cell mean is below it, and how many residents that is.
    python fig_exposure_world.py [threshold 37.5] [city1 ... city6]"""
import sys

import cartopy.crs as ccrs
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm
from matplotlib.patches import ConnectionPatch, Rectangle

from fig_exposure_zoom import ghap_box, pop_box
from nature_style import DOUBLE, HERE, MM, letter, plt, save, world_axes
from scan_exposure_cities import CITIES, box

HID = "#D55E00"
args = [a for a in sys.argv[1:]]
THR = float(args[0]) if args and args[0].replace(".", "").isdigit() else 37.5
PICK = [a for a in args if not a.replace(".", "").isdigit()] or ["Lahore", "Beijing", "Dhaka", "Mexico City", "Lagos", "Cairo"]


def city_panel(ax, name, doy, cmap, norm):
    la, lo = CITIES[name]
    r0, r1, c0, c1 = box(la, lo)
    x = ghap_box(1826 + doy, r0, r1, c0, c1); pop = pop_box(r0, r1, c0, c1)
    ok = x > 0; nr, nc = (r1 - r0) // 40, (c1 - c0) // 40
    with np.errstate(all="ignore"):
        cell = np.where(ok, x, 0).reshape(nr, 40, nc, 40).sum((1, 3)) / ok.reshape(nr, 40, nc, 40).sum((1, 3))
    x40 = np.repeat(np.repeat(cell, 40, 0), 40, 1)
    w = np.where(ok & np.isfinite(x40), pop, 0.0)
    hid_px = (x > THR) & ~(x40 > THR)
    hid = (w * hid_px).sum() / 1e6
    hid_cells = (w * hid_px).reshape(nr, 40, nc, 40).sum((1, 3)) > 1000           # cells hiding > 1,000 exposed residents
    ext = (-180 + c0 * 0.01, -180 + c1 * 0.01, 84 - r1 * 0.01, 84 - r0 * 0.01)
    ax.imshow(np.where(ok, x, np.nan), extent=ext, origin="upper", cmap=cmap, norm=norm, interpolation="nearest")
    for cc in range(1, nc):
        ax.axvline(ext[0] + 0.4 * cc, color="white", lw=0.3, alpha=0.7)
    for rr in range(1, nr):
        ax.axhline(ext[3] - 0.4 * rr, color="white", lw=0.3, alpha=0.7)
    for i in range(nr):
        for j in range(nc):
            if hid_cells[i, j]:
                ax.add_patch(Rectangle((ext[0] + 0.4 * j, ext[3] - 0.4 * (i + 1)), 0.4, 0.4, fill=False, ec=HID, lw=1.1, zorder=4))
    ax.plot(lo, la, "o", ms=2.4, mfc="white", mec="k", mew=0.4, zorder=5)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect(1 / np.cos(np.deg2rad(la)))
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(True); s_.set_linewidth(0.5); s_.set_edgecolor("#333333")
    date = np.datetime64("2022-01-01") + np.timedelta64(int(doy), "D")
    ax.set_title(f"{name}, {date.astype(object).strftime('%-d %b')}", fontsize=6.5, loc="left", pad=2)
    ax.text(0.03, 0.04, f"{hid:.1f} million hidden", transform=ax.transAxes, fontsize=6, color=HID, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
    return hid


def main():
    cz = np.load(HERE / "data" / "exposure_cities.npz")
    names, thr_list, res = list(cz["names"]), list(cz["thr"]), cz["res"]
    kt = thr_list.index(THR)
    ez = np.load(HERE / "data" / "exposure_2022.npz")
    kt2 = list(ez["thr"]).index(THR)
    hidden_map = ez["maps"][1, kt2]                                                  # (360, 900) person-days hidden per cell
    exposed_map = ez["maps"][0, kt2]
    R = [str(r) for r in ez["regions"]]
    tot_exp = ez["res"][:, R.index("World"), 0, kt2].sum(); tot_hid = ez["res"][:, R.index("World"), 2, kt2].sum()
    fig = plt.figure(figsize=(DOUBLE, 150 * MM))
    # --- the world map (middle)
    ax = world_axes(fig, [0.0, 0.27, 1.0, 0.46])
    lat_e = 84.0 - 0.4 * np.arange(361); lon_e = -180.0 + 0.4 * np.arange(901)
    cmap_w = LinearSegmentedColormap.from_list("hid", ["#FDE0C5", "#F7A35C", "#E4572E", "#A3123A", "#4A0B2E"], N=256)
    v = np.where(hidden_map > 1e4, hidden_map, np.nan)
    pm = ax.pcolormesh(lon_e, lat_e, np.ma.masked_invalid(v), cmap=cmap_w, norm=LogNorm(1e4, 1e8), transform=ccrs.PlateCarree(),
                       rasterized=True, zorder=3)
    letter(ax, "a", x=0.02, y=0.98)
    cax = fig.add_axes([0.33, 0.29, 0.34, 0.012])
    cb = fig.colorbar(pm, cax=cax, orientation="horizontal", extend="both")
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.5, length=2)
    cb.set_label(f"Person-days of 2022 above {THR:g} µg m$^{{-3}}$ invisible on 40 km cells, per cell", fontsize=6)
    ax.text(0.5, 1.02, f"{tot_hid / 1e9:.1f} billion of the {tot_exp / 1e9:.1f} billion person-days above the WHO 24-h interim "
            f"target ({100 * tot_hid / tot_exp:.0f} %) are invisible at 40 km", transform=ax.transAxes, ha="center", fontsize=7)
    # --- six call-outs (3 above, 3 below), PM2.5 at 1 km
    cmap = LinearSegmentedColormap.from_list("pmthr", [(0.0, "#FFF6DD"), (0.4999, "#F59E3D"), (0.5, "#D7301F"), (1.0, "#3B0A24")], N=512)
    norm = TwoSlopeNorm(vmin=5, vcenter=THR, vmax=150)
    slots = [(0.01, 0.76), (0.345, 0.76), (0.68, 0.76), (0.01, 0.025), (0.345, 0.025), (0.68, 0.025)]
    for k, (name, (x0, y0)) in enumerate(zip(PICK, slots)):
        cax_ = fig.add_axes([x0, y0, 0.31, 0.21])
        doy = int(res[names.index(name), :, kt, 2].argmax())
        city_panel(cax_, name, doy, cmap, norm)
        letter(cax_, "bcdefg"[k], x=-0.01, y=1.02)
        la, lo = CITIES[name]
        xy = ax.projection.transform_point(lo, la, ccrs.PlateCarree())
        anchor = (0.5, 0.0) if y0 > 0.5 else (0.5, 1.0)
        con = ConnectionPatch(xyA=anchor, coordsA=cax_.transAxes, xyB=xy, coordsB=ax.transData, color="#444444", lw=0.5, zorder=10)
        fig.add_artist(con)
        ax.plot(lo, la, "o", ms=2.5, mfc="white", mec="k", mew=0.5, transform=ccrs.PlateCarree(), zorder=11)
    cax2 = fig.add_axes([0.70, 0.29, 0.25, 0.012])
    cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax2, orientation="horizontal", extend="max",
                       ticks=[5, 15, THR, 75, 150])
    cb2.outline.set_linewidth(0.4); cb2.ax.tick_params(labelsize=5.5, length=2)
    cb2.set_label("Daily PM$_{2.5}$ at 1 km (µg m$^{-3}$), panels b-g", fontsize=6)
    fig.legend(handles=[plt.Line2D([], [], marker="s", ls="", mfc="none", mec=HID, mew=1.1, ms=5,
                                   label="40 km cell below the target that holds residents above it")],
               loc="lower left", bbox_to_anchor=(0.02, 0.255), fontsize=5.5, frameon=False)
    save(fig, f"fig_exposure_world_{THR:g}")


if __name__ == "__main__":
    main()
