#!/usr/bin/env python3
"""Fig. 1 (teaser, v6). a, one Foehn forecast of the whole Earth at 1 km: 21 January 2022, issued on 20 January (ensemble
mean; eval_global_green.py --doys 19 --leads 1 --save-1km 19). The day follows the rule of fig1_overview.py, fixed in
advance: the day of 2022 with the highest population-weighted PM2.5 (GHAP).
b-d, Beijing-Tianjin that day, data only (fig_exposure_row1.py): GHAP at 1 km; the same field on 40 km cells, the size of
the CAMS and Aurora cells, i.e. what a perfect 40 km forecast would show; the residents above the WHO 24-h interim target
(75 ug m-3) seen or hidden by the 40 km grid. No station dots: on that day the Beijing stations read 50-65 ug m-3 where
GHAP has > 75 (a GHAP error of the day; the stations evaluate the forecasts in Fig. 2). No figure title.
    python fig1_teaser_v6.py"""
import os
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as pe
import numpy as np

import fig_exposure_row1 as R1
from nature_style import DOUBLE, MM, PM_BOUNDS, plt, pm_cmap, save

G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
EV = Path("/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km/runs/"
          "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12/eval_jan21")
D0, H = 19, 1                                                                # issued 20 January for 21 January 2022


def tiles(name, k):
    with open(G / name, "rb") as f:
        np.lib.format.read_magic(f); np.lib.format.read_array_header_1_0(f); hdr = f.tell()
    n = 29 * 71 * 512 * 512
    fd = os.open(G / name, os.O_RDONLY)
    a = np.frombuffer(os.pread(fd, n * 2, hdr + k * n * 2), np.float16).reshape(29, 71, 512, 512)
    os.close(fd)
    return a.transpose(0, 2, 1, 3).reshape(29 * 512, 71 * 512)[:14400, :36000]


def main():
    cmap, norm = pm_cmap()
    W, Hf = DOUBLE, 150 * MM
    fig = plt.figure(figsize=(W, Hf))
    # a: the Earth at 1 km
    ax = fig.add_axes([0.0, 0.385, 1.0, 0.59], projection=ccrs.Robinson())
    ax.set_extent([-180, 180, -58, 84], crs=ccrs.PlateCarree())
    ax.spines["geo"].set_visible(False)                                        # white background, no frame
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#ECECEC", edgecolor="none", zorder=0)
    m = np.load(EV / f"map1km_d{D0}_h{H}.npy").astype(np.float32)
    m[~(tiles("static1km_tiles.npy", 1) > 0)] = np.nan
    ax.imshow(m, origin="upper", extent=(-180, 180, -60, 84), transform=ccrs.PlateCarree(), cmap=cmap, norm=norm,
              regrid_shape=4400, interpolation="nearest", zorder=1)
    del m
    la, lo = 0.5 * (R1.N + R1.S), 0.5 * (R1.W + R1.E)
    ax.add_patch(plt.Rectangle((lo - 4, la - 3.5), 8, 7, fill=False, ec="#111111", lw=0.8, zorder=5, transform=ccrs.PlateCarree()))
    ax.text(lo + 5.0, la + 2.5, "b–d", fontsize=6.5, fontweight="bold", zorder=6, transform=ccrs.PlateCarree(),
            path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
    fig.text(0.004, 0.993, "a", fontsize=8, fontweight="bold", va="top")
    fig.text(0.028, 0.993, "Foehn forecast for 21 January 2022, issued on 20 January: 1 km over the whole Earth", fontsize=7, va="top")
    cax = ax.inset_axes([0.02, 0.15, 0.27, 0.03])                            # over the South Pacific, label above
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="max",
                      spacing="uniform", ticks=PM_BOUNDS)
    cb.ax.set_xticklabels([f"{t:g}" for t in PM_BOUNDS]); cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.4, pad=1)
    cb.outline.set_linewidth(0.4); cb.ax.xaxis.set_label_position("top")
    cb.set_label("Daily PM$_{2.5}$ (µg m$^{-3}$)", fontsize=6, labelpad=2)
    # b-d: Beijing-Tianjin that day (GHAP, residents), with the stations
    gs = fig.add_gridspec(1, 3, left=0.06, right=0.99, top=0.37, bottom=0.09, wspace=0.12)
    axs, im, n_seen, n_hid = R1.row1(fig, gs, letters="bcd")
    axs[0].set_title("Daily PM$_{2.5}$ at 1 km (GHAP)", fontsize=7, loc="left")
    axs[1].text(0.97, 0.97, "= a perfect 40 km forecast", transform=axs[1].transAxes, ha="right", va="top", fontsize=5.5,
                bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.9), zorder=7)
    fig.canvas.draw()
    p0, p1 = axs[0].get_position(), axs[1].get_position()
    cax = fig.add_axes([p0.x0 + 0.02, 0.035, p1.x1 - p0.x0 - 0.04, 0.011])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both", ticks=[45, 55, 65, 75, 85, 95, 105])
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.5, length=2)
    cb.ax.axvline(R1.THR, color="k", lw=0.8)
    axs[0].legend(handles=[plt.Line2D([], [], color="k", lw=0.8, label="WHO interim target")],
                  loc="lower left", fontsize=5.5, frameon=True, framealpha=0.9, edgecolor="none", borderpad=0.4, handlelength=1.5)
    cb.set_label("Daily PM$_{2.5}$ on 21 January 2022 (µg m$^{-3}$)", fontsize=6)
    save(fig, "fig1_teaser_v6")
    print(f"seen {n_seen:.2f} M, hidden {n_hid:.2f} M")


if __name__ == "__main__":
    main()
