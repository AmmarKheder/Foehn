#!/usr/bin/env python3
"""Fig. 1 (teaser, v5): a, Foehn over the whole Earth: annual mean 2022 of its forecasts for the next day at 1 km (ensemble
mean; figdata_teaser_fc.py, shown on 0.1 deg means), Robinson, white background; the frame marks b-d.
b-d, why the kilometre matters (fig_exposure_row1.py, data only): Beijing-Tianjin on 21 January 2022, daily PM2.5 at 1 km
(GHAP), the same field on 40 km cells, and the residents above the WHO 24-h interim target seen or hidden by the 40 km
grid. No figure title.
    python fig1_teaser_v5.py      (until the 1 km annual run exists: the 0.25 deg block means of eval_final_e02, every
                                   second start day, are used for a and the script says so)"""
import glob
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

import fig_exposure_row1 as R1
from nature_style import DOUBLE, HERE, MM, plt, save

PM_CMAP = LinearSegmentedColormap.from_list("ghap", ["#313695", "#4575B4", "#74ADD1", "#ABD9E9", "#E0F3F8", "#FFFFBF", "#FEE090",
                                                     "#FDAE61", "#F46D43", "#D73027", "#A50026"], N=256)
PM_B = [0, 5, 10, 15, 25, 35, 50, 75, 100]                                   # WHO 2021 annual guideline 5, interim targets 10-35
PM_NORM = BoundaryNorm(PM_B, PM_CMAP.N, extend="max")


def foehn_annual():
    """(field, extent, label): the 1 km annual mean on 0.1 deg if computed, else the 0.25 deg means of eval_final_e02 (lead 1)"""
    from pathlib import Path
    w = HERE / "data" / "fc_annual_2022" / "world01.npz"
    if w.exists():
        return np.load(w)["foehn_mean"], "1 km"
    ev = Path("/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km/runs/"
              "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12/eval_final_e02")
    s = np.zeros((576, 1440)); n = np.zeros((576, 1440))
    for p in sorted(glob.glob(str(ev / "part_*.npz"))):
        z = np.load(p)
        for k, (d, h, t) in enumerate(z["keys"]):
            if h == 1:
                m = z["map25"][k].astype(np.float32); ok = np.isfinite(m)
                s[ok] += m[ok]; n[ok] += 1
    print("PROVISIONAL a: 0.25 deg block means of eval_final_e02, lead 1, every second start day", file=sys.stderr)
    with np.errstate(all="ignore"):
        return np.where(n > 0, s / n, np.nan), "0.25 deg (provisional)"


def main():
    f, res = foehn_annual()
    PM_CMAP.set_bad((1, 1, 1, 0))
    W, H = DOUBLE, 150 * MM
    fig = plt.figure(figsize=(W, H))
    # a: the world
    ax = fig.add_axes([0.0, 0.43, 1.0, 0.535], projection=ccrs.Robinson())
    ax.set_extent([-180, 180, -58, 80], crs=ccrs.PlateCarree())
    ax.imshow(f, extent=(-180, 180, -60, 84), origin="upper", transform=ccrs.PlateCarree(), cmap=PM_CMAP, norm=PM_NORM,
              interpolation="nearest", regrid_shape=(1800, 3600), zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.25, edgecolor="#333333", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.15, edgecolor="#8A8A8A", zorder=3)
    ax.spines["geo"].set_visible(False)                                        # white background, no frame
    la, lo = 0.5 * (R1.N + R1.S), 0.5 * (R1.W + R1.E)
    ax.add_patch(plt.Rectangle((lo - 4, la - 3.5), 8, 7, fill=False, ec="#111111", lw=0.8, zorder=5, transform=ccrs.PlateCarree()))
    ax.text(lo + 5.5, la + 3.0, "b–d", fontsize=6.5, fontweight="bold", zorder=6, transform=ccrs.PlateCarree(),
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
    fig.text(0.005, 0.995, "a", fontsize=8, fontweight="bold", va="top")
    fig.text(0.03, 0.995, "Foehn forecast for the next day, annual mean 2022, 1 km over the whole Earth"
             + ("" if res == "1 km" else "  [PROVISIONAL: 0.25° means, every second day]"), fontsize=7, va="top")
    cax = ax.inset_axes([0.035, 0.11, 0.27, 0.033])                           # South Pacific, label above
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=PM_NORM, cmap=PM_CMAP), cax=cax, orientation="horizontal", ticks=PM_B, extend="max")
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
    cb.ax.xaxis.set_label_position("top")
    cb.set_label("PM$_{2.5}$ (µg m$^{-3}$); 5: WHO annual guideline, 10–35: interim targets", fontsize=6, labelpad=2)
    # b-d: Beijing, 21 January 2022 (data only)
    gs = fig.add_gridspec(1, 3, left=0.06, right=0.99, top=0.38, bottom=0.075, wspace=0.12)
    axs, im, n_seen, n_hid = R1.row1(fig, gs, letters="bcd")
    fig.canvas.draw()
    p0, p1 = axs[0].get_position(), axs[1].get_position()
    cax = fig.add_axes([p0.x0 + 0.02, 0.028, p1.x1 - p0.x0 - 0.04, 0.010])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both", ticks=[45, 55, 65, 75, 85, 95, 105])
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.5, length=2)
    cb.ax.axvline(R1.THR, color="k", lw=0.8)
    axs[0].legend(handles=[plt.Line2D([], [], color="k", lw=0.8, label="WHO interim target")], loc="lower left",
                  fontsize=5.5, frameon=True, framealpha=0.9, edgecolor="none", borderpad=0.4, handlelength=1.5)
    cb.set_label("Daily PM$_{2.5}$ on 21 January 2022 (µg m$^{-3}$)", fontsize=6)
    save(fig, "fig1_teaser_v5")
    print(f"a: {res} | b-d: seen {n_seen:.2f} M, hidden {n_hid:.2f} M")


if __name__ == "__main__":
    main()
