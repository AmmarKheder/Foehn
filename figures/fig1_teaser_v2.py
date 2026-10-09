#!/usr/bin/env python3
"""Fig. 1 (teaser, v2): the same world seen three ways. Data only (no model).
a, residents per km2 (GHS-POP 2020); b, annual mean PM2.5 of 2022 on 40 km cells (the CAMS / Aurora grid); c, the same at
1 km (GHAP), Robinson. d-f, three zones on three continents (Nile delta, Bengal delta, Lima), in the same three views,
aligned with the maps; the 40 km cells are drawn in every zoom. No title in the figure (Nature: the title opens the legend).
    python fig1_teaser_v2.py"""
import cartopy.crs as ccrs
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.patches import Rectangle

from fig_exposure_zoom import pop_box
from nature_style import DOUBLE, HERE, MM, letter, plt, save, world_axes

GH, GW = 14400, 36000
ZONES = {"Nile delta": (31.6, 29.2, 29.6, 32.4), "Bengal delta": (25.2, 22.4, 88.8, 91.6), "Lima": (-10.8, -13.6, -78.4, -75.6)}
PM_B = [0, 5, 10, 15, 25, 35, 50, 75, 100]                                   # WHO 2021 annual guideline 5, interim targets 10-35
PM_C = ["#FFF8E1", "#FDE3A7", "#FBC26B", "#F79A3F", "#E8663A", "#C8323B", "#8E1B44", "#4E0D3A"]
POP_C = ["#F7F4F9", "#E2D9EC", "#C3B3D8", "#9E8BC2", "#7A62A8", "#57408E", "#3A2268"]


def blockmean(a, f):
    h, w = a.shape[0] // f * f, a.shape[1] // f * f
    a = a[:h, :w].reshape(h // f, f, w // f, f)
    with np.errstate(all="ignore"):
        return np.nanmean(a, (1, 3))


def main():
    pm1 = np.load(HERE / "data" / "pm_annual_2022.npy", mmap_mode="r")
    pm40 = np.load(HERE / "data" / "pm_annual_2022_40km.npy")
    pop = np.load("/scratch/project_462001140/ammar/eccv/data/world/global1km/pop1km_global.npy", mmap_mode="r")
    pmcmap, pmnorm = ListedColormap(PM_C), BoundaryNorm(PM_B, len(PM_C), extend="max")
    pmcmap.set_bad((0, 0, 0, 0))
    popcmap = ListedColormap(POP_C); popnorm = LogNorm(1, 10000); popcmap.set_bad((0, 0, 0, 0))
    fig = plt.figure(figsize=(DOUBLE, 176 * MM))
    ext_g = (-180, 180, -60, 84)
    rows = [0.665, 0.355, 0.045]                                                 # bottom of the three map rows
    hmap = 0.30
    # global fields for display: population density at 0.1 deg (sum / km2), PM at 40 km cells and at ~2 km (display)
    popd = blockmean(np.asarray(pop[:, :], np.float32), 10)                      # mean residents per 1 km pixel ~ per km2
    pm1d = blockmean(np.asarray(pm1, np.float32), 2)
    fields = [(popd, popcmap, popnorm, "Residents per km$^2$"), (pm40, pmcmap, pmnorm, "Annual PM$_{2.5}$ on 40 km cells"),
              (pm1d, pmcmap, pmnorm, "Annual PM$_{2.5}$ at 1 km")]
    for k, ((f, cm, nm, title), y0) in enumerate(zip(fields, rows)):
        ax = world_axes(fig, [0.0, y0, 0.62, hmap], extent=(-180, 180, -60, 84))
        ax.imshow(np.ma.masked_invalid(np.where(f > (0.5 if k == 0 else 0), f, np.nan)), origin="upper", extent=ext_g,
                  transform=ccrs.PlateCarree(), cmap=cm, norm=nm, interpolation="nearest", regrid_shape=(1400, 3200), zorder=3)
        for zn, (N, S, W, E) in ZONES.items():
            ax.add_patch(Rectangle((W, S), E - W, N - S, fill=False, ec="#111111", lw=0.8, transform=ccrs.PlateCarree(), zorder=6))
        ax.text(0.01, 0.97, title, transform=ax.transAxes, fontsize=6.5, fontweight="bold", va="top")
        letter(ax, "abc"[k], x=0.0, y=1.0)
    # zooms: one column per zone, one row per view, the 40 km cells drawn
    for j, (zn, (N, S, W, E)) in enumerate(ZONES.items()):
        r0, r1 = int(round((84 - N) * 100)), int(round((84 - S) * 100)); c0, c1 = int(round((W + 180) * 100)), int(round((E + 180) * 100))
        p = pop_box(r0, r1, c0, c1)
        x1 = np.asarray(pm1[r0:r1, c0:c1], np.float32)
        x40 = np.repeat(np.repeat(pm40[r0 // 40:r1 // 40, c0 // 40:c1 // 40], 40, 0), 40, 1)
        x40 = np.where(np.isfinite(x1), x40, np.nan)
        for k, (f, cm, nm) in enumerate(((np.where(p > 0.5, p, np.nan), popcmap, popnorm), (x40, pmcmap, pmnorm), (x1, pmcmap, pmnorm))):
            ax = fig.add_axes([0.635 + 0.122 * j, rows[k] + 0.02, 0.115, hmap - 0.04])
            ax.imshow(np.ma.masked_invalid(f), extent=(W, E, S, N), origin="upper", cmap=cm, norm=nm, interpolation="nearest")
            ax.set_facecolor("#E8EEF3")
            for c_ in np.arange(W + 0.4, E - 1e-6, 0.4):
                ax.axvline(c_, color="white" if k else "#BBBBBB", lw=0.3, alpha=0.9)
            for r_ in np.arange(S + 0.4, N - 1e-6, 0.4):
                ax.axhline(r_, color="white" if k else "#BBBBBB", lw=0.3, alpha=0.9)
            ax.set_xlim(W, E); ax.set_ylim(S, N); ax.set_aspect(1 / np.cos(np.deg2rad(0.5 * (N + S))))
            ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(True); s_.set_linewidth(0.5); s_.set_edgecolor("#333333")
            if k == 0:
                ax.set_title(zn, fontsize=6.5, fontweight="bold", pad=3)
                letter(ax, "def"[j], x=-0.02, y=1.12)
    # colour bars
    cax = fig.add_axes([0.07, rows[0] - 0.012, 0.26, 0.008])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=popnorm, cmap=popcmap), cax=cax, orientation="horizontal", extend="max")
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.5, length=2); cb.set_label("Residents per km$^2$ (a, top row of d–f)", fontsize=6)
    cax2 = fig.add_axes([0.07, 0.03, 0.40, 0.008])
    cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=pmnorm, cmap=pmcmap), cax=cax2, orientation="horizontal", ticks=PM_B)
    cb2.outline.set_linewidth(0.4); cb2.ax.tick_params(labelsize=5.5, length=2)
    cb2.set_label("Annual mean PM$_{2.5}$ in 2022 (µg m$^{-3}$; WHO guideline 5, interim targets 10–35) (b, c, d–f)", fontsize=6)
    save(fig, "fig1_teaser_v2")


if __name__ == "__main__":
    main()
