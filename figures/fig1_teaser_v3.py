#!/usr/bin/env python3
"""Fig. 1 (teaser, v3, GHAP-paper style): the same world three ways, data only.
a, residents per km2 (GHS-POP 2020); b, annual mean PM2.5 of 2022 on the 40 km cells of the CAMS / Aurora grid; c, the same at
1 km (GHAP). Red frames: the three zones shown on the right in the same three views (rows aligned with a-c), the 40 km cells
drawn. Plate Carree, tight frames, continuous blue-yellow-red PM2.5 scale, colour bars inside the maps. No figure title.
    python fig1_teaser_v3.py"""
import cartopy.crs as ccrs
import matplotlib.patheffects  # noqa: F401
import cartopy.feature as cfeature
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

from nature_style import DOUBLE, HERE, MM, plt, save

ZONES = {"Cairo": (31.2, 28.8, 29.6, 32.4), "Bengal delta": (25.2, 22.8, 88.8, 91.6), "Lima": (-10.8, -13.2, -78.4, -76.0)}
PM_CMAP = LinearSegmentedColormap.from_list("ghap", ["#313695", "#4575B4", "#74ADD1", "#ABD9E9", "#E0F3F8", "#FFFFBF", "#FEE090",
                                                     "#FDAE61", "#F46D43", "#D73027", "#A50026"], N=256)
POP_CMAP = LinearSegmentedColormap.from_list("pop", ["#F7FCFD", "#BFD3E6", "#8C96C6", "#8C6BB1", "#88419D", "#6E016B", "#3F0045"], N=256)
PM_NORM = Normalize(0, 60)
POP_NORM = LogNorm(1, 10000)
EXT = (-180, 180, -58, 80)
MM_W = DOUBLE / MM


def blockmean(a, f):
    h, w = a.shape[0] // f * f, a.shape[1] // f * f
    with np.errstate(all="ignore"):
        return np.nanmean(a[:h, :w].reshape(h // f, f, w // f, f), (1, 3))


def frame(ax, lw=0.6):
    for s_ in ax.spines.values():
        s_.set_visible(True); s_.set_linewidth(lw); s_.set_edgecolor("#111111")


def main():
    pm1 = np.load(HERE / "data" / "pm_annual_2022.npy", mmap_mode="r")
    pm40 = np.load(HERE / "data" / "pm_annual_2022_40km.npy")
    pop = np.load("/scratch/project_462001140/ammar/eccv/data/world/global1km/pop1km_global.npy", mmap_mode="r")
    POP_CMAP.set_bad((1, 1, 1, 0)); PM_CMAP.set_bad((1, 1, 1, 0))
    # geometry (figure fractions): maps on the left, 3 x 3 square zooms on the right, rows of equal height
    W, Hf = DOUBLE, 118 * MM
    fig = plt.figure(figsize=(W, Hf))                                            # height set below from the geometry
    gap = 1.2; asp = (EXT[1] - EXT[0]) / (EXT[3] - EXT[2])                       # Plate Carree frame aspect
    map_w = (MM_W - 1.5 - 3.0 - 2 * gap - 1.0) / (1 + 3 / asp)                     # maps + 3 square zooms fill the width
    s_mm = map_h = map_w / asp
    Hf = (3 * map_h + 2 * gap + 9.0) * MM
    fig.set_size_inches(W, Hf)
    fx = lambda mm: mm / MM_W; fy = lambda mm: mm / (Hf / MM)
    top = Hf / MM - 6.5
    popd = blockmean(np.asarray(pop, np.float32), 10)                            # display at 0.1 deg (map ~85 mm wide)
    pm1d = blockmean(np.asarray(pm1, np.float32), 10)
    pm40d = np.repeat(np.repeat(pm40, 4, 0), 4, 1)                               # 0.4 deg cells shown on the 0.1 deg display grid
    views = [(np.where(popd > 0.5, popd, np.nan), POP_CMAP, POP_NORM, "a", "Where people live"),
             (pm40d, PM_CMAP, PM_NORM, "b", "PM$_{2.5}$ on a 40 km grid (CAMS, Aurora)"),
             (pm1d, PM_CMAP, PM_NORM, "c", "PM$_{2.5}$ at 1 km")]
    maps = []
    for k, (f, cm, nm, lt, title) in enumerate(views):
        y0 = top - (k + 1) * map_h - k * gap
        ax = fig.add_axes([fx(1.5), fy(y0), fx(map_w), fy(map_h)], projection=ccrs.PlateCarree())
        ax.set_extent(EXT, crs=ccrs.PlateCarree())
        ax.imshow(f, extent=(-180, 180, -60, 84), origin="upper", transform=ccrs.PlateCarree(), cmap=cm, norm=nm,
                  interpolation="nearest", zorder=2)
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.25, edgecolor="#222222", zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale("110m"), lw=0.15, edgecolor="#777777", zorder=3)
        for z_, (zn, (N, S, Wz, E)) in enumerate(ZONES.items()):           # zones are ~3 deg: frames enlarged to be visible
            la, lo = 0.5 * (N + S), 0.5 * (Wz + E)
            ax.add_patch(plt.Rectangle((lo - 4.5, la - 4.5), 9, 9, fill=False, ec="#E3001B", lw=0.9, zorder=5, transform=ccrs.PlateCarree()))
            ax.text(lo + 6, la + 3, str(z_ + 1), color="#E3001B", fontsize=6, fontweight="bold", zorder=6, transform=ccrs.PlateCarree(),
                    path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=1.2, foreground="white")])
        frame(ax)
        ax.text(0.008, 0.965, f"$\\bf{{{lt}}}$  {title}", transform=ax.transAxes, fontsize=6.5, va="top", zorder=6,
                bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="none", alpha=0.85))
        maps.append(ax)
    # colour bars inside the maps, over the South Pacific
    for ax, cm, nm, lab, ticks in ((maps[0], POP_CMAP, POP_NORM, "residents km$^{-2}$", [1, 10, 100, 1000, 10000]),
                                   (maps[2], PM_CMAP, PM_NORM, "annual PM$_{2.5}$ 2022 (µg m$^{-3}$)", [0, 10, 20, 30, 40, 50, 60])):
        cax = ax.inset_axes([0.05, 0.17, 0.21, 0.05])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=nm, cmap=cm), cax=cax, orientation="horizontal", ticks=ticks,
                          extend="max" if cm is PM_CMAP else "both")
        cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5, length=1.5, pad=1)
        cb.set_label(lab, fontsize=5.5, labelpad=1)
        if cm is POP_CMAP:
            cb.set_ticklabels(["1", "10", "100", "1k", "10k"])
    # zooms
    for j, (zn, (N, S, Wz, E)) in enumerate(ZONES.items()):
        r0, r1 = int(round((84 - N) * 100)), int(round((84 - S) * 100)); c0, c1 = int(round((Wz + 180) * 100)), int(round((E + 180) * 100))
        p = np.asarray(pop[r0:r1, c0:c1], np.float32)
        x1 = np.asarray(pm1[r0:r1, c0:c1], np.float32)
        x40 = np.where(np.isfinite(x1), np.repeat(np.repeat(pm40[r0 // 40:r1 // 40, c0 // 40:c1 // 40], 40, 0), 40, 1), np.nan)
        for k, (f, cm, nm) in enumerate(((np.where(p > 0.5, p, np.nan), POP_CMAP, POP_NORM), (x40, PM_CMAP, PM_NORM), (x1, PM_CMAP, PM_NORM))):
            y0 = top - (k + 1) * map_h - k * gap
            x0 = 1.5 + map_w + 3.0 + j * (s_mm + gap)
            ax = fig.add_axes([fx(x0), fy(y0), fx(s_mm), fy(map_h)])
            ax.imshow(f, extent=(Wz, E, S, N), origin="upper", cmap=cm, norm=nm, interpolation="nearest")
            ax.set_facecolor("#DCE6EE")
            if k == 1:                                                          # the 40 km cells, only where they are the data
                for c_ in np.arange(Wz + 0.4, E - 1e-6, 0.4):
                    ax.axvline(c_, color="#FFFFFF", lw=0.35, alpha=0.9)
                for r_ in np.arange(S + 0.4, N - 1e-6, 0.4):
                    ax.axhline(r_, color="#FFFFFF", lw=0.35, alpha=0.9)
            ax.set_xlim(Wz, E); ax.set_ylim(S, N); ax.set_aspect("auto"); ax.set_xticks([]); ax.set_yticks([]); frame(ax, 0.6)
            if k == 0:
                ax.set_title(f"{j + 1}  {zn}", fontsize=6.5, fontweight="bold", pad=2)
                lat_c, lon_c = 0.5 * (N + S), 0.5 * (Wz + E)
                ax.text(0.97, 0.04, f"{abs(lat_c):.1f}°{'N' if lat_c > 0 else 'S'} {abs(lon_c):.1f}°{'E' if lon_c > 0 else 'W'}",
                        transform=ax.transAxes, ha="right", fontsize=5, color="#333333",
                        bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.8))
            if j == 0:
                dl = 50.0 / (111.32 * np.cos(np.deg2rad(0.5 * (N + S))))
                if k == 2:
                    ax.plot([Wz + 0.15, Wz + 0.15 + dl], [S + 0.15, S + 0.15], color="k", lw=1.0, solid_capstyle="butt")
                    ax.text(Wz + 0.15 + dl / 2, S + 0.22, "50 km", ha="center", va="bottom", fontsize=5,
                            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.7))
    save(fig, "fig1_teaser_v3")


if __name__ == "__main__":
    main()
