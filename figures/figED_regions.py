#!/usr/bin/env python3
"""Extended Data figure: kilometre-scale structure in four regions, 21 January 2022 (forecasts for that day issued at the end of
20 January / 00 UTC 21 January). Columns: GHAP at 1 km (reference), CAMS and Aurora (0.4 deg), Foehn's ensemble mean (1 km), and
Foehn minus GHAP. Each row has its own colour range (2nd-98th percentiles of GHAP); numbers: mean absolute error and spatial
correlation against GHAP over the land pixels of the panel (0.4 deg forecasts read at the 1 km pixels).
    python figED_regions.py"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import zarr
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

from fig1_teaser_v6 import tiles
import final_model as FM
from nature_style import DOUBLE, HERE, MM, plt, save

EV = FM.JAN21_DIR
AURD = HERE.parent / "benchmark" / "aurora"
T = 20
REGIONS = {"North China Plain": (41.0, 38.0, 114.5, 118.5), "Delhi and the Ganges plain": (30.3, 27.3, 75.6, 79.6),
           "Po Valley": (46.3, 44.0, 7.3, 12.3), "Seoul and the Han basin": (38.3, 36.5, 126.0, 128.3)}
from nature_style import PM_COLORS  # noqa: E402
import gc_style as GC  # noqa: E402
CM = LinearSegmentedColormap.from_list("pmc", PM_COLORS, N=256)            # the paper's PM2.5 colours, continuous per row


def coarse_at(field, lat0, dlat, lon0, dlon, lats, lons):
    i = np.clip(np.round((lat0 - lats) / dlat).astype(int), 0, field.shape[0] - 1)
    j = np.round(((lons - lon0) % 360.0) / dlon).astype(int) % field.shape[1]
    return field[i[:, None], j[None, :]]


def main():
    CM.set_bad((1, 1, 1, 0))
    land = tiles("static1km_tiles.npy", 1) > 0
    gh = tiles("ghap1km_tiles.npy", 1826 + T).astype(np.float32); gh[(gh <= 0) | ~land] = np.nan
    fm = np.load(EV / "map1km_d19_h1.npy", mmap_mode="r")
    cams = np.asarray(zarr.open("/scratch/project_462001140/ammar/eccv/data/world/cams_forecast_daily/2022.zarr", mode="r")[T, 0, 0], np.float32)
    for p in sorted((AURD / "out_12h_all").glob("part_*.npz")):
        z = np.load(p)
        if T in list(z["days"]):
            A = z["pred"][list(z["days"]).index(T)].astype(np.float32); alat, alon = z["lat"], z["lon"]
    an = np.load(AURD / "raw" / f"{T:03d}.npz")["surf"][0, 5, :A.shape[1]].astype(np.float32) * 1e9
    aur = np.maximum(0.25 * an + 0.5 * A[0] + 0.25 * A[1], 0)
    MW = 29.3                                                                 # map width (mm); height from the extent, rows packed
    MH = [min(MW * (N - S) / (E - W_), 31.0) for (N, S, W_, E) in REGIONS.values()]
    W, H = DOUBLE, (9 + sum(MH) + 8.5 * (len(MH) - 1) + 4) * MM
    XC = [0.085, 0.255, 0.425, 0.595, 0.80]                                 # gaps after Foehn and the error map for the colour bars
    DCM = LinearSegmentedColormap.from_list("div", GC.DIV, N=256)
    fig = plt.figure(figsize=(W, H))
    cols = ["GHAP, 1 km (reference)", "CAMS, 0.4°", "Aurora, 0.4°", "Foehn, 1 km", "Foehn − GHAP"]
    for i, (rn, (N, S, Wl, E)) in enumerate(REGIONS.items()):
        r0, r1, c0, c1 = int(round((84 - N) * 100)), int(round((84 - S) * 100)), int(round((Wl + 180) * 100)), int(round((E + 180) * 100))
        lats, lons = 84 - 0.01 * (np.arange(r0, r1) + 0.5), -180 + 0.01 * (np.arange(c0, c1) + 0.5)
        g = gh[r0:r1, c0:c1]; ok = np.isfinite(g)
        f = np.where(ok, np.asarray(fm[r0:r1, c0:c1], np.float32), np.nan)
        c = np.where(ok, coarse_at(cams, 90.0, 0.4, 0.0, 0.4, lats, lons), np.nan)
        a = np.where(ok, coarse_at(aur, float(alat[0]), float(alat[0] - alat[1]), float(alon[0]), float(alon[1] - alon[0]), lats, lons), np.nan)
        lo_, hi_ = np.nanpercentile(g, [2, 98]); nm = Normalize(lo_, hi_)
        err = f - g; em = np.nanpercentile(np.abs(err), 98); en = TwoSlopeNorm(0, -em, em)
        ytop = H / MM - 9 - sum(MH[:i]) - 8.5 * i; y0 = (ytop - MH[i]) / (H / MM); bb = {}
        for j, (fld, title) in enumerate(zip((g, c, a, f, err), cols)):
            ax = fig.add_axes([XC[j], y0, 0.16, MH[i] / (H / MM)], projection=ccrs.PlateCarree())
            ax.set_extent([Wl, E, S, N], crs=ccrs.PlateCarree())
            ax.imshow(fld, extent=(Wl, E, S, N), origin="upper", transform=ccrs.PlateCarree(), interpolation="nearest",
                      cmap=DCM if j == 4 else CM, norm=en if j == 4 else nm, zorder=1)
            ax.add_feature(cfeature.COASTLINE.with_scale("10m"), lw=0.3, edgecolor="#333333", zorder=3)
            ax.add_feature(cfeature.BORDERS.with_scale("10m"), lw=0.2, edgecolor="#777777", zorder=3)
            ax.spines["geo"].set_linewidth(0.4)
            GC.geo_ticks(ax, left=j == 0, bottom=True)          # each row its own region: longitudes under every map
            if i == 0:
                ax.set_title(title, fontsize=6.2, fontweight="bold" if j == 3 else "normal", pad=3)
            if j == 0:
                ax.text(-0.22, 0.5, rn.replace(" and the ", ",\n"), transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=6.2, fontweight="bold")
            if j in (1, 2, 3):
                m = ok & np.isfinite(fld)
                mae = np.mean(np.abs(fld[m] - g[m])); rr = np.corrcoef(fld[m], g[m])[0, 1]
                ax.text(0.97, 0.04, f"MAE {mae:.1f} | R {rr:.2f}", transform=ax.transAxes, ha="right", fontsize=5, zorder=6,
                        fontweight="bold" if j == 3 else "normal", bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
            ax.apply_aspect(); bb[j] = ax.get_position()                         # the drawn map, for the letter and the colour bars
            fig.text(bb[j].x0 - 0.004, bb[j].y1 + 0.003, "abcdefghijklmnopqrstuvwxy"[5 * i + j], fontsize=7.5, fontweight="bold", va="bottom")
        for jj, (norm_, cm_, lab) in enumerate(((nm, CM, "µg m$^{-3}$"), (en, DCM, "µg m$^{-3}$"))):
            b_ = bb[3 + jj]
            cax = fig.add_axes([b_.x1 + 0.004, b_.y0, 0.006, b_.height])
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm_, cmap=cm_), cax=cax); cb.outline.set_linewidth(0.3)
            cb.ax.tick_params(labelsize=4.5, length=1.2, pad=0.8)
    save(fig, "figED_regions")


if __name__ == "__main__":
    main()
