#!/usr/bin/env python3
"""Fig. 2: one day over the globe, 21 January 2022 (the day of 2022 with the highest population-weighted PM2.5, a rule fixed
in advance), daily PM2.5. a, stations of the six official networks (observed); b, GHAP at 1 km (the reference product, not a
forecast); c, CAMS and d, Aurora (0.4 deg), issued at 00 UTC on 21 January; e, Foehn's ensemble mean (four members) and f, one
member, at 1 km, issued at the end of 20 January. Boxes: RMSE, correlation and bias at the stations (forecasts read at the
station's 1 km pixel for Foehn and GHAP, bilinearly for CAMS and Aurora).
    python fig_gc2_world.py"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import zarr
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

from fig1_teaser_v6 import tiles
import final_model as FM
from nature_style import DOUBLE, HERE, MM, plt, save

G = "/scratch/project_462001140/ammar/eccv/data/world/global1km"
EV = FM.JAN21_DIR
AURD = HERE.parent / "benchmark" / "aurora"
T = 20                                                                         # 21 January 2022 (0-based day of year)
BOUNDS = [0, 5, 10, 15, 20, 30, 40, 50, 70, 100, 150, 500]
COLORS = ["#313695", "#4575B4", "#74ADD1", "#ABD9E9", "#E0F3F8", "#FFFFBF", "#FEE090", "#FDAE61", "#F46D43", "#D73027", "#A50026"]
NETS = [0, 1, 3, 4, 5, 6]


def bil(field, la, lo, lat0, dlat, lon0, dlon):
    nlat, nlon = field.shape
    fi = (lat0 - la) / dlat; fj = ((lo - lon0) % 360.0) / dlon
    i0 = np.clip(np.floor(fi).astype(int), 0, nlat - 2); j0 = np.floor(fj).astype(int) % nlon; j1 = (j0 + 1) % nlon
    wi, wj = np.clip(fi - i0, 0, 1), fj - np.floor(fj)
    return (1 - wi) * ((1 - wj) * field[i0, j0] + wj * field[i0, j1]) + wi * ((1 - wj) * field[i0 + 1, j0] + wj * field[i0 + 1, j1])


def scores(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    e = x[ok] - y[ok]
    return np.sqrt(np.mean(e ** 2)), np.corrcoef(x[ok], y[ok])[0, 1], np.mean(e), ok.sum()


def main():
    from nature_style import PM_BOUNDS, pm_cmap
    from gc_style import LAND_BG, land_mask_04
    cmap, norm = pm_cmap(); BOUNDS_ = PM_BOUNDS                                 # the paper's PM2.5 palette
    land04 = land_mask_04()
    st = np.load(f"{G}/stations_2022.npz")
    sel = np.isin(st["source"], NETS) & np.isfinite(st["daily"][:, T])
    la, lo, ob = st["lat"][sel], st["lon"][sel], st["daily"][sel, T].astype(np.float32)
    r = np.clip(((84.0 - la) / 0.01).astype(int), 0, 14399); c = np.clip(((lo + 180.0) / 0.01).astype(int), 0, 35999)
    land = tiles("static1km_tiles.npy", 1) > 0
    gh = tiles("ghap1km_tiles.npy", 1826 + T).astype(np.float32); gh[(gh <= 0) | ~land] = np.nan
    fm = np.load(EV / "map1km_d19_h1.npy").astype(np.float32); fm[~land] = np.nan
    f0 = np.load(EV / "map1km_d19_h1_member0.npy").astype(np.float32); f0[~land] = np.nan
    cams = np.asarray(zarr.open("/scratch/project_462001140/ammar/eccv/data/world/cams_forecast_daily/2022.zarr", mode="r")[T, 0, 0], np.float32)
    aur = None
    for p in sorted((AURD / "out_12h_all").glob("part_*.npz")):
        z = np.load(p)
        if T in list(z["days"]):
            A = z["pred"][list(z["days"]).index(T)].astype(np.float32); alat, alon = z["lat"], z["lon"]
            an = np.load(AURD / "raw" / f"{T:03d}.npz")["surf"][0, 5, :A.shape[1]].astype(np.float32) * 1e9
            aur = 0.25 * an + 0.5 * A[0] + 0.25 * A[1]
    S = {"GHAP": scores(gh[r, c], ob), "CAMS": scores(bil(cams, la, lo, 90.0, 0.4, 0.0, 0.4), ob),
         "Aurora": scores(bil(aur, la, lo, float(alat[0]), float(alat[0] - alat[1]), float(alon[0]), float(alon[1] - alon[0])), ob),
         "Foehn": scores(fm[r, c], ob), "Foehn member": scores(f0[r, c], ob)}
    W, H = DOUBLE, 158 * MM
    fig = plt.figure(figsize=(W, H))
    panels = [("Stations (observed)", None), ("GHAP, 1 km (reference product, not a forecast)", gh),
              ("CAMS forecast, 0.4°, issued 00 UTC 21 January", (cams, (0, 360, -90, 90))),
              ("Aurora forecast, 0.4°, issued 00 UTC 21 January", (aur, (float(alon[0]), float(alon[-1]) + 0.4, float(alat[-1]) - 0.2, float(alat[0]) + 0.2))),
              ("Foehn ensemble mean, 1 km, issued end of 20 January", fm), ("Foehn, one member, 1 km", f0)]
    keys = [None, "GHAP", "CAMS", "Aurora", "Foehn", "Foehn member"]
    for k, ((title, data), key) in enumerate(zip(panels, keys)):
        row, col = k // 2, k % 2
        ax = fig.add_axes([0.005 + 0.5 * col, 0.70 - 0.305 * row, 0.49, 0.265], projection=ccrs.Robinson())
        ax.set_extent([-180, 180, -60, 84], crs=ccrs.PlateCarree()); ax.spines["geo"].set_linewidth(0.4); ax.spines["geo"].set_edgecolor("#9A9A9A")
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="white", edgecolor="none", zorder=2)   # white ocean in every map
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=LAND_BG, edgecolor="none", zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.2, edgecolor="#777777", zorder=4)
        ax.add_feature(cfeature.BORDERS.with_scale("110m"), lw=0.12, edgecolor="#AAAAAA", zorder=4)
        if data is None:
            o = np.argsort(ob)
            ax.scatter(lo[o], la[o], c=ob[o], cmap=cmap, norm=norm, s=2.0, linewidths=0, transform=ccrs.PlateCarree(), zorder=5, rasterized=True)
            txt = None                                                    # no text on the map: the caption says it
        else:
            if isinstance(data, tuple):
                f, ext = data
                f = np.where(np.isfinite(f), np.maximum(f, 0), np.nan)                 # Aurora's few negative values shown as 0
                if ext[0] == 0:                                               # CAMS: 0..360 -> -180..180
                    f = np.roll(f, f.shape[1] // 2, axis=1); ext = (-180, 180, ext[2], ext[3])
                else:
                    f = np.roll(f, f.shape[1] // 2, axis=1); ext = (-180, 180, ext[2], ext[3])
                ax.imshow(f, origin="upper", extent=ext, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, interpolation="nearest",
                          regrid_shape=2000, zorder=1)
            else:
                ax.imshow(data, origin="upper", extent=(-180, 180, -60, 84), transform=ccrs.PlateCarree(), cmap=cmap, norm=norm,
                          interpolation="nearest", regrid_shape=4000, zorder=1)
            rm, rr, bi, n = S[key]
            txt = f"RMSE {rm:.1f} µg m$^{{-3}}$ | R {rr:.2f} | bias {bi:+.1f}"
        ax.set_title(title, fontsize=7, pad=3, fontweight="bold" if key and key.startswith("Foehn") else "normal")
        if txt:
            ax.text(0.02, 0.04, txt, transform=ax.transAxes, fontsize=5.6, va="bottom", zorder=8,
                    bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="#9A9A9A", lw=0.4))
        fig.text(0.005 + 0.5 * col, 0.70 - 0.305 * row + 0.275, "abcdef"[k], fontsize=8, fontweight="bold", va="bottom")
    cax = fig.add_axes([0.2, 0.045, 0.6, 0.016])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="max", spacing="uniform", ticks=BOUNDS_)
    cb.ax.set_xticklabels([f"{t:g}" for t in BOUNDS_]); cb.ax.tick_params(labelsize=5.5, length=2); cb.outline.set_linewidth(0.4)
    cb.set_label("Daily mean PM$_{2.5}$ (µg m$^{-3}$)", fontsize=6.5)
    save(fig, "fig_gc2_world")
    for k_, v in S.items():
        print(f"{k_:13s} RMSE {v[0]:.1f} R {v[1]:.2f} bias {v[2]:+.1f} n {v[3]}")


if __name__ == "__main__":
    main()
