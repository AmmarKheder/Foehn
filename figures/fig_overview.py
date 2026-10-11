#!/usr/bin/env python3
"""Main figure: one forecast of the whole planet at 1 km (the final model; after fig1_overview.py).
a, Foehn's forecast (ensemble mean of four members) issued at the end of 20 January 2022 for 21 January, the day of 2022 with
   the highest population-weighted PM2.5 (rule fixed in advance), with the four regions of b;
b, the same forecast in four regions on four continents, next to the CAMS forecast issued at the same time (00 UTC run of
   21 January, 0.4 deg); dots: the measurements of that day at the stations of the six official networks. Numbers: RMSE
   over 2022 at those stations, one day ahead, on identical station-days. The station-verdict panel (78% of 5,959
   stations, three orthographic views) now opens fig_gc3_scorecards.py.
    python fig_overview.py"""
import glob

import cartopy.crs as ccrs
import numpy as np
import zarr

import final_model as FM
from fig1_teaser_v6 import tiles
from gc_style import geo_ticks, map_axes, world_coords
from nature_style import DOUBLE, HERE, PM_BOUNDS, letter, plt, pm_cmap, save

D0, H = 19, 1                                                   # start day 20 January (0-based 19), lead 1 day
DT = D0 + H
G = "/scratch/project_462001140/ammar/eccv/data/world/global1km"
CENTRES = {"Delhi and the Ganges plain": (28.7, 77.6, 700), "Po Valley and the Alps": (45.35, 10.0, 600),
           "Los Angeles basin": (34.05, -117.9, 300), "Seoul and the Han basin": (37.56, 127.0, 300)}
ASPECT = 1.5                                                    # width / height of every zoom (km)


def main():
    ours = np.load(FM.JAN21_DIR / f"map1km_d{D0}_h{H}.npy", mmap_mode="r")
    ghap = tiles("ghap1km_tiles.npy", 1826 + DT); land = tiles("static1km_tiles.npy", 1) > 0
    cz = zarr.open("/scratch/project_462001140/ammar/eccv/data/world/cams_forecast_daily/2022.zarr", mode="r")
    cams = np.asarray(cz[DT, 0, 0], np.float32)                 # run of 00 UTC 21 January, its lead day 0: equal issuance
    st = np.load(f"{G}/stations_2022.npz"); off = st["source"] != 2  # the six official networks (OpenAQ excluded, as everywhere)
    obs = st["daily"][:, DT]; okst = np.isfinite(obs) & off
    slat, slon, sobs = st["lat"][okst], st["lon"][okst], obs[okst]
    cmap, norm = pm_cmap()
    # RMSE over 2022, lead 1 day, at every station: Foehn (ensemble mean), CAMS (equal issuance), GHAP; identical station-days
    la_all, lo_all = st["lat"], st["lon"]
    cache = HERE / "data" / f"overview_cache_{FM.LABEL}.npz"
    if cache.exists():
        zc_ = np.load(cache); SSE, NSD = zc_["SSE"], zc_["NSD"]
    else:
        Pp = [np.load(p) for p in sorted(glob.glob(str(FM.EVAL_DIR / "part_*.npz")))]
        kk = np.concatenate([p["keys"] for p in Pp]); ee = np.concatenate([p["st_ens"] for p in Pp]); gg = np.concatenate([p["st_ghap"] for p in Pp])
        fi = (90.0 - la_all) / 0.4; fj = (lo_all % 360.0) / 0.4
        i0 = np.clip(np.floor(fi).astype(int), 0, 449); j0 = np.floor(fj).astype(int) % 900; j1 = (j0 + 1) % 900; wi, wj = fi - i0, fj - np.floor(fj)
        SSE = np.zeros((len(la_all), 3)); NSD = np.zeros(len(la_all))
        for n in np.where(kk[:, 1] == 1)[0]:
            t = int(kk[n, 2])
            c = np.asarray(cz[t, 0, 0], np.float32)
            cs = (1 - wi) * ((1 - wj) * c[i0, j0] + wj * c[i0, j1]) + wi * ((1 - wj) * c[i0 + 1, j0] + wj * c[i0 + 1, j1])
            X = np.stack([ee[n].astype(np.float32).mean(-1), cs, gg[n].astype(np.float32)]); o = st["daily"][:, t]
            m = np.isfinite(o) & np.all(np.isfinite(X), 0) & off
            SSE[m] += ((X[:, m] - o[m]) ** 2).T; NSD[m] += 1
        np.savez(cache, SSE=SSE, NSD=NSD)
    print("station RMSE (lead 1) on", int(NSD.sum()), "station-days")
    REG = {}
    for nm, (la, lo, wkm) in CENTRES.items():
        dla = wkm / ASPECT / 111.2 / 2; dlo = wkm / (111.2 * np.cos(np.deg2rad(la))) / 2
        REG[nm] = (la + dla, la - dla, lo - dlo, lo + dlo, wkm)
    FH = 183 / 25.4
    fig = plt.figure(figsize=(DOUBLE, FH))
    # a: the globe at 1 km (ensemble mean) and the regions of b
    ax = map_axes(fig, [0.035, 91 / 183, 0.93, 88 / 183])                 # room for the latitudes on both sides
    m = np.asarray(ours, np.float32); m[~land] = np.nan
    ax.imshow(m, origin="upper", extent=(-180, 180, -60, 84), transform=ccrs.PlateCarree(), cmap=cmap, norm=norm,
              regrid_shape=4400, zorder=1, interpolation="nearest")
    del m
    for k, (nm, (n, s, w, e, _)) in enumerate(REG.items()):
        t_ = np.linspace(0, 1, 40)
        bx = np.concatenate([w + (e - w) * t_, np.full(40, e), e - (e - w) * t_, np.full(40, w)])
        by = np.concatenate([np.full(40, n), n - (n - s) * t_, np.full(40, s), s + (n - s) * t_])
        ax.plot(bx, by, transform=ccrs.PlateCarree(), color="#111111", lw=0.7, zorder=6)
        ax.text(e + 1.2, n + 0.5, str(k + 1), transform=ccrs.PlateCarree(), fontsize=6.5, fontweight="bold", va="center", zorder=7,
                bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec="#111111", lw=0.5))
    world_coords(ax, size=5.5)
    letter(ax, "a", x=-0.005, y=0.985)
    cax = fig.add_axes([0.385, 101 / 183, 0.285, 0.0075])   # fully inside the Southern Ocean of a
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="max", spacing="uniform", ticks=PM_BOUNDS)
    cb.ax.set_xticklabels([f"{t:g}" for t in PM_BOUNDS]); cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.4, pad=1)
    cb.outline.set_linewidth(0.4)
    cb.ax.xaxis.set_label_position("top")
    cb.set_label("Daily mean PM$_{2.5}$ on 21 January 2022 (µg m$^{-3}$)", fontsize=5.8, labelpad=2)
    # b: four regions x (CAMS 40 km, Foehn 1 km); the GHAP zooms are in the ED regions figure
    rows = (("CAMS forecast (40 km)", "cams"), ("Foehn (1 km)", "ours"), ("GHAP (1 km)", "ghap"))
    cw = 0.205; ch = cw * DOUBLE / FH / ASPECT; x0 = 0.056; y0 = 82 / 183 - ch       # 7 mm gaps: each region has its latitudes
    for j, (nm, (n, s, w, e, wkm)) in enumerate(REG.items()):
        r0, r1 = int((84 - n) * 100), int((84 - s) * 100); c0, c1 = int((w + 180) * 100), int((e + 180) * 100)
        sel = (slat <= n) & (slat >= s) & (slon >= w) & (slon <= e)
        inbox = (la_all <= n) & (la_all >= s) & (lo_all >= w) & (lo_all <= e) & (NSD > 0) & off
        for i, (rt, key) in enumerate(rows):
            a = fig.add_axes([x0 + j * 0.2435, y0 - i * (ch + 0.012), cw, ch]); a.set_facecolor("white")
            if key == "cams":
                ci0, ci1 = int((90 - n) / 0.4), int((90 - s) / 0.4) + 1; cj0, cj1 = int((w % 360) / 0.4), int((e % 360) / 0.4) + 1
                lw0 = cj0 * 0.4 - 0.2; lw0 = lw0 - 360.0 if lw0 > 180 else lw0
                cc = cams[ci0:ci1, cj0:cj1].copy()
                la_c = 90 - 0.4 * np.arange(ci0, ci1); lo_c = (0.4 * np.arange(cj0, cj1) + 180) % 360 - 180
                rr = np.clip(((84 - la_c) * 100).astype(int), 0, land.shape[0] - 1); cc_ = np.clip(((lo_c + 180) * 100).astype(int), 0, land.shape[1] - 1)
                lm = land[rr[:, None], cc_[None, :]]                  # the 40 km cells shown over land, like the 1 km rows
                cc[~lm] = np.nan
                a.imshow(cc, extent=(lw0, lw0 + (cj1 - cj0) * 0.4, 90 - ci1 * 0.4 + 0.2, 90 - ci0 * 0.4 + 0.2),
                         cmap=cmap, norm=norm, interpolation="nearest", origin="upper")
            else:
                src = ours if key == "ours" else ghap
                v = np.asarray(src[r0:r1, c0:c1], np.float32); v[~land[r0:r1, c0:c1]] = np.nan
                if key == "ghap":
                    v[v <= 0] = np.nan
                a.imshow(v, extent=(w, e, s, n), cmap=cmap, norm=norm, interpolation="nearest", origin="upper")
            if sel.any():
                a.scatter(slon[sel], slat[sel], c=sobs[sel], cmap=cmap, norm=norm, s=6, linewidths=0.35, edgecolors="#111111", zorder=5)
            if inbox.sum() >= 3:
                rm = np.sqrt(SSE[inbox, {"ours": 0, "cams": 1, "ghap": 2}[key]].sum() / NSD[inbox].sum())
                a.text(0.03, 0.04, f"RMSE {rm:.1f} µg m$^{{-3}}$" + (f" ({int(inbox.sum())} stations)" if i == 0 else ""),
                       transform=a.transAxes, fontsize=5.2, zorder=8, fontweight="bold" if key == "ours" else "normal",
                       bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.85))
                print(f"{nm:28s} {key:5s} RMSE 2022 {rm:.1f} ({int(inbox.sum())} stations)")
            a.set_xlim(w, e); a.set_ylim(s, n); a.set_aspect(1 / np.cos(np.deg2rad(0.5 * (n + s))), adjustable="box")
            if i == 2:                                                 # scale bar
                L = {700: 200, 600: 100, 400: 100, 300: 50}[wkm]; dlo = L / (111.2 * np.cos(np.deg2rad(0.5 * (n + s))))
                xb, yb = e - 0.06 * (e - w) - dlo, s + 0.08 * (n - s)
                a.plot([xb, xb + dlo], [yb, yb], color="#111111", lw=1.0, zorder=9, solid_capstyle="butt")
                a.text(xb + dlo / 2, yb + 0.03 * (n - s), f"{L} km", ha="center", va="bottom", fontsize=5.2, zorder=9,
                       bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.7))
            geo_ticks(a, left=True, bottom=i == 2)                         # latitudes on every panel, longitudes under the last row
            for sp in a.spines.values():
                sp.set_visible(True); sp.set_linewidth(0.5); sp.set_color("#333333")
            if i == 0:
                a.set_title(f"{j + 1}  {nm}", fontsize=6.5, pad=2, fontweight="bold")
            if j == 0:
                a.text(-0.17, 0.5, rt, transform=a.transAxes, rotation=90, ha="right", va="center", fontsize=6,
                       fontweight="bold" if key == "ours" else "normal")
            if i == 0 and j == 0:
                letter(a, "b", x=-0.12, y=1.04)
    save(fig, "fig_overview")


if __name__ == "__main__":
    main()
