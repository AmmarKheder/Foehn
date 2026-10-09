#!/usr/bin/env python3
"""Fig. 4: where the air comes from, and why computing it helps (FLEXGREEN Green's functions of the 2022 forecasts;
figdata_transport.py, figdata_transport_gain.py).
a, share of the air arriving over land three days ahead that descended from above the mixed layer of the start day
   (0.5 deg receptors, mean over every fourth start day of 2022);
b, that share by lead time, all land and regions; c, median distance of the air's origin on the start day;
d, the pollution follows the air: daily PM2.5 (GHAP, 0.25 deg) over East Asia on 5-8 January 2022 and, on each of the first
   three days, where the air reaching Seoul on 8 January was then (FLEXGREEN origins of the forecasts issued 3, 2 and 1 days
   before, 256 particles in each of the 3 x 3 receptor cells; data/transport_seoul.npz);
e, the gain from computing the transport follows the memory of the start day: reduction of the fair CRPS against GHAP
   (2022, land) of the base model with the transport inputs over the same network trained identically without them,
   against the share of the arriving air that was in the start-day mixed layer, per region and lead time (1-7 days).
    python fig4_transport.py"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

from nature_style import COLOR, DOUBLE, HERE, MM, PM_BOUNDS, plt, pm_cmap, save  # noqa: F401

G = "/scratch/project_462001140/ammar/eccv/data/world/global1km"
REG_GREY = "#ABABAB"                                     # single regions: grey (the paper's colours are Foehn blue, CAMS black,
PART = "#4F8FD6"                                          # Aurora orange, climatology grey); particles: light Foehn blue
HALO = [pe.withStroke(linewidth=1.6, foreground="white")]


def clean(ax):
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(labelsize=5.5, length=2)


def main():
    z = np.load(HERE / "data" / "transport_fig.npz")
    tg = np.load(HERE / "data" / "transport_gain.npz")
    R = [str(r) for r in z["regions"]]; GR = [str(r) for r in tg["regions"]]
    land = np.load(f"{G}/meta.npz")["land"].astype(bool).reshape(288, 2, 720, 2).any((1, 3))
    W, H = DOUBLE, 150 * MM
    fig = plt.figure(figsize=(W, H))
    fx = lambda mm: mm / (W / MM); fy = lambda mm: mm / (H / MM)
    # a: share from aloft three days ahead
    ax = fig.add_axes([fx(0), fy(94), fx(122), fy(49)], projection=ccrs.Robinson())
    ax.set_extent((-180, 180, -58, 84), crs=ccrs.PlateCarree()); ax.spines["geo"].set_visible(False)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#EDEDED", edgecolor="none", zorder=0)
    cmap = LinearSegmentedColormap.from_list("aloft", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"], N=12)
    norm = BoundaryNorm(np.arange(20, 81, 6), cmap.N, extend="both")
    lat_e = 84.0 - 0.5 * np.arange(289); lon_e = -180.0 + 0.5 * np.arange(721)
    a3 = np.where(land, 100 * z["aloft_map"][2], np.nan)
    ax.pcolormesh(lon_e, lat_e, np.ma.masked_invalid(a3), cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), rasterized=True, zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.2, edgecolor="#555555", zorder=3)
    cax = ax.inset_axes([0.02, 0.05, 0.17, 0.035])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="both", ticks=[20, 32, 44, 56, 68, 80])
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5, length=1.5, pad=1); cb.ax.xaxis.set_label_position("top")
    cb.set_label("Air from aloft, 3 days ahead (%)", fontsize=5.5, labelpad=1.5)
    fig.text(fx(2), fy(143.5), "$\\bf{a}$", fontsize=8, va="bottom")
    # b, c: by lead (all land black, single regions grey)
    L = z["leads"]
    for k, (key, ylab, y0, lim) in enumerate((("aloft", "From aloft (%)", 124, (0, 100)),
                                               ("dist", "Origin distance (km)", 101, (0, 3000)))):
        a_ = fig.add_axes([fx(136), fy(y0), fx(41), fy(18)])
        for g, r in enumerate(R):
            v = np.nanmean(z[key][:, :, g], 0) * (100 if key == "aloft" else 1)
            big = r == "Global"
            a_.plot(L, v, color="#111111" if big else REG_GREY, lw=1.4 if big else 0.6, marker="o" if big else "", ms=2.3,
                    zorder=5 if big else 3, label="All land" if big else ("Single regions" if g == 1 else None))
            if big:
                for h in (1, 3, 7):
                    a_.annotate(f"{v[h - 1]:.0f}{'%' if key == 'aloft' else ''}", (h, v[h - 1]), textcoords="offset points",
                                xytext={1: (4, -2), 3: (-3, 3), 7: (4, -1)}[h], ha="right" if h == 3 else "left",
                                va={1: "top", 3: "bottom", 7: "center"}[h], fontsize=5, color="#111111", path_effects=HALO, zorder=7)
        a_.set_xlim(0.7, 7.3); a_.set_ylim(*lim); a_.set_xticks(range(1, 8)); a_.set_ylabel(ylab, fontsize=5.5); clean(a_)
        if k == 0:
            a_.set_xticklabels([]); a_.legend(fontsize=4.8, loc="lower right", handlelength=1.4, frameon=False)
        else:
            a_.set_xlabel("Lead time (days)", fontsize=6, labelpad=1)
        fig.text(fx(126), fy(y0 + 18.5), f"$\\bf{{{'bc'[k]}}}$", fontsize=8, va="bottom")
    # d: the pollution follows the air, Seoul 8 January 2022
    zs = np.load(HERE / "data" / "transport_seoul.npz"); T = int(zs["T"]); sla, slo = float(zs["lat"]), float(zs["lon"])
    pmz = np.load(f"{G}/pm.npy", mmap_mode="r"); pcm, pnorm = pm_cmap()
    ext = (110, 132, 30, 45)
    for k, h in enumerate((3, 2, 1, 0)):
        dday = T - h
        am = fig.add_axes([fx(2 + 45.5 * k), fy(50), fx(43), fy(36)], projection=ccrs.PlateCarree())
        am.set_extent(ext, crs=ccrs.PlateCarree())
        am.imshow(np.asarray(pmz[1826 + dday], np.float32), extent=(-180.125, 179.875, -59.875, 84.125), origin="upper", cmap=pcm,
                  norm=pnorm, transform=ccrs.PlateCarree(), interpolation="nearest", zorder=1)
        am.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.3, edgecolor="#555555", zorder=3)
        am.spines["geo"].set_linewidth(0.4)
        if h > 0:
            o = zs[f"org_{h}"]; keep = np.nan_to_num(o[:, 3]) > 0.05
            am.scatter(o[keep, 0], o[keep, 1], s=0.5, color=PART, alpha=0.55, linewidths=0, transform=ccrs.PlateCarree(), zorder=4,
                       rasterized=True)
        am.plot(slo, sla, marker="*", ms=7, color="white", mec="#111111", mew=0.5, transform=ccrs.PlateCarree(), zorder=6)
        if k == 0:
            am.text(slo + 0.6, sla - 1.6, "Seoul", fontsize=5.5, fontweight="bold", transform=ccrs.PlateCarree(), zorder=6, path_effects=HALO)
        day = np.datetime64("2022-01-01") + np.timedelta64(dday, "D")
        am.set_title(f"{int(str(day)[8:10])} January" + (f": the air of 8 January, {h} day{'s' if h > 1 else ''} earlier" if h else ": arrival in Seoul"),
                     fontsize=5.6, pad=2)
    fig.text(fx(2), fy(89), "$\\bf{d}$", fontsize=8, va="bottom")
    cax = fig.add_axes([fx(40), fy(45.5), fx(105), fy(1.5)])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=pnorm, cmap=pcm), cax=cax, orientation="horizontal", extend="max", spacing="uniform", ticks=PM_BOUNDS)
    cb.ax.set_xticklabels([f"{t:g}" for t in PM_BOUNDS]); cb.ax.tick_params(labelsize=5, length=1.5, pad=1); cb.outline.set_linewidth(0.4)
    cb.set_label("Daily PM$_{2.5}$ (µg m$^{-3}$); blue dots: where the air reaching Seoul on 8 January was on that day", fontsize=5.5, labelpad=1)
    # e: gain from the transport vs memory of the start day
    ae = fig.add_axes([fx(16), fy(8), fx(150), fy(26)])
    for r in ("East Asia", "South Asia", "Europe", "North America", "Africa", "Global"):
        g, gg = R.index(r), GR.index("All regions" if r == "Global" else r)
        mem = 100 - 100 * np.nanmean(z["aloft"][:, :, g], 0)                        # share still from the start-day mixed layer
        gain = tg["gain"][gg, :, 0]
        big = r == "Global"
        ae.plot(mem, gain, color="#111111" if big else REG_GREY, lw=1.4 if big else 0.7, marker="o", ms=3.0 if big else 1.8,
                zorder=6 if big else 3, label="All regions" if big else ("Single regions" if r == "East Asia" else None))
        if big:
            lo, hi = tg["gain"][gg, :, 1], tg["gain"][gg, :, 2]
            ae.vlines(mem, lo, hi, color="#111111", lw=0.8, zorder=5)
            for h in (1, 3, 7):
                ae.annotate(f"{h} d", (mem[h - 1], gain[h - 1]), textcoords="offset points", xytext=(4, 4) if h == 1 else (-4, 5),
                            ha="left" if h == 1 else "right", fontsize=5.2, fontweight="bold", path_effects=HALO, zorder=8)
        elif r in ("North America", "South Asia", "Africa"):
            ae.annotate(r, (mem[0], gain[0]), textcoords="offset points", xytext=(4, 0), va="center", fontsize=4.8, color="#7A7A7A")
    ae.axhline(0, color="#999999", lw=0.5, zorder=1)
    ae.set_xlabel("Arriving air that was in the start-day mixed layer (%)", fontsize=6, labelpad=1)
    ae.set_ylabel("CRPS reduction by\nthe transport (%)", fontsize=6); clean(ae)
    ae.legend(fontsize=5, loc="upper left", frameon=False, handlelength=1.4)
    fig.text(fx(2), fy(36), "$\\bf{e}$", fontsize=8, va="bottom")
    save(fig, "fig4_transport")
    g = R.index("Global")
    print("aloft % by lead:", np.round(100 * np.nanmean(z["aloft"][:, :, g], 0), 1), "dist km:", np.round(np.nanmean(z["dist"][:, :, g], 0)),
          "gain all regions:", np.round(tg["gain"][GR.index("All regions"), :, 0], 1))


if __name__ == "__main__":
    main()
