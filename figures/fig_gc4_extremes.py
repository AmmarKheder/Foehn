#!/usr/bin/env python3
"""Fig. 4: extreme pollution events.
a, residents per day in 2022 above the WHO 24-h interim target (37.5 ug m-3) at their 1 km pixel whose 40 km cell is below it,
   i.e. people a perfect 40 km forecast could not warn, per 40 km cell (GHAP 1 km x GHS-POP 1 km, no forecast; figdata_exposure.py).
b-d, Beijing-Tianjin on 21 January 2022, the most polluted day of the year for the world's population: daily PM2.5 at 1 km
   (GHAP) with the 75 ug m-3 contour, the same field on 40 km cells, residents above 75 ug m-3 seen or hidden by the 40 km cells
   (fig_exposure_row1.py).
e, f, relative economic value of warnings of days above the local 95th (e) and 99th (f) percentile at the stations, one (solid)
   and three days ahead (dashed): Foehn's ensemble, CAMS and Aurora issued at the same time (extremes_stations.py).
(the SEDI of the land-wide extremes is in figED_sedi.py)
    python fig_gc4_extremes.py"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

import fig_exposure_row1 as R1
from gc_style import AURORA, CAMS, FOEHN, PERS, clean, label
import final_model as FM
from nature_style import DOUBLE, HERE, MM, PM_BOUNDS, plt, save  # noqa: F401


def rev_curves(cnt, alphas):
    """cnt (bins, 2): counts by forecast probability bin and outcome -> REV(alpha), envelope over probability thresholds"""
    n0, n1 = cnt[:, 0].astype(float), cnt[:, 1].astype(float)
    s = n1.sum() / (n0.sum() + n1.sum())
    H = np.cumsum(n1[::-1])[::-1] / n1.sum(); F = np.cumsum(n0[::-1])[::-1] / n0.sum()
    keep = (H > 0) | (F > 0)
    H, F = H[keep][1:] if keep.sum() > 1 else H[keep], F[keep][1:] if keep.sum() > 1 else F[keep]
    a = alphas[:, None]
    V = (np.minimum(a, s) - (a * (H * s + F * (1 - s)) + (1 - H) * s)) / (np.minimum(a, s) - a * s)
    return np.max(V, 1)


def sedi(c):
    h, m, f, n = np.moveaxis(c, -1, 0)
    with np.errstate(all="ignore"):
        H = np.clip(h / (h + m), 1e-6, 1 - 1e-6); F = np.clip(f / (f + n), 1e-6, 1 - 1e-6)
        return (np.log(F) - np.log(H) - np.log(1 - F) + np.log(1 - H)) / (np.log(F) + np.log(H) + np.log(1 - F) + np.log(1 - H))


def main():
    import matplotlib.patheffects  # noqa: F401  (used by the Beijing row)
    es = np.load(HERE / "data" / f"extremes_stations_{FM.LABEL}.npz"); CE = es["C"]; ES = [str(s) for s in es["systems"]]
    te = np.load(HERE / "data" / f"tail_events_{FM.LABEL}.npz"); CT = te["C"]; TS = [str(s) for s in te["systems"]]; TR = [str(r) for r in te["regions"]]
    ex = np.load(HERE / "data" / "exposure_2022.npz"); XT = list(ex["thr"]); XR = [str(r) for r in ex["regions"]]
    W, H = DOUBLE, 174 * MM
    fig = plt.figure(figsize=(W, H))
    fx = lambda mm: mm / (W / MM); fy = lambda mm: mm / (H / MM)
    # a: people a 40 km grid hides, per 40 km cell
    k375 = XT.index(37.5); nd = ex["res"].shape[0]
    hid = ex["maps"][1, k375] / nd
    tot = ex["res"][:, XR.index("World"), 2, k375].sum() / nd
    ax = fig.add_axes([fx(8), fy(102), fx(167), fy(68)], projection=ccrs.Robinson())
    ax.set_extent([-180, 180, -58, 84], crs=ccrs.PlateCarree()); ax.spines["geo"].set_visible(False)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#ECECEC", edgecolor="none", zorder=0)
    cm = LinearSegmentedColormap.from_list("hid", ["#FBE3EC", "#F29BBD", "#E0457F", "#C2185B", "#5C0A2C"], N=256)   # hidden = pink (d)
    lat_e = 84.0 - 0.4 * np.arange(361); lon_e = -180.0 + 0.4 * np.arange(901)
    pm = ax.pcolormesh(lon_e, lat_e, np.ma.masked_less(hid, 10), cmap=cm, norm=LogNorm(10, 1e5), transform=ccrs.PlateCarree(),
                       rasterized=True, zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.2, edgecolor="#666666", zorder=3)
    cax = fig.add_axes([fx(14), fy(108), fx(30), fy(1.8)])
    cb = fig.colorbar(pm, cax=cax, orientation="horizontal", extend="both"); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=5, length=1.5, pad=1); cb.ax.xaxis.set_label_position("top")
    cb.set_label("Residents per day above 37.5 µg m$^{-3}$ at 1 km\nbut not in their 40 km cell, per 40 km cell", fontsize=5.3, labelpad=1.5)
    label(fig, fx(1), fy(168), "a")
    # b-d: Beijing, 21 January 2022
    gs = fig.add_gridspec(1, 3, left=0.06, right=0.99, top=94 / 174, bottom=50 / 174, wspace=0.12)
    axs, im, n_seen, n_hid = R1.row1(fig, gs, letters="bcd")
    axs[0].set_title("Daily PM$_{2.5}$ at 1 km (GHAP), 21 Jan 2022", fontsize=6.5, loc="left")
    axs[1].text(0.97, 0.97, "= a perfect 40 km forecast", transform=axs[1].transAxes, ha="right", va="top", fontsize=5.5,
                bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.9), zorder=7)
    for a_ in axs:
        a_.title.set_fontsize(6.5)
    fig.canvas.draw()
    p0, p1 = axs[0].get_position(), axs[1].get_position()
    cax = fig.add_axes([p0.x0 + 0.02, 44 / 174, p1.x1 - p0.x0 - 0.04, 0.007])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both", ticks=[40, 50, 60, 75, 90, 110])
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5, length=1.5, pad=1); cb.ax.axvline(R1.THR, color="k", lw=0.8)
    cb.set_label("Daily PM$_{2.5}$ (µg m$^{-3}$); black line: WHO interim target 1",
                 fontsize=5.5, labelpad=1)
    # e, f: REV at the stations
    alphas = np.logspace(np.log10(0.003), np.log10(0.6), 200)
    SYS = [("Foehn", FM.LABEL, FOEHN, 1.6), ("CAMS", "CAMS", CAMS, 1.0), ("Aurora", "Aurora", AURORA, 1.0)]
    for k, (tier, title) in enumerate(((1, "Warnings of days > local 95th percentile"), (2, "Warnings of days > local 99th percentile"))):
        ax = fig.add_axes([fx(14 + 92 * k), fy(9), fx(74), fy(23)])
        for nm, key, col, lw in SYS:
            for h, ls in ((0, "-"), (2, (0, (3, 1.6)))):
                v = rev_curves(CE[0, tier, h, ES.index(key)], alphas)
                ax.plot(alphas, np.clip(v, -0.1, 1), color=col, lw=lw, ls=ls, label=nm if h == 0 else None, zorder=5 if nm == "Foehn" else 3)
        ax.set_xscale("log"); ax.set_xlim(alphas[0], alphas[-1]); ax.set_ylim(0, 0.4 if tier == 1 else 0.3)
        ax.set_xlabel("Cost/loss ratio", fontsize=6, labelpad=1); clean(ax)
        ax.set_ylabel("Relative economic value", fontsize=6)
        if k == 0:
            ax.text(0.02, 0.97, "solid: 1 day ahead\ndashed: 3 days ahead", transform=ax.transAxes, fontsize=5, va="top", color="#444444")
        else:
            ax.legend(fontsize=5, loc="upper right", frameon=False, handlelength=1.6)
        ax.set_title(title + ", stations", fontsize=6.2, loc="left", pad=3)
        label(fig, fx(2 + 92 * k), fy(35.5), "ef"[k])
    save(fig, "fig_gc4_extremes")
    print(f"hidden {tot / 1e6:.0f} M/day; Beijing seen {n_seen:.1f} hidden {n_hid:.1f}")


if __name__ == "__main__":
    main()
