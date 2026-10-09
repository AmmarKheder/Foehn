#!/usr/bin/env python3
"""Fig. 2: Foehn against the operational forecast and the AI foundation model.
a, the stations of six official networks (evaluation only), coloured by the change of Foehn's error against the CAMS
   forecast issued at the same time, one day ahead, 2022 (blue: Foehn better; compare_fair.py);
b, Foehn trained like Aurora on the CAMS analyses (0.4 deg, 12-hour steps): change of error against Aurora on Aurora's own
   benchmark, three days ahead, per 0.4 deg cell (May-November 2022, truth: CAMS analysis; eval_cams12h.py);
c, error at the stations by lead time, identical station-days, every forecast issued at the end of day d; persistence of
   GHAP (the information Foehn starts from) and, for reference, of the station's own measurement;
d, error on Aurora's benchmark by lead time (latitude-weighted, 12-hour steps).
    python fig2_skill.py"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from nature_style import COLOR, DOUBLE, HERE, MM, plt, save

BLUE_RED = ["#08519C", "#3182BD", "#6BAED6", "#BDD7E7", "#F7F7F7", "#FCBBA1", "#FB6A4A", "#CB181D", "#67000D"]
B_DIFF = [-60, -40, -25, -15, -5, 5, 15, 25, 40, 60]
LINES = {"Foehn": (COLOR["ours"], "-", "o", 1.5, 3.2), "CAMS": (COLOR["cams"], "-", "s", 1.0, 2.6),
         "Aurora": (COLOR["aurora"], "-", "D", 1.0, 2.6), "Persistence": ("#111111", (0, (3, 2)), "", 0.9, 0),
         "Station persistence": ("#9A9A9A", (0, (1, 1.5)), "", 0.9, 0)}


def map_axes(fig, rect, extent):
    ax = fig.add_axes(rect, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#EDEDED", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.25, edgecolor="#555555", zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.15, edgecolor="#999999", zorder=4)
    for s_ in ax.spines.values():
        s_.set_linewidth(0.5); s_.set_edgecolor("#222222")
    return ax


def badge(ax, txt):
    ax.text(0.012, 0.04, txt, transform=ax.transAxes, fontsize=6.5, fontweight="bold", color="#08519C", zorder=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.9))


def main():
    fz = np.load(HERE / "data" / "compare_fair_e02.npz")
    SYS = [str(s) for s in fz["systems"]]
    iF, iC, iA = SYS.index("e02 (1 km)"), SYS.index("CAMS (run d+1, equal issuance)"), SYS.index("Aurora (00 UTC d+1, equal issuance)")
    iP, iG = SYS.index("persistence (day d)"), SYS.index("GHAP persistence (day d)")
    SST, lat, lon, S = fz["SST"], fz["lat"], fz["lon"], fz["S"]
    cz = np.load(HERE / "data" / "cams12h_scores_foehn_emis.npz")
    CS = [str(s) for s in cz["systems"]]
    cmap, norm = ListedColormap(BLUE_RED), BoundaryNorm(B_DIFF, len(BLUE_RED))
    cmap.set_under(BLUE_RED[0]); cmap.set_over(BLUE_RED[-1])
    W, H = DOUBLE, 100 * MM
    fig = plt.figure(figsize=(W, H))
    fx = lambda mm: mm / (W / MM); fy = lambda mm: mm / (H / MM)
    # a: stations, one day ahead, against CAMS issued at the same time
    ax = map_axes(fig, [fx(2), fy(70), fx(124), fy(25.3)], (-128, 146, 6, 62))
    n = SST[iF, 0, :, 1]; ok = n >= 30
    rf = np.sqrt(SST[iF, 0, :, 0] / np.maximum(n, 1)); rc = np.sqrt(SST[iC, 0, :, 0] / np.maximum(SST[iC, 0, :, 1], 1))
    with np.errstate(all="ignore"):
        ch = 100 * (rf / rc - 1)
    o = np.argsort(np.abs(ch[ok]))                                               # strongest changes on top
    ax.scatter(lon[ok][o], lat[ok][o], c=ch[ok][o], cmap=cmap, norm=norm, s=2.2, linewidths=0, transform=ccrs.PlateCarree(),
               zorder=5, rasterized=True)
    badge(ax, f"Foehn more accurate at {100 * np.mean(ch[ok] < 0):.0f}% of {ok.sum():,} stations")
    fig.text(fx(2), fy(96.5), "$\\bf{a}$  Against the CAMS forecast issued at the same time, 1 day ahead (stations)", fontsize=6.5,
             va="bottom")
    # b: Aurora's benchmark, 72 h, per 0.4 deg cell
    ab = map_axes(fig, [fx(2), fy(13), fx(124), fy(47.5)], (-180, 180, -58, 80))
    k72 = list(cz["leads_h"]).index(72)
    rfm = np.sqrt(cz["mse_map"][CS.index("Foehn-CAMS (mean)"), k72]); ram = np.sqrt(cz["mse_map"][CS.index("Aurora"), k72])
    with np.errstate(all="ignore"):
        chm = 100 * (rfm / np.maximum(ram, 1e-3) - 1)
    lat_e = 84.0 - 0.4 * np.arange(362) + 0.2; lon_e = -180.0 + 0.4 * np.arange(901) - 0.2
    ab.pcolormesh(lon_e, lat_e, np.ma.masked_invalid(chm), cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), rasterized=True, zorder=2)
    wl = np.cos(np.deg2rad(84.0 - 0.4 * np.arange(361)))[:, None]
    badge(ab, f"Foehn more accurate over {100 * np.nansum(wl * (chm < 0)) / np.nansum(wl * np.isfinite(chm)):.0f}% of the Earth")
    fig.text(fx(2), fy(62.5), "$\\bf{b}$  Against Aurora on its own benchmark, same training data, 3 days ahead", fontsize=6.5, va="bottom")
    cax = fig.add_axes([fx(36), fy(3.0), fx(56), fy(1.6)])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=B_DIFF, extend="both")
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5, length=1.5, pad=1)
    cb.ax.set_title("Change of Foehn's RMSE (%), blue: Foehn better", fontsize=5.5, pad=2)
    # c: stations by lead time
    ac = fig.add_axes([fx(141), fy(58), fx(40), fy(37)])
    g = S.shape[0] - 1
    for nm, i in (("Station persistence", iP), ("Persistence", iG), ("CAMS", iC), ("Aurora", iA), ("Foehn", iF)):
        col, ls, mk, lw, ms = LINES[nm]
        ac.plot(range(1, 5), [np.sqrt(S[g, h, i, 2] / S[g, h, i, 0]) for h in range(4)], color=col, ls=ls, marker=mk or None,
                ms=ms, lw=lw, label={"Persistence": "GHAP persistence"}.get(nm, nm), zorder=6 if nm == "Foehn" else 3)
    ac.set_xticks(range(1, 5)); ac.set_xlabel("Lead time (days)", fontsize=6, labelpad=1)
    ac.set_ylabel("RMSE at the stations (µg m$^{-3}$)", fontsize=6, labelpad=1.5)
    ac.tick_params(labelsize=5.5, length=2); ac.legend(fontsize=5, loc="lower right", frameon=False, handlelength=1.8)
    for s_ in ("top", "right"):
        ac.spines[s_].set_visible(False)
    fig.text(fx(132), fy(96.5), "$\\bf{c}$", fontsize=6.5, va="bottom")
    # d: Aurora's benchmark by lead time
    ad = fig.add_axes([fx(141), fy(10), fx(40), fy(37)])
    L = cz["leads_h"]; g0 = list(cz["regions"]).index("Global")
    for nm, key in (("Persistence", "persistence"), ("CAMS", "CAMS forecast"), ("Aurora", "Aurora"), ("Foehn", "Foehn-CAMS (mean)")):
        col, ls, mk, lw, ms = LINES[nm]
        ad.plot(L, cz["rmse"][CS.index(key), :, g0], color=col, ls=ls, marker=mk or None, ms=ms * 0.8, lw=lw, label=nm,
                zorder=6 if nm == "Foehn" else 3)
    ad.set_xticks([24, 48, 72, 96, 120]); ad.set_xlabel("Lead time (hours)", fontsize=6, labelpad=1)
    ad.set_ylabel("RMSE against the CAMS analysis (µg m$^{-3}$)", fontsize=6, labelpad=1.5)
    ad.tick_params(labelsize=5.5, length=2)
    for s_ in ("top", "right"):
        ad.spines[s_].set_visible(False)
    fig.text(fx(132), fy(50), "$\\bf{d}$", fontsize=6.5, va="bottom")
    save(fig, "fig2_skill")
    r = lambda i, h: np.sqrt(S[g, h, i, 2] / S[g, h, i, 0])
    print("stations J+1..4 Foehn/CAMS/Aurora/GHAPpers/stationpers:",
          [f"{r(iF, h):.1f}/{r(iC, h):.1f}/{r(iA, h):.1f}/{r(iG, h):.1f}/{r(iP, h):.1f}" for h in range(4)])


if __name__ == "__main__":
    main()
