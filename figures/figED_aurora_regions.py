#!/usr/bin/env python3
"""Extended Data figure: Foehn in Aurora's setup, region by region. For six regions, near-surface PM2.5 three days ahead of
the forecasts initialised at 00 UTC on 1 September 2022 (valid 4 September 00 UTC): the CAMS analysis (truth), Foehn-CAMS
(ensemble mean), Aurora and the CAMS forecast, over land, with the latitude-weighted RMSE of the map over the region; and the
latitude-weighted RMSE by lead time over May-November 2022 (214 initialisations) for the three forecasts and the climatology
of the CAMS analyses (2018-2021); beside it, the zoom on days 3-5 (shaded) with the change of Foehn-CAMS's RMSE relative to
Aurora three days ahead and its 95% interval (7-day moving-block bootstrap over initialisations).
    python figED_aurora_regions.py"""
import sys
from pathlib import Path

import cartopy.crs as ccrs
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "global1km"))
from eval_cams12h import LAT, NX, REGIONS  # noqa: E402
from fig_gc3_scorecards import boot_inits  # noqa: E402
from fig_gc5_aurora_setup import fields  # noqa: E402
import fig_gc5_aurora_setup as F5  # noqa: E402
from gc_style import AURORA, CAMS, FOEHN, geo_ticks, land_mask_04, map_axes  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402
from nature_style import DOUBLE, HERE, MM, plt, pm_cmap, save  # noqa: E402

REG = ["Europe", "North America", "South America", "Africa", "South Asia", "East Asia"]
CLIMC = "#9A9A9A"


def main():
    F5.LEADS = (72,)
    Fm = fields()[72]
    cmap, norm = pm_cmap(); land = land_mask_04()
    cz = np.load(HERE / "data" / "cams12h_scores_foehn_emis.npz"); CS = [str(s) for s in cz["systems"]]; CR = [str(r) for r in cz["regions"]]
    cl = np.load(HERE / "data" / "cams12h_clim.npz")
    Lh = cz["leads_h"] / 24; k72 = list(cz["leads_h"]).index(72)
    w = np.cos(np.deg2rad(LAT))[:, None] * np.ones((1, NX))
    LO2 = -180.0 + 0.4 * np.arange(NX)
    W, H = DOUBLE, 168 * MM
    fig = plt.figure(figsize=(W, H))
    fx = lambda mm: mm / (W / MM); fy = lambda mm: mm / (H / MM)
    cols = ["CAMS analysis", "Foehn-CAMS", "Aurora", "CAMS forecast"]
    mw, mh, top = 29.5, 21.0, 168 - 8
    for i, r in enumerate(REG):
        N, S, Wl, E = REGIONS[r]
        y0 = top - (i + 1) * mh - i * 4.6                                      # room for the longitudes under each row
        box = (LAT[:, None] <= N) & (LAT[:, None] > S) & (LO2[None, :] >= Wl) & (LO2[None, :] < E)
        T = Fm["CAMS analysis"]
        for j, nm in enumerate(cols):
            ax = map_axes(fig, [fx(16 + j * (mw + 1.5)), fy(y0), fx(mw), fy(mh)], proj="plate", extent=(Wl, E, S, N), land=False)
            ax.imshow(np.where(land, np.maximum(Fm[nm], 0), np.nan), origin="upper", extent=(-180, 180, -60.2, 84.2),
                      transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, interpolation="nearest", zorder=1)
            geo_ticks(ax, left=j == 0, bottom=True, size=4.6)                     # each row its own region
            if nm != "CAMS analysis":
                e = np.sqrt((w * box * (Fm[nm] - T) ** 2).sum() / (w * box).sum())
                ax.text(0.03, 0.05, f"RMSE {e:.1f}", transform=ax.transAxes, fontsize=4.8, zorder=6,
                        fontweight="bold" if nm == "Foehn-CAMS" else "normal", bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.8))
            if i == 0:
                ax.set_title(nm, fontsize=6.2, pad=2, fontweight="bold" if nm == "Foehn-CAMS" else "normal")
            if j == 0:
                ax.text(0, 0.5, r, transform=offset_copy(ax.transAxes, fig=fig, x=-18.5, y=0, units="points"), rotation=90,
                        ha="right", va="center", fontsize=6, fontweight="bold")      # 6.5 mm left of the map, clear of its latitudes
        g = CR.index(r)
        SY = (("CAMS forecast", "CAMS forecast", CAMS, 0.9, "s"), ("Aurora", "Aurora", AURORA, 0.9, "D"), ("Foehn-CAMS (mean)", "Foehn-CAMS", FOEHN, 1.4, "o"))
        # main curves (all leads, climatology) and, beside them, the zoom on days 3-5 (shaded in the main panel)
        ac = fig.add_axes([fx(145), fy(y0 + 1.5), fx(20), fy(mh - 3)])
        ac.axvspan(2.8, 5.2, color="#F2F2F2", zorder=0, lw=0)
        for key, nm, col, lw, mk in SY:
            ac.plot(Lh, cz["rmse"][CS.index(key), :, g], color=col, lw=lw, marker=mk, ms=1.3, label=nm, zorder=6 if nm == "Foehn-CAMS" else 3)
        ac.plot(Lh, cl["rmse"][:, list(cl["regions"]).index(r)], color=CLIMC, lw=0.9, ls=(0, (3, 2)), label="Climatology", zorder=2)
        az = fig.add_axes([fx(168), fy(y0 + 1.5), fx(10.5), fy(mh - 3)])
        vv = []
        for key, nm, col, lw, mk in SY:
            v = cz["rmse"][CS.index(key), :, g]; sel = Lh >= 2.9
            az.plot(Lh[sel], v[sel], color=col, lw=lw * 0.9, marker=mk, ms=1.3, zorder=6 if col == FOEHN else 3); vv.append(v[sel])
        vv = np.concatenate(vv); rg = vv.max() - vv.min()
        az.set_xlim(2.75, 5.25); az.set_ylim(vv.min() - 0.08 * rg, vv.max() + 0.55 * rg); az.set_xticks([3, 4, 5])
        az.yaxis.tick_right(); az.set_facecolor("#F2F2F2")
        pt, lo, hi = boot_inits(cz["rmse_per_init"][CS.index("Foehn-CAMS (mean)"), :, k72, g], cz["rmse_per_init"][CS.index("Aurora"), :, k72, g])
        cc = "#08519C" if hi < 0 else "#444444"
        az.text(0.5, 0.97, f"{pt:+.0f}%".replace("-", "−"), transform=az.transAxes, ha="center", va="top", fontsize=5.6, fontweight="bold", color=cc)
        az.text(0.5, 0.80, f"[{lo:+.0f}, {hi:+.0f}]".replace("-", "−"), transform=az.transAxes, ha="center", va="top", fontsize=4.3, color=cc)
        cmax = cl["rmse"][:, list(cl["regions"]).index(r)].max(); fmin = min(cz["rmse"][CS.index(k_), :, g].min() for k_, *_ in SY)
        ac.set_ylim(fmin - 0.08 * (cmax - fmin), cmax + 0.06 * (cmax - fmin))
        for a_ in (ac, az):
            a_.tick_params(labelsize=4.6, length=1.3, pad=0.8)
            for s_ in ("top", "right") if a_ is ac else ("top", "left"):
                a_.spines[s_].set_visible(False)
            a_.grid(lw=0.3, color="#E3E3E3"); a_.set_axisbelow(True)
        ac.set_xticks([1, 2, 3, 4, 5]); ac.set_xlim(0.3, 5.25); ac.set_ylabel("RMSE (µg m$^{-3}$)", fontsize=5, labelpad=1)
        if i == len(REG) - 1:
            ac.set_xlabel("Lead time (days)", fontsize=5.5, labelpad=1)
            ac.legend(fontsize=4.8, loc="upper left", bbox_to_anchor=(-0.35, -0.42), ncol=2, frameon=False, handlelength=1.8, columnspacing=0.8)
        if i == 0:
            ac.set_title("May–November 2022", fontsize=5.8, pad=2)
            az.set_title("Days 3–5", fontsize=5.8, pad=2)
        fig.text(fx(1), fy(y0 + mh + 0.6), "abcdef"[i], fontsize=8, fontweight="bold", va="bottom")
    cax = fig.add_axes([fx(30), fy(3.0), fx(100), fy(1.6)])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="max", spacing="uniform",
                      ticks=list(norm.boundaries))
    cb.ax.set_xticklabels([f"{t:g}" for t in norm.boundaries]); cb.ax.tick_params(labelsize=5, length=1.5, pad=1); cb.outline.set_linewidth(0.4)
    cb.set_label("PM$_{2.5}$ near the surface on 4 September 2022 00 UTC, forecasts issued 3 days earlier (µg m$^{-3}$)", fontsize=5.8, labelpad=1)
    save(fig, "figED_aurora_regions")


if __name__ == "__main__":
    main()
