#!/usr/bin/env python3
"""Fig. 5: Foehn in Aurora's own setup (Foehn-CAMS: trained on the 0.4 deg CAMS analyses, 12-hour steps to 120 h, the inputs of
Aurora's air-pollution model; global1km/foehn_cams.py, scored by eval_cams12h.py). Layout after Aurora's Fig. 2.
a, near-surface PM2.5 of the forecasts initialised at 00 UTC on 1 September 2022, 1 and 5 days ahead: Foehn-CAMS (ensemble mean
   of eight members), Aurora run from the same analyses, the operational CAMS forecast, and the CAMS analysis (the truth of the
   benchmark); shown over land (all systems are scored over the whole domain, 84 N - 60 S).
b, RMSE relative to the CAMS forecast, by region and lead time, for Foehn-CAMS and Aurora (latitude-weighted, May-November 2022,
   214 initialisations).
c, latitude-weighted RMSE by lead time over the whole domain, with Foehn-CAMS given the weather forecast of the previous day.
    python fig_gc5_aurora_setup.py"""
import sys
from pathlib import Path

import cartopy.crs as ccrs
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "global1km"))
from eval_cams12h import AUR, CFC, LAT, NX, RAW, to_domain  # noqa: E402
from gc_style import AURORA, CAMS, FOEHN, land_mask_04, letter_left_of, map_axes, world_coords  # noqa: E402
from nature_style import DOUBLE, HERE, MM, PM_BOUNDS, plt, pm_cmap, save  # noqa: E402

RUNS = HERE.parent / "global1km" / "runs" / "cams04_flow_shift2.6_emis"
D0 = 243                                                                       # 1 September 2022 (0-based day of year)
LEADS = (24, 120)
ROWS = ["Foehn-CAMS", "Aurora", "CAMS forecast", "CAMS analysis"]


def fields():
    w = np.cos(np.deg2rad(LAT))[:, None] * np.ones((1, NX))
    fd = list(np.load(RUNS / "cams12h" / "doys.npy"))
    E = np.asarray(np.load(RUNS / "cams12h" / "pred.npy", mmap_mode="r")[fd.index(D0)], np.float32).mean(0)
    for p in sorted(AUR.glob("part_*.npz")):
        z = np.load(p)
        if D0 in list(z["days"]):
            A = to_domain(z["pred"][list(z["days"]).index(D0)].astype(np.float32), z["lat"], z["lon"])
    zc = np.load(CFC / "2022_09.npz"); Cf = to_domain(zc["pm25"][0].astype(np.float32), zc["lat"], zc["lon"])
    out = {}
    for L in LEADS:
        k = L // 12 - 1; vt = 2 * D0 + L // 12
        zt = np.load(RAW / f"{vt // 2:03d}.npz"); T = to_domain(zt["surf"][:, 5] * 1e9, zt["lat"], zt["lon"])[vt % 2].astype(np.float32)
        out[L] = {"Foehn-CAMS": E[k], "Aurora": A[k], "CAMS forecast": Cf[k], "CAMS analysis": T}
        for nm in ROWS[:3]:
            out[L][nm + "_rmse"] = float(np.sqrt((w * (out[L][nm] - T) ** 2).sum() / w.sum()))
    return out


def main():
    cmap, norm = pm_cmap()
    land = land_mask_04()
    F = fields()
    gz = np.load(HERE / "data" / "aurora_grid.npz"); GS = [str(x) for x in gz["systems"]]; GR = [str(x) for x in gz["regions"]]
    W, H = DOUBLE, 150 * MM
    fig = plt.figure(figsize=(W, H))
    fx = lambda mm: mm / (W / MM); fy = lambda mm: mm / (H / MM)
    # ---- a: maps, 2 lead times x 4 systems (top band)
    mw, mh, top = 40.5, 21.0, 150 - 8
    cols = ["Foehn-CAMS", "Aurora", "CAMS forecast", "CAMS analysis"]
    amaps, atitle = [], None
    for i, L in enumerate(LEADS):
        vt = np.datetime64("2022-09-01T00") + np.timedelta64(L, "h")
        for j, nm in enumerate(cols):
            x0, y0 = 14 + j * (mw + 1.8), top - (i + 1) * mh - i * 1.5
            ax = map_axes(fig, [fx(x0), fy(y0), fx(mw), fy(mh)], land=False)
            ax.imshow(np.where(land, np.maximum(F[L][nm], 0), np.nan), origin="upper", extent=(-180, 180, -60.2, 84.2),
                      transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, interpolation="nearest", regrid_shape=1200, zorder=1)
            world_coords(ax, left=j == 0, right=j == len(cols) - 1, bottom=i == len(LEADS) - 1, size=4.6)   # outer maps labelled
            if nm != "CAMS analysis":
                ax.text(0.02, 0.10, f"RMSE {F[L][nm + '_rmse']:.1f}", transform=ax.transAxes, fontsize=4.8, zorder=6,
                        fontweight="bold" if nm == "Foehn-CAMS" else "normal")
            if i == 0 and j == 0:
                atitle = ax.title
            if i == 0:
                ax.set_title(nm + (" (truth)" if nm == "CAMS analysis" else ""), fontsize=6.2, pad=2,
                             fontweight="bold" if nm == "Foehn-CAMS" else "normal")
            if j == 0:
                amaps.append(ax)
                ax.text(-0.15, 0.5, f"+{L // 24} d\n{int(str(vt)[8:10])} Sep", transform=ax.transAxes, rotation=90, ha="right",
                        va="center", fontsize=5.8)
    yb = top - 2 * mh - 1.5 - 7.5
    cax = fig.add_axes([fx(40), fy(yb), fx(110), fy(1.5)])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="max", spacing="uniform",
                      ticks=PM_BOUNDS)
    cb.ax.set_xticklabels([f"{t:g}" for t in PM_BOUNDS]); cb.ax.tick_params(labelsize=5, length=1.5, pad=1); cb.outline.set_linewidth(0.4)
    cb.set_label("PM$_{2.5}$ near the surface (µg m$^{-3}$), forecasts initialised at 00 UTC on 1 September 2022, shown over land",
                 fontsize=5.8, labelpad=1)
    # ---- b: RMSE, CRPS and spread-skill ratio x regions (MOSAIC / WeatherBench layout)
    STY = {"Climatology": ("#9A9A9A", (0, (3, 2)), None, 0.9), "CAMS forecast": (CAMS, "-", "s", 0.9), "Aurora": (AURORA, "-", "D", 0.9),
           "Foehn-CAMS": (FOEHN, "-", "o", 1.4)}
    k00 = np.arange(1, len(gz["leads_h"]), 2)                                       # 24, 48, ..., 120 h: valid at 00 UTC, as the inits
    Lh = gz["leads_h"][k00] / 24
    gw, gh, gxs, gys, y_top = 31.0, 30.0, 5.2, 6.0, yb - 9
    rows = [("rmse", "RMSE (µg m$^{-3}$)"), ("crps", "CRPS (µg m$^{-3}$)")]
    FC = ("CAMS forecast", "Aurora", "Foehn-CAMS")
    bcol, btitle = [], None
    for ri, (key, ylab) in enumerate(rows):
        for ci, r in enumerate(GR):
            ax = fig.add_axes([fx(14 + ci * (gw + gxs)), fy(y_top - (ri + 1) * gh - ri * gys), fx(gw), fy(gh)])
            g = GR.index(r)
            if key == "ssr":
                ax.axhline(1.0, color="#111111", lw=0.9, ls=(0, (3, 2)))
                ax.plot(Lh, gz["ssr"][k00, g], color=FOEHN, lw=1.4, marker="o", ms=2.0)
                ax.set_ylim(0.4, 1.2)
            else:
                V = {sn: gz[key][GS.index(sn), k00, g] for sn in ("Climatology",) + FC}
                for sn in ("Climatology",) + FC:
                    col, ls, mk, lw = STY[sn]
                    ax.plot(Lh, V[sn], color=col, ls=ls, marker=mk, ms=1.8, lw=lw, label=sn, zorder=6 if sn == "Foehn-CAMS" else 3)
                fmin = min(V[sn].min() for sn in FC); fmax = max(V[sn].max() for sn in FC); cmax = V["Climatology"].max()
                ymin = fmin - 0.15 * (fmax - fmin)
                if key == "rmse":                  # room for the zoom (axes x <= 0.40, y 0.53-0.86) above the curves, climatology above it
                    xr = 0.7 + 0.40 * 4.6
                    need = ymin + (max(np.interp(xr, Lh, V[sn]) for sn in FC) - ymin) / 0.50
                    top_ok = ymin + (V["Climatology"].min() - ymin) / 0.90
                    ymax = max(need, cmax + 0.05 * (cmax - ymin))
                    if ymax > top_ok and need <= top_ok:
                        ymax = top_ok
                else:
                    ymax = max(cmax, fmax) + 0.08 * (max(cmax, fmax) - ymin)
                ax.set_ylim(ymin, ymax)
                if key == "rmse":
                    ins = ax.inset_axes([0.07, 0.53, 0.33, 0.33])                      # zoom on days 3-5, upper left
                    for sn in FC:
                        col, ls, mk, lw = STY[sn]
                        ins.plot(Lh[2:], V[sn][2:], color=col, ls=ls, marker=mk, ms=1.5, lw=lw * 0.9, zorder=6 if sn == "Foehn-CAMS" else 3)
                    vv = np.concatenate([V[sn][2:] for sn in FC]); pad = 0.1 * (vv.max() - vv.min())
                    ins.set_xlim(2.8, 5.2); ins.set_ylim(vv.min() - pad, vv.max() + pad)
                    ins.set_xticks([3, 4, 5]); ins.tick_params(labelsize=3.8, length=1, pad=0.5)
                    for s_ in ins.spines.values():
                        s_.set_linewidth(0.4)
            ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xlim(0.7, 5.3); ax.tick_params(labelsize=4.8, length=1.5, pad=1)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            ax.grid(lw=0.3, color="#E6E6E6"); ax.set_axisbelow(True)
            if ri == 0 and ci == 0:
                btitle = ax.title
            if ci == 0:
                bcol.append(ax)
            if ri == 0:
                ax.set_title(r, fontsize=6.3, pad=2, fontweight="bold" if r == "Global" else "normal")
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=5.6, labelpad=1)
            if ri == len(rows) - 1:
                ax.set_xlabel("Lead time (days)", fontsize=5.6, labelpad=1)
            else:
                ax.set_xticklabels([])
            if ri == 0 and ci == 0:
                hl = ax.get_legend_handles_labels()
    fig.legend(*hl, fontsize=5.4, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False, handlelength=2.2, columnspacing=1.4)
    fig.canvas.draw()                                         # a and b on the lines of their titles, 1 mm left of their labels
    letter_left_of(fig, amaps, "a", atitle)
    letter_left_of(fig, bcol, "b", btitle)
    save(fig, "fig_gc5_aurora_setup")
    print({L: {k: round(v, 2) for k, v in F[L].items() if k.endswith("_rmse")} for L in LEADS})


if __name__ == "__main__":
    main()
