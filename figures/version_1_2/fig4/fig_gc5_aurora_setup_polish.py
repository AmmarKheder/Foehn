#!/usr/bin/env python3
"""Fig. 5: Foehn in Aurora's own setup (Foehn-CAMS: trained on the 0.4 deg CAMS analyses, 12-hour steps to 120 h, the inputs of
Aurora's air-pollution model; global1km/foehn_cams.py, scored by eval_cams12h.py). Layout after Aurora's Fig. 2.
a, near-surface PM2.5 of the forecasts initialised at 00 UTC on 1 September 2022, 1 and 5 days ahead: Foehn-CAMS (ensemble mean
   of eight members), Aurora run from the same analyses, the operational CAMS forecast, and the CAMS analysis (the truth of the
   benchmark); shown over land (all systems are scored over the whole domain, 84 N - 60 S).
b, RMSE relative to the CAMS forecast, by region and lead time, for Foehn-CAMS and Aurora (latitude-weighted, May-November 2022,
   214 initialisations).
c, latitude-weighted RMSE by lead time over the whole domain, with Foehn-CAMS given the weather forecast of the previous day.
    python fig_gc5_aurora_setup.py

POLISH (2026-11-21, figs_show/polish/fig4): copy of experiments/figures/fig_gc5_aurora_setup.bak_20261014_letters.py, the
version that renders work/figs/fig_gc5_aurora_setup.pdf pixel for pixel (150 dpi). Execution only: every change is marked
'POLISH:'; data, numbers, panels, dates, axis and colour ranges and all words are those of the present figure.
    $FOEHN_ENVS/world/bin/python fig_gc5_aurora_setup_polish.py   (writes the PDF next to this file)"""
import os  # POLISH: atomic save next to this script
import sys
from pathlib import Path

import cartopy.crs as ccrs
import numpy as np

FIGDIR = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/figures"))      # POLISH: the copy lives outside FIG
sys.path[:0] = [str(FIGDIR), str(FIGDIR.parent / "global1km")]                         # POLISH: (was parents[1] / "global1km")
from eval_cams12h import AUR, CFC, LAT, NX, RAW, to_domain  # noqa: E402
from gc_style import AURORA, CAMS, FOEHN, land_mask_04, letter_left_of, map_axes, world_coords  # noqa: E402  POLISH: letter_left_of
from matplotlib.ticker import MaxNLocator  # noqa: E402  POLISH: keep the tick values of the present figure
from nature_style import HERE, MM, PM_BOUNDS, plt, pm_cmap  # noqa: E402  POLISH: own save below (stored outputs untouched)

RUNS = HERE.parent / "global1km" / "runs" / "cams04_flow_shift2.6_emis"
D0 = 243                                                                       # 1 September 2022 (0-based day of year)
LEADS = (24, 120)
ROWS = ["Foehn-CAMS", "Aurora", "CAMS forecast", "CAMS analysis"]
POLISH_OUT = Path(__file__).resolve().parent                                   # POLISH: output folder


def save(fig, name):                                                           # POLISH: PDF on the designed 180 mm page,
    out = POLISH_OUT / f"{name}.pdf"; tmp = out.with_name("." + out.name + ".tmp")   # written via temp file + fsync + rename
    fig.savefig(tmp, format="pdf", bbox_inches=fig.bbox_inches)
    fd = os.open(tmp, os.O_RDONLY); os.fsync(fd); os.close(fd); os.replace(tmp, out)
    print("saved", out)


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
    # POLISH: one grid in printed millimetres (page = 180 mm, the Nat Commun double column, so set sizes are printed sizes;
    # was 183 mm drawn 189 mm wide, printed at 95 %). a and b share the left and right frame edges X0, X1; the map axes take
    # the Robinson aspect of 84N-60S (no empty band between the map rows); positions are measured from the top of the page.
    X0, X1 = 12.5, 175.8                                                       # POLISH: frame edges (labels hang outside)
    gm = 1.2; mw = (X1 - X0 - 3 * gm) / 4; mh = mw * 0.4322; t_map = 3.7; gmr = 2.0   # POLISH: maps, gutters, row gap
    gx, gh, gys = 5.0, 32.0, 3.0; gw = (X1 - X0 - 4 * gx) / 5                  # POLISH: b panels on the same grid
    cb_top = t_map + 2 * mh + gmr + 5.8                                         # POLISH: colour bar under the longitudes
    leg_c = cb_top + 1.6 + 5.0 + 3.5 + 1.2                                      # POLISH: centre of the legend line above b
    y_top = cb_top + 16.7                                                       # POLISH: top of the b frames
    Hmm = y_top + 2 * gh + gys + 5.7                                            # POLISH: page height
    W, H = 180 * MM, Hmm * MM                                                   # POLISH: (was DOUBLE, 150 mm)
    fig = plt.figure(figsize=(W, H))
    fx = lambda mm: mm / (W / MM); fy = lambda mm: mm / (H / MM)
    ft = lambda mm: (Hmm - mm) / Hmm                                            # POLISH: figure y of a distance from the top
    # ---- a: maps, 2 lead times x 4 systems (top band)
    cols = ["Foehn-CAMS", "Aurora", "CAMS forecast", "CAMS analysis"]
    amaps, atitle = [], None                                                    # POLISH: for letter_left_of
    for i, L in enumerate(LEADS):
        vt = np.datetime64("2022-09-01T00") + np.timedelta64(L, "h")
        for j, nm in enumerate(cols):
            x0, ytp = X0 + j * (mw + gm), t_map + i * (mh + gmr)                # POLISH: grid position (top edge)
            ax = map_axes(fig, [fx(x0), ft(ytp + mh), fx(mw), fy(mh)], land=False)
            ax.imshow(np.where(land, np.maximum(F[L][nm], 0), np.nan), origin="upper", extent=(-180, 180, -60.2, 84.2),
                      transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, interpolation="nearest", regrid_shape=1200, zorder=1)
            world_coords(ax, left=j == 0, right=j == len(cols) - 1, bottom=i == len(LEADS) - 1, size=5.0)   # POLISH: 4.6 -> 5 pt
            if nm != "CAMS analysis":                                           # POLISH: 4.8 -> 5.5 pt, inside the map frame
                ax.text(0.07, 0.09, f"RMSE {F[L][nm + '_rmse']:.1f}", transform=ax.transAxes, fontsize=5.5, zorder=6,
                        fontweight="bold" if nm == "Foehn-CAMS" else "normal")
            if i == 0:
                ax.set_title(nm + (" (truth)" if nm == "CAMS analysis" else ""), fontsize=6.5, pad=2,   # POLISH: 6.2 -> 6.5 pt
                             fontweight="bold" if nm == "Foehn-CAMS" else "normal")
                if j == 0:
                    atitle = ax.title                                           # POLISH: letter a on this baseline
            if j == 0:
                amaps.append(ax)                                                # POLISH
                ax.text(-4.4 / mw, 0.5, f"+{L // 24} d\n{int(str(vt)[8:10])} Sep", transform=ax.transAxes, rotation=90, ha="right",
                        va="center", fontsize=6.0)                              # POLISH: 4.4 mm left of the frame; 5.8 -> 6 pt
    cbw = 104.0                                                                 # POLISH: colour bar centred on the map block
    cax = fig.add_axes([fx((X0 + X1 - cbw) / 2), ft(cb_top + 1.6), fx(cbw), fy(1.6)])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="max", spacing="uniform",
                      ticks=PM_BOUNDS)
    cb.ax.set_xticklabels([f"{t:g}" for t in PM_BOUNDS]); cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.4, pad=1)  # POLISH: 5 -> 5.5 pt
    cb.outline.set_linewidth(0.4)
    cb.set_label("PM$_{2.5}$ near the surface (µg m$^{-3}$), forecasts initialised at 00 UTC on 1 September 2022, shown over land",
                 fontsize=6.0, labelpad=1.5)                                    # POLISH: 5.8 -> 6 pt
    # ---- b: RMSE, CRPS and spread-skill ratio x regions (MOSAIC / WeatherBench layout)
    # POLISH: Foehn-CAMS first (1.6 pt, gc_style.SYS), references lighter (0.8 pt) with smaller markers, no marker rims
    STY = {"Climatology": ("#9A9A9A", (0, (3, 2)), None, 0.8), "CAMS forecast": (CAMS, "-", "s", 0.8), "Aurora": (AURORA, "-", "D", 0.8),
           "Foehn-CAMS": (FOEHN, "-", "o", 1.6)}
    MS = {"Climatology": 0, "CAMS forecast": 2.3, "Aurora": 2.5, "Foehn-CAMS": 3.0}   # POLISH: marker sizes (were 1.8 + 1 pt rim)
    k00 = np.arange(1, len(gz["leads_h"]), 2)                                       # 24, 48, ..., 120 h: valid at 00 UTC, as the inits
    Lh = gz["leads_h"][k00] / 24
    rows = [("rmse", "RMSE (µg m$^{-3}$)"), ("crps", "CRPS (µg m$^{-3}$)")]
    FC = ("CAMS forecast", "Aurora", "Foehn-CAMS")
    bcol, btitle = [], None                                                     # POLISH: for letter_left_of
    for ri, (key, ylab) in enumerate(rows):
        for ci, r in enumerate(GR):
            ax = fig.add_axes([fx(X0 + ci * (gw + gx)), ft(y_top + (ri + 1) * gh + ri * gys), fx(gw), fy(gh)])   # POLISH: grid
            g = GR.index(r)
            if key == "ssr":
                ax.axhline(1.0, color="#111111", lw=0.9, ls=(0, (3, 2)))
                ax.plot(Lh, gz["ssr"][k00, g], color=FOEHN, lw=1.4, marker="o", ms=2.0)
                ax.set_ylim(0.4, 1.2)
            else:
                V = {sn: gz[key][GS.index(sn), k00, g] for sn in ("Climatology",) + FC}
                for sn in ("Climatology",) + FC:
                    col, ls, mk, lw = STY[sn]
                    ax.plot(Lh, V[sn], color=col, ls=ls, marker=mk, ms=MS[sn], mew=0, lw=lw, label=sn,   # POLISH: MS, mew=0
                            zorder=6 if sn == "Foehn-CAMS" else 3)
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
                ax.yaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))   # POLISH: the ticks of the present figure
                if key == "rmse":                  # POLISH: zoom placed clear of every curve with its 5 pt labels (same ranges)
                    ins = ax.inset_axes([0.14, 0.605, 0.23, 0.255])                    # zoom on days 3-5, upper left
                    for sn in FC:
                        col, ls, mk, lw = STY[sn]
                        ins.plot(Lh[2:], V[sn][2:], color=col, ls=ls, marker=mk, ms=MS[sn] * 0.85, mew=0, lw=lw * 0.9,   # POLISH
                                 zorder=6 if sn == "Foehn-CAMS" else 3)
                    vv = np.concatenate([V[sn][2:] for sn in FC]); pad = 0.1 * (vv.max() - vv.min())
                    ins.set_xlim(2.8, 5.2); ins.set_ylim(vv.min() - pad, vv.max() + pad)
                    ins.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))  # POLISH: ticks of the present figure
                    ins.set_xticks([3, 4, 5]); ins.tick_params(labelsize=5.0, length=1, width=0.4, pad=0.8)   # POLISH: 3.8 -> 5 pt
                    for s_ in ins.spines.values():
                        s_.set_linewidth(0.4)
            ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xlim(0.7, 5.3); ax.tick_params(labelsize=5.5, length=1.5, pad=1)  # POLISH: 4.8 -> 5.5 pt
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            ax.grid(lw=0.3, color="#E6E6E6"); ax.set_axisbelow(True)
            if ri == 0:
                ax.set_title(r, fontsize=6.5, pad=2, fontweight="bold" if r == "Global" else "normal")   # POLISH: 6.3 -> 6.5 pt
                if ci == 0:
                    btitle = ax.title                                           # POLISH: letter b on this baseline
            if ci == 0:
                bcol.append(ax)                                                 # POLISH
                ax.set_ylabel(ylab, fontsize=6.0, labelpad=1.5)                 # POLISH: 5.6 -> 6 pt
            if ri == len(rows) - 1:
                ax.set_xlabel("Lead time (days)", fontsize=6.0, labelpad=1)     # POLISH: 5.6 -> 6 pt
            else:
                ax.set_xticklabels([])
            if ri == 0 and ci == 0:
                hl = ax.get_legend_handles_labels()
    order = [3, 2, 1, 0]                                                        # POLISH: Foehn-CAMS first, as the map columns
    fig.legend([hl[0][k] for k in order], [hl[1][k] for k in order], fontsize=6.0, loc="center",   # POLISH: above the b frames
               bbox_to_anchor=(fx((X0 + X1) / 2), ft(leg_c)), ncol=4, frameon=False, handlelength=2.4, columnspacing=2.0,
               borderaxespad=0, borderpad=0)
    fig.canvas.draw()                              # POLISH: letters with gc_style.letter_left_of on the title baselines, in one column
    letter_left_of(fig, amaps + bcol, "a", atitle)                              # 1 mm left of the leftmost label of a and b
    letter_left_of(fig, amaps + bcol, "b", btitle)
    save(fig, "fig_gc5_aurora_setup_polish")                                    # POLISH: own name and folder
    print({L: {k: round(v, 2) for k, v in F[L].items() if k.endswith("_rmse")} for L in LEADS})


if __name__ == "__main__":
    main()
