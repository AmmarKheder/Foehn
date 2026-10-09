#!/usr/bin/env python3
"""Fig. 2: why the kilometre matters - the people a 40 km forecast cannot warn (no forecast involved).
a, residents per day in 2022 above the WHO 24-h interim target (37.5 ug m-3) at their 1 km pixel whose 40 km cell is below it,
   per 40 km cell (GHAP 1 km x GHS-POP 1 km; figdata_exposure.py -> data/exposure_2022.npz);
b-d, Beijing-Tianjin on 21 January 2022, the most polluted day of the year for the world's population (rule fixed in
   advance): daily PM2.5 at 1 km (GHAP) with the 75 ug m-3 contour, the same field on 40 km cells (what a perfect 40 km forecast gives), and the residents
   above 75 ug m-3 seen or hidden by the 40 km cells (fig_exposure_row1.row1).
    python fig_hidden.py"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

import fig_exposure_row1 as R1
from gc_style import label, map_axes
from nature_style import DOUBLE, HERE, MM, PM_BOUNDS, plt, save

PINK = ["#FBE3EC", "#F29BBD", "#E0457F", "#C2185B", "#5C0A2C"]



def main():
    import matplotlib.patheffects  # noqa: F401  (used by the Beijing row)
    ex = np.load(HERE / "data" / "exposure_2022.npz"); XT = list(ex["thr"]); XR = [str(r) for r in ex["regions"]]
    W, H = DOUBLE, 172 * MM
    fig = plt.figure(figsize=(W, H))
    fx = lambda mm: mm / (W / MM); fy = lambda mm: mm / (H / MM)
    k375 = XT.index(37.5); nd = ex["res"].shape[0]
    hid = ex["maps"][1, k375] / nd
    ax = map_axes(fig, [fx(8), fy(100), fx(167), fy(68)])                  # Robinson, white ocean, grey land: every map of the paper
    cm = LinearSegmentedColormap.from_list("hid", PINK, N=256)
    lat_e = 84.0 - 0.4 * np.arange(361); lon_e = -180.0 + 0.4 * np.arange(901)
    pm = ax.pcolormesh(lon_e, lat_e, np.ma.masked_less(hid, 10), cmap=cm, norm=LogNorm(10, 1e5), transform=ccrs.PlateCarree(),
                       rasterized=True, zorder=2)
    import cartopy.crs as ccrs_                                       # colour bar inside the frame, over the empty South Pacific
    rob = ax.projection
    xa, ya = rob.transform_point(-162, -40, ccrs_.PlateCarree()); xb, yb = rob.transform_point(-98, -36.5, ccrs_.PlateCarree())
    cax = ax.inset_axes([xa, ya, xb - xa, yb - ya], transform=ax.transData)
    cb = fig.colorbar(pm, cax=cax, orientation="horizontal", extend="both"); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=5, length=1.5, pad=1); cb.ax.xaxis.set_label_position("top")
    cb.set_label("Residents per day hidden by the\n40 km grid, per 40 km cell", fontsize=5.3, labelpad=1.5)
    label(fig, fx(1), fy(166), "a")
    gs = fig.add_gridspec(1, 3, left=0.06, right=0.99, top=92 / 172, bottom=54 / 172, wspace=0.12)
    axs, im, n_seen, n_hid = R1.row1(fig, gs, letters="bcd")   # Beijing-Tianjin, 21 January 2022 (fig_exposure_row1 defaults)
    axs[0].set_title("Beijing–Tianjin, GHAP (1 km), 21 Jan 2022", fontsize=6.5, loc="left")
    axs[1].set_title("The same field on 40 km cells", fontsize=6.5, loc="left")
    for a_ in axs:
        a_.title.set_fontsize(6.5)
    fig.canvas.draw()
    p0, p1 = axs[0].get_position(), axs[1].get_position()
    cax = fig.add_axes([p0.x0 + 0.02, 47.5 / 172, p1.x1 - p0.x0 - 0.04, 0.0067])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="max", spacing="uniform", ticks=PM_BOUNDS)
    cb.ax.set_xticklabels([f"{t:g}" for t in PM_BOUNDS])
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5, length=1.5, pad=1)
    cb.set_label("Daily PM$_{2.5}$ (µg m$^{-3}$); black contour: 75 µg m$^{-3}$, WHO interim target 1", fontsize=5.5, labelpad=1)
    # e, f: detection of local extremes over land (tail_events.py: days above the local 99th percentile of the 2017-2020
    # climatology of each 0.25 deg cell, identical cells and days); g, h: the value of warning these people (REV at the
    # stations; extremes_stations.py)
    import final_model as FM
    from fig_gc4_extremes import rev_curves, sedi
    from gc_style import AURORA, CAMS, FOEHN, clean
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    te = np.load(HERE / "data" / f"tail_events_{FM.LABEL}.npz"); CT = te["C"]
    TS = [str(x) for x in te["systems"]]; TR = [str(r) for r in te["regions"]]
    SYSE = [("Foehn", f"{FM.LABEL} (mean)", FOEHN, 1.65, "o", 2.4), ("CAMS", "CAMS", CAMS, 1.0, "s", 1.9),
            ("Aurora", "Aurora", AURORA, 1.0, "D", 1.8)]
    L = np.arange(1, 8); gl = TR.index("Global")
    ae = fig.add_axes([fx(11), fy(12.5), fx(33), fy(22)])
    for nm, key, col, lw, mk, ms in SYSE:
        c = CT[gl, :, 2, TS.index(key)]; v = sedi(c); ok = np.isfinite(v) & (c[:, :2].sum(-1) > 0)
        ae.plot(L[ok], v[ok], color=col, lw=lw, marker=mk, ms=ms, zorder=5 if nm == "Foehn" else 3)
    ae.set_xticks(L); ae.set_xlim(0.6, 7.4); ae.set_ylim(0.25, 0.68)
    ae.set_xlabel("Lead time (days)", fontsize=6, labelpad=1); ae.set_ylabel("SEDI, all land", fontsize=6)
    clean(ae); ae.tick_params(labelsize=5.5)
    ae.set_title("Days above the local P99", fontsize=6.3, loc="left", pad=3)
    label(fig, fx(1), fy(36), "e")
    af = fig.add_axes([fx(66), fy(12.5), fx(24), fy(22)])
    regs = [r for r in TR if r != "Global"]; yy = np.arange(len(regs))[::-1]
    for y_, r in zip(yy, regs):
        vals = [float(sedi(CT[TR.index(r), 0, 2, TS.index(key)])) for _, key, *_ in SYSE]
        af.plot([min(vals), max(vals)], [y_, y_], color="#D5DBE0", lw=2.2, solid_capstyle="round", zorder=1)
        for (nm, key, col, lw, mk, ms), v in zip(SYSE, vals):
            af.plot([v], [y_], marker=mk, ms=ms + (0.9 if nm == "Foehn" else 0.4), color=col, mec="white", mew=0.35,
                    ls="none", zorder=5 if nm == "Foehn" else 3)
        print(f"SEDI >P99 1 d {r:14s}", " ".join(f"{nm} {v:.2f}" for (nm, *_), v in zip(SYSE, vals)))
    af.set_yticks(yy); af.set_yticklabels(regs, fontsize=5.5); af.set_ylim(-0.6, len(regs) - 0.4)
    af.set_xlim(0.2, 0.72); af.set_xticks([0.2, 0.4, 0.6])
    af.set_xlabel("SEDI, 1 day ahead", fontsize=6, labelpad=1)
    clean(af); af.tick_params(labelsize=5.5); af.tick_params(axis="y", length=0)
    af.grid(axis="y", visible=False)
    af.set_title("By region", fontsize=6.3, loc="left", pad=3)
    label(fig, fx(45), fy(36), "f")
    es = np.load(HERE / "data" / f"extremes_stations_{FM.LABEL}.npz"); CE = es["C"]; ES = [str(x) for x in es["systems"]]
    alphas = np.logspace(np.log10(0.003), np.log10(0.6), 250)
    SYSR = [("Foehn", FM.LABEL, FOEHN, 1.65), ("CAMS", "CAMS", CAMS, 1.0), ("Aurora", "Aurora", AURORA, 1.0)]
    for k, (tier, title, ymax) in enumerate(((1, "Warning value, days above P95", 0.4), (2, "Warning value, days above P99", 0.3))):
        axr = fig.add_axes([fx(104 + 44 * k), fy(12.5), fx(34), fy(22)])
        for nm, key, col, lw in SYSR:
            for h, ls in ((0, "-"), (2, (0, (4, 2)))):
                v = rev_curves(CE[0, tier, h, ES.index(key)], alphas)
                axr.plot(alphas, v, color=col, lw=lw, ls=ls, zorder=5 if nm == "Foehn" else 3)
        axr.set_xscale("log"); axr.set_xlim(0.01, 0.6); axr.set_ylim(0, ymax)
        axr.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 0.5]))
        axr.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}")); axr.xaxis.set_minor_formatter(NullFormatter())
        axr.set_xlabel("Cost / avoided loss", fontsize=6, labelpad=1)
        if k == 0:
            axr.set_ylabel("Relative economic value", fontsize=6)
        clean(axr); axr.tick_params(labelsize=5.5)
        axr.set_title(title, fontsize=6.3, loc="left", pad=3)
        label(fig, fx(93 + 44 * k), fy(36), "gh"[k])
    handles = [Line2D([0], [0], color=c, linewidth=1.6, label=nm) for nm, _, c, _ in SYSR]
    handles += [Line2D([0], [0], color="#56636D", linewidth=1.1, linestyle=ls, label=lab)
                for lab, ls in (("1 day ahead", "-"), ("3 days ahead", (0, (4, 2))))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, fy(0.6)), ncol=5, fontsize=5.5,
               columnspacing=1.0, handlelength=1.8)
    save(fig, "fig_hidden")
    tot = ex["res"][:, XR.index("World"), 2, k375].sum() / nd
    print(f"hidden {tot / 1e6:.0f} M/day; Beijing seen {n_seen:.1f} M, hidden {n_hid:.1f} M")


if __name__ == "__main__":
    main()
