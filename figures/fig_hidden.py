#!/usr/bin/env python3
"""Fig. 2: why the kilometre matters - the people a 40 km forecast cannot warn (no forecast involved).
a, residents per day in 2022 above the WHO 24-h interim target (37.5 ug m-3) at their 1 km pixel whose 40 km cell is below it,
   per 40 km cell (GHAP 1 km x GHS-POP 1 km; figdata_exposure.py -> data/exposure_2022.npz);
b-d, Beijing-Tianjin on 21 January 2022, the most polluted day of the year for the world's population (rule fixed in
   advance): daily PM2.5 at 1 km (GHAP) with the 75 ug m-3 contour, the same field on 40 km cells (what a perfect 40 km forecast gives), and the residents
   above 75 ug m-3 seen or hidden by the 40 km cells (fig_exposure_row1.row1);
e, share of the person-days above 15, 37.5 and 75 ug m-3 at 1 km that the 40 km cells hide, by region (regions with at least a
   billion person-days above the threshold). The extremes and warnings panels (e-h until 15 October 2026) are in fig_exposure.py.
    python fig_hidden.py"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

import fig_exposure_row1 as R1
from gc_style import clean, letter_before, letter_left_of, map_axes, world_coords
from nature_style import DOUBLE, HERE, MM, PM_BOUNDS, plt, save

PINK = ["#FBE3EC", "#F29BBD", "#E0457F", "#C2185B", "#5C0A2C"]
PINK3 = ["#F29BBD", "#C2185B", "#5C0A2C"]                          # > 15, 37.5, 75 (as the Extended Data figure)



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
    world_coords(ax, size=5.5)
    import cartopy.crs as ccrs_                                       # colour bar inside the frame, over the empty South Pacific
    rob = ax.projection
    xa, ya = rob.transform_point(-162, -40, ccrs_.PlateCarree()); xb, yb = rob.transform_point(-98, -36.5, ccrs_.PlateCarree())
    cax = ax.inset_axes([xa, ya, xb - xa, yb - ya], transform=ax.transData)
    cb = fig.colorbar(pm, cax=cax, orientation="horizontal", extend="both"); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=5, length=1.5, pad=1); cb.ax.xaxis.set_label_position("top")
    cb.set_label("Residents per day hidden by the\n40 km grid, per 40 km cell", fontsize=5.3, labelpad=1.5)
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
    # e: share of the exposure above each threshold that a 40 km grid hides, by region (the panel a of the Extended Data
    # figure until 15 October 2026; the extremes and warnings panels went there, extremes_panels.py)
    res = ex["res"]
    regs = ["World", "South Asia", "East Asia", "Africa", "Middle East", "Southeast Asia", "Europe", "North America", "South America"]
    ab = fig.add_axes([fx(14), fy(8), fx(166), fy(26)])
    for j, (thr, col, who) in enumerate(zip((15.0, 37.5, 75.0), PINK3, ("WHO guideline", "interim target 3", "interim target 1"))):
        k = XT.index(thr)
        share = np.array([100 * res[:, XR.index(r), 2, k].sum() / max(res[:, XR.index(r), 0, k].sum(), 1) for r in regs])
        keep = np.array([res[:, XR.index(r), 0, k].sum() for r in regs]) > 1e9          # at least a billion person-days above
        xb = np.arange(len(regs)) + (j - 1) * 0.27
        ab.bar(xb[keep], share[keep], width=0.25, color=col, label=f"> {thr:g} µg m$^{{-3}}$, {who}", zorder=3)
        print(f"hidden share > {thr:g}:", {r: round(float(v), 1) for r, v, kk in zip(regs, share, keep) if kk})
    ab.set_xticks(range(len(regs))); ab.set_xticklabels(regs, fontsize=5.6); ab.set_ylim(0, 60)
    ab.set_ylabel("Person-days hidden (%)", fontsize=6); clean(ab); ab.tick_params(labelsize=5.5); ab.tick_params(axis="x", length=0)
    ab.legend(fontsize=5.4, frameon=False, loc="upper left", ncol=3, handlelength=1.0, columnspacing=1.2)
    ab.set_title("Share of the exposure that a 40 km grid hides, by region (2022)", fontsize=6.3, loc="left", pad=5)
    marks = [(ab._left_title, "e")]
    fig.canvas.draw()                                         # letters: a beside the map, e just before its title, like b-d
    letter_left_of(fig, ax, "a", fy(166))
    for t, s_ in marks:
        letter_before(fig, t, s_)
    save(fig, "fig_hidden")
    tot = ex["res"][:, XR.index("World"), 2, k375].sum() / nd
    print(f"hidden {tot / 1e6:.0f} M/day; Beijing seen {n_seen:.1f} M, hidden {n_hid:.1f} M")


if __name__ == "__main__":
    main()
