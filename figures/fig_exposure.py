#!/usr/bin/env python3
"""Extended Data figure: exposure that a 40 km grid hides (GHAP 2022 at 1 km, GHS-POP 2020 residents; no forecast).
a, share of the person-days above 15, 37.5 and 75 ug m-3 at 1 km that a 40 km grid hides (the cell mean is below), by region
   (figdata_exposure.py -> data/exposure_2022.npz; regions with at least a billion person-days above the threshold);
b-d, Cairo on 2 January 2022 and e-g, Santiago de Chile on 20 June 2022 (the largest hidden city-day of Africa and of the
Americas among twenty cities fixed in advance), as Fig. 2b-d: daily PM2.5 at 1 km with the 75 ug m-3 contour,
   the same field on 40 km cells, and the residents above 75 ug m-3 seen or hidden by the 40 km cells (fig_exposure_row1.row1).
    python fig_exposure.py"""
import numpy as np

import fig_exposure_row1 as R1
from gc_style import clean, label
from nature_style import DOUBLE, HERE, MM, PM_BOUNDS, plt, save

# the largest hidden city-day above 75 ug m-3 of Africa and of the Americas, among twenty large cities of the Americas,
# Africa, Europe and the Middle East fixed in advance (scratch search exposure_cities.py; boxes of 4 x 5 cells of 0.4 deg)
CASES = [dict(name="Cairo", doy=1, box=(30.8, 29.2, 30.0, 32.0),
              cities={"Cairo": (30.04, 31.24), "Faiyum": (29.31, 30.84), "Zagazig": (30.59, 31.50)}),
         dict(name="Santiago", doy=170, box=(-32.8, -34.4, -72.0, -70.0), scalebar="left",
              cities={"Santiago": (-33.45, -70.65), "Valparaíso": (-33.05, -71.62), "Rancagua": (-34.17, -70.74)})]
PINK3 = ["#F29BBD", "#C2185B", "#5C0A2C"]


def main():
    import matplotlib.patheffects  # noqa: F401
    z = np.load(HERE / "data" / "exposure_2022.npz"); THR = list(z["thr"]); R = [str(r) for r in z["regions"]]; res = z["res"]
    W, H = DOUBLE, 172 * MM
    fig = plt.figure(figsize=(W, H))
    fx = lambda mm: mm / (W / MM); fy = lambda mm: mm / (H / MM)
    # a: share hidden by region (vertical grouped bars)
    regs = ["World", "South Asia", "East Asia", "Africa", "Middle East", "Southeast Asia", "Europe", "North America", "South America"]
    ax = fig.add_axes([fx(14), fy(134), fx(166), fy(30)])
    for j, (thr, col) in enumerate(zip((15.0, 37.5, 75.0), PINK3)):
        k = THR.index(thr)
        share = np.array([100 * res[:, R.index(r), 2, k].sum() / max(res[:, R.index(r), 0, k].sum(), 1) for r in regs])
        keep = np.array([res[:, R.index(r), 0, k].sum() for r in regs]) > 1e9
        x = np.arange(len(regs)) + (j - 1) * 0.27
        ax.bar(x[keep], share[keep], width=0.25, color=col, label=f"> {thr:g} µg m$^{{-3}}$", zorder=3)
    ax.set_xticks(range(len(regs))); ax.set_xticklabels(regs, fontsize=5.6); ax.set_ylim(0, 60)
    ax.set_ylabel("Exposure hidden by\na 40 km grid (%)", fontsize=6); clean(ax); ax.tick_params(axis="x", length=0)
    ax.legend(fontsize=5.4, frameon=False, loc="upper left", ncol=3, handlelength=1.0, columnspacing=1.0)
    label(fig, fx(1), fy(166), "a")
    # b-g: Delhi and Karachi
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.99, top=121 / 172, bottom=16 / 172, wspace=0.12, hspace=0.32)
    for i, c in enumerate(CASES):
        axs, im, n_seen, n_hid = R1.row1(fig, gs, letters="bcd" if i == 0 else "efg", row=i, case=c)
        axs[0].set_title(f"{c['name']}, GHAP (1 km), {['2 Jan', '20 Jun'][i]} 2022", fontsize=6.3, loc="left")
        axs[1].set_title("The same field on 40 km cells", fontsize=6.3, loc="left")
        axs[2].set_title("Residents above 75 µg m$^{-3}$", fontsize=6.3, loc="left")
        print(f"{c['name']}: seen {n_seen:.1f} M, hidden {n_hid:.1f} M")
    fig.canvas.draw()
    p0 = axs[0].get_position(); p1 = axs[1].get_position()
    cax = fig.add_axes([p0.x0 + 0.02, 6 / 172, p1.x1 - p0.x0 - 0.04, 0.008])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="max", spacing="uniform", ticks=PM_BOUNDS)
    cb.ax.set_xticklabels([f"{t:g}" for t in PM_BOUNDS]); cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5, length=1.5, pad=1)
    cb.set_label("Daily PM$_{2.5}$ (µg m$^{-3}$); black contour: 75 µg m$^{-3}$", fontsize=5.5, labelpad=1)
    save(fig, "fig_exposure")


if __name__ == "__main__":
    main()
