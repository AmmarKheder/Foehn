#!/usr/bin/env python3
"""Extended Data figure: exposure that a 40 km grid hides (GHAP 2022 at 1 km, GHS-POP 2020 residents; no forecast), and the
local extremes that a 1 km forecast catches.
a-c, Cairo on 2 January 2022 and d-f, Santiago de Chile on 20 June 2022 (the largest hidden city-day of Africa and of the
Americas among twenty cities fixed in advance), as Fig. 2b-d: daily PM2.5 at 1 km with the 75 ug m-3 contour,
   the same field on 40 km cells, and the residents above 75 ug m-3 seen or hidden by the 40 km cells (fig_exposure_row1.row1);
g-j, detection of the days above the local P99 (SEDI by lead time and by region) and the value of warnings at the stations
   (extremes_panels.py; Fig. 2e-h until 15 October 2026, when the share hidden by region moved to Fig. 2e).
    python fig_exposure.py"""
import fig_exposure_row1 as R1
from extremes_panels import extremes_row
from gc_style import letter_before
from nature_style import DOUBLE, MM, PM_BOUNDS, plt, save

# the largest hidden city-day above 75 ug m-3 of Africa and of the Americas, among twenty large cities of the Americas,
# Africa, Europe and the Middle East fixed in advance (scratch search exposure_cities.py; boxes of 4 x 5 cells of 0.4 deg)
CASES = [dict(name="Cairo", doy=1, box=(30.8, 29.2, 30.0, 32.0),
              cities={"Cairo": (30.04, 31.24), "Faiyum": (29.31, 30.84), "Zagazig": (30.59, 31.50)}),
         dict(name="Santiago", doy=170, box=(-32.8, -34.4, -72.0, -70.0), scalebar="left",
              cities={"Santiago": (-33.45, -70.65), "Valparaíso": (-33.05, -71.62), "Rancagua": (-34.17, -70.74)})]
PINK3 = ["#F29BBD", "#C2185B", "#5C0A2C"]


def main():
    import matplotlib.patheffects  # noqa: F401
    W, H = DOUBLE, 178 * MM
    fig = plt.figure(figsize=(W, H))
    fx = lambda mm: mm / (W / MM); fy = lambda mm: mm / (H / MM)
    # a-f: Cairo and Santiago, as Fig. 2b-d
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.99, top=172 / 178, bottom=58 / 178, wspace=0.12, hspace=0.32)
    for i, c in enumerate(CASES):
        axs, im, n_seen, n_hid = R1.row1(fig, gs, letters="abc" if i == 0 else "def", row=i, case=c)
        axs[0].set_title(f"{c['name']}, GHAP (1 km), {['2 Jan', '20 Jun'][i]} 2022", fontsize=6.3, loc="left")
        axs[1].set_title("The same field on 40 km cells", fontsize=6.3, loc="left")
        axs[2].set_title("Residents above 75 µg m$^{-3}$", fontsize=6.3, loc="left")
        print(f"{c['name']}: seen {n_seen:.1f} M, hidden {n_hid:.1f} M")
    fig.canvas.draw()
    p0 = axs[0].get_position(); p1 = axs[1].get_position()
    cax = fig.add_axes([p0.x0 + 0.02, 48.5 / 178, p1.x1 - p0.x0 - 0.04, 1.4 / 178])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="max", spacing="uniform", ticks=PM_BOUNDS)
    cb.ax.set_xticklabels([f"{t:g}" for t in PM_BOUNDS]); cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5, length=1.5, pad=1)
    cb.set_label("Daily PM$_{2.5}$ (µg m$^{-3}$); black contour: 75 µg m$^{-3}$", fontsize=5.5, labelpad=1)
    # g-j: local extremes and the value of warnings (Fig. 2e-h until 15 October 2026), under the maps
    axe = extremes_row(fig, fx, fy, y0=12.5, h=22.0, legend_y=0.6)
    fig.canvas.draw()                                         # letters just before the titles, like a-f
    for ax_, s_ in zip(axe, "ghij"):
        letter_before(fig, ax_._left_title, s_)
    save(fig, "fig_exposure")


if __name__ == "__main__":
    main()
