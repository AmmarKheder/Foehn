#!/usr/bin/env python3
"""COPY of experiments/figures/fig_exposure.py (Extended Data Fig. 4). Two changes only:
  * panels g-j come from extremes_panels_eqiss.py (g, h: CAMS and Aurora issued at the same time as Foehn, leads 1-5);
  * the figure is written to ../figs/fig_exposure_eqiss.{pdf,png} with fig.savefig (nature_style.save() would overwrite the
    figure of the Nature folder).
Panels a-f (Cairo, Santiago) and i, j are those of the submitted figure.
    python -B fig_exposure_eqiss.py            (TE_ISSUANCE=d redraws the submitted panels -> fig_exposure_runD_check)
    python -B fig_exposure_eqiss.py row        only the row g-j -> ../figs/extremes_row_eqiss.{pdf,png}"""
import os
import sys
from pathlib import Path

FIG = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/figures"))
sys.path.append(str(FIG))
OUTD = Path(__file__).resolve().parent.parent / "figs"
if os.environ.get("FOEHN_DESIGN"):   # A16 rebuild: renders of a noise design go to natcomms/rebuild/figs_<D>
    OUTD = Path(__file__).resolve().parents[3] / "rebuild" / f"figs_{os.environ['FOEHN_DESIGN']}"
ISS = os.environ.get("TE_ISSUANCE", "d1")

from extremes_panels_eqiss import extremes_row  # noqa: E402
from gc_style import letter_before  # noqa: E402
from nature_style import DOUBLE, MM, PM_BOUNDS, plt  # noqa: E402

# the largest hidden city-day above 75 ug m-3 of Africa and of the Americas, among twenty large cities of the Americas,
# Africa, Europe and the Middle East fixed in advance (scratch search exposure_cities.py; boxes of 4 x 5 cells of 0.4 deg)
CASES = [dict(name="Cairo", doy=1, box=(30.8, 29.2, 30.0, 32.0),
              cities={"Cairo": (30.04, 31.24), "Faiyum": (29.31, 30.84), "Zagazig": (30.59, 31.50)}),
         dict(name="Santiago", doy=170, box=(-32.8, -34.4, -72.0, -70.0), scalebar="left",
              cities={"Santiago": (-33.45, -70.65), "Valparaíso": (-33.05, -71.62), "Rancagua": (-34.17, -70.74)})]


def out(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUTD / f"{name}.{ext}")
    print("saved", OUTD / name)


def row_only():
    W, H = DOUBLE, 44 * MM
    fig = plt.figure(figsize=(W, H))
    fx = lambda mm: mm / (W / MM); fy = lambda mm: mm / (H / MM)
    axe = extremes_row(fig, fx, fy, y0=12.5, h=22.0, legend_y=0.6)
    fig.canvas.draw()
    for ax_, s_ in zip(axe, "ghij"):
        letter_before(fig, ax_._left_title, s_)
    out(fig, "extremes_row_eqiss" if ISS == "d1" else "extremes_row_runD_check")


def main():
    import matplotlib.patheffects  # noqa: F401
    import fig_exposure_row1 as R1
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
    # g-j: local extremes and the value of warnings, under the maps
    axe = extremes_row(fig, fx, fy, y0=12.5, h=22.0, legend_y=0.6)
    fig.canvas.draw()                                         # letters just before the titles, like a-f
    for ax_, s_ in zip(axe, "ghij"):
        letter_before(fig, ax_._left_title, s_)
    out(fig, "fig_exposure_eqiss" if ISS == "d1" else "fig_exposure_runD_check")


if __name__ == "__main__":
    row_only() if len(sys.argv) > 1 and sys.argv[1] == "row" else main()
