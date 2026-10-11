#!/usr/bin/env python3
"""Main figure 'Kilometre-scale exposure': what a 40 km grid hides from the people exposed.
Row 1 (Beijing-Tianjin, 21 January 2022, GHAP 1 km): a, daily PM2.5 at 1 km with the WHO interim-target contour;
b, the same field on the 40 km cells of the CAMS / Aurora grid; c, residents above the interim target, seen or hidden by
the 40 km grid. Row 2 (all land, 2022, figdata_exposure.py): d, person-days above the target hidden by 40 km cells per
cell; e, share of exposure hidden, by region.
    python fig_exposure_row1.py --row1
(rebuilt on 7 October 2026 from the version that drew paper/nature/figs/fig_exposure_row1; fig_exposure.py is now the
exposure figure with the Delhi and Karachi zooms; row1() is also drawn by fig1_teaser_v5.py)"""
import sys

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


def gaussian_filter(a, sigma):
    """separable Gaussian blur (numpy only), edges reflected"""
    r = int(3 * sigma); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    for ax_ in (0, 1):
        a = np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode="reflect"), k, mode="valid"), ax_, a)
    return a

from fig_exposure_zoom import ghap_box, pop_box
from gc_style import fmt_lat, fmt_lon
from nature_style import COLOR, DOUBLE, HERE, MM, PM_COLORS, letter, plt, save

THR = 75.0
DOY, N, S, W, E = 20, 40.4, 38.8, 115.6, 117.6                              # 21 January 2022, box snapped to 0.4 deg cells
CITIES = {"Beijing": (39.90, 116.40), "Tianjin": (39.13, 117.20), "Langfang": (39.52, 116.70), "Zhuozhou": (39.49, 115.97)}
HID = "#C2185B"                                                              # pink: hidden by the 40 km grid
SEEN = "#3B3B3B"


def row1(fig, gs, letters="abc", row=0, case=None):
    """one row (1 km field, 40 km cells, residents seen / hidden); case = dict(doy, box=(N, S, W, E) on 0.4 deg cells, cities)"""
    doy, (N_, S_, W_, E_), cities = (DOY, (N, S, W, E), CITIES) if case is None else (case["doy"], case["box"], case["cities"])
    r0 = int(round((84.0 - N_) / 0.4)) * 40; r1 = int(round((84.0 - S_) / 0.4)) * 40
    c0 = int(round((W_ + 180.0) / 0.4)) * 40; c1 = int(round((E_ + 180.0) / 0.4)) * 40
    x = ghap_box(1826 + doy, r0, r1, c0, c1); pop = pop_box(r0, r1, c0, c1)
    ok = x > 0; nr, nc = (r1 - r0) // 40, (c1 - c0) // 40
    with np.errstate(all="ignore"):
        cell = np.where(ok, x, 0).reshape(nr, 40, nc, 40).sum((1, 3)) / ok.reshape(nr, 40, nc, 40).sum((1, 3))
    x40 = np.repeat(np.repeat(cell, 40, 0), 40, 1)
    ext = (-180 + c0 * 0.01, -180 + c1 * 0.01, 84 - r1 * 0.01, 84 - r0 * 0.01)
    # below the interim target: cream -> orange; above: red -> deep burgundy (break at 75 so that the threshold reads at once)
    from nature_style import pm_cmap
    cmap, norm = pm_cmap()                                                     # the paper's PM2.5 classes (RdYlBu_r), as every map
    axs = [fig.add_subplot(gs[row, k]) for k in range(3)]
    for ax, f, title in ((axs[0], x, "Daily PM$_{2.5}$ at 1 km"), (axs[1], x40, "The same field on 40 km cells")):
        im = ax.imshow(np.where(ok, f, np.nan), extent=ext, origin="upper", cmap=cmap, norm=norm, interpolation="nearest")
        fs = gaussian_filter(np.where(ok, f, np.nanmean(f[ok])), 1.5) if f is x else f   # contour of the lightly smoothed field
        ax.contour(np.where(ok, fs, np.nan), levels=[THR], extent=ext, origin="upper", colors="k", linewidths=0.5)
        ax.set_title(title, fontsize=7, loc="left")
    # c: residents above the interim target, seen / hidden by the 40 km grid; shade = residents per km2 (log)
    above, seen = ok & (x > THR), ok & (x > THR) & (x40 > THR)
    lp = np.clip((np.log10(np.maximum(pop, 1)) - 1.0) / 3.5, 0, 1)            # 10 .. 30,000 residents per km2 -> 0 .. 1
    rgb = np.ones(x.shape + (3,))
    for m, col in ((ok & ~above, "#BDBDBD"), (seen, SEEN), (above & ~seen, HID)):
        c = np.array(plt.matplotlib.colors.to_rgb(col))
        a = (0.15 + 0.85 * lp)[..., None] * m[..., None]
        rgb = rgb * (1 - a) + c * a
    axs[2].imshow(rgb, extent=ext, origin="upper", interpolation="nearest")
    axs[2].set_title(f"Residents above {THR:g} µg m$^{{-3}}$", fontsize=7, loc="left")
    w = np.where(ok, pop, 0.0)
    n_seen, n_hid = (w * seen).sum() / 1e6, (w * (above & ~seen)).sum() / 1e6
    h1 = plt.Line2D([], [], marker="s", ls="", color=SEEN, ms=4, label="seen at 40 km")
    h2 = plt.Line2D([], [], marker="s", ls="", color=HID, ms=4, label="hidden at 40 km")       # numbers in the caption
    axs[2].legend(handles=[h1, h2], loc="lower left", fontsize=5.5, frameon=True, framealpha=0.9, edgecolor="none",
                  borderpad=0.4, handletextpad=0.3)
    for k, ax in enumerate(axs):
        for cn, (la, lo) in cities.items():
            ax.plot(lo, la, "o", ms=2.2, mfc="white", mec="k", mew=0.4, zorder=5)
            ax.text(lo + 0.04, la + 0.04, cn, fontsize=5.5, zorder=5, color="k",
                    path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=1.2, foreground="white")])
        for cc in range(1, nc):
            ax.axvline(ext[0] + 0.4 * cc, color="white" if k < 2 else "#9E9E9E", lw=0.35, alpha=0.8)
        for rr in range(1, nr):
            ax.axhline(ext[3] - 0.4 * rr, color="white" if k < 2 else "#9E9E9E", lw=0.35, alpha=0.8)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect(1 / np.cos(np.deg2rad(0.5 * (N_ + S_))))
        xt = [v for v in range(int(np.ceil(W_)), int(np.floor(E_)) + 1)]; yt = [v for v in range(int(np.ceil(S_)), int(np.floor(N_)) + 1)]
        ax.set_xticks(xt); ax.set_xticklabels([fmt_lon(v) for v in xt]); ax.set_yticks(yt)
        ax.set_yticklabels([fmt_lat(v) for v in yt] if k == 0 else []); ax.tick_params(length=2, labelsize=5.5)
        for s_ in ax.spines.values():
            s_.set_visible(True); s_.set_linewidth(0.5)
        letter(ax, letters[k], x=-0.02, y=1.02)
    # scale bar 50 km on panel a
    dl = 50.0 / (111.32 * np.cos(np.deg2rad(0.5 * (N_ + S_))))
    x0, y0 = (ext[0] + 0.12, ext[2] + 0.10) if (case or {}).get("scalebar") == "left" else (ext[1] - 0.12 - dl, ext[2] + 0.10)
    axs[0].plot([x0, x0 + dl], [y0, y0], color="k", lw=1.2, solid_capstyle="butt")
    axs[0].text(x0 + dl / 2, y0 + 0.05, "50 km", ha="center", va="bottom", fontsize=5.5,
                path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=1.2, foreground="white")])
    return axs, im, n_seen, n_hid


def main():
    import matplotlib.patheffects  # noqa: F401
    rows = 1 if "--row1" in sys.argv else 2
    fig = plt.figure(figsize=(DOUBLE, (68 if rows == 1 else 140) * MM))
    gs = fig.add_gridspec(rows, 3, left=0.06, right=0.99, top=0.93 if rows == 1 else 0.96, bottom=0.17 if rows == 1 else 0.06,
                          wspace=0.12, hspace=0.35)
    axs, im, n_seen, n_hid = row1(fig, gs)
    fig.canvas.draw()
    p0, p1 = axs[0].get_position(), axs[1].get_position()
    cax = fig.add_axes([p0.x0 + 0.02, p0.y0 - 0.085 if rows == 1 else p0.y0 - 0.045, p1.x1 - p0.x0 - 0.04, 0.022 if rows == 1 else 0.011])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both", ticks=[45, 55, 65, 75, 85, 95, 105])
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.5, length=2)
    cb.ax.axvline(THR, color="k", lw=0.8)
    axs[0].legend(handles=[plt.Line2D([], [], color="k", lw=0.8, label="WHO interim target")], loc="lower left",
                  fontsize=5.5, frameon=True, framealpha=0.9, edgecolor="none", borderpad=0.4, handlelength=1.5)
    cb.set_label("Daily PM$_{2.5}$ on 21 January 2022 (µg m$^{-3}$)", fontsize=6)
    save(fig, "fig_exposure" + ("_row1" if rows == 1 else ""))
    print(f"seen {n_seen:.2f} M, hidden {n_hid:.2f} M")


if __name__ == "__main__":
    main()
