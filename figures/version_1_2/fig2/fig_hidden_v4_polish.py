#!/usr/bin/env python3
"""Fig. 2, version 4 (2026-11-19): the figure of the submitted version (a-e, code of fig_hidden.bak_20261015_warned.py, unchanged
apart from the positions) with ONE small panel added, f: share of the people exposed who are warned and share of the warnings that are correct, as bars for Foehn, the CAMS forecast and Aurora, above 37.5 and 75 ug m-3, issued 1 day ahead
(numbers of panel f: rebuild/print/warn8/out_<label>/fig2f.json of build_warn8.py, Foehn with eight members on the 149 confirmatory start days, keys systems/thr*/lead1/World/<system>/reach_pct and correct_pct).
    python fig_hidden_v4_m8.py   -> out_<label>/fig_hidden_v4.pdf, .png (label: stored, or FOEHN_DESIGN=ar)"""
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, (os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/figures"))
WARN8 = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms/rebuild/print/warn8"))   # POLISH: the copy lives in polish/fig2; inputs stay in warn8
sys.path.insert(1, str(WARN8))                                      # POLISH: build_warn8.safe_write is imported from warn8 (read only)
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

import fig_exposure_row1 as R1
from gc_style import AURORA, CAMS, FOEHN, clean, letter_before, letter_left_of, map_axes, world_coords
from nature_style import DOUBLE, HERE, MM, PM_BOUNDS, plt, save
from build_warn8 import safe_write  # warn8: temp file + fsync + read-back + os.replace

PINK = ["#FBE3EC", "#F29BBD", "#E0457F", "#C2185B", "#5C0A2C"]
OUTDIR = WARN8                                                       # POLISH: inputs read from warn8 (was Path(__file__).parent)
SFX = f"_{os.environ['FOEHN_DESIGN']}" if os.environ.get("FOEHN_DESIGN") else ""   # A16 rebuild: design label
V2 = OUTDIR / f"out_{os.environ.get('FOEHN_DESIGN') or 'stored'}" / "fig2f.json"   # warn8: panel f only (build_warn8.py)
FIGOUT = Path(os.environ.get("POLISH_OUT", Path(__file__).resolve().parent))   # POLISH: outputs in polish/fig2 (or POLISH_OUT), never in warn8/out_*
NAME = os.environ.get("POLISH_NAME", "fig_hidden_v4_polish")       # POLISH: output name (the reproduction uses another)
D = 12                                                              # mm added below panels a-d for the taller bottom row
PINK3 = ["#F29BBD", "#C2185B", "#5C0A2C"]                          # > 15, 37.5, 75 (as the Extended Data figure)
# POLISH: one vertical rhythm (about 3 mm between the blocks a / b-d with their colour bar / e-f) and larger Beijing maps
# POLISH: (38 -> 44 mm tall), paid by the blank band above e-f (7.0 -> 3.1 mm) and 3.0 mm of page height (177.9 -> 180.9 mm)
MAPH = 44.0                                                         # POLISH: height of the maps b-d (mm; was 92 - 54 = 38)
DEAD = 3.73                                                         # POLISH: mm removed from the band between the PM colour bar and e-f
GAPA = 0.76                                                         # POLISH: mm added between a's longitude labels and the titles of b-d (gap 3.1 mm, as above e-f)
PX = 25.4 / 600                                                     # POLISH: one pixel of the 600 dpi rasters (mm)
LIFT = round((MAPH - 38.0 - DEAD + GAPA) / PX) * PX                 # POLISH: rise of map a and of the page top (mm), whole pixels: the raster of a stays the same array
XA = round(2.64 / PX) * PX                                          # POLISH: map a moved right to sit centred on the page (labels 6 mm from both edges), whole pixels
FS = 5.5                                                            # POLISH: smallest base text (was 5.0 / 5.3 in colour bars, legends, bar values)


def _mt_rise(t, dpi):
    """POLISH: how far (inches) matplotlib draws a mathtext title above its baseline: it rounds the box height and depth up to
    whole units of the renderer (points in the PDF), 0.42 pt for d's title, which then sat above the titles of b, c and its letter"""
    from matplotlib.mathtext import MathTextParser
    P, fp = MathTextParser("path"), t.get_fontproperties()
    a, b = P.parse(t.get_text(), dpi=dpi, prop=fp), P.parse(t.get_text(), dpi=1000 * dpi, prop=fp)
    return ((a.height - b.height / 1000) - (a.depth - b.depth / 1000)) / dpi



def main():
    import matplotlib.patheffects  # noqa: F401  (used by the Beijing row)
    ex = np.load(HERE / "data" / "exposure_2022.npz"); XT = list(ex["thr"]); XR = [str(r) for r in ex["regions"]]
    W, H = DOUBLE, (172 + D + LIFT) * MM                               # POLISH: + LIFT
    fig = plt.figure(figsize=(W, H))
    fx = lambda mm: mm / (W / MM); fy = lambda mm: mm / (H / MM)
    k375 = XT.index(37.5); nd = ex["res"].shape[0]
    hid = ex["maps"][1, k375] / nd
    ax = map_axes(fig, [fx(8 + XA), fy(100 + D + LIFT), fx(167), fy(68)])     # POLISH: + XA, + LIFT (same size)
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
    cb.ax.tick_params(labelsize=FS, length=1.5, pad=1); cb.ax.xaxis.set_label_position("top")   # POLISH: ticks 5 -> FS (exponents 3.5 -> 3.85 pt)
    cb.set_label("Residents per day hidden by the\n40 km grid, per 40 km cell", fontsize=FS, labelpad=1.5)   # POLISH: 5.3 -> FS
    # POLISH: b-d on the frame grid of e-f (b's left edge = e's axis, d's right edge = f's right end), c centred between them
    gs = fig.add_gridspec(1, 3, left=fx(14), right=fx(181), top=fy(54 + D - DEAD + MAPH), bottom=fy(54 + D - DEAD), wspace=0.12)
    axs, im, n_seen, n_hid = R1.row1(fig, gs, letters="bcd")   # Beijing-Tianjin, 21 January 2022 (fig_exposure_row1 defaults)
    axs[0].set_anchor("W"); axs[2].set_anchor("E")                    # POLISH: outer maps flush with the frame grid
    for a_, s_ in zip(axs, "bcd"):                                    # POLISH: drop row1's letters (0.3 mm above the title baseline);
        for t_ in [t for t in a_.texts if t.get_text() == s_ and t.get_fontweight() == "bold"]:   # POLISH: redrawn with
            t_.remove()                                               # POLISH: gc_style.letter_before below, as e and f
    axs[0].set_title("Beijing–Tianjin, GHAP (1 km), 21 Jan 2022", fontsize=6.5, loc="left")
    axs[1].set_title("The same field on 40 km cells", fontsize=6.5, loc="left")
    for a_ in axs:
        a_._left_title.set_fontsize(6.5)                              # POLISH: was a_.title (the empty centre title): d stayed at 7 pt
    fig.canvas.draw()
    p0, p1 = axs[0].get_position(), axs[1].get_position()
    cax = fig.add_axes([p0.x0 + 0.02, fy(47.5 + D - DEAD), p1.x1 - p0.x0 - 0.04, fy(0.0067 * 172)])   # POLISH: - DEAD (same size, in mm)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="max", spacing="uniform", ticks=PM_BOUNDS)
    cb.ax.set_xticklabels([f"{t:g}" for t in PM_BOUNDS])
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=FS, length=1.5, pad=1)   # POLISH: 5 -> FS
    cb.set_label("Daily PM$_{2.5}$ (µg m$^{-3}$); black contour: 75 µg m$^{-3}$, WHO interim target 1", fontsize=5.5, labelpad=1)
    # e: share of the exposure above each threshold that a 40 km grid hides, by region (the panel a of the Extended Data
    # figure until 15 October 2026; the extremes and warnings panels went there, extremes_panels.py)
    res = ex["res"]
    regs = ["World", "South Asia", "East Asia", "Africa", "Middle East", "Southeast Asia", "Europe", "North America", "South America"]
    ab = fig.add_axes([fx(14), fy(10), fx(104), fy(34)])
    for j, (thr, col, who) in enumerate(zip((15.0, 37.5, 75.0), PINK3, ("WHO guideline", "interim target 3", "interim target 1"))):
        k = XT.index(thr)
        share = np.array([100 * res[:, XR.index(r), 2, k].sum() / max(res[:, XR.index(r), 0, k].sum(), 1) for r in regs])
        keep = np.array([res[:, XR.index(r), 0, k].sum() for r in regs]) > 1e9          # at least a billion person-days above
        xb = np.arange(len(regs)) + (j - 1) * 0.27
        ab.bar(xb[keep], share[keep], width=0.25, color=col, label=f"> {thr:g} µg m$^{{-3}}$, {who}", zorder=3)
        print(f"hidden share > {thr:g}:", {r: round(float(v), 1) for r, v, kk in zip(regs, share, keep) if kk})
    ab.set_xticks(range(len(regs))); ab.set_xticklabels([r.replace(" ", "\n", 1) if len(r) > 8 else r for r in regs], fontsize=5.6, linespacing=0.95); ab.set_ylim(0, 68)
    ab.set_ylabel("Person-days hidden (%)", fontsize=6); clean(ab); ab.tick_params(labelsize=5.5); ab.tick_params(axis="x", length=0)
    ab.xaxis.grid(False)                                              # POLISH: no vertical grid lines (they ran through the middle bar of every region and through the legend)
    ab.legend(fontsize=FS, frameon=False, loc="upper left", ncol=3, handlelength=1.0, columnspacing=0.9, handletextpad=0.4, borderaxespad=0.2)   # POLISH: 5.3 -> FS
    ab.set_title("Share of the exposure that a 40 km grid hides, by region (2022)", fontsize=6.5, loc="left", pad=5)   # POLISH: 6.3 -> 6.5 as the titles of b-d
    # f: the warnings, ONE small and simple panel: bars, three systems, issued 1 day ahead (share of the people exposed who are
    # warned, and share of the warnings that are correct, above 37.5 and 75 ug m-3)
    N = json.loads(V2.read_text())["systems"]
    af = fig.add_axes([fx(134), fy(10), fx(47), fy(34)])
    # panel f only: softer tints of the house colours (Foehn blue in front; the two references lighter)
    SYS = (("foehn_m8", "Foehn", "#2F6FB7"), ("cams", "CAMS forecast", "#A9B1BC"), ("aurora", "Aurora", "#F2B27A"))
    groups = (("37.5", "reach_pct", "reached"), ("37.5", "correct_pct", "correct"), ("75", "reach_pct", "reached"), ("75", "correct_pct", "correct"))
    xg = [0, 1, 2.25, 3.25]
    for g, (thr, key, _) in zip(xg, groups):
        for k, (sk, name, col) in enumerate(SYS):
            v = N[f"thr{thr}"]["lead1"]["World"][sk][key][0]
            af.bar(g + (k - 1) * 0.28, v, width=0.26, color=col, zorder=3, label=name if g == 0 else None)
            af.text(g + (k - 1) * 0.28, v + 1.2, f"{v:.0f}", fontsize=FS, ha="center", va="bottom", color="#1F4E86" if k == 0 else "#5B626C", fontweight="bold" if k == 0 else "normal")   # POLISH: 5.0 -> FS
            print(f"f: > {thr}, 1 d, {sk}, {key}: {v:.1f} %")
    af.set_xticks(xg); af.set_xticklabels([g[2] for g in groups], fontsize=5.5); af.set_ylim(0, 108); af.set_yticks([0, 20, 40, 60, 80, 100])
    af.set_xlim(-0.6, 3.85)
    af.set_ylabel("Share (%)", fontsize=6, labelpad=2); clean(af); af.tick_params(labelsize=5.5); af.tick_params(axis="x", length=0, pad=2)
    af.xaxis.grid(False)                                              # POLISH: no vertical grid lines (they ran through the CAMS bar of every group)
    for xc, t in ((0.5, "> 37.5 µg m$^{-3}$"), (2.75, "> 75 µg m$^{-3}$")):
        af.text(xc, -0.125, t, fontsize=5.5, ha="center", va="top", transform=af.get_xaxis_transform())
    af.legend(fontsize=FS, frameon=True, facecolor="white", edgecolor="none", framealpha=1, borderpad=0.25,   # POLISH: 5.3 -> FS; white box: the
              loc="upper right", ncol=3, handlelength=0.9, columnspacing=0.7, handletextpad=0.3, borderaxespad=0.0)   # POLISH: 100 % grid line no longer runs through the legend
    af.set_title("Warnings issued 1 day ahead", fontsize=6.5, loc="left", pad=5)   # POLISH: 6.3 -> 6.5
    marks = [(a_._left_title, s_) for a_, s_ in zip(axs, "bcd")] + [(ab._left_title, "e"), (af._left_title, "f")]   # POLISH: b-d too
    fig.canvas.draw()                                         # letters: a beside the map, e just before its title, like b-d
    letter_left_of(fig, ax, "a", fy(166 + D + LIFT))                   # POLISH: + LIFT
    for t, s_ in marks:
        letter_before(fig, t, s_)
    from matplotlib.transforms import ScaledTranslation                # POLISH: d's title (mathtext) on the baseline of b, c and its
    t3 = axs[2]._left_title; base3 = t3.get_transform()               # POLISH: letter in the PDF (the PNG preview is not corrected:
    for ext in ("pdf", "png"):                                         # POLISH: Agg rounds to its own pixels)
        t3.set_transform(base3 + ScaledTranslation(0, -_mt_rise(t3, 72), fig.dpi_scale_trans) if ext == "pdf" else base3)   # POLISH
        buf = io.BytesIO(); fig.savefig(buf, format=ext)
        safe_write(FIGOUT / f"{NAME}.{ext}", buf.getvalue())          # POLISH: output name
    print("saved", FIGOUT / f"{NAME}.pdf")                             # POLISH: output name
    tot = ex["res"][:, XR.index("World"), 2, k375].sum() / nd
    print(f"hidden {tot / 1e6:.0f} M/day; Beijing seen {n_seen:.1f} M, hidden {n_hid:.1f} M")


if __name__ == "__main__":
    main()
