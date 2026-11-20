#!/usr/bin/env python3
"""Fig. 2, version 4 (2026-11-19): the figure of the submitted version (a-e, code of fig_hidden.bak_20261015_warned.py, unchanged
apart from the positions) with ONE small panel added, f: share of the people exposed who are warned and share of the warnings that are correct, as bars for Foehn, the CAMS forecast and Aurora, above 37.5 and 75 ug m-3, issued 1 day ahead
(numbers: analyses/hidden/v2/results/numbers_v2.json, keys systems/thr*/lead*/World/<system>/reach_pct and correct_pct).
    python fig_hidden_v4.py   -> fig_hidden_v4.pdf, .png (this folder)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/project_462001804/ammar/eccv/cran-pm/experiments/figures")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

import fig_exposure_row1 as R1
from gc_style import AURORA, CAMS, FOEHN, clean, letter_before, letter_left_of, map_axes, world_coords
from nature_style import DOUBLE, HERE, MM, PM_BOUNDS, plt, save

PINK = ["#FBE3EC", "#F29BBD", "#E0457F", "#C2185B", "#5C0A2C"]
OUTDIR = Path(__file__).resolve().parent
V2 = OUTDIR.parents[1] / "analyses" / "hidden" / "v2" / "results" / "numbers_v2.json"
D = 12                                                              # mm added below panels a-d for the taller bottom row
PINK3 = ["#F29BBD", "#C2185B", "#5C0A2C"]                          # > 15, 37.5, 75 (as the Extended Data figure)



def main():
    import matplotlib.patheffects  # noqa: F401  (used by the Beijing row)
    ex = np.load(HERE / "data" / "exposure_2022.npz"); XT = list(ex["thr"]); XR = [str(r) for r in ex["regions"]]
    W, H = DOUBLE, (172 + D) * MM
    fig = plt.figure(figsize=(W, H))
    fx = lambda mm: mm / (W / MM); fy = lambda mm: mm / (H / MM)
    k375 = XT.index(37.5); nd = ex["res"].shape[0]
    hid = ex["maps"][1, k375] / nd
    ax = map_axes(fig, [fx(8), fy(100 + D), fx(167), fy(68)])                  # Robinson, white ocean, grey land: every map of the paper
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
    gs = fig.add_gridspec(1, 3, left=0.06, right=0.99, top=(92 + D) / (172 + D), bottom=(54 + D) / (172 + D), wspace=0.12)
    axs, im, n_seen, n_hid = R1.row1(fig, gs, letters="bcd")   # Beijing-Tianjin, 21 January 2022 (fig_exposure_row1 defaults)
    axs[0].set_title("Beijing–Tianjin, GHAP (1 km), 21 Jan 2022", fontsize=6.5, loc="left")
    axs[1].set_title("The same field on 40 km cells", fontsize=6.5, loc="left")
    for a_ in axs:
        a_.title.set_fontsize(6.5)
    fig.canvas.draw()
    p0, p1 = axs[0].get_position(), axs[1].get_position()
    cax = fig.add_axes([p0.x0 + 0.02, (47.5 + D) / (172 + D), p1.x1 - p0.x0 - 0.04, 0.0067 * 172 / (172 + D)])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="max", spacing="uniform", ticks=PM_BOUNDS)
    cb.ax.set_xticklabels([f"{t:g}" for t in PM_BOUNDS])
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5, length=1.5, pad=1)
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
    ab.legend(fontsize=5.3, frameon=False, loc="upper left", ncol=3, handlelength=1.0, columnspacing=0.9, handletextpad=0.4, borderaxespad=0.2)
    ab.set_title("Share of the exposure that a 40 km grid hides, by region (2022)", fontsize=6.3, loc="left", pad=5)
    # f: the warnings, ONE small and simple panel: bars, three systems, issued 1 day ahead (share of the people exposed who are
    # warned, and share of the warnings that are correct, above 37.5 and 75 ug m-3)
    N = json.loads(V2.read_text())["systems"]
    af = fig.add_axes([fx(134), fy(10), fx(47), fy(34)])
    # panel f only: softer tints of the house colours (Foehn blue in front; the two references lighter), asked by the author 2026-11-19
    SYS = (("foehn_p50", "Foehn", "#2F6FB7"), ("cams", "CAMS forecast", "#A9B1BC"), ("aurora", "Aurora", "#F2B27A"))
    groups = (("37.5", "reach_pct", "reached"), ("37.5", "correct_pct", "correct"), ("75", "reach_pct", "reached"), ("75", "correct_pct", "correct"))
    xg = [0, 1, 2.25, 3.25]
    for g, (thr, key, _) in zip(xg, groups):
        for k, (sk, name, col) in enumerate(SYS):
            v = N[f"thr{thr}"]["lead1"]["World"][sk][key][0]
            af.bar(g + (k - 1) * 0.28, v, width=0.26, color=col, zorder=3, label=name if g == 0 else None)
            af.text(g + (k - 1) * 0.28, v + 1.2, f"{v:.0f}", fontsize=5.0, ha="center", va="bottom", color="#1F4E86" if k == 0 else "#5B626C", fontweight="bold" if k == 0 else "normal")
            print(f"f: > {thr}, 1 d, {sk}, {key}: {v:.1f} %")
    af.set_xticks(xg); af.set_xticklabels([g[2] for g in groups], fontsize=5.5); af.set_ylim(0, 108); af.set_yticks([0, 20, 40, 60, 80, 100])
    af.set_xlim(-0.6, 3.85)
    af.set_ylabel("Share (%)", fontsize=6, labelpad=2); clean(af); af.tick_params(labelsize=5.5); af.tick_params(axis="x", length=0, pad=2)
    for xc, t in ((0.5, "> 37.5 µg m$^{-3}$"), (2.75, "> 75 µg m$^{-3}$")):
        af.text(xc, -0.125, t, fontsize=5.5, ha="center", va="top", transform=af.get_xaxis_transform())
    af.legend(fontsize=5.3, frameon=False, loc="upper right", ncol=3, handlelength=0.9, columnspacing=0.7, handletextpad=0.3, borderaxespad=0.0)
    af.set_title("Warnings issued 1 day ahead", fontsize=6.3, loc="left", pad=5)
    marks = [(ab._left_title, "e"), (af._left_title, "f")]
    fig.canvas.draw()                                         # letters: a beside the map, e just before its title, like b-d
    letter_left_of(fig, ax, "a", fy(166 + D))
    for t, s_ in marks:
        letter_before(fig, t, s_)
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"fig_hidden_v4.{ext}")
    print("saved", OUTDIR / "fig_hidden_v4.pdf")
    tot = ex["res"][:, XR.index("World"), 2, k375].sum() / nd
    print(f"hidden {tot / 1e6:.0f} M/day; Beijing seen {n_seen:.1f} M, hidden {n_hid:.1f} M")


if __name__ == "__main__":
    main()
