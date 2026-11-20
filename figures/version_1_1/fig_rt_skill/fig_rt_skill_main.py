#!/usr/bin/env python3
"""Main figure: the real-time 1 km forecasts verified at the monitors (house style of the paper; companion of fig_rt_day).
a, where the monitors are: 2-degree cells coloured by the change in RMSE of the 1 km forecast against the CAMS forecast of
   the same cycle (blue: the 1 km forecast is closer to the measurements; grey: fewer than 100 station-days); boxes = the
   regions of c; bars above the colour classes = share of the coloured cells in each class;
b, RMSE by lead day of the four systems and, below, the change of the 1 km forecast against each of the three others
   (95 % intervals, two-way block bootstrap of the scorer);
c, the same change by region against the CAMS forecast and against the climatology (filled: interval excludes zero);
d, the 1 km forecast read 1, 3, 10 km away from the monitor's pixel and as the mean of its 0.25 degree cell, against CAMS;
e, the record: one value per day of issue (live days shaded; filled: the interval of the 2-degree-cell bootstrap excludes zero);
f, the eight members at the monitors: rank of the measurement, and error against spread (a second ensemble, e.g. a
   calibrated one, is drawn as a second series when rt_skill_data.py was run with --arm2).
In b-e the colour and the symbol name the system that the 1 km forecast is compared with.
Reads only numbers[_tag].json and data/rt_skill[_tag].npz (written by rt_skill_data.py) and the paper's style modules.
Vector text (Liberation Sans, TrueType); nothing is rasterised.
    python -B fig_rt_skill.py [tag]        -> fig_rt_skill[_tag].pdf / .png (300 dpi) beside this script (or ./render.sh)"""
import datetime as dt
# VARIANT FOR THE MAIN TEXT (2026-11-19, lead): identical to fig_rt_skill.py without panel f; panel e spans the bottom row.
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/scratch/project_462001804/ammar/eccv/cran-pm/experiments/figures")   # the paper's style modules (imported only)

import cartopy.crs as ccrs  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon, Rectangle  # noqa: E402

from gc_style import CAMS, CLIM, FOEHN, SC_BOUNDS, SC_COLORS, clean, letter_before, letter_left_of, map_axes, sc_cmap, world_coords  # noqa: E402
from nature_style import DOUBLE, MM, plt  # noqa: E402

_argv, sys.argv = sys.argv, sys.argv[:1]            # (the companion script reads its own first argument at import)
try:                                                # names and colours of the companion figure (one charter for the section)
    sys.path.insert(1, str(HERE.parent / "fig_rt_day"))
    from fig_rt_day import INK, NAME_04, NAME_KM, RT04
except Exception:                                   # the same values, if that script has moved
    INK, NAME_04, NAME_KM, RT04 = "#111111", "Foehn in real time (40 km)", "Foehn in real time (1 km)", "#3E8E86"
finally:
    sys.argv = _argv
NAME_CAMS, NAME_CLIM = "CAMS forecast", "Climatology (1 km)"
ARM2 = "#8FB2DD"                                    # a second ensemble in f (light blue, as the longer lead of the companion figure)
PC = ccrs.PlateCarree()
W_MM, H_MM = 183.0, 190.5
EDGE, MUTED, LIVE_BG = "#6A6A6A", "#555555", "#E9EEF6"
FEW = "#BDBDBD"                                     # a cell with monitors but too few station-days to be coloured
YCH = (-27.5, 3.5)                                  # one scale for every "change in RMSE" axis of b, d, e


def whisker(a, x, d, col, mk, ms=3.0, lw=0.8, z=4, horizontal=False, lim=None, fill=None):
    """estimate with its 95 % interval; the marker is filled when the interval excludes zero (or as fill says)"""
    lo, hi = d["ci"]
    solid = (hi < 0 or lo > 0) if fill is None else fill
    cl, ch = (lo, hi) if lim is None else (max(lo, lim[0]), min(hi, lim[1]))
    if horizontal:
        a.plot([cl, ch], [x, x], color=col, lw=lw, zorder=z, solid_capstyle="butt")
        if lim is not None and hi > lim[1]:
            a.plot(lim[1] - 0.6, x, marker=">", ms=2.6, color=col, mew=0, zorder=z, clip_on=False)
        a.plot(d["est"], x, marker=mk, ms=ms, mfc=col if solid else "white", mec=col, mew=0.7, ls="", zorder=z + 1)
    else:
        a.plot([x, x], [cl, ch], color=col, lw=lw, zorder=z, solid_capstyle="butt")
        a.plot(x, d["est"], marker=mk, ms=ms, mfc=col if solid else "white", mec=col, mew=0.7, ls="", zorder=z + 1)


def halo_text(a, x, y, s, lw=1.3, **kw):
    """text on a white halo; the halo is an outlined copy underneath, so that the text itself stays text in the PDF"""
    z = kw.pop("zorder", 7)
    a.text(x, y, s, zorder=z, **{**kw, "color": "white"}, path_effects=[pe.withStroke(linewidth=lw, foreground="white")])
    return a.text(x, y, s, zorder=z + 0.1, **kw)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    suf = f"_{tag}" if tag else ""
    num = json.loads((HERE / f"numbers{suf}.json").read_text())
    Z = np.load(HERE / "data" / f"rt_skill{suf}.npz")
    A, A2 = num["arm"], num.get("arm2")
    fig = plt.figure(figsize=(DOUBLE, H_MM * MM))
    fx = lambda mm: mm / W_MM; fy = lambda mm: mm / H_MM
    R = lambda x, y, w, h: [fx(x), fy(y), fx(w), fy(h)]
    SYS = {A: (FOEHN, "-", 1.6, "o", 3.0), "rt04": (RT04, "-", 1.0, "^", 2.8), "cams": (CAMS, "-", 1.0, "s", 2.5), "clim": (CLIM, (0, (3, 2)), 0.9, "o", 2.3)}
    titles = []

    # ------------------------------------------------------------------ a: where the 1 km forecast is closer than the CAMS forecast
    M = num["a_map"]
    rob = ccrs.Robinson()
    xr = rob.transform_point(180, 0, PC)[0]; yt = rob.transform_point(0, 84, PC)[1]; yb = rob.transform_point(0, -60, PC)[1]
    wa = 170.0; ha = wa * (yt - yb) / (2 * xr); ya = H_MM - 1.5 - ha
    ax = map_axes(fig, R(6.0, ya, wa, ha))
    world_coords(ax, size=5.5)
    lat0, lon0 = Z["cell_lat0"], Z["cell_lon0"]
    chg = 100.0 * (Z[f"cell_rmse_{A}"] / Z["cell_rmse_cams"] - 1.0)
    okc = Z["cell_n"] >= M["min_station_days_to_colour"]
    cm, nm = sc_cmap()
    t3 = np.linspace(0.0, 1.0, 3)

    def quad(lo, la, d=2.0):
        xs = np.concatenate([lo + d * t3, np.full(3, lo + d), lo + d - d * t3, np.full(3, lo)])
        ys = np.concatenate([np.full(3, la), la + d * t3, np.full(3, la + d), la + d - d * t3])
        return rob.transform_points(PC, xs, ys)[:, :2]
    verts = [quad(lo, la) for lo, la in zip(lon0, lat0)]
    ax.add_collection(PolyCollection([v for v, k in zip(verts, okc) if not k], facecolors=FEW, edgecolors=EDGE, linewidths=0.15, zorder=2), autolim=False)
    ax.add_collection(PolyCollection([v for v, k in zip(verts, okc) if k], facecolors=cm(nm(chg[okc])), edgecolors=EDGE, linewidths=0.15, zorder=2.2), autolim=False)
    place = {"Europe": (-12.5, 49.0, "right", "center"), "East Asia": (136.5, 23.5, "left", "center"), "South Asia": (77.5, 3.2, "center", "top"),
             "North America": (-131.5, 30.0, "right", "center"), "Africa": (-16.5, -13.0, "left", "center"), "South America": (-83.5, -27.0, "right", "center")}
    tt = np.linspace(0.0, 1.0, 24)
    for bx in M["region_boxes"]:
        if bx["name"] not in place:                                           # north of 60 N: the 60 N parallel of the graticule
            continue
        (s, n), (w, e) = bx["lat"], bx["lon"]
        ax.plot(np.concatenate([w + (e - w) * tt, np.full(24, e), e - (e - w) * tt, np.full(24, w)]),
                np.concatenate([np.full(24, n), n - (n - s) * tt, np.full(24, s), s + (n - s) * tt]), transform=PC, color="#3A3A3A", lw=0.45, zorder=5)
        lo, la, ha_, va_ = place[bx["name"]]
        halo_text(ax, lo, la, bx["name"], lw=1.4, transform=PC, fontsize=5.4, color="#2A2A2A", ha=ha_, va=va_, zorder=7)
    # the colour classes with the share of cells in each (scorecard classes of the paper: open-ended beyond +-20 %)
    nb = len(SC_COLORS); cw = 4.4; x0 = 78.0
    cnt = np.array(M["cells_per_class"], float)
    shr = 100.0 * np.concatenate([[cnt[0] + cnt[1]], cnt[2:-2], [cnt[-2] + cnt[-1]]]) / cnt.sum()
    al = fig.add_axes(R(x0, ya + 0.35, nb * cw, 12.0)); al.set_xlim(0, nb); al.set_ylim(0, 12.0); al.axis("off"); al.patch.set_alpha(0)
    yb0, yb1, ybar, sc_ = 2.4, 3.7, 3.95, 2.6 / shr.max()
    for k in range(nb):
        if k == 0:
            pts = [(0.0, 0.5 * (yb0 + yb1)), (0.4, yb0), (1, yb0), (1, yb1), (0.4, yb1)]
        elif k == nb - 1:
            pts = [(k, yb0), (k + 0.6, yb0), (nb, 0.5 * (yb0 + yb1)), (k + 0.6, yb1), (k, yb1)]
        else:
            pts = [(k, yb0), (k + 1, yb0), (k + 1, yb1), (k, yb1)]
        al.add_patch(Polygon(pts, closed=True, fc=SC_COLORS[k], ec="#4A4A4A", lw=0.35, zorder=3))
        al.add_patch(Rectangle((k + 0.14, ybar), 0.72, shr[k] * sc_, fc=SC_COLORS[k], ec="#4A4A4A", lw=0.3, zorder=3))
        halo_text(al, k + 0.5, ybar + shr[k] * sc_ + 0.3, f"{shr[k]:.0f}%", fontsize=5.0, ha="center", va="bottom", color="#2A2A2A", zorder=4)
    for k, v in enumerate(SC_BOUNDS[1:-1]):
        halo_text(al, k + 1, yb0 - 0.4, f"{v:+d}".replace("-", "−"), fontsize=5.2, ha="center", va="top", color="#2A2A2A", zorder=4)
    halo_text(al, 1.5, 7.3, "1 km forecast closer", fontsize=5.2, ha="left", va="bottom", color=SC_COLORS[1], zorder=4)
    halo_text(al, nb, 7.3, "CAMS forecast closer", fontsize=5.2, ha="right", va="bottom", color=SC_COLORS[-2], zorder=4)
    halo_text(al, 0.5 * nb, 9.3, "Change in RMSE against the CAMS forecast (%)", fontsize=5.8, ha="center", va="bottom", color=INK, zorder=4)
    al.add_patch(Rectangle((nb + 0.75, yb0), 0.36, yb1 - yb0, fc=FEW, ec="#4A4A4A", lw=0.3, clip_on=False, zorder=3))      # key of the grey cells
    halo_text(al, nb + 1.3, 0.5 * (yb0 + yb1), "too few data", fontsize=5.2, ha="left", va="center", color="#2A2A2A", zorder=4, clip_on=False)

    # ------------------------------------------------------------------ geometry of the two rows of panels
    y2, h2 = 54.5, 42.0                                                        # row 2: b, c, d
    y3, h3 = 9.0, 33.0                                                         # row 3: e, f
    lead = np.arange(1, 6)

    # ------------------------------------------------------------------ b: by lead day
    Bl = num["b_lead"]
    hb2 = 15.0; gapb = 2.4
    ab = fig.add_axes(R(13.0, y2 + hb2 + gapb, 44.0, h2 - hb2 - gapb))
    for key in ("cams", "rt04", "clim", A):
        col, ls, lw, mk, ms = SYS[key]
        ab.plot(lead, Bl["rmse"][key], color=col, ls=ls, lw=lw, marker=mk, ms=ms, mec=col, mfc=col, zorder=5 if key == A else 3, clip_on=False)
    ab.set_xlim(0.65, 5.35); ab.set_xticks(lead); ab.set_ylim(15.0, 21.5); ab.set_yticks([15, 17, 19, 21])
    clean(ab); ab.tick_params(labelbottom=False)
    ab.set_ylabel("RMSE (µg m$^{-3}$)", fontsize=5.8, labelpad=2)
    titles.append((ab.set_title("By lead day", fontsize=6.5, loc="left", pad=2.5), "b"))
    ab2 = fig.add_axes(R(13.0, y2, 44.0, hb2))
    ab2.axhline(0, color="#333333", lw=0.5, zorder=2)
    for j, b in enumerate(("cams", "rt04", "clim")):
        col, ls, lw, mk, ms = SYS[b]
        xs = lead + (j - 1) * 0.14
        ab2.plot(xs, [v["est"] for v in Bl["change_pooled"][b]], color=col, ls=ls, lw=0.7, zorder=3)
        for x_, v in zip(xs, Bl["change_pooled"][b]):
            whisker(ab2, x_, v, col, mk, ms=ms)
    ab2.set_xlim(0.65, 5.35); ab2.set_xticks(lead); ab2.set_ylim(*YCH); ab2.set_yticks([-20, -10, 0])
    clean(ab2)
    ab2.set_xlabel("Lead day", fontsize=5.8, labelpad=1.5)
    ab2.set_ylabel("Change in\nRMSE (%)", fontsize=5.8, labelpad=2)
    fig.align_ylabels([ab, ab2])

    # ------------------------------------------------------------------ c: by region (forest plot, one column per reference)
    Cr = num["c_regions"]
    regs = sorted(Cr["order_of_scorer"], key=lambda g: Cr["change"]["cams"][g]["est"])
    rowsc = [("all", "All monitors", 0.0), ("regmean", "Mean of the regions", 1.0)] + [(g, Cr["figure_names"][g], 2.55 + i) for i, g in enumerate(regs)]
    xc, mmu, gc_ = 82.0, 0.37, 2.4                                             # one scale (mm per %) for the two columns
    cols_c = (("cams", (-57.0, 30.0), [-40, -20, 0, 20], "against CAMS forecast"), ("clim", (-33.0, 10.0), [-20, 0], "against climatology"))
    acs = []
    for b, xl, xt, head in cols_c:
        col, ls, lw, mk, ms = SYS[b]
        a = fig.add_axes(R(xc, y2, mmu * (xl[1] - xl[0]), h2))
        xc += mmu * (xl[1] - xl[0]) + gc_
        a.axhspan(-0.6, 1.55, color="#F3F3F3", lw=0, zorder=0)
        a.axvline(0, color="#333333", lw=0.5, zorder=2)
        for key, _, y_ in rowsc:
            whisker(a, y_, Cr["change"][b][key], col, mk, ms=ms + 0.5, lw=0.9, horizontal=True, lim=xl)
        a.set_xlim(*xl); a.set_ylim(rowsc[-1][2] + 0.6, -0.6); a.set_xticks(xt)
        clean(a); a.grid(axis="y", visible=False); a.tick_params(axis="y", length=0, labelsize=5.8, pad=3)
        a.set_yticks([r[2] for r in rowsc]); a.set_yticklabels([r[1] for r in rowsc] if not acs else [])
        a.spines["left"].set_visible(False)
        a.set_title(head, fontsize=5.6, loc="center", pad=2.5, color=col if b == "cams" else "#6E6E6E")
        acs.append(a)
    ac = acs[0]
    xm_ = 0.5 * (ac.get_position().x0 + acs[1].get_position().x1)
    xlab_c = fig.text(xm_, fy(y2 - 6.6), "Change in RMSE of the 1 km forecast (%)", fontsize=5.8, ha="center", va="center")
    mons = {"all": num["sample"]["monitors"], **Cr["monitors"]}
    xn = 1.0 + (11.0 / (mmu * 43.0))                                            # right edge of the column of counts, in axes widths of the second column
    acs[1].annotate("monitors", xy=(xn, 1.0), xycoords="axes fraction", xytext=(0, 2.5), textcoords="offset points", fontsize=5.6, color=MUTED,
                    ha="right", va="baseline", annotation_clip=False)
    for key, _, y_ in rowsc:
        if key in mons:
            acs[1].text(xn, y_, f"{mons[key]:,}", transform=acs[1].get_yaxis_transform(), fontsize=5.3, color="#2A2A2A", ha="right", va="center")

    # ------------------------------------------------------------------ d: away from the monitor's pixel
    Dr = num["d_rings"]
    ad = fig.add_axes(R(154.0, y2, 27.0, h2))
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.25])
    ad.axhline(0, color="#333333", lw=0.5, zorder=2)
    for j, b in enumerate(("cams", "rt04")):
        col, ls, lw, mk, ms = SYS[b]
        xx = xs + (j - 0.5) * 0.26
        ad.plot(xx[:4], [v["est"] for v in Dr["vs"][b]["pooled"][:4]], color=col, lw=0.7, zorder=3)
        for x_, v in zip(xx, Dr["vs"][b]["pooled"]):
            whisker(ad, x_, v, col, mk, ms=ms)
    ad.set_xlim(-0.55, 4.8); ad.set_ylim(*YCH); ad.set_yticks([-20, -10, 0])
    ad.set_xticks(xs); ad.set_xticklabels(["0", "1", "3", "10", "cell"])
    clean(ad)
    ad.set_xlabel("Distance from the monitor (km)", fontsize=5.8, labelpad=1.5)
    ad.set_ylabel("Change in RMSE (%)", fontsize=5.8, labelpad=2)
    titles.append((ad.set_title("Away from the monitor", fontsize=6.5, loc="left", pad=2.5), "d"))

    # ------------------------------------------------------------------ e: the record, one value per day of issue
    E = num["e_days"]
    d0 = dt.date.fromisoformat(E["issue_day"][0])
    xd = np.array([(dt.date.fromisoformat(s) - d0).days for s in E["issue_day"]])
    live = np.array(E["live"])
    seg = [xd <= 20, xd > 20]
    lims = [(xd[seg[0]].min() - 0.8, xd[seg[0]].max() + 0.8), (xd[seg[1]].min() - 0.8, xd[seg[1]].max() + 0.8)]
    xe, we, ge = 13.0, 168.0, 2.6        # main-text variant: panel e spans the row (the ensemble panel is a Supplementary figure)
    per = (we - ge) / sum(b_ - a_ for a_, b_ in lims)
    axe = []
    for k in range(2):
        w_ = per * (lims[k][1] - lims[k][0])
        a = fig.add_axes(R(xe if k == 0 else xe + per * (lims[0][1] - lims[0][0]) + ge, y3, w_, h3))
        a.axhline(0, color="#333333", lw=0.5, zorder=2)
        if live[seg[k]].any():
            xa = xd[seg[k] & live]
            a.axvspan(xa.min() - 0.5, xa.max() + 0.5, color=LIVE_BG, lw=0, zorder=0)
            a.text(0.5 * (xa.min() + xa.max()), YCH[1] - 0.9, "live", fontsize=5.4, color="#2A2A2A", ha="center", va="top")
            xo = xd[seg[k] & ~live]
            a.text(0.5 * (xo.min() + xo.max()), YCH[1] - 0.9, "reforecasts", fontsize=5.4, color=MUTED, ha="center", va="top")
        else:
            a.text(0.5 * (xd[seg[k]].min() + xd[seg[k]].max()), YCH[1] - 0.9, "reforecasts", fontsize=5.4, color=MUTED, ha="center", va="top")
        for b in ("clim", "cams"):
            col, ls, lw, mk, ms = SYS[b]
            est = np.array([v["est"] for v in E["change_pooled"][b]])
            sig = np.array([v["ci"][1] < 0 or v["ci"][0] > 0 for v in E["change_pooled"][b]])      # as in b-d: filled = interval excludes zero
            s_ = seg[k]
            a.plot(xd[s_], est[s_], color=col, ls=ls, lw=0.8, zorder=3)
            a.plot(xd[s_ & ~sig], est[s_ & ~sig], ls="", marker=mk, ms=ms, mfc="white", mec=col, mew=0.6, zorder=4)
            a.plot(xd[s_ & sig], est[s_ & sig], ls="", marker=mk, ms=ms, mfc=col, mec=col, mew=0.6, zorder=4)
        a.set_xlim(*lims[k]); a.set_ylim(*YCH); a.set_yticks([-20, -10, 0])
        tk = [d for d in xd[seg[k]] if (d0 + dt.timedelta(int(d))).weekday() == 6]          # Sundays
        a.set_xticks(tk); a.set_xticklabels([f"{(d0 + dt.timedelta(int(d))).day} {(d0 + dt.timedelta(int(d))):%b}" for d in tk])
        clean(a)
        if k == 1:
            a.spines["left"].set_visible(False); a.tick_params(axis="y", left=False, labelleft=False)
        axe.append(a)
    axe[0].set_ylabel("Change in RMSE of\nthe 1 km forecast (%)", fontsize=5.8, labelpad=2)
    xm = fx(xe + per * (lims[0][1] - lims[0][0]) + 0.5 * ge)
    for dx in (-0.45, 0.45):                                                   # break of the time axis
        fig.add_artist(Line2D([xm + fx(dx - 0.45), xm + fx(dx + 0.45)], [fy(y3 - 0.9), fy(y3 + 0.9)], color="#222222", lw=0.5))
    fig.text(fx(xe + 0.5 * we), fy(y3 - 6.6), f"Day of issue ({d0.year})", fontsize=5.8, ha="center", va="center")
    titles.append((axe[0].set_title("By day of issue", fontsize=6.5, loc="left", pad=2.5), "e"))

    # ------------------------------------------------------------------ the four systems (one legend for b-e)
    hs = [Line2D([], [], color=SYS[k][0], ls=SYS[k][1], lw=SYS[k][2], marker=SYS[k][3], ms=SYS[k][4], mfc=SYS[k][0], mec=SYS[k][0]) for k in (A, "rt04", "cams", "clim")]
    fig.legend(hs, [NAME_KM, NAME_04, NAME_CAMS, NAME_CLIM], fontsize=6.0, ncol=4, loc="center", bbox_to_anchor=(fx(6.0 + 0.5 * wa), fy(y2 + h2 + 8.6)), frameon=False,
               handlelength=2.4, columnspacing=2.2, handletextpad=0.6, borderaxespad=0)

    # ------------------------------------------------------------------ letters (after the layout is drawn)
    fig.canvas.draw()
    r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    tb = titles[0][0]
    ybase = inv.transform(tb.get_transform().transform(tb.get_position()))[1]
    xlab = inv.transform((ac.get_tightbbox(r).x0, 0))[0]                       # c: title above the names of the regions
    titles.insert(1, (fig.text(xlab, ybase, "By region", fontsize=6.5, ha="left", va="baseline"), "c"))
    bb = ab2.xaxis.label.get_window_extent(r)                                  # c: axis label on the line of the labels of b and d
    xlab_c.set_y(inv.transform((0, 0.5 * (bb.y0 + bb.y1)))[1])
    fig.canvas.draw()
    letter_left_of(fig, ax, "a", fy(ya + ha - 3.0))
    for t, s_ in titles:
        letter_before(fig, t, s_)
    fig.savefig(HERE / f"fig_rt_skill_main{suf}.pdf")
    fig.savefig(HERE / f"fig_rt_skill_main{suf}.png", dpi=300)
    print("saved", HERE / f"fig_rt_skill_main{suf}.pdf")


if __name__ == "__main__":
    main()
