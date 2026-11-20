#!/usr/bin/env python3
"""Main figure: a real-time forecast at 1 km (house style of fig_overview.py and fig_hidden.py; data from rt_day_data.py).
a,   the world at 1 km: real-time forecast (ensemble mean of 8 members) issued on the case day for that day (lead day 1);
b-e, the region of the case on that day: the CAMS operational forecast of the same cycle (b), the real-time 0.4 degree
     forecast (c) and the real-time 1 km forecast (d), each with the measurements of that day at the monitors in the same
     colours and RMSE | R | bias at those monitors (0.4 degree fields read bilinearly, as in the live verification);
     e, the 1 km forecasts for the same day from the cycles two and four days earlier (lead days 3 and 5);
f-g, daily series at the monitors of the two cities over the live period (open squares: cycles computed one day late;
     band: 10th-90th percentile of the 8 members at each monitor pixel, averaged over the monitors);
h-j, hidden in real time: 1 km forecast with the contour of 37.5 ug m-3, the 0.4 degree forecast, residents above 37.5 ug m-3
     at 1 km in 0.4 degree cells forecast above it (grey) or not (pink).
Cases by rule (rt_day_data.py). Nothing is fitted to the monitors: they are drawn and scored only.
Daily means (rt_day_data.py, rt_loaders.py): the CAMS forecast (map and scores of b, black lines of f, g) is the mean of its
eight 3-hourly steps 00 .. 21 UTC of the day, the rule of the paper's Methods (three instants until 2026-11-19: files
*_3instants.*); the real-time 0.4 degree forecast (c, i, green lines) keeps its three instants 00, 12, 24 UTC with weights
1/4, 1/2, 1/4, because its members exist every 12 h only. This script reads data/ and numbers*.json and knows neither.
Vector text (Liberation Sans, TrueType); only the 1 km images are rasterised (600 dpi); the 0.4 degree cells of b, c are vector.
    python fig_rt_day.py [variant]        -> fig_rt_day[_variant].pdf / .png beside this script (or ./render.sh)
The two files are written through rt_safe_io (memory buffer, temporary file, fsync, read back, os.replace)."""
import datetime as dt
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/scratch/project_462001804/ammar/eccv/cran-pm/experiments/figures")   # the paper's style modules (imported only)

import cartopy.crs as ccrs  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from gc_style import CAMS, FOEHN, LAND_BG, clean, geo_ticks, letter_before, letter_left_of, map_axes, world_coords  # noqa: E402
from nature_style import DOUBLE, MM, PM_BOUNDS, plt, pm_cmap  # noqa: E402

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "strict"
SUF = "" if VARIANT == "strict" else f"_{VARIANT}"
NAME_CAMS, NAME_04, NAME_KM = "CAMS forecast (40 km)", "Foehn in real time (40 km)", "Foehn in real time (1 km)"
RT04 = "#3E8E86"                                    # the real-time 0.4 degree forecast (teal of the paper's palette)
LEAD3 = "#8FB2DD"                                   # the 1 km forecast at a longer lead: light blue, dashed (as Fig. 4)
HID, SEEN = "#C2185B", "#3B3B3B"                    # hidden / seen residents, as Fig. 2
THR = 37.5
PC = ccrs.PlateCarree()
W_MM, H_MM = 183.0, 184.0
INK = "#111111"


def gauss(a, sigma):
    r = int(3 * sigma); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    for ax_ in (0, 1):
        a = np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode="reflect"), k, mode="valid"), ax_, a)
    return a


def frame(a):
    for sp in a.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.5); sp.set_color("#333333")


def main():
    Z = np.load(HERE / "data" / f"rt_day{SUF}.npz")
    num = json.loads((HERE / f"numbers{SUF}.json").read_text())
    day1, day2 = dt.date.fromisoformat(str(Z["day1"])), dt.date.fromisoformat(str(Z["day2"]))
    city1, cityn, city2 = str(Z["city1"]), str(Z["neighbour"]), str(Z["city2"])
    cmap, norm = pm_cmap()
    fig = plt.figure(figsize=(DOUBLE, H_MM * MM))
    fx = lambda mm: mm / W_MM; fy = lambda mm: mm / H_MM
    R = lambda x, y, w, h: [fx(x), fy(y), fx(w), fy(h)]
    f32 = lambda k: Z[k].astype(np.float32)
    marks = []

    # ------------------------------------------------------------------ a: the world at 1 km
    rob = ccrs.Robinson()
    xr = rob.transform_point(180, 0, PC)[0]; yt = rob.transform_point(0, 84, PC)[1]; yb = rob.transform_point(0, -60, PC)[1]
    wa = 170.0; ha = wa * (yt - yb) / (2 * xr); ya = H_MM - 1.5 - ha
    ax = map_axes(fig, R(6.0, ya, wa, ha))
    ax.imshow(f32("world"), origin="upper", extent=(-180, 180, -60, 84), transform=PC, cmap=cmap, norm=norm, regrid_shape=4400,
              zorder=1, interpolation="nearest")
    world_coords(ax, size=5.5)
    W_, E_, S_, N_ = Z["region"]; s2, n2, w2, e2 = Z["box2"]
    tr = PC._as_mpl_transform(ax)
    for (w, e, s, n), lab, (tx, ty, ha_) in (((W_, E_, S_, N_), "b–e", (E_ + 2.5, N_ + 3.5, "left")), ((w2, e2, s2, n2), "h–j", (w2 - 3.0, s2 - 5.5, "right"))):
        t_ = np.linspace(0, 1, 12)
        bx = np.concatenate([w + (e - w) * t_, np.full(12, e), e - (e - w) * t_, np.full(12, w)])
        by = np.concatenate([np.full(12, n), n - (n - s) * t_, np.full(12, s), s + (n - s) * t_])
        ax.plot(bx, by, transform=PC, color=INK, lw=0.6, zorder=6)
        ax.annotate(lab, xy=((e if ha_ == "left" else w), (n if ha_ == "left" else s)), xytext=(tx, ty), xycoords=tr, textcoords=tr,
                    fontsize=6, fontweight="bold", ha=ha_, va="center", zorder=7,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=INK, shrinkA=0, shrinkB=0),
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=INK, lw=0.5))
    cax = fig.add_axes(R(70.0, ya + 2.4, 47.0, 1.4))                          # inside the Southern Ocean, as Fig. 3a
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="max", spacing="uniform", ticks=PM_BOUNDS)
    cb.ax.set_xticklabels([f"{t:g}" for t in PM_BOUNDS]); cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.4, pad=1)
    cb.outline.set_linewidth(0.4); cb.ax.xaxis.set_label_position("top")
    cb.set_label(f"Daily mean PM$_{{2.5}}$ on {day1.day} {day1:%B} {day1.year} (µg m$^{{-3}}$)", fontsize=5.8, labelpad=2)

    # ------------------------------------------------------------------ b-e: the region of the case
    lat_mid = 0.5 * (N_ + S_); asp = (E_ - W_) * np.cos(np.deg2rad(lat_mid)) / (N_ - S_)
    ph = 43.0; pw = ph * asp; x0 = 8.0; gap = 2.3; y2 = 57.5
    ext = (W_, E_, S_, N_)
    lon, lat, obs = Z["reg_lon"], Z["reg_lat"], Z["reg_obs"]
    sc = num["case1"]["region"]["scores_at_monitors"]                    # at the monitors of the window on that day

    def region(rect, field, title, left, small=False, score=None, bold=False):
        a = fig.add_axes(rect); a.set_facecolor(LAND_BG)
        if field.shape[0] < 100:                                         # 0.4 degree cells: vector
            a.pcolormesh(np.linspace(W_, E_, field.shape[1] + 1), np.linspace(N_, S_, field.shape[0] + 1), field, cmap=cmap, norm=norm,
                         edgecolors="face", linewidth=0.2, zorder=1)
        else:                                                            # 1 km field: image
            a.imshow(field, extent=ext, cmap=cmap, norm=norm, interpolation="nearest", origin="upper", zorder=1)
        a.scatter(lon, lat, c=obs, cmap=cmap, norm=norm, s=2.2 if small else 5.5, linewidths=0.2 if small else 0.35, edgecolors=INK, zorder=5)
        for key in ("box1", "boxn"):
            s, n, w, e = Z[key]
            a.plot([w, e, e, w, w], [s, s, n, n, s], color=INK, lw=0.35 if small else 0.5, zorder=6)
        a.set_xlim(W_, E_); a.set_ylim(S_, N_); a.set_aspect(1 / np.cos(np.deg2rad(lat_mid)), adjustable="box")
        if small:
            geo_ticks(a, xt=(74, 77), yt=(28, 30, 32), left=True, bottom=left)
        else:
            geo_ticks(a, left=left, bottom=True)
        frame(a)
        t = a.set_title(title, fontsize=6.0 if small else 6.5, loc="left", pad=2.5)
        if score is not None:                                            # over the mountains: no monitor under the box
            a.text(0.955 if small else 0.975, 0.945 if small else 0.97, score, transform=a.transAxes, fontsize=5.0 if small else 5.2, zorder=8,
                   ha="right", va="top", linespacing=1.25, fontweight="bold" if bold else "normal",
                   bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.85))
        return a, t

    box = lambda q: (f"RMSE {q['rmse']:.1f} µg m$^{{-3}}$\nR {q['r']:.2f} | bias " + f"{q['bias']:+.1f}".replace("-", "−"))   # as the paper's day maps
    panels = (("cams", NAME_CAMS, box(sc["cams_lead1"]), False), ("rt04", NAME_04, box(sc["rt04_lead1"]), False), ("km1", NAME_KM, box(sc["km_lead1"]), True))
    axr = []
    for j, (key, title, score, bold) in enumerate(panels):
        a, t = region(R(x0 + j * (pw + gap), y2, pw, ph), f32(key), title, left=j == 0, score=score, bold=bold)
        marks.append((t, "bcd"[j])); axr.append(a)
        for key2, name in (("box1", city1), ("boxn", cityn)):
            s, n, w, e = Z[key2]
            a.text(0.5 * (w + e), n + 0.08, name, fontsize=5.5, ha="center", va="bottom", zorder=7,
                   path_effects=[pe.withStroke(linewidth=1.3, foreground="white")])
    L = 100.0; dlo = L / (111.2 * np.cos(np.deg2rad(lat_mid)))                                        # scale bar on d
    xb, yb_ = W_ + 0.03 * (E_ - W_), S_ + 0.055 * (N_ - S_)                                         # a corner without monitors
    axr[2].plot([xb, xb + dlo], [yb_, yb_], color=INK, lw=1.0, zorder=9, solid_capstyle="butt")
    axr[2].text(xb + dlo + 0.012 * (E_ - W_), yb_, f"{L:g} km", ha="left", va="center", fontsize=5.2, zorder=9,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.7))
    sg = 5.5; sh = (ph - sg) / 2; sw = sh * asp; xe = x0 + 3 * pw + 2 * gap + 8.5                    # e: two small maps
    for k, L_ in enumerate((3, 5)):
        d = day1 - dt.timedelta(L_ - 1)
        a, t = region(R(xe, y2 + (1 - k) * (sh + sg), sw, sh), f32(f"km{L_}"), f"1 km, cycle of {d.day} {d:%b}", left=k == 1, small=True,
                      score=f"RMSE {sc[f'km_lead{L_}']['rmse']:.1f}")
        if k == 0:
            marks.append((t, "e"))

    # ------------------------------------------------------------------ f-g: series at the monitors of the two cities
    days = [dt.date.fromisoformat(str(s)) for s in Z["days"]]; x = np.array([d.day for d in days])
    late = np.array([not num["rule"]["lead1_computed_on_issue_day"].get(str(d), True) for d in days])
    late04 = np.array([not num["rule"]["rt04_computed_on_issue_day"].get(str(d), True) for d in days])
    xs, ws, hs = 13.5, 55.5, 15.5; yg = 8.0; yf = yg + hs + 3.7
    axs = []
    for k, (city, y_) in enumerate(((city1, yf), (cityn, yg))):
        S = {q: Z[f"s_{city}_{q}"] for q in ("obs", "km1", "km1_p10", "km1_p90", "km3", "rt1", "cams1")}
        a = fig.add_axes(R(xs, y_, ws, hs))
        a.axvline(day1.day, color="#C9C9C9", lw=0.6, zorder=1)
        a.fill_between(x, S["km1_p10"], S["km1_p90"], color=FOEHN, alpha=0.18, lw=0, zorder=2)
        a.plot(x, S["cams1"], color=CAMS, lw=0.9, zorder=3)
        a.plot(x, S["rt1"], color=RT04, lw=1.0, zorder=4)
        a.plot(x, S["km3"], color=LEAD3, lw=1.0, ls=(0, (3, 1.6)), zorder=4.5)
        a.plot(x, S["km1"], color=FOEHN, lw=1.5, zorder=5)
        a.plot(x[late], S["km1"][late], ls="", marker="s", ms=2.3, mfc="white", mec=FOEHN, mew=0.7, zorder=5.5)
        a.plot(x[late04], S["rt1"][late04], ls="", marker="s", ms=2.1, mfc="white", mec=RT04, mew=0.6, zorder=4.2)
        a.plot(x, S["obs"], marker="o", ms=2.6, color=INK, mfc="white", mew=0.7, lw=0, zorder=6)
        a.set_xlim(2.5, 15.5); a.set_xticks(np.arange(3, 16, 2)); a.set_ylim(0, 300); a.set_yticks([0, 100, 200, 300])
        clean(a); a.tick_params(labelsize=5.5)
        a.set_ylabel(f"{city}\nPM$_{{2.5}}$ (µg m$^{{-3}}$)", fontsize=5.8, labelpad=2)
        if k == 0:
            a.tick_params(labelbottom=False)
        else:
            a.set_xlabel(f"{day1:%B} {day1.year}", fontsize=5.8, labelpad=1.5)
        axs.append(a)
    hd = [Line2D([], [], marker="o", ms=2.6, color=INK, mfc="white", mew=0.7, lw=0), Line2D([], [], color=FOEHN, lw=1.5),
          Patch(fc=FOEHN, alpha=0.18, lw=0), Line2D([], [], color=LEAD3, lw=1.0, ls=(0, (3, 1.6))), Line2D([], [], color=RT04, lw=1.0),
          Line2D([], [], color=CAMS, lw=0.9)]
    lb = ["Measured (monitors)", "Foehn 1 km, lead day 1", "10th–90th percentile of members", "Foehn 1 km, lead day 3", "Foehn 40 km, lead day 1",
          "CAMS forecast, lead day 1"]
    axs[0].legend(hd, lb, fontsize=5.2, ncol=2, loc="lower left", bbox_to_anchor=(-0.02, 1.04), frameon=False, handlelength=1.6,
                  columnspacing=1.2, handletextpad=0.5, labelspacing=0.35, borderaxespad=0)

    # ------------------------------------------------------------------ h-j: hidden in real time
    hx = f32("h_km"); hext = tuple(Z["h_ext"]); ok = np.isfinite(hx); pop = Z["h_pop"]; c04 = Z["h_rt04"]
    hm = 32.0; xh = 78.5; yh = 15.5; gh = 2.2
    above = ok & (hx > THR); seen = above & (c04 > THR)
    lp = np.clip((np.log10(np.maximum(pop, 1)) - 1.0) / 3.5, 0, 1)             # 10 .. 30,000 residents per km2 -> 0 .. 1
    rgb = np.ones(hx.shape + (4,)); rgb[..., 3] = ok
    for m, col in ((ok & ~above, "#BDBDBD"), (seen, SEEN), (above & ~seen, HID)):
        c = np.array(plt.matplotlib.colors.to_rgb(col)); al = (0.15 + 0.85 * lp)[..., None] * m[..., None]
        rgb[..., :3] = rgb[..., :3] * (1 - al) + c * al
    coast = gauss(ok.astype(float), 0.8)                                       # coast of the 1 km mask, lightly smoothed
    titles = (f"{city2}, 1 km, {day2.day} {day2:%b} {day2.year}", NAME_04, f"Residents above {THR:g} µg m$^{{-3}}$")
    yc = 84.2 - 0.4 * np.arange(np.ceil((84.2 - n2) / 0.4 - 1e-6), np.floor((84.2 - s2) / 0.4 + 1e-6) + 1)   # 0.4 degree cell edges
    xc = 0.4 * np.arange(np.ceil((w2 + 180.2) / 0.4 - 1e-6), np.floor((e2 + 180.2) / 0.4 + 1e-6) + 1) - 180.2
    axh = []
    for k in range(3):
        a = fig.add_axes(R(xh + k * (hm + gh), yh, hm, hm)); a.set_facecolor("white")
        if k < 2:
            a.imshow(np.where(ok, hx if k == 0 else c04, np.nan), extent=hext, origin="upper", cmap=cmap, norm=norm, interpolation="nearest", zorder=1)
        else:
            a.imshow(rgb, extent=hext, origin="upper", interpolation="nearest", zorder=1)
        if k == 0 and above.any():
            fs = gauss(np.where(ok, hx, np.nanmean(hx[ok])), 1.5)
            a.contour(np.where(ok, fs, np.nan), levels=[THR], extent=hext, origin="upper", colors="k", linewidths=0.5, zorder=3)
        a.contour(coast, levels=[0.5], extent=hext, origin="upper", colors="#6E6E6E", linewidths=0.3, zorder=4)
        for v in yc:
            a.axhline(v, color="white" if k < 2 else "#9E9E9E", lw=0.35, alpha=0.8, zorder=2)
        for v in xc:
            a.axvline(v, color="white" if k < 2 else "#9E9E9E", lw=0.35, alpha=0.8, zorder=2)
        a.scatter(Z["h_lon"], Z["h_lat"], c=Z["h_obs"], cmap=cmap, norm=norm, s=5.5, linewidths=0.35, edgecolors=INK, zorder=5)
        a.set_xlim(w2, e2); a.set_ylim(s2, n2); a.set_aspect(1 / np.cos(np.deg2rad(0.5 * (n2 + s2))), adjustable="box")
        geo_ticks(a, xt=xc, yt=yc[(yc > s2 + 0.05) & (yc < n2 - 0.05)], left=k == 0, bottom=True); frame(a)
        marks.append((a.set_title(titles[k], fontsize=6.5, loc="left", pad=2.5), "hij"[k])); axh.append(a)
    L = 25.0; dlo = L / (111.2 * np.cos(np.deg2rad(0.5 * (n2 + s2))))
    xb, yb_ = w2 + 0.07 * (e2 - w2), s2 + 0.09 * (n2 - s2)
    axh[0].plot([xb, xb + dlo], [yb_, yb_], color=INK, lw=1.0, zorder=9, solid_capstyle="butt")
    axh[0].text(xb + dlo / 2, yb_ + 0.025 * (n2 - s2), f"{L:g} km", ha="center", va="bottom", fontsize=5.2, zorder=9)
    hd = [Line2D([], [], color="k", lw=0.6), Patch(fc=HID, lw=0)]; lb = [f"{THR:g} µg m$^{{-3}}$, WHO interim target 3", "hidden at 40 km"]
    if seen.any():
        hd.insert(1, Patch(fc=SEEN, lw=0)); lb.insert(1, "seen at 40 km")
    fig.legend(hd, lb, fontsize=5.5, ncol=3, loc="upper center", bbox_to_anchor=(fx(xh + 1.5 * hm + gh), fy(yh - 5.2)), frameon=False,
               handlelength=1.4, columnspacing=1.6, handletextpad=0.5, borderaxespad=0)

    # ------------------------------------------------------------------ letters (after the layout is drawn)
    fig.canvas.draw()
    letter_left_of(fig, ax, "a", fy(ya + ha - 3.0))
    for t, s_ in marks:
        letter_before(fig, t, s_)
    letter_left_of(fig, axs, "f", fy(yf + hs - 2.0))
    letter_left_of(fig, axs, "g", fy(yg + hs - 2.0))
    import rt_safe_io                                                         # beside this script; imported here so that the constants above import alone
    rt_safe_io.savefig(fig, HERE / f"fig_rt_day{SUF}.pdf")
    rt_safe_io.savefig(fig, HERE / f"fig_rt_day{SUF}.png", dpi=300)
    print("saved", HERE / f"fig_rt_day{SUF}.pdf")


if __name__ == "__main__":
    main()
