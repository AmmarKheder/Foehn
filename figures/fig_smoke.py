#!/usr/bin/env python3
"""Main figure: forecasting the smoke that harms people and wildlife (the Gironde fires, July 2022).
a,   GHAP daily PM2.5 at 1 km on 18 July 2022, the day of the highest smoke at the Teich ornithological reserve (rule in
     Methods), with the GFAS fire cells of the day (triangles: the La Teste-de-Buch fire on the Arcachon basin and the
     Landiras fires) and the EEA stations (circles, coloured by their measurement);
b-e, Foehn's probability of PM2.5 above 25 ug m-3 (WHO interim target 4) on 18 July at 1 km (16 members), issued seven,
     six, five and one days before; triangles: the fire cells in the inputs of that forecast (GFAS of the start day);
     the fires ignited on 12 July (eval_case_gironde_crps_e01 for leads 1-3, eval_case_gironde_long_crps_e01 for 4-7);
f,   GHAP daily PM2.5 at the reserve (20 km box around it), 8-26 July;
g,   Foehn's probability of exceeding 25 ug m-3 at the reserve (ensemble members averaged over the same box), by target
     day, for the forecasts issued one, three, five and seven days before.
    python fig_smoke.py"""
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

import final_model as FM
from fig1_teaser_v6 import tiles
from fig_exposure_zoom import ghap_box
from gc_style import clean, label
from nature_style import DOUBLE, MM, PM_BOUNDS, plt, pm_cmap, save

G = "/scratch/project_462001140/ammar/eccv/data/world/global1km"
SHORT, LONG = FM.RUN_DIR / "eval_case_gironde_crps_e01", FM.RUN_DIR / "eval_case_gironde_long_crps_e01"
N_, S_, W_, E_ = 46.0, 43.4, -2.6, 1.0                        # display box (the evaluated crop is 47-43 N, -3-1 E)
CR0, CC0 = 3700, 17700                                        # first row and column of the stored crop on the 1 km grid
R0, R1 = int((84 - N_) * 100), int((84 - S_) * 100); C0, C1 = int((W_ + 180) * 100), int((E_ + 180) * 100)
T = 198                                                       # 18 July 2022, the highest smoke at the reserve
THR = 25.0                                                    # WHO 24-h interim target 4
BORD = (44.84, -0.58); TEICH = (44.64, -1.02)
HALO = [pe.withStroke(linewidth=1.6, foreground="white")]
ASPECT = 1 / np.cos(np.deg2rad(0.5 * (N_ + S_)))
FIRE = "#7E1239"


def day(d):
    return (np.datetime64("2022-01-01") + np.timedelta64(int(d), "D")).astype("datetime64[D]").item()


def members(t, h):
    return np.load((SHORT if h <= 3 else LONG) / f"crop0_d{t - h}_h{h}.npz")["members"].astype(np.float32)


def kmap(fig, rect, field, cmap, norm, title, bold=False, coast=None):
    ax = fig.add_axes(rect)
    if coast is not None:
        yy_ = np.linspace(N_, S_, coast.shape[0]); xx_ = np.linspace(W_, E_, coast.shape[1])
        ax.contour(xx_, yy_, coast.astype(float), levels=[0.5], colors="#9AA5AE", linewidths=0.35, zorder=3)
    ax.imshow(field, extent=(W_, E_, S_, N_), origin="upper", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xlim(W_, E_); ax.set_ylim(S_, N_); ax.set_aspect(ASPECT); ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("white")
    for s_ in ax.spines.values():
        s_.set_linewidth(0.4)
    ax.set_title(title, fontsize=6.2, pad=2.5, fontweight="bold" if bold else "normal")
    return ax


def places(ax, fire_pts, labels=False):
    if len(fire_pts):
        ax.scatter(fire_pts[:, 1], fire_pts[:, 0], marker="^", s=11, color=FIRE, edgecolors="white", linewidths=0.4, zorder=6)
    ax.plot(*BORD[::-1], marker="*", ms=6.5, color="white", mec="#111111", mew=0.6, zorder=7)
    ax.plot(*TEICH[::-1], marker="o", ms=3.4, mfc="none", mec="#111111", mew=0.85, zorder=7)
    if labels:
        ax.text(TEICH[1] - 0.06, TEICH[0] - 0.12, "Teich bird\nreserve", fontsize=4.9, ha="right", va="top", zorder=7,
                path_effects=HALO)
        ax.text(BORD[1] + 0.08, BORD[0] + 0.06, "Bordeaux", fontsize=5.4, fontweight="bold", zorder=7, path_effects=HALO)


def main():
    cmap, norm = pm_cmap()
    land = tiles("static1km_tiles.npy", 1)
    lnd = land[R0:R1, C0:C1] > 0
    fires = np.load(f"{G}/fires.npy", mmap_mode="r")
    st = np.load(f"{G}/stations_2022.npz")
    sel = (st["lat"] <= N_) & (st["lat"] >= S_) & (st["lon"] >= W_) & (st["lon"] <= E_) & (st["source"] == 0)
    fr0, fc0 = int((84 - N_) / 0.25), int((W_ + 180) / 0.25); fr1, fc1 = int((84 - S_) / 0.25) + 1, int((E_ + 180) / 0.25) + 1

    def fire_pts(d):
        fr = np.asarray(fires[1826 + d, 0, fr0:fr1, fc0:fc1], np.float32)
        return np.array([(84 - 0.25 * (fr0 + i) - 0.125, -180 + 0.25 * (fc0 + j) + 0.125) for i, j in np.argwhere(fr > 3)])

    tr, tc = int((84 - TEICH[0]) * 100), int((TEICH[1] + 180) * 100)

    def at_reserve(m):                                         # members averaged over the 20 km box around the reserve
        return m[:, tr - CR0 - 10:tr - CR0 + 10, tc - CC0 - 10:tc - CC0 + 10].mean((1, 2))

    mw = 33.4; mh = mw * (N_ - S_) * ASPECT / (E_ - W_)
    yb, yc, ym = 9.0, 53.5, 62.0                               # bottoms of the series row, the colour bars and the maps (mm)
    W, H = DOUBLE, (ym + mh + 8.0) * MM
    fig = plt.figure(figsize=(W, H))
    fx = lambda mm: mm / (W / MM); fy = lambda mm: mm / (H / MM)
    xs = [3.5 + 35.5 * k for k in range(5)]

    # a: the smoke on 18 July in the 1 km estimates
    g = ghap_box(1826 + T, R0, R1, C0, C1).astype(np.float32)
    ax = kmap(fig, [fx(xs[0]), fy(ym), fx(mw), fy(mh)], np.where(lnd & (g > 0), g, np.nan), cmap, norm,
              f"{day(T):%-d %B}, GHAP (1 km)", bold=True, coast=lnd)
    ax.scatter(st["lon"][sel], st["lat"][sel], c=st["daily"][sel, T].astype(np.float32), cmap=cmap, norm=norm, s=9,
               linewidths=0.5, edgecolors="#111111", zorder=6)
    places(ax, fire_pts(T), labels=True)
    label(fig, fx(xs[0] - 1.5), fy(ym + mh + 3.2), "a")
    from scipy.ndimage import uniform_filter
    gsm = uniform_filter(np.where(lnd, g, 0.0).astype(np.float32), 5)      # the observed smoke, smoothed over 5 km

    # b-e: the chance of smoke on 18 July, issued 7, 6, 5 and 1 days before, with the fires the forecast knew of
    pcm = LinearSegmentedColormap.from_list("prob", ["#FFFFFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"], N=10)
    pn = BoundaryNorm(np.linspace(0, 1, 11), 10)
    for k, h in enumerate((7, 6, 5, 1)):
        m = members(T, h)[:, R0 - CR0:R1 - CR0, C0 - CC0:C1 - CC0]
        pr = np.where(lnd, (m > THR).mean(0), np.nan)
        ax = kmap(fig, [fx(xs[k + 1]), fy(ym), fx(mw), fy(mh)], pr, pcm, pn,
                  f"Issued {h} day{'s' if h > 1 else ''} before ({day(T - h):%-d %b})", bold=h == 5, coast=lnd)
        ax.contour(np.linspace(W_, E_, g.shape[1]), np.linspace(N_, S_, g.shape[0]), gsm, levels=[THR], colors="#111111",
                   linewidths=0.55, zorder=4)
        places(ax, fire_pts(T - h))
        label(fig, fx(xs[k + 1] - 1.5), fy(ym + mh + 3.2), "bcde"[k])
        vt = at_reserve(members(T, h))
        print(f"issued {h} d before ({day(T - h):%d %b}, {len(fire_pts(T - h))} fire cells in the inputs): reserve P>{THR:g} "
              f"{np.mean(vt > THR):.2f} (members {np.round(np.sort(vt)).astype(int)}); basin cells P>0.5 {np.nanmean(pr > 0.5):.2f}")
    cax = fig.add_axes([fx(xs[0] + 1), fy(yc), fx(mw - 2), fy(1.5)])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="max",
                      spacing="uniform", ticks=PM_BOUNDS[::2])
    cb.ax.set_xticklabels([f"{t:g}" for t in PM_BOUNDS[::2]]); cb.ax.tick_params(labelsize=4.8, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4); cb.set_label("Daily PM$_{2.5}$ (µg m$^{-3}$)", fontsize=5.4, labelpad=1)
    cax = fig.add_axes([fx(xs[2]), fy(yc), fx(xs[3] + mw - xs[2]), fy(1.5)])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=pn, cmap=pcm), cax=cax, orientation="horizontal",
                      ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
    cb.ax.tick_params(labelsize=4.8, length=1.5, pad=1); cb.outline.set_linewidth(0.4)
    cb.set_label(f"Probability of PM$_{{2.5}}$ > {THR:g} µg m$^{{-3}}$ from the 16 members (b–e, on {day(T):%-d %B}; g, at the reserve)", fontsize=5.4, labelpad=1)

    # f: the smoke at the reserve; g: the chance of it, by target day (columns) and lead time (rows); h: the 16 members
    # at the reserve on 18 July, lead by lead
    days_f = np.arange(193, 207)                               # 13-26 July
    gr = []
    for d in days_f:
        v = ghap_box(1826 + d, tr - 10, tr + 10, tc - 10, tc + 10).astype(np.float32)
        gr.append(float(v[v > 0].mean()))
    cal = lambda d: np.asarray(d) - 181 + 1                    # day of July
    def on_fires(d):                                           # GFAS cells over the two fires (44.25-44.75 N, 1.25-0.25 W)
        pts = fire_pts(d)
        return len(pts) > 0 and bool(np.any((pts[:, 0] > 44.25) & (pts[:, 0] < 44.75) & (pts[:, 1] > -1.25) & (pts[:, 1] < -0.25)))
    fire_days = [d for d in days_f if on_fires(d)]
    print("days with GFAS cells over the two fires:", [f"{day(d):%d %b}" for d in fire_days])
    af = fig.add_axes([fx(19), fy(yb + 19.5), fx(98), fy(10.5)])
    ag = fig.add_axes([fx(19), fy(yb), fx(98), fy(17)])
    for d in fire_days:
        af.axvspan(cal(d) - 0.5, cal(d) + 0.5, color="#F6E3E8", lw=0, zorder=0)
    af.axvline(cal(T), color="#9AA5AE", lw=0.5, ls=(0, (2, 1.5)), zorder=1)
    af.plot(cal(days_f), gr, color="#111111", lw=1.0, marker="o", ms=2.3, mfc="white", mew=0.7, zorder=4)
    af.axhline(THR, color="#7C8994", lw=0.5, ls=(0, (3, 2)), zorder=2)
    af.set_xlim(cal(193) - 0.5, cal(206) + 0.5); af.set_ylim(0, 40); af.set_yticks([0, 25, 40])
    af.set_xticks(cal(days_f)); af.tick_params(axis="x", labelbottom=False, length=1.5)
    af.set_ylabel("PM$_{2.5}$ at the\nreserve (µg m$^{-3}$)", fontsize=6)
    clean(af); af.tick_params(labelsize=5.5)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hf = [Line2D([0], [0], color="#111111", lw=1.0, marker="o", ms=2.3, mfc="white", mew=0.7, label="GHAP, 20 km around the reserve"),
          Patch(color="#F6E3E8", label="days with fire cells (GFAS)"),
          Line2D([0], [0], color="#7C8994", lw=0.6, ls=(0, (3, 2)), label=f"{THR:g} µg m$^{{-3}}$, WHO interim target 4"),
          Line2D([0], [0], color="#9AA5AE", lw=0.6, ls=(0, (2, 1.5)), label=f"{day(T):%-d %B}, the day of a–e"),
          Patch(color="#EDEDED", label="no forecast in this evaluation (g)")]
    af.legend(handles=hf, loc="lower left", bbox_to_anchor=(0.0, 1.04), ncol=3, fontsize=5.0, frameon=False,
              handlelength=1.8, columnspacing=1.2, labelspacing=0.3, borderaxespad=0.0)
    label(fig, fx(2), fy(yb + 30.5), "f")
    LD = np.arange(1, 8); P = np.full((len(LD), len(days_f)), np.nan)
    for i_, h in enumerate(LD):
        for j_, t in enumerate(days_f):
            f_ = (SHORT if h <= 3 else LONG) / f"crop0_d{t - h}_h{h}.npz"
            if f_.exists():
                P[i_, j_] = float(np.mean(at_reserve(members(t, h)) > THR))
    pcm_g = pcm.copy(); pcm_g.set_bad("#EDEDED")
    ag.imshow(np.ma.masked_invalid(P), cmap=pcm_g, norm=pn, aspect="auto", interpolation="nearest",
              extent=(cal(193) - 0.5, cal(206) + 0.5, 7.5, 0.5), zorder=2)
    for i_ in range(len(LD)):
        for j_ in range(len(days_f)):
            if np.isfinite(P[i_, j_]) and P[i_, j_] >= 0.18:
                ag.text(cal(days_f[j_]), LD[i_], f"{P[i_, j_]:.2f}"[1:], ha="center", va="center", fontsize=4.3,
                        color="white" if P[i_, j_] >= 0.5 else "#111111", zorder=5)
    ag.set_xticks(np.arange(0.5, 15, 1) + cal(193) - 1, minor=True); ag.set_yticks(np.arange(0.5, 8, 1), minor=True)
    ag.grid(which="minor", color="white", lw=0.6); ag.tick_params(which="minor", length=0)
    ag.axvline(cal(T), color="#9AA5AE", lw=0.5, ls=(0, (2, 1.5)), zorder=6)
    ag.set_xlim(cal(193) - 0.5, cal(206) + 0.5); ag.set_ylim(7.5, 0.5)
    ag.set_xticks(cal(days_f)); ag.set_yticks(LD); ag.set_yticklabels([f"{h} d" for h in LD])
    ag.tick_params(labelsize=5.5, length=1.5, pad=1.5)
    ag.set_xlabel("July 2022 (target day)", fontsize=6, labelpad=1)
    ag.set_ylabel("Issued before", fontsize=6)
    for s_ in ag.spines.values():
        s_.set_visible(False)
    label(fig, fx(2), fy(yb + 17.5), "g")
    ah = fig.add_axes([fx(137), fy(yb), fx(41), fy(28)])
    rng = np.random.default_rng(0)
    for h in LD:
        v = at_reserve(members(T, h)); x_ = 8 - h
        ah.scatter(x_ + rng.uniform(-0.18, 0.18, v.size), v, s=6, color="#1F5AA6", alpha=0.75, linewidths=0, zorder=4)
        ah.plot([x_ - 0.3, x_ + 0.3], [v.mean()] * 2, color="#111111", lw=1.1, zorder=5)
        print(f"members 18 Jul, {h} d before: {int((v > THR).sum())}/16 above {THR:g}, mean {v.mean():.0f}, max {v.max():.0f}")
    ah.axhline(gr[list(days_f).index(T)], color="#555555", lw=0.8, ls=(0, (1, 1.2)), zorder=3)
    ah.axhline(THR, color="#7C8994", lw=0.5, ls=(0, (3, 2)), zorder=2)
    ah.set_xlim(0.4, 7.6); ah.set_xticks(range(1, 8)); ah.set_xticklabels([f"{8 - x} d" for x in range(1, 8)])
    ah.set_ylim(0, 55); ah.set_yticks([0, 25, 50])
    ah.set_xlabel(f"Forecast of {day(T):%-d %B}, issued before", fontsize=6, labelpad=1)
    ah.set_ylabel("PM$_{2.5}$ at the reserve,\n16 members (µg m$^{-3}$)", fontsize=6)
    clean(ah); ah.tick_params(labelsize=5.5)
    hh_ = [Line2D([0], [0], marker="o", ms=3.0, color="#1F5AA6", alpha=0.75, lw=0, label="16 members"),
           Line2D([0], [0], color="#111111", lw=1.1, label="ensemble mean"),
           Line2D([0], [0], color="#555555", lw=0.8, ls=(0, (1, 1.2)), label=f"GHAP on {day(T):%-d %B}"),
           Line2D([0], [0], color="#7C8994", lw=0.6, ls=(0, (3, 2)), label=f"{THR:g} µg m$^{{-3}}$")]
    ah.legend(handles=hh_, loc="lower left", bbox_to_anchor=(0.0, 1.04), ncol=2, fontsize=5.0, frameon=False,
              handlelength=1.8, columnspacing=1.2, labelspacing=0.3, borderaxespad=0.0)
    label(fig, fx(126), fy(yb + 28.8), "h")
    print("GHAP at the reserve:", " ".join(f"{day(d):%d}:{v:.0f}" for d, v in zip(days_f, gr)))
    save(fig, "fig_smoke")


if __name__ == "__main__":
    main()
