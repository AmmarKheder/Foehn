#!/usr/bin/env python3
"""Main figure 'The pollution follows the air' (FLEXGREEN Green's functions and Foehn's 1 km forecasts of one case).
a-d, GHAP (1 km) on the three days before the arrival day and on it, and, on each of the first three days, where the air reaching
     the city on the arrival day was then (origins in the Green's functions of the forecasts started that day, leads 3, 2, 1;
     256 particles in each of the 3 x 3 receptor cells; data/transport_<case>.npz);
e-g, Foehn's probability of exceeding the threshold on the arrival day at 1 km (16 members), issued 3, 2 and 1 days before
     (eval_case_* crops: crop0_d<start>_h<lead>.npz);
h,   share of the air arriving over land that was in the mixed layer of the start day, by lead time, all land and regions
     (figdata_transport.py);
i,   reduction of the fair CRPS against GHAP brought by the computed transport, per start day of 2022 (violins; all regions),
     with the reduction over all forecasts and its 95% interval (figdata_transport_gain.py);
j,   that reduction against the share of h, per region and lead time, with the least-squares line.
Case: Kano, arrival 16 May 2022, chosen by a rule fixed before any forecast was examined (scratch transport_rule.py: among ten
African cities, the arrival day with the largest rise of PM2.5 explained by the air, min(PM2.5 at the origins two days before,
PM2.5 on arrival) - PM2.5 two days before).
    python fig_transport.py"""
import matplotlib.patheffects as pe
import numpy as np

import final_model as FM
from fig_exposure_zoom import ghap_box
from gc_style import clean, geo_ticks, label
from nature_style import DOUBLE, HERE, MM, PM_BOUNDS, plt, pm_cmap, save

REGC = {"East Asia": "#D55E00", "South Asia": "#E69F00", "Europe": "#0072B2", "North America": "#CC79A7", "Africa": "#009E73"}
BOX = {"East Asia": (54, 18, 97, 146), "South Asia": (37, 5, 60, 97), "Europe": (72, 35, -25, 45), "North America": (72, 15, -170, -50),
       "Africa": (37, -35, -20, 52)}
HALO = [pe.withStroke(linewidth=1.6, foreground="white")]
CASES = {"seoul": dict(npz="transport_seoul.npz", box=(42.0, 33.0, 112.0, 131.0), crop="eval_case_eastasia_crps_e01", city="Seoul",
                      thr=50, txt=(-1.0, 0.5)),
         "kano": dict(npz="transport_kano.npz", box=(28.0, 6.0, 0.0, 26.0), crop="eval_case_kano_crps_e01", city="Kano",
                     thr=100, txt=(-1.6, 0.6))}
CASE = CASES["kano"]
N_, S_, W_, E_ = CASE["box"]
R0, R1, C0, C1 = int(round((84 - N_) * 100)), int(round((84 - S_) * 100)), int(round((W_ + 180) * 100)), int(round((E_ + 180) * 100))
ASPECT = 1 / np.cos(np.deg2rad(0.5 * (N_ + S_)))
X0, DX = 9.5, 43.5                                                    # maps: left edge and step (mm); latitudes left of a and e


def kmap(fig, rect, field, cmap, norm, title, bold=False, positive=True, left=False, bottom=False):
    ax = fig.add_axes(rect)
    f = field[::2, ::2]                                       # 2 km display (the 1 km structure stays visible at print size)
    ax.imshow(np.where(np.nan_to_num(f) > 0, f, np.nan) if positive else f, extent=(W_, E_, S_, N_), origin="upper", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xlim(W_, E_); ax.set_ylim(S_, N_); ax.set_aspect(ASPECT)
    geo_ticks(ax, (5, 10, 15, 20), (10, 15, 20, 25), left=left, bottom=bottom)   # one extent: labels on the outer maps
    ax.set_facecolor("white")
    for s_ in ax.spines.values():
        s_.set_visible(True); s_.set_linewidth(0.4)
    ax.set_title(title, fontsize=6.2, pad=2, fontweight="bold" if bold else "normal")
    return ax


def main():
    z = np.load(HERE / "data" / "transport_fig.npz"); tg = np.load(HERE / "data" / "transport_gain.npz")
    R = [str(r) for r in z["regions"]]; GR = [str(r) for r in tg["regions"]]
    zs = np.load(HERE / "data" / CASE["npz"]); T = int(zs["T"]); sla, slo = float(zs["lat"]), float(zs["lon"])
    cmap, norm = pm_cmap()
    from fig1_teaser_v6 import tiles
    land1 = tiles("static1km_tiles.npy", 1)[R0:R1, C0:C1]
    day = lambda d: (np.datetime64("2022-01-01") + np.timedelta64(int(d), "D")).astype("datetime64[D]").item()
    mw = 41.5; mh = mw * (N_ - S_) * ASPECT / (E_ - W_)
    ym, yt = 45.0, 45.0 + mh + 9.0                                     # bottoms of the Foehn row and of the GHAP row (mm)
    W, H = DOUBLE, (yt + mh + 6.5) * MM
    fig = plt.figure(figsize=(W, H))
    fx = lambda mm: mm / (W / MM); fy = lambda mm: mm / (H / MM)
    # a-d: GHAP and where the air of the arrival day was
    for k, h in enumerate((3, 2, 1, 0)):
        d = T - h
        ax = kmap(fig, [fx(X0 + DX * k), fy(yt), fx(mw), fy(mh)], ghap_box(1826 + d, R0, R1, C0, C1).astype(np.float32),
                  cmap, norm, f"{day(d):%-d %B}" + (", arrival" if h == 0 else ""), bold=h == 0, left=k == 0, bottom=k == 0)
        if h > 0:
            o = zs[f"org_{h}"]; keep = np.nan_to_num(o[:, 3]) > 0.05
            ax.scatter(o[keep, 0], o[keep, 1], s=0.4, color="#111111", alpha=0.45, linewidths=0, zorder=4, rasterized=True)
        ax.plot(slo, sla, marker="*", ms=7, color="white", mec="#111111", mew=0.6, zorder=6)
        if k == 3:
            ax.text(slo + CASE["txt"][1], sla + CASE["txt"][0], CASE["city"], fontsize=5.8, fontweight="bold", zorder=6, path_effects=HALO)
        label(fig, fx(X0 + DX * k), fy(yt + mh + 3.6), "abcd"[k])
    fig.text(fx(1.5), fy(yt + mh / 2), "GHAP (1 km)", rotation=90, ha="left", va="center", fontsize=6.2)
    # e-g: Foehn's chance of exceeding the threshold on the arrival day, issued 3, 2 and 1 days before
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
    pcm = LinearSegmentedColormap.from_list("prob", ["#FFFFFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"], N=10)
    pn = BoundaryNorm(np.linspace(0, 1, 11), 10)
    thr = CASE["thr"]
    for k, h in enumerate((3, 2, 1)):
        z_ = np.load(FM.RUN_DIR / CASE["crop"] / f"crop0_d{T - h}_h{h}.npz")
        assert int(z_["rows"][0]) == R0 and int(z_["cols"][0]) == C0, (z_["rows"], z_["cols"])
        m = z_["members"].astype(np.float32)
        pr = np.where(land1 > 0, (m > thr).mean(0), np.nan)
        ax = kmap(fig, [fx(X0 + DX * (k + 1)), fy(ym), fx(mw), fy(mh)], pr, pcm, pn,
                  f"{day(T):%-d %B}, issued {h} day{'s' if h > 1 else ''} before", bold=True, positive=False, left=k == 0, bottom=True)
        ax.plot(slo, sla, marker="*", ms=7, color="white", mec="#111111", mew=0.6, zorder=6)
        label(fig, fx(X0 + DX * (k + 1)), fy(ym + mh + 3.6), "efg"[k])
        iy, ix = int((N_ - sla) * 100), int((slo - W_) * 100)
        v = m[:, iy - 15:iy + 15, ix - 15:ix + 15].mean((1, 2))
        print(f"issued {h} d before: members at {CASE['city']} {np.round(np.sort(v)).astype(int)} | P>{thr} {np.mean(v > thr):.2f}")
    fig.text(fx(X0 + 0.5), fy(ym + mh - 0.5), f"Foehn: chance of PM$_{{2.5}}$\n> {thr:g} µg m$^{{-3}}$ on {day(T):%-d %B}", fontsize=6.0,
             fontweight="bold", va="top", color="#1F5AA6")
    cax = fig.add_axes([fx(X0 + 0.5), fy(ym + mh - 10.5), fx(35.5), fy(1.6)])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=pn, cmap=pcm), cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
    cb.ax.tick_params(labelsize=4.8, length=1.5, pad=1); cb.outline.set_linewidth(0.4)
    cb.set_label("Probability (16 members)", fontsize=5.0, labelpad=1)
    cax = fig.add_axes([fx(X0 + 0.5), fy(ym + mh - 19.5), fx(35.5), fy(1.6)])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="max", spacing="uniform",
                      ticks=PM_BOUNDS[::2])
    cb.ax.set_xticklabels([f"{t:g}" for t in PM_BOUNDS[::2]]); cb.ax.tick_params(labelsize=4.8, length=1.5, pad=1); cb.outline.set_linewidth(0.4)
    cb.set_label("Daily PM$_{2.5}$ (µg m$^{-3}$) in a-d", fontsize=5.0, labelpad=1)
    # h, where the arriving air was on the start day (near the ground, in the mixed layer, or higher up), by lead time;
    # i, the same share place by place three days ahead; j, the gain brought by computing the transport, by lead time.
    from matplotlib.patches import Patch
    L = z["leads"]; g0 = R.index("Global")
    near = 100 - 100 * np.nanmean(z["aloft"][:, :, g0], 0)                       # all land, mean over the 90 start days
    dist = z["dist"][:, :, g0]; dmed = np.nanmean(dist, 0); d25, d75 = np.nanpercentile(dist, [25, 75], axis=0)
    ah = fig.add_axes([fx(13), fy(8), fx(46), fy(27)])
    ai = fig.add_axes([fx(72), fy(8), fx(46), fy(27)])
    aj = fig.add_axes([fx(133), fy(8), fx(45), fy(27)])
    ah.fill_between(L, 0, near, color="#1F5AA6", alpha=0.85, lw=0, zorder=2)
    ah.fill_between(L, near, 100, color="#F5BFD3", lw=0, zorder=1)
    ah.plot(L, near, color="#111111", lw=1.2, marker="o", ms=2.4, zorder=4)
    for hh, dx, ha in ((1, 3, "left"), (3, 0, "center"), (7, -3, "right")):
        ah.annotate(f"{near[hh - 1]:.0f}%", (hh, near[hh - 1]), textcoords="offset points", xytext=(dx, -9), ha=ha,
                    va="top", fontsize=5.6, fontweight="bold", color="white", zorder=7)
    ah.set_xlim(1, 7); ah.set_ylim(0, 100); ah.set_xticks(L)
    ah.set_xlabel("Lead time (days)", fontsize=6, labelpad=1)
    ah.set_ylabel("Arriving air (%)", fontsize=6)
    ah.legend(handles=[Patch(color="#1F5AA6", alpha=0.85, label="near the ground on the start day"),
                       Patch(color="#F5BFD3", label="higher up, or farther back")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1, fontsize=5.2, frameon=False, handlelength=1.4,
              borderaxespad=0.2, labelspacing=0.25)
    import cartopy.crs as ccrs
    from gc_style import map_axes, world_coords
    land05 = np.load("/scratch/project_462001140/ammar/eccv/data/world/global1km/meta.npz")["land"].astype(bool).reshape(288, 2, 720, 2).any((1, 3))
    near_map = np.where(land05, 100 - 100 * z["aloft_map"][2], np.nan)             # three days ahead
    lat_e = 84.125 - 0.5 * np.arange(289); lon_e = -180.125 + 0.5 * np.arange(721)
    ai.remove()
    ai = map_axes(fig, [fx(65), fy(12), fx(52), fy(26)])                         # room for its coordinates
    pm_ = ai.pcolormesh(lon_e, lat_e, near_map, cmap="Blues", vmin=20, vmax=70, transform=ccrs.PlateCarree(), rasterized=True, zorder=2)
    world_coords(ai, size=4.6)
    cax_ = fig.add_axes([fx(74), fy(8.2), fx(34), fy(1.4)])
    cb_ = fig.colorbar(pm_, cax=cax_, orientation="horizontal", ticks=[20, 30, 40, 50, 60, 70], extend="both"); cb_.outline.set_linewidth(0.4)
    cb_.ax.tick_params(labelsize=4.8, length=1.5, pad=1)
    cb_.set_label("Arriving air near the ground on the start day,\n3 days ahead (%)", fontsize=5.2, labelpad=1.5)
    print("map 3 d: land percentiles", np.nanpercentile(near_map, [5, 25, 50, 75, 95]).round(0))
    # j: the gain per start day, coloured violins (tails trimmed at the 2nd-98th percentiles), point and 95% interval
    ga = GR.index("All regions"); gain, lo, hi = tg["gain"][ga, :, 0], tg["gain"][ga, :, 1], tg["gain"][ga, :, 2]
    gd = tg["gain_day"][ga]                                                  # (leads, start days)
    vcol = plt.cm.viridis(np.linspace(0.15, 0.85, 7))
    data = []
    for v in gd:
        v = v[np.isfinite(v)]; qa, qb = np.percentile(v, [2, 98]); data.append(v[(v >= qa) & (v <= qb)])
    parts = aj.violinplot(data, positions=L, widths=0.75, showextrema=False, showmedians=False)
    for b, c in zip(parts["bodies"], vcol):
        b.set_facecolor(c); b.set_edgecolor("none"); b.set_alpha(0.9)
    aj.errorbar(L, gain, yerr=[gain - lo, hi - gain], fmt="o", color="#111111", ms=2.2, elinewidth=0.8, capsize=1.5, zorder=6)
    aj.axhline(0, color="#888888", lw=0.5, zorder=1)
    aj.set_ylim(-8, 13.5); aj.set_xticks(L)
    aj.set_xlabel("Lead time (days)", fontsize=6, labelpad=1)
    aj.set_ylabel("CRPS reduction by the\ncomputed transport (%)", fontsize=6)
    for a_ in (ah, aj):
        clean(a_); a_.tick_params(labelsize=5.5)
    for k in range(3):
        label(fig, fx((2, 61, 122)[k]), fy(37), "hij"[k])
    print("near-ground share, all land:", np.round(near, 1), "| distance km:", np.round(dmed), "| gain:", np.round(gain, 1))
    save(fig, "fig_transport")


if __name__ == "__main__":
    main()
