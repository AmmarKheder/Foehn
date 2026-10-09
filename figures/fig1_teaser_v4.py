#!/usr/bin/env python3
"""Fig. 1 (teaser, v4, GHAP-paper style): what each forecast can see, and what it changes for alerts.
a-e, annual mean PM2.5 of 2022 over the valid days 2 January to 31 December (figdata_teaser_fc.py): a, residents per km2
(GHS-POP 2020); b, GHAP at 1 km, the reference (pixel-days of the forecasts); c, CAMS forecast for the next day on its 40 km
grid; d, Aurora forecast for the next day on the same grid; e, Foehn forecast for the next day at 1 km (ensemble mean). Every
forecast is issued at 00 UTC of the day it predicts. Red frames: three zones with dense official networks on three continents,
shown on the right in the same views, the 40 km cells drawn; dots: the monitoring stations (evaluation only), coloured by
their observed annual mean.
f, alerts for the next day over 2022 (figdata_alerts.py): residents above the WHO 24-h interim target at their 1 km pixel
(GHAP) warned the day before or missed, and residents warned below it (false alerts), per region; a perfect 40 km forecast
is GHAP of the day averaged over the 40 km cell.
g, the zones on their episode day (figdata_teaser_episodes.py: the day of 2022 with the largest exposure above the target
hidden by the 40 km cells, chosen from GHAP alone): outcome of the alert for every resident, CAMS (40 km) and Foehn (1 km).
No figure title.
    python fig1_teaser_v4.py [--preview]     (--preview: layout only, before the Foehn run; Foehn fields left empty)"""
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, LogNorm

from nature_style import COLOR, DOUBLE, HERE, MM, plt, save

ZONES = {"Beijing": (41.1, 38.7, 114.63, 117.77), "Po Valley and Alps": (46.6, 44.2, 8.29, 11.71),
         "Los Angeles": (35.25, 32.85, -119.45, -116.55)}                   # ~270 km squares
PM_CMAP = LinearSegmentedColormap.from_list("ghap", ["#313695", "#4575B4", "#74ADD1", "#ABD9E9", "#E0F3F8", "#FFFFBF", "#FEE090",
                                                     "#FDAE61", "#F46D43", "#D73027", "#A50026"], N=256)
POP_CMAP = LinearSegmentedColormap.from_list("pop", ["#F7FCFD", "#BFD3E6", "#8C96C6", "#8C6BB1", "#88419D", "#6E016B", "#3F0045"], N=256)
PM_B = [0, 5, 10, 15, 25, 35, 50, 75, 100]                                   # WHO 2021 annual guideline 5, interim targets 10-35
PM_NORM = BoundaryNorm(PM_B, PM_CMAP.N, extend="max")
POP_NORM = LogNorm(1, 10000)
OUTCOME = {"warned": "#2B2B2B", "missed": "#E66101", "false alert": "#4393C3"}  # residents above and warned / above, no alert / below, warned
EXT = (-180, 180, -58, 80)
MM_W = DOUBLE / MM
D = HERE / "data" / "fc_annual_2022"
G = "/scratch/project_462001140/ammar/eccv/data/world/global1km/"
NETS = [0, 1, 3, 4, 5, 6]                                                    # EEA, CNEMC, EPA, AirKorea, NIES, CPCB
ROWS = [("pop", "Where people live"), ("ghap", "PM$_{2.5}$ at 1 km (GHAP reference)"),
        ("cams", "CAMS forecast for the next day, 40 km"), ("aurora", "Aurora forecast for the next day, 40 km"),
        ("foehn", "Foehn forecast for the next day, 1 km")]
THR = 37.5                                                                   # WHO 2021 24-h interim target 3
POP_MIN = 100                                                                # residents per km2 shown in g (as figdata_alerts)


def blockmean(a, f):
    h, w = a.shape[0] // f * f, a.shape[1] // f * f
    with np.errstate(all="ignore"):
        return np.nanmean(a[:h, :w].reshape(h // f, f, w // f, f), (1, 3))


def nearest(field, grid, lats, lons):
    """value of the 0.4 deg cell (centres lat0 - dlat i, lon0 + dlon j) that contains each point of the target grid"""
    lat0, dlat, lon0, dlon = grid
    i = np.clip(np.round((lat0 - lats) / dlat).astype(int), 0, field.shape[0] - 1)
    j = np.round(((lons - lon0) % 360.0) / dlon).astype(int) % field.shape[1]
    return field[i[:, None], j[None, :]]


def frame(ax, lw=0.6):
    for s_ in ax.spines.values():
        s_.set_visible(True); s_.set_linewidth(lw); s_.set_edgecolor("#111111")


def cell_lines(ax, N, S, W, E):
    """the 0.4 deg cells of CAMS and Aurora (edges at 0.2 + 0.4 n deg)"""
    for c_ in np.arange(np.floor((W - 0.2) / 0.4) * 0.4 + 0.2, E, 0.4):
        if W + 1e-6 < c_ < E - 1e-6:
            ax.axvline(c_, color="#FFFFFF", lw=0.35, alpha=0.9)
    for r_ in np.arange(np.floor((S - 0.2) / 0.4) * 0.4 + 0.2, N, 0.4):
        if S + 1e-6 < r_ < N - 1e-6:
            ax.axhline(r_, color="#FFFFFF", lw=0.35, alpha=0.9)


def alerts_panel(ax):
    """f: share of the residents above the target warned / missed, and false alerts, per region and system"""
    z = np.load(HERE / "data" / "alerts_2022_h1.npz")
    SY, RG = [str(s) for s in z["systems"]], [str(r) for r in z["regions"]]
    k = list(z["thr"]).index(THR); nd = len(z["days"])
    regs = ["World", "South Asia", "East Asia", "Africa", "Middle East"]
    show = [("Foehn (1 km)", "Foehn, 1 km"), ("CAMS (40 km)", "CAMS, 40 km"), ("Aurora (40 km)", "Aurora, 40 km"),
            ("perfect 40 km forecast", "perfect 40 km")]
    y, ticks, labels = 0.0, [], []
    for rg in regs:
        E = z["S"][0, RG.index(rg), k, 0]
        ax.text(-2, y - 0.15, f"{rg} · {E / nd / 1e6:,.0f} M per day", fontsize=5.8, fontweight="bold", ha="left", va="bottom")
        y -= 0.9
        for key, lab in show:
            _, H, F = z["S"][SY.index(key), RG.index(rg), k]
            h, m, f = 100 * H / E, 100 * (E - H) / E, 100 * F / E
            ax.barh(y, h, height=0.72, color=OUTCOME["warned"], zorder=3)
            ax.barh(y, m, left=h, height=0.72, color=OUTCOME["missed"], zorder=3)
            ax.barh(y, min(f, 60), left=101.5, height=0.72, color=OUTCOME["false alert"], zorder=3)
            ax.text(h / 2, y, f"{h:.0f}%", ha="center", va="center", fontsize=4.8, color="white", zorder=4)
            ax.text(101.5 + min(f, 60) + 1, y, f"{f:.0f}%{' >' if f > 60 else ''}", va="center", fontsize=4.8, zorder=4)
            ticks.append(y); labels.append(lab)
            y -= 0.82
        y -= 0.35
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=5.2)
    for t_, (key, _) in zip(ax.get_yticklabels(), show * len(regs)):
        t_.set_fontweight("bold" if key.startswith("Foehn (1") else "normal")
    ax.set_xlim(0, 175); ax.set_ylim(y + 0.3, 0.6)
    ax.set_xticks([0, 50, 100, 101.5, 131.5, 161.5]); ax.set_xticklabels(["0", "50", "100%", "0", "30", "60%"], fontsize=5)
    ax.tick_params(axis="y", length=0, pad=1); ax.tick_params(axis="x", length=1.5, pad=1)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.text(50, y - 0.2, "residents above the target:\nwarned / missed", ha="center", va="top", fontsize=5.5)
    ax.text(131.5, y - 0.2, "warned below it\n(false alerts, % of above)", ha="center", va="top", fontsize=5.5)


def outcome_map(ax, ghap, fc, pop, ext):
    """g: every populated pixel coloured by the alert outcome of its resident"""
    ok = np.isfinite(ghap); above = ok & (ghap > THR); warn = ok & (np.nan_to_num(fc) > THR); live = pop >= POP_MIN
    bg = np.where(ok, np.log10(np.clip(pop, 1, 1e4)) / 4, np.nan)
    ax.imshow(bg, extent=ext, origin="upper", cmap=LinearSegmentedColormap.from_list("g", ["#FFFFFF", "#BDBDBD"]), vmin=0, vmax=1,
              interpolation="nearest")
    cat = np.full(ghap.shape, np.nan)
    cat[live & above & warn] = 0; cat[live & above & ~warn] = 1; cat[live & ~above & warn] = 2
    ax.imshow(cat, extent=ext, origin="upper", cmap=ListedColormap(list(OUTCOME.values())), vmin=-0.5, vmax=2.5, interpolation="nearest")
    w = np.where(ok, pop, 0)
    return [(w * m).sum() / 1e6 for m in (above & warn, above & ~warn, ~above & warn)]


def main(preview):
    co = np.load(D / "coarse.npz"); grid = co["grid"]
    pop = np.load(G + "pop1km_global.npy", mmap_mode="r")
    ep = dict(np.load(HERE / "data" / "teaser_episodes.npz"))
    if preview:                                                                # GHAP of the whole year; no Foehn field yet
        g1 = np.load(HERE / "data" / "pm_annual_2022.npy", mmap_mode="r"); g01 = blockmean(np.asarray(g1, np.float32), 10)
        f1, f01 = None, np.full_like(g01, np.nan)
    else:
        g1, f1 = np.load(D / "ghap_mean.npy", mmap_mode="r"), np.load(D / "foehn_mean.npy", mmap_mode="r")
        w = np.load(D / "world01.npz"); g01, f01 = w["ghap_mean"], w["foehn_mean"]
    st = np.load(G + "stations_2022.npz")
    s_lat, s_lon, src, obs = st["lat"], st["lon"], st["source"], st["daily"][:, 1:365].astype(np.float32)
    with np.errstate(all="ignore"):
        s_mean = np.nanmean(obs, 1)
    s_ok = np.isin(src, NETS) & (np.isfinite(obs).sum(1) >= 180)             # observed annual mean: >= half of the days
    POP_CMAP.set_bad((1, 1, 1, 0)); PM_CMAP.set_bad((1, 1, 1, 0))
    lat01, lon01 = 84.0 - 0.1 * (np.arange(1440) + 0.5), -180.0 + 0.1 * (np.arange(3600) + 0.5)
    land01 = np.isfinite(g01)
    popd = blockmean(np.asarray(pop, np.float32), 10)                            # display at 0.1 deg (map ~80 mm wide)
    world = {"pop": np.where(popd > 0.5, popd, np.nan), "ghap": g01, "foehn": np.where(land01, f01, np.nan),
             "cams": np.where(land01, nearest(co["cams_mean"], grid, lat01, lon01), np.nan),
             "aurora": np.where(land01, nearest(co["aurora_mean"], grid, lat01, lon01), np.nan)}
    # geometry (mm): maps on the left, 3 square zooms per row on the right, rows of equal height; 2 alert rows below
    nr, gap, asp = len(ROWS), 1.2, (EXT[1] - EXT[0]) / (EXT[3] - EXT[2])
    map_w = (MM_W - 1.5 - 3.0 - 2 * gap - 1.0) / (1 + 3 / asp)
    s_mm = map_h = map_w / asp
    gap_alert = 9.0                                                             # room for the headings and dates of f, g
    H_mm = 6.5 + (nr + 2) * map_h + (nr + 1) * gap + gap_alert + 6.0
    W, Hf = DOUBLE, H_mm * MM
    fig = plt.figure(figsize=(W, Hf))
    fx = lambda mm: mm / MM_W; fy = lambda mm: mm / H_mm
    top = H_mm - 6.5
    y_row = lambda k: top - (k + 1) * map_h - k * gap - (gap_alert if k >= nr else 0.0)
    x_zoom = lambda j: 1.5 + map_w + 3.0 + j * (s_mm + gap)
    maps = []
    for k, (key, title) in enumerate(ROWS):
        cm, nm = (POP_CMAP, POP_NORM) if key == "pop" else (PM_CMAP, PM_NORM)
        ax = fig.add_axes([fx(1.5), fy(y_row(k)), fx(map_w), fy(map_h)], projection=ccrs.PlateCarree())
        ax.set_extent(EXT, crs=ccrs.PlateCarree())
        ax.imshow(world[key], extent=(-180, 180, -60, 84), origin="upper", transform=ccrs.PlateCarree(), cmap=cm, norm=nm,
                  interpolation="nearest", zorder=2)
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.25, edgecolor="#222222", zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale("110m"), lw=0.15, edgecolor="#777777", zorder=3)
        for z_, (N, S, Wz, E) in enumerate(ZONES.values()):                # zones are ~3 deg: frames enlarged to be visible
            la, lo = 0.5 * (N + S), 0.5 * (Wz + E)
            ax.add_patch(plt.Rectangle((lo - 4.5, la - 4.5), 9, 9, fill=False, ec="#E3001B", lw=0.9, zorder=5, transform=ccrs.PlateCarree()))
            ax.text(lo + 6, la + 3, str(z_ + 1), color="#E3001B", fontsize=6, fontweight="bold", zorder=6, transform=ccrs.PlateCarree(),
                    path_effects=[pe.withStroke(linewidth=1.2, foreground="white")])
        frame(ax)
        ax.text(0.008, 0.965, f"$\\bf{{{'abcde'[k]}}}$  {title}", transform=ax.transAxes, fontsize=6.5, va="top", zorder=6,
                bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="none", alpha=0.85))
        if key == "foehn" and preview:
            ax.text(0.5, 0.45, "production running", transform=ax.transAxes, ha="center", fontsize=8, color="#999999")
        maps.append(ax)
    for ax, cm, nm, lab_, ticks, wd in ((maps[0], POP_CMAP, POP_NORM, "residents km$^{-2}$", [1, 10, 100, 1000, 10000], 0.21),
                                        (maps[1], PM_CMAP, PM_NORM, "annual mean (µg m$^{-3}$)", PM_B, 0.28)):
        cax = ax.inset_axes([0.012, 0.17, wd, 0.05])                             # over the South Pacific
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=nm, cmap=cm), cax=cax, orientation="horizontal", ticks=ticks,
                          extend="max" if cm is PM_CMAP else "both")
        cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5, length=1.5, pad=1)
        cb.set_label(lab_, fontsize=5.5, labelpad=1)
        if cm is POP_CMAP:
            cb.set_ticklabels(["1", "10", "100", "1k", "10k"])
    for j, (zn, (N, S, Wz, E)) in enumerate(ZONES.items()):
        r0, r1 = int(round((84 - N) * 100)), int(round((84 - S) * 100)); c0, c1 = int(round((Wz + 180) * 100)), int(round((E + 180) * 100))
        lats, lons = 84.0 - 0.01 * (np.arange(r0, r1) + 0.5), -180.0 + 0.01 * (np.arange(c0, c1) + 0.5)
        p = np.asarray(pop[r0:r1, c0:c1], np.float32)
        x1 = np.asarray(g1[r0:r1, c0:c1], np.float32); land = np.isfinite(x1)
        zf = {"pop": np.where(p > 0.5, p, np.nan), "ghap": x1,
              "foehn": np.full_like(x1, np.nan) if f1 is None else np.where(land, np.asarray(f1[r0:r1, c0:c1], np.float32), np.nan),
              "cams": np.where(land, nearest(co["cams_mean"], grid, lats, lons), np.nan),
              "aurora": np.where(land, nearest(co["aurora_mean"], grid, lats, lons), np.nan)}
        box = (s_lat < N) & (s_lat > S) & (s_lon > Wz) & (s_lon < E)
        dots = np.flatnonzero(box & s_ok); dots = dots[np.argsort(s_mean[dots])]       # the most polluted on top
        for k, (key, _) in enumerate(ROWS):
            cm, nm = (POP_CMAP, POP_NORM) if key == "pop" else (PM_CMAP, PM_NORM)
            ax = fig.add_axes([fx(x_zoom(j)), fy(y_row(k)), fx(s_mm), fy(map_h)])
            ax.imshow(zf[key], extent=(Wz, E, S, N), origin="upper", cmap=cm, norm=nm, interpolation="nearest")
            ax.set_facecolor("#DCE6EE")
            if key in ("cams", "aurora"):
                cell_lines(ax, N, S, Wz, E)
            if key != "pop":                                                   # stations, coloured by their observed annual mean
                ax.scatter(s_lon[dots], s_lat[dots], c=s_mean[dots], cmap=PM_CMAP, norm=PM_NORM, s=7, edgecolors="#111111",
                           linewidths=0.3, zorder=5)
            ax.set_xlim(Wz, E); ax.set_ylim(S, N); ax.set_aspect("auto"); ax.set_xticks([]); ax.set_yticks([]); frame(ax, 0.6)
            if k == 0:
                ax.set_title(f"{j + 1}  {zn}", fontsize=6.5, fontweight="bold", pad=2)
                ax.text(0.97, 0.04, f"{len(dots)} stations", transform=ax.transAxes, ha="right", fontsize=5, color="#333333",
                        bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.8))
            if j == 0 and k == 1:                                               # scale bar on the reference row
                dl = 50.0 / (111.32 * np.cos(np.deg2rad(0.5 * (N + S))))
                ax.plot([Wz + 0.15, Wz + 0.15 + dl], [S + 0.15, S + 0.15], color="k", lw=1.0, solid_capstyle="butt", zorder=6)
                ax.text(Wz + 0.15 + dl / 2, S + 0.22, "50 km", ha="center", va="bottom", fontsize=5, zorder=6,
                        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.7))
        # g: the episode day, alert outcome of every resident (CAMS 40 km, then Foehn 1 km)
        t = int(ep[f"day_{j}"]); date = (np.datetime64("2022-01-01") + np.timedelta64(t, "D")).astype(object)
        gh, pp = ep[f"ghap_{j}"], ep[f"pop_{j}"]
        n_above = (np.where(np.isfinite(gh), pp, 0) * (np.nan_to_num(gh) > THR)).sum() / 1e6
        for q, (key, lab) in enumerate((("cams", "CAMS, 40 km"), ("foehn", "Foehn, 1 km"))):
            ax = fig.add_axes([fx(x_zoom(j)), fy(y_row(nr + q)), fx(s_mm), fy(map_h)])
            ax.set_facecolor("#DCE6EE")
            if key == "foehn" and f"foehn_{j}" not in ep:
                ax.text(0.5, 0.5, "after the run", transform=ax.transAxes, ha="center", fontsize=6, color="#999999")
            else:
                h_, m_, f_ = outcome_map(ax, gh, ep[f"{key}_{j}"], pp, (Wz, E, S, N))
                ax.text(0.03, 0.04, f"warned {h_:.1f} M · missed {m_:.1f} M\nfalse alerts {f_:.1f} M", transform=ax.transAxes, fontsize=4.8,
                        va="bottom", fontweight="bold" if key == "foehn" else "normal", zorder=6,
                        bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
            if key == "cams":
                cell_lines(ax, N, S, Wz, E)
            ax.text(0.03, 0.96, lab, transform=ax.transAxes, fontsize=5.5, va="top", fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
            ax.set_xlim(Wz, E); ax.set_ylim(S, N); ax.set_aspect("auto"); ax.set_xticks([]); ax.set_yticks([]); frame(ax, 0.6)
            if q == 0:
                ax.set_title(f"{j + 1}  {date.day} {date:%B} · {n_above:.1f} M above {THR:g}", fontsize=5.8, pad=2)
        print(f"{zn}: {len(dots)} stations, observed annual means {np.nanmin(s_mean[dots]):.0f}-{np.nanmax(s_mean[dots]):.0f};"
              f" episode {date:%d %b}, {n_above:.1f} M above", flush=True)
    # f: alerts over 2022, left of the two alert rows; legend of the outcomes above g
    af = fig.add_axes([fx(14.0), fy(y_row(nr + 1) + 6.0), fx(map_w - 13.5), fy(2 * map_h + gap - 6.0)])
    fig.text(fx(1.5), fy(y_row(nr) + map_h + gap_alert - 0.3), "$\\bf{f}$  Next-day alerts in 2022 (WHO 24-h interim target, "
             f"{THR:g} µg m$^{{-3}}$)", fontsize=6.5, va="top")
    if preview:
        af.text(0.5, 0.5, "after the Foehn run", ha="center", transform=af.transAxes, fontsize=8, color="#999999"); af.axis("off")
    else:
        alerts_panel(af)
    fig.text(fx(x_zoom(0)), fy(y_row(nr) + map_h + gap_alert - 0.3), "$\\bf{g}$  Alert outcome of every resident on the zone's episode day",
             fontsize=6.5, va="top")
    hs = [plt.Rectangle((0, 0), 1, 1, color=c) for c in OUTCOME.values()]
    fig.legend(hs, ["above, warned", "above, missed", "below, warned (false alert)"], loc="lower right", ncol=3, fontsize=5.2, frameon=False,
               bbox_to_anchor=(fx(MM_W - 1.0), fy(y_row(nr + 1) - 5.5)), handlelength=1.0, columnspacing=1.0, handletextpad=0.4)
    if preview:
        fig.savefig("/tmp/claude-10026109/-pfs-lustrep1-scratch-project-462001140-ammar/876cd2f9-d218-4ea1-acb1-e871a631f22d/scratchpad/teaser_v4_preview.png", dpi=300)
    else:
        save(fig, "fig1_teaser_v4")


if __name__ == "__main__":
    main("--preview" in sys.argv)
