#!/usr/bin/env python3
"""Fig. 5: where the air comes from (FLEXGREEN Green's functions of the 2022 forecasts; figdata_transport.py).
a, share of the air arriving over land three days ahead that descended from above the mixed layer of the start day
   (Robinson, 0.5 deg receptors, mean over every fourth start day of 2022);
b, that share by lead time, all land and regions; c, median distance of the air's origin;
d, origins three days earlier of the air arriving in Beijing on 25 January 2022, when a cold outbreak ended a week of haze
   (256 particles per 0.5 deg cell, 3 x 3 cells), by height relative to the mixed layer;
e, the forecasts of the 25th issued 1-7 days ahead (members' station mean at the 12 CNEMC stations), with and without the
   physical transport, and the observations.
    python fig5_transport.py [run with transport] [label]"""
import glob
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from nature_style import COLOR, DOUBLE, HERE, MM, letter, plt, save, world_axes

RUNS = HERE.parent / "global1km" / "runs"
RUN = sys.argv[1] if len(sys.argv) > 1 else "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12"
NOTR = "green_flow_shift25_global_L7_noant_whres_notransport"
G = "/scratch/project_462001140/ammar/eccv/data/world/global1km"
REGC = {"Global": "#111111", "East Asia": "#0072B2", "South Asia": "#E69F00", "Europe": "#009E73", "North America": "#CC79A7",
        "Africa": "#56B4E9"}
ALOFT, MIXED = "#2166AC", "#E08214"
T = 24                                                                      # 25 January (0-based)


def beijing_forecasts(run):
    st = np.load(f"{G}/stations_2022.npz")
    sel = np.where((st["source"] == 1) & (np.abs(st["lat"] - 39.9) < 0.3) & (np.abs(st["lon"] - 116.4) < 0.4))[0]
    E = np.full((7, 16), np.nan)
    for p in sorted(glob.glob(str(RUNS / run / "case_beijing" / "part_*.npz"))):
        z = np.load(p)
        for n, (d, h, t) in enumerate(z["keys"]):
            if int(t) == T:
                m = np.nanmean(z["st_ens"][n][sel].astype(np.float32), 0); E[int(h) - 1, :len(m)] = m[:16]
    return E, np.nanmean(st["daily"][sel].astype(np.float32), 0)


def main():
    z = np.load(HERE / "data" / "transport_fig.npz")
    R = [str(r) for r in z["regions"]]
    land = np.load(f"{G}/meta.npz")["land"].astype(bool).reshape(288, 2, 720, 2).any((1, 3))
    fig = plt.figure(figsize=(DOUBLE, 132 * MM))
    # a: world map of the share from aloft at J+3
    ax = world_axes(fig, [0.0, 0.43, 0.68, 0.55])
    cmap = LinearSegmentedColormap.from_list("aloft", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"], N=256)
    lat_e = 84.0 - 0.5 * np.arange(289); lon_e = -180.0 + 0.5 * np.arange(721)
    a3 = np.where(land, 100 * z["aloft_map"][2], np.nan)
    pm = ax.pcolormesh(lon_e, lat_e, np.ma.masked_invalid(a3), cmap=cmap, vmin=0, vmax=100, transform=ccrs.PlateCarree(),
                       rasterized=True, zorder=3)
    letter(ax, "a", x=0.02, y=0.98)
    cax = fig.add_axes([0.17, 0.455, 0.34, 0.013])
    cb = fig.colorbar(pm, cax=cax, orientation="horizontal", ticks=range(0, 101, 20))
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.5, length=2)
    cb.set_label("Air arriving 3 days ahead that descended from above the mixed layer (%)", fontsize=6)
    # b, c: by lead
    L = z["leads"]
    for k, (key, ylab, rect, lim) in enumerate((("aloft", "From above the\nmixed layer (%)", [0.755, 0.76, 0.23, 0.21], (0, 100)),
                                                 ("dist", "Median distance of\nthe origin (km)", [0.755, 0.47, 0.23, 0.21], (0, 3200)))):
        a_ = fig.add_axes(rect)
        for g, r in enumerate(R):
            v = np.nanmean(z[key][:, :, g], 0) * (100 if key == "aloft" else 1)
            a_.plot(L, v, color=REGC[r], lw=1.4 if r == "Global" else 0.8, marker="o" if r == "Global" else "", ms=2.5, zorder=5 if r == "Global" else 3,
                    label="All land" if r == "Global" else r)
            if r == "Global":
                for h in (1, 3, 7):
                    a_.annotate(f"{v[h - 1]:.0f}{'%' if key == 'aloft' else ''}", (h, v[h - 1]), textcoords="offset points",
                                xytext=(0, 5 if key == "aloft" else -9), ha="center", fontsize=5.5, color=REGC[r])
        a_.set_xlim(0.7, 7.3); a_.set_ylim(*lim); a_.set_xticks(range(1, 8)); a_.tick_params(labelsize=5.5)
        a_.set_ylabel(ylab, fontsize=6)
        if k == 1:
            a_.set_xlabel("Lead time (days)", fontsize=6)
        else:
            a_.set_xticklabels([])
            a_.legend(fontsize=5, loc="lower right", ncol=2, handlelength=1.2, columnspacing=0.8)
        letter(a_, "bc"[k], x=-0.3)
    # d: Beijing origins, lead 3 (start 22 January)
    ad = fig.add_axes([0.01, 0.03, 0.40, 0.36], projection=ccrs.PlateCarree())
    ad.set_extent((88, 136, 30, 60), crs=ccrs.PlateCarree())
    ad.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#F2F0EB", zorder=0)
    ad.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#E3EEF6", zorder=0)
    ad.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.3, edgecolor="#777777")
    ad.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.25, edgecolor="#999999", linestyle=(0, (2, 1.5)))
    o = z["bj_org_3"]; lon, lat, zr, sv = o.T; keep = sv > 0.05
    for msk, col, lab in ((keep & (zr <= 1), MIXED, "in the mixed layer"), (keep & (zr > 1), ALOFT, "above it")):
        ad.scatter(lon[msk], lat[msk], s=1.0, color=col, alpha=0.55, linewidths=0, transform=ccrs.PlateCarree(), zorder=4,
                   rasterized=True, label=lab)
    ad.plot(116.4, 39.9, marker="*", ms=8, color=COLOR["ours"], mec="white", mew=0.5, transform=ccrs.PlateCarree(), zorder=6)
    ad.text(117.6, 38.3, "Beijing", fontsize=6, color=COLOR["ours"], fontweight="bold", transform=ccrs.PlateCarree(), zorder=6)
    share = (sv * (zr > 1)).sum() / sv.sum()
    ad.text(0.02, 0.97, f"Origins on 22 January of the air\narriving in Beijing on the 25th:\n{share:.0%} from above the mixed layer",
            transform=ad.transAxes, va="top", fontsize=5.8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))
    lg = ad.legend(loc="lower right", fontsize=5.5, markerscale=5, frameon=True, framealpha=0.85, edgecolor="none", handletextpad=0.2)
    letter(ad, "d", x=0.0, y=1.01)
    # e: forecasts of the 25th by lead, with and without transport
    ae = fig.add_axes([0.52, 0.09, 0.46, 0.27])
    Ew, obs = beijing_forecasts(RUN); Eo, _ = beijing_forecasts(NOTR)
    o25, o24 = float(obs[T]), float(obs[T - 1])
    ae.axhspan(0, 75, color="#F4F4F4", zorder=0)
    ae.axhline(o25, color="#111111", lw=0.9); ae.text(7.55, o25, f"observed on the 25th ({o25:.0f})", fontsize=5.5, va="center")
    ae.axhline(o24, color="#111111", lw=0.7, ls=(0, (2, 2))); ae.text(7.55, o24, f"day before ({o24:.0f})", fontsize=5.5, va="center")
    for k, (E, col, lab) in enumerate(((Ew, COLOR["ours"], "Foehn"), (Eo, "#7F7F7F", "same network, no transport"))):
        x = np.arange(1, 8) + (k - 0.5) * 0.22
        ae.vlines(x, np.nanmin(E, 1), np.nanmax(E, 1), color=col, lw=1.0, alpha=0.5)
        ae.plot(x, np.nanmean(E, 1), "o", color=col, ms=3.5, label=lab, zorder=5)
    ae.set_xticks(range(1, 8)); ae.set_xlim(0.5, 7.5); ae.set_ylim(20, 165)
    ae.set_xlabel("Forecast issued (days before 25 January)", fontsize=6); ae.set_ylabel("Forecast PM$_{2.5}$ in Beijing\n(µg m$^{-3}$)", fontsize=6)
    ae.tick_params(labelsize=5.5); ae.legend(fontsize=5.5, loc="upper left", ncol=2, handletextpad=0.2)
    letter(ae, "e", x=-0.12)
    save(fig, "fig5_transport")


if __name__ == "__main__":
    main()
