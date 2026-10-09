#!/usr/bin/env python3
"""Fig. 3: forecasts of locally extreme pollution, 2022.
a, the local threshold of the extreme tier: 99th percentile of the daily PM2.5 of 2017-2020 for the calendar day (+-15 days),
   at 0.25 deg, annual mean of the daily thresholds (tail_events.py thresholds; GHAP of the training years only);
b, symmetric extremal dependence index (SEDI, base-rate independent) of the forecasts of extremes against GHAP, by lead
   time, over all land (tail_events.py: 0.25 deg block means, identical cells and days); Foehn's ensemble mean;
c, the same events at the stations of the six official networks (thresholds: GHAP 2017-2020 at the station pixel):
   area under the ROC curve of Foehn's probability (share of members above the threshold) and of the yes/no forecasts of
   CAMS and Aurora issued at the same time and of the station's own persistence (extremes_stations.py);
d, what the physical transport adds: SEDI three days ahead by region, the base model with and without the FLEXGREEN
   transport inputs, trained identically (above the local 95th percentile).
    python fig3_extremes.py"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

from nature_style import COLOR, DOUBLE, HERE, MM, plt, save

TH = "/scratch/project_462001140/ammar/eccv/data/world/global1km/clim/tail_thresholds_2017_2020.npy"
TH_B = [5, 10, 15, 25, 37.5, 50, 75, 100, 150, 250]
TH_C = LinearSegmentedColormap.from_list("th", ["#fffbe9", "#fde8a7", "#fcc566", "#f89d3c", "#ec6a2a", "#d33a24", "#ab1a2e",
                                                "#7e1239", "#55103f", "#2e0a3b"], N=11)
STY = {"Foehn": (COLOR["ours"], "-", "o", 1.5, 3.0), "CAMS": (COLOR["cams"], "-", "s", 1.0, 2.5),
       "Aurora": (COLOR["aurora"], "-", "D", 1.0, 2.5), "Persistence": ("#111111", (0, (3, 2)), "", 0.9, 0)}


def sedi(c):
    """c (..., 4): hits, misses, false alarms, correct negatives -> SEDI (Ferro & Stephenson 2011)"""
    h, m, f, n = np.moveaxis(c, -1, 0)
    with np.errstate(all="ignore"):
        H = np.clip(h / (h + m), 1e-6, 1 - 1e-6); F = np.clip(f / (f + n), 1e-6, 1 - 1e-6)
        return (np.log(F) - np.log(H) - np.log(1 - F) + np.log(1 - H)) / (np.log(F) + np.log(H) + np.log(1 - F) + np.log(1 - H))


def roc_area(cnt):
    """cnt (bins, 2): counts by forecast probability bin and outcome"""
    H = np.concatenate([[1.0], np.cumsum(cnt[::-1, 1])[::-1][1:] / cnt[:, 1].sum(), [0.0]])
    F = np.concatenate([[1.0], np.cumsum(cnt[::-1, 0])[::-1][1:] / cnt[:, 0].sum(), [0.0]])
    return -np.trapezoid(H, F)


def clean(ax):
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(labelsize=5.5, length=2)


def main():
    te = np.load(HERE / "data" / "tail_events_e02.npz"); C = te["C"]; TS = [str(s) for s in te["systems"]]; TR = [str(r) for r in te["regions"]]
    es = np.load(HERE / "data" / "extremes_stations_e02.npz"); CE = es["C"]; ES = [str(s) for s in es["systems"]]
    W, H = DOUBLE, 104 * MM
    fig = plt.figure(figsize=(W, H))
    fx = lambda mm: mm / (W / MM); fy = lambda mm: mm / (H / MM)
    # a: local threshold
    th = np.asarray(np.load(TH, mmap_mode="r")[:, 2], np.float32)                 # (365, 576, 1440) P99
    with np.errstate(all="ignore"):
        tm = np.nanmean(th, 0)
    del th
    ax = fig.add_axes([fx(2), fy(52), fx(110), fy(45)], projection=ccrs.PlateCarree())
    ax.set_extent((-180, 180, -58, 80), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#EDEDED", edgecolor="none", zorder=0)
    norm = BoundaryNorm(TH_B, TH_C.N, extend="both")
    ax.imshow(tm, extent=(-180, 180, -60, 84), origin="upper", transform=ccrs.PlateCarree(), cmap=TH_C, norm=norm,
              interpolation="nearest", zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.2, edgecolor="#555555", zorder=3)
    for s_ in ax.spines.values():
        s_.set_linewidth(0.5)
    cax = ax.inset_axes([0.02, 0.1, 0.27, 0.04])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=TH_C), cax=cax, orientation="horizontal", extend="both",
                      ticks=[5, 15, 37.5, 75, 150, 250])
    cb.ax.set_xticklabels(["5", "15", "37.5", "75", "150", "250"])
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=4.8, length=1.5, pad=1); cb.ax.xaxis.set_label_position("top")
    cb.set_label("local 99th percentile (µg m$^{-3}$)", fontsize=5.5, labelpad=1.5)
    fig.text(fx(2), fy(98.5), "$\\bf{a}$  What is extreme depends on where: the local threshold of an extreme day", fontsize=6.5, va="bottom")
    # b: SEDI vs lead, all land, against GHAP, three tiers
    ab = fig.add_axes([fx(124), fy(57), fx(57), fy(40)])
    g = TR.index("Global"); L = np.arange(1, 8)
    for nm, key in (("Persistence", "persistence"), ("CAMS", "CAMS"), ("Aurora", "Aurora"), ("Foehn", "e02 (mean)")):
        col, ls, mk, lw, ms = STY[nm]
        for t, alpha in ((2, 1.0),):
            v = sedi(C[g, :, t, TS.index(key)]); keep = np.isfinite(v) & (C[g, :, t, TS.index(key), :2].sum(-1) > 0)
            ab.plot(L[keep], v[keep], color=col, ls=ls, marker=mk or None, ms=ms if t == 2 else 0, lw=lw, alpha=alpha,
                    label=nm if t == 2 else None, zorder=6 if nm == "Foehn" else 3)
    ab.set_xticks(L); ab.set_xlabel("Lead time (days)", fontsize=6, labelpad=1); ab.set_ylabel("SEDI against GHAP", fontsize=6)
    ab.legend(fontsize=5, loc="lower left", frameon=False, handlelength=1.8); clean(ab)
    fig.text(fx(116), fy(98.5), "$\\bf{b}$  Days above the local P99, all land", fontsize=6.5, va="bottom")
    # c: ROC area at the stations, P95 tier, thresholds from GHAP 2017-2020
    ac = fig.add_axes([fx(14), fy(9), fx(60), fy(33)])
    for nm, key in (("Persistence", "persistence"), ("CAMS", "CAMS"), ("Aurora", "Aurora"), ("Foehn", "e02")):
        col, ls, mk, lw, ms = STY[nm]
        v = [roc_area(CE[0, 1, h, ES.index(key)]) for h in range(4)]
        ac.plot(range(1, 5), v, color=col, ls=ls, marker=mk or None, ms=ms, lw=lw, label={"Persistence": "station persistence"}.get(nm, nm),
                zorder=6 if nm == "Foehn" else 3)
    ac.axhline(0.5, color="#BBBBBB", lw=0.5, zorder=1)
    ac.set_xticks(range(1, 5)); ac.set_xlabel("Lead time (days)", fontsize=6, labelpad=1)
    ac.set_ylabel("ROC area at the stations", fontsize=6); ac.set_ylim(0.48, 0.78); clean(ac)
    ac.legend(fontsize=5, loc="upper right", frameon=False, handlelength=1.8)
    fig.text(fx(2), fy(45.5), "$\\bf{c}$  The same events measured at 6,612 stations (> local P95)", fontsize=6.5, va="bottom")
    # d: transport ablation, SEDI three days ahead by region
    ad = fig.add_axes([fx(104), fy(9), fx(77), fy(33)])
    regs = ["Global", "East Asia", "South Asia", "Europe", "North America", "Africa"]
    x = np.arange(len(regs))
    for k, (key, col, lab) in enumerate((("C (mean)", "#C9B8A8", "without transport"), ("A (mean)", COLOR["ours"], "with FLEXGREEN transport"))):
        v = [sedi(C[TR.index(r), 2, 1, TS.index(key)]) for r in regs]
        ad.bar(x + (k - 0.5) * 0.36, v, width=0.34, color=col, label=lab, zorder=3)
    ad.set_xticks(x); ad.set_xticklabels([r.replace("North America", "N. America") for r in regs], fontsize=5.5)
    ad.set_ylabel("SEDI, 3 days ahead", fontsize=6); ad.set_ylim(0.25, 0.6); clean(ad)
    ad.legend(fontsize=5, loc="upper right", frameon=False, ncol=2)
    fig.text(fx(96), fy(45.5), "$\\bf{d}$  What the computed transport adds (same network, > local P95)", fontsize=6.5, va="bottom")
    save(fig, "fig3_extremes")
    print("SEDI P99 global e02/CAMS/Aurora/pers J+1:", [round(float(sedi(C[g, 0, 2, TS.index(k)])), 2) for k in ("e02 (mean)", "CAMS", "Aurora", "persistence")],
          "J+3:", [round(float(sedi(C[g, 2, 2, TS.index(k)])), 2) for k in ("e02 (mean)", "CAMS", "Aurora", "persistence")])
    print("transport J+3 P95 SEDI with/without:", {r: (round(float(sedi(C[TR.index(r), 2, 1, TS.index("A (mean)")])), 2),
                                                      round(float(sedi(C[TR.index(r), 2, 1, TS.index("C (mean)")])), 2)) for r in regs})
    print("station ROC P95 J+1..4:", {k: [round(roc_area(CE[0, 1, h, ES.index(k)]), 3) for h in range(4)] for k in ES})


if __name__ == "__main__":
    main()
