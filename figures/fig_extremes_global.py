#!/usr/bin/env python3
"""Main figure: detection of locally extreme pollution (tail_events.py: thresholds = percentiles of each cell's GHAP
climatology of 2017-2020 for the calendar day, +-15 days; 2022 forecasts against the GHAP block means of the target day).
  a  local threshold of the extreme tier (P99) on 15 January, 0.25 deg
  b  SEDI of the extreme tier (> P99) versus lead time: Foehn (ensemble mean), the same network without transport, CAMS,
     Aurora (leads 1-4), persistence
  c  share of the extreme events detected three days ahead, by region, with and without the physical transport, and CAMS
Input: data/tail_events_<label>.npz written by tail_events.py evaluate <label>=... C=... (C: no transport)
    python fig_extremes_global.py [label A]"""
import sys

import numpy as np

from nature_style import COLOR, DOUBLE, HERE, MM, STYLE, letter, plt, pm_cmap, save, world_axes

LAB = sys.argv[1] if len(sys.argv) > 1 else "A"
z = np.load(HERE / "data" / f"tail_events_{LAB}.npz")
C, SYS, REG = z["C"], [str(s) for s in z["systems"]], [str(r) for r in z["regions"]]
TIER = 2                                                           # extreme (> P99)
NOTR = "#4C7DB8"


def scores(c):
    a, b, f, d = c
    H, Fa = a / max(a + b, 1), f / max(f + d, 1)
    Hc, Fc = np.clip(H, 1e-6, 1 - 1e-6), np.clip(Fa, 1e-6, 1 - 1e-6)
    sedi = (np.log(Fc) - np.log(Hc) - np.log(1 - Fc) + np.log(1 - Hc)) / (np.log(Fc) + np.log(Hc) + np.log(1 - Fc) + np.log(1 - Hc))
    return H, sedi, a + b


lines = [(f"{LAB} (mean)", "ours", None), ("C (mean)", None, dict(label="Same network, no transport", color=NOTR, ls="-", marker="o",
                                                                     lw=0.9, ms=2.4, z=8)),
         ("CAMS", "cams", None), ("Aurora", "aurora", None), ("persistence", "persistence", None)]
fig = plt.figure(figsize=(DOUBLE, 64 * MM))
# --- a: local threshold
thr = np.load(HERE.parents[2] / "data" / "world" / "global1km" / "clim" / "tail_thresholds_2017_2020.npy", mmap_mode="r")
p99 = np.asarray(thr[14, 2], np.float32)                           # 15 January, P99, on pm.npy points (84 - 0.25 k)
ax = world_axes(fig, [0.0, 0.08, 0.42, 0.86])
import cartopy.crs as ccrs  # noqa: E402
cmap, norm = pm_cmap()
lat = 84.0 - 0.25 * np.arange(576); lon = -180.0 + 0.25 * np.arange(1440)
ax.pcolormesh(lon, lat, np.ma.masked_invalid(p99), cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading="nearest",
              rasterized=True, zorder=2)
cax = fig.add_axes([0.06, 0.10, 0.30, 0.025])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=[0, 15, 37.5, 75, 150, 250])
cb.ax.set_xticklabels(["0", "15", "37.5", "75", "150", "250"]); cb.outline.set_linewidth(0.4)
cb.set_label("Local extreme threshold, 99th percentile (µg m$^{-3}$), 15 January")
letter(ax, "a", x=0.02, y=0.98)
# --- b: SEDI vs lead, extreme tier, global
ax = fig.add_axes([0.49, 0.20, 0.22, 0.70])
g = REG.index("Global")
for name, key, sty in lines:
    if name not in SYS:
        continue
    k = SYS.index(name); st = dict(STYLE[key]) if key else sty
    hs = [h for h in range(1, 8) if C[g, h - 1, TIER, k].sum() > 0]
    ax.plot(hs, [scores(C[g, h - 1, TIER, k])[1] for h in hs], color=st["color"], ls=st["ls"], marker=st["marker"],
            lw=st["lw"], ms=st["ms"], zorder=st["z"], label=st["label"])
ax.set_xlabel("Lead time (days)"); ax.set_ylabel("SEDI, events above the local P99")
ax.set_xticks(range(1, 8)); ax.set_ylim(0.2, 0.75)
ax.legend(loc="upper right", fontsize=5.5, handlelength=1.8)
letter(ax, "b", x=-0.18)
# --- c: detection by region three days ahead
ax = fig.add_axes([0.80, 0.20, 0.19, 0.70])
regs = [r for r in REG if r != "Global"][::-1]
H = 3
bars = [(f"{LAB} (mean)", COLOR["ours"], "Foehn"), ("C (mean)", NOTR, "No transport"), ("CAMS", COLOR["cams"], "CAMS")]
w = 0.26
for j, (name, col, lab) in enumerate(bars):
    k = SYS.index(name)
    v = [100 * scores(C[REG.index(r), H - 1, TIER, k])[0] for r in regs]
    ax.barh(np.arange(len(regs)) + (1 - j) * w, v, height=w * 0.92, color=col, label=lab, zorder=3)
ax.set_yticks(range(len(regs))); ax.set_yticklabels(regs)
ax.set_xlabel("Extreme events detected\nthree days ahead (%)")
ax.legend(loc="lower right", fontsize=5.5)
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
letter(ax, "c", x=-0.42)
save(fig, f"fig_extremes_global_{LAB}")
for r in REG:
    row = []
    for name, _, _ in lines:
        if name in SYS:
            Hh, s, n = scores(C[REG.index(r), 2, TIER, SYS.index(name)])
            row.append(f"{name}: POD {Hh:.2f} SEDI {s:.2f}")
    print(f"J+3 {r} ({int(n)} events): " + " | ".join(row))
