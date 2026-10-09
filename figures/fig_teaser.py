#!/usr/bin/env python3
"""Teaser. (a) Satellite-derived PM2.5 at 1 km on 15 March 2022, when Saharan dust reached Iberia, with the back-trajectories
that FOEHN's footprint layer integrates through the HRES forecast wind for air arriving in eight cities (lead 2 days).
(b) Daily PM2.5 over February-April 2022 in the city with the highest observed peak among them: observation, FOEHN
2-day-ahead ensemble (16 members: 10-90 % band and mean) and the best baseline (ConvLSTM)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import matplotlib.patheffects as pe

from fig_style import COL2, EXP, STYLE, save

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_footprint_traj as FT  # noqa: E402   (trajectories of fig_footprint, arrival 15 March 12 UTC)

D = Path(__file__).resolve().parent / "data"
DAY_MAP = 73                                     # 15 March 2022
meta = np.load(EXP / "dejavu/data/meta.npz")
i0 = int(np.where(meta["year"] == 2022)[0][0])
pm = np.load(EXP / "dejavu/data/pm.npy", mmap_mode="r")

# ---------------------------------------------------------------- (a) 1 km map
FINE = EXP / "fullimage/data/ghap1km.npy"
with open(FINE, "rb") as f:
    np.lib.format.read_magic(f); np.lib.format.read_array_header_1_0(f); HDR = f.tell()
OFF22 = 365 * 4 + 1 + 365
full = np.memmap(FINE, dtype=np.float16, mode="r", offset=HDR + (OFF22 + DAY_MAP) * 4192 * 6992 * 2, shape=(4192, 6992))
LON0, LON1, LAT0, LAT1 = -12.0, 20.0, 31.0, 53.0
r0, r1 = int((72 - LAT1) / 0.01), int((72 - LAT0) / 0.01)
c0, c1 = int((LON0 + 25) / 0.01), int((LON1 + 25) / 0.01)
crop = np.asarray(full[r0:r1, c0:c1], np.float32)
crop = np.where(crop > 0, crop, np.nan)
paths, _ = FT.trajectories()

fig = plt.figure(figsize=(COL2, 2.3))
gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1], wspace=0.22)
ax = fig.add_subplot(gs[0])
ax.set_facecolor("#dfe7f1")
im = ax.imshow(crop, extent=[LON0, LON1, LAT0, LAT1], cmap="RdYlBu_r", vmin=0, vmax=60, interpolation="none", zorder=1)
for name, (p, w) in paths.items():
    ax.plot(p[:, 0], p[:, 1], color="white", lw=1.1, zorder=4, path_effects=[pe.Stroke(linewidth=2.1, foreground="#111111"), pe.Normal()])
    ax.plot(p[4::4, 0], p[4::4, 1], "o", ms=1.7, color="#111111", zorder=5)
    ax.plot(*p[0], "o", ms=3.6, mfc="white", mec="#111111", mew=0.8, zorder=6)
    off = {"Munich": (-14, 6), "Paris": (5, 2), "Bordeaux": (-3, 3), "Lyon": (3, 5), "Vienna": (-2, 5), "Milan": (3, -9),
           "Rome": (4, 3), "Madrid": (4, 4)}.get(name, (3, 2))
    ha = {"Bordeaux": "right", "Vienna": "center"}.get(name, "left")
    ax.annotate(name, p[0], xytext=off, textcoords="offset points", fontsize=5.8, color="#111111", zorder=7, ha=ha,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
ax.set_xlim(LON0, LON1); ax.set_ylim(LAT0, LAT1); ax.set_aspect(1.3); ax.grid(False)
ax.set_xticks([-10, 0, 10, 20]); ax.set_xticklabels(["10°W", "0°", "10°E", "20°E"])
ax.set_yticks([35, 40, 45, 50]); ax.set_yticklabels(["35°N", "40°N", "45°N", "50°N"])
ax.set_title("(a) Following the air: 15 March 2022, 1 km")
cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.09, aspect=40, extend="max")
cb.set_label("PM$_{2.5}$ (µg m$^{-3}$), satellite-derived, 1 km", fontsize=6.3); cb.ax.tick_params(labelsize=5.8)

# ---------------------------------------------------------------- (b) episode extent over Europe
fz, cz = np.load(D / "series_foehn.npz"), np.load(D / "series_convlstm.npz")
fd = {int(d): k for k, d in enumerate(fz["days"])}
cd = {int(d): k for k, d in enumerate(cz["days"])}
F, Cv = fz["pred"].astype(np.float32)[:, 0], cz["pred"].astype(np.float32)[:, 0, 0]
H = 2
land = meta["land"].astype(bool)
LATG = (72.0 - 0.25 * np.arange(168))[:, None] * np.ones((1, 280))
eu = land & (LATG > 36.5)
days = sorted(set(fd) & set(cd))
obs, fmean, flo, fhi, conv = [], [], [], [], []
for d in days:
    t = np.asarray(pm[i0 + d + H], np.float32); m = eu & np.isfinite(t)
    obs.append(100 * (t[m] > 25).mean())
    mem = [100 * (F[fd[d], k][m] > 25).mean() for k in range(F.shape[1])]
    fmean.append(100 * (F[fd[d]].mean(0)[m] > 25).mean()); flo.append(np.percentile(mem, 10)); fhi.append(np.percentile(mem, 90))
    conv.append(100 * (Cv[cd[d]][m] > 25).mean())
obs, fmean, conv = map(np.array, (obs, fmean, conv))
r = lambda x: np.corrcoef(x, obs)[0, 1]
dates = [(np.datetime64("2022-01-01") + np.timedelta64(d + H, "D")).astype("datetime64[D]").astype(object) for d in days]
axb = fig.add_subplot(gs[1])
axb.fill_between(dates, flo, fhi, color=STYLE["foehn"]["color"], alpha=0.2, lw=0, label="FOEHN, 10–90 % of members")
axb.plot(dates, fmean, color=STYLE["foehn"]["color"], lw=1.6, label=f"FOEHN (r = {r(fmean):.2f})")
axb.plot(dates, conv, color=STYLE["convlstm"]["color"], lw=1.0, ls="--", label=f"ConvLSTM, best baseline (r = {r(conv):.2f})")
axb.plot(dates, obs, color="#111111", lw=1.1, label="Observed")
axb.xaxis.set_major_locator(mdates.MonthLocator()); axb.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axb.set_ylabel("Land area above 25 µg m$^{-3}$ (%)")
axb.set_title("(b) Pollution episodes over Europe, 2 days ahead")

axb.set_ylim(0, 27)
axb.legend(loc="upper left", fontsize=5.0, handlelength=1.4, labelspacing=0.2, borderpad=0.3, handletextpad=0.4)
save(fig, "fig_teaser")
