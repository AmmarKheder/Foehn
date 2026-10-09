#!/usr/bin/env python3
"""The MOSAIC 'hurricane tracks' figure, for the most extensive pollution episode of 2022: on 25 March 2022 (the day with
the largest land area above 25 ug/m3), forecasts from three initialisation days (real HRES weather). Shading: FOEHN
probability of exceeding 25 ug/m3 (32 members); dark blue: its 50 % contour; black: observed exceedance (GHAP);
dashed: the best baseline (ConvLSTM, deterministic). The map is framed on the observed episode (rule fixed in advance)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from fig_style import COL2, EXP, STYLE, save

THR, TARGET = 25.0, 83
D = Path(__file__).resolve().parent / "data"
E = np.load(D / "episode83.npz")["pred"].astype(np.float32)[:, 0]            # (3 inits, 32, 168, 280)
C = np.load(D / "episode_convlstm83.npz")["pred"].astype(np.float32)[:, 0, 0]  # (3 inits, 168, 280)
meta = np.load(EXP / "dejavu/data/meta.npz")
land = meta["land"].astype(bool)
i0 = int(np.where(meta["year"] == 2022)[0][0])
truth = np.asarray(np.load(EXP / "dejavu/data/pm.npy", mmap_mode="r")[i0 + TARGET], np.float32)
lat = 72.0 - 0.25 * np.arange(168); lon = -25.0 + 0.25 * np.arange(280)
LON, LAT = np.meshgrid(lon, lat)
OBS, DARK = "#111111", STYLE["foehn"]["color"]

# frame: the observed exceedance area, with a margin
europe = land & (LAT > 36.5)
ex = np.where(europe, truth, 0) > THR
la, lo = LAT[ex], LON[ex]
x0, x1 = np.percentile(lo, 1) - 3, np.percentile(lo, 99) + 3
y0, y1 = max(np.percentile(la, 1) - 2, 36.5), np.percentile(la, 99) + 2


def smooth(a):
    x = np.where(land, np.nan_to_num(a), 0.0); w = land.astype(float)
    k = lambda f: sum(np.roll(np.roll(f, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1))
    return np.where(land, k(x) / np.maximum(k(w), 1), np.nan)


fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.3), constrained_layout=True)
for ax, k, h in zip(axes, range(3), (1, 2, 3)):
    ax.set_facecolor("#eef3fa")
    ax.contourf(LON, LAT, land.astype(float), levels=[0.5, 1.5], colors=["#f6f5f2"], zorder=0)
    prob = np.where(land, (E[k] > THR).mean(0), np.nan)
    cf = ax.contourf(LON, LAT, prob, levels=[0.1, 0.3, 0.5, 0.7, 0.9, 1.01], cmap="Blues", vmin=-0.2, vmax=1.0, zorder=1)
    ax.contour(LON, LAT, land.astype(float), levels=[0.5], colors="#6f6f6f", linewidths=0.4, zorder=2)
    ax.contour(LON, LAT, prob, levels=[0.5], colors=DARK, linewidths=1.3, zorder=3)
    ax.contour(LON, LAT, smooth(C[k]), levels=[THR], colors=STYLE["convlstm"]["color"], linewidths=1.0, linestyles="--", zorder=4)
    ax.contour(LON, LAT, smooth(truth), levels=[THR], colors=OBS, linewidths=1.3, zorder=5)
    obs = europe & (truth > THR)
    csi = lambda f: (f & obs).sum() / max((f | obs).sum(), 1)
    ax.text(0.02, 0.03, f"CSI  FOEHN {csi(europe & (prob >= 0.5)):.2f}   ConvLSTM {csi(europe & (C[k] > THR)):.2f}",
            transform=ax.transAxes, fontsize=6, bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="#999999", lw=0.5), zorder=6)
    init = np.datetime64("2022-01-01") + np.timedelta64(TARGET - h, "D")
    ax.set_title(f"({'abc'[k]}) Init: {str(init)[8:10]} Mar 2022, lead {h} day{'s' if h > 1 else ''}")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect(1.4)
    ax.grid(True, ls="--", lw=0.4, color="#b9b9b9")
    xt = np.arange(np.ceil(x0 / 10) * 10, x1, 10); yt = np.arange(np.ceil(y0 / 4) * 4, y1, 4)
    ax.set_xticks(xt); ax.set_yticks(yt)
    ax.set_xticklabels([f"{abs(v):.0f}°{'W' if v < 0 else 'E'}" for v in xt], fontsize=5.8)
    ax.set_yticklabels([f"{v:.0f}°N" for v in yt], fontsize=5.8)
cb = fig.colorbar(cf, ax=axes, shrink=0.8, pad=0.01, aspect=25, ticks=[0.1, 0.3, 0.5, 0.7, 0.9])
cb.set_label("FOEHN probability of PM$_{2.5}$ > 25 µg m$^{-3}$", fontsize=6.5); cb.ax.tick_params(labelsize=5.8)
fig.legend(handles=[Line2D([], [], color=DARK, lw=1.3, label="FOEHN, 50 % probability"),
                    Line2D([], [], color=STYLE["convlstm"]["color"], lw=1.0, ls="--", label="ConvLSTM (best baseline)"),
                    Line2D([], [], color=OBS, lw=1.3, label="Observed (GHAP)")],
           loc="outside lower center", ncol=3, frameon=False)
save(fig, "fig_episode")
