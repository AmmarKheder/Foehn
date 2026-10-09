#!/usr/bin/env python3
"""Seams at kilometre scale. (a-c) 1 km PM2.5 around a corner of the 768 x 1536 px tiling (lead 1 day): observation,
Earthformer trained natively at 1 km and applied tile by tile, and FOEHN-km, our flow-matching network trained on the whole 1 km map without tiles; arrows: tile edges.
The window is centred on a tile corner (49 N 30 E) where the tiled forecast has a clear seam (illustration; local RMSE
shown); red: the tiles, outlined slightly inside their edges so that the seam itself stays visible. (d) Jump across tile edges
divided by the jump elsewhere over the whole of Europe (4 days, all lines of the tiling): 1 = no seam."""
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fig_style import COL2, EXP, OURS_KM, STYLE, save

D, H = int(__import__('os').environ.get('KM_D', 40)), 1    # start day 40 (10 Feb 2022), lead 1 day -> 11 Feb
ROWS, COLS = [768, 1536, 2304, 3072, 3424], [1536, 3072, 4608, 5456]
FINE = EXP / "fullimage/data/ghap1km.npy"
with open(FINE, "rb") as f:
    np.lib.format.read_magic(f); np.lib.format.read_array_header_1_0(f); HDR = f.tell()
OFF22 = 365 * 4 + 1 + 365
truth = np.asarray(np.memmap(FINE, dtype=np.float16, mode="r", offset=HDR + (OFF22 + D + H) * 4192 * 6992 * 2, shape=(4192, 6992)), np.float32)
tiled = np.load(EXP / f"regional/runs/earthformer/tiled_europe_d{D}_h{H}.npy").astype(np.float32)
free = None
for p in sorted(glob.glob(str(EXP / "fullimage/runs/flow_shift25/eval_hres/part_*.npz"))):
    z = np.load(p)
    if f"pred_d{D}_h{H}" in z.files:
        free = z[f"pred_d{D}_h{H}"].astype(np.float32)
land = np.load(EXP / "fullimage/data/static1km.npz")["land"]
mask = lambda a: np.where(land > 0, a, np.nan)

# window centred on a tile edge in western Russia (56.6 N, 33.6 E) where the tiled forecast has a clear seam and FOEHN-km
# is much closer to the observation (search over all tile edges of continental Europe, days 40/100/200/300, ranked by
# seam ratio x RMSE gain; illustration)
import os
R, C, W = int(os.environ.get("KM_R", 1536)), int(os.environ.get("KM_C", 5860)), 260
sl = (slice(R - W, R + W), slice(C - W, C + W))
ext = [-25 + 0.01 * (C - W), -25 + 0.01 * (C + W), 72 - 0.01 * (R + W), 72 - 0.01 * (R - W)]
ok = (land[sl] > 0) & (truth[sl] > 0)
rmse = lambda a: float(np.sqrt(((a[sl] - truth[sl])[ok] ** 2).mean()))
vmax = float(np.nanpercentile(mask(truth)[sl], 99))
vmin = float(np.floor(np.nanpercentile(np.concatenate([mask(x)[sl].ravel() for x in (truth, tiled, free)]), 1)))   # contrast: the 1-99% range
lat = lambda r: 72 - 0.01 * r
lon = lambda c: -25 + 0.01 * c

fig = plt.figure(figsize=(COL2, 2.05))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25], wspace=0.12, left=0.055, right=0.99, top=0.9, bottom=0.3)
panels = [("(a) Observed (GHAP)", truth, None), ("(b) Earthformer, tiled", tiled, rmse(tiled)), ("(c) " + OURS_KM, free, rmse(free))]
axes = []
dx, dy = 0.06 * (ext[1] - ext[0]), 0.06 * (ext[3] - ext[2])
arrow = dict(arrowstyle="-|>", color="#111111", lw=0.8)
for k, (title, a, e) in enumerate(panels):
    ax = fig.add_subplot(gs[k]); axes.append(ax)
    ax.set_facecolor("#dfe7f1")
    from matplotlib.colors import LinearSegmentedColormap as _LSC
    from nature_style import PM_COLORS
    im = ax.imshow(mask(a)[sl], extent=ext, cmap=_LSC.from_list("pmc", PM_COLORS, N=256), vmin=vmin, vmax=vmax, interpolation="none")
    for r in ROWS:                                           # tile edges marked outside the image, so the seam stays visible
        if R - W < r < R + W:
            ax.annotate("", xy=(ext[0], lat(r)), xytext=(ext[0] - dx, lat(r)), arrowprops=arrow, annotation_clip=False)
            ax.annotate("", xy=(ext[1], lat(r)), xytext=(ext[1] + dx, lat(r)), arrowprops=arrow, annotation_clip=False)
    for c in COLS:
        if C - W < c < C + W:
            ax.annotate("", xy=(lon(c), ext[2]), xytext=(lon(c), ext[2] - dy), arrowprops=arrow, annotation_clip=False)
            ax.annotate("", xy=(lon(c), ext[3]), xytext=(lon(c), ext[3] + dy), arrowprops=arrow, annotation_clip=False)
    if e is not None:
        ax.text(0.03, 0.04, f"RMSE {e:.2f}", transform=ax.transAxes, fontsize=6.2, ha="left", va="bottom", zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.92),
                fontweight="bold" if k == 2 else "normal")
    ax.set_title(title, fontsize=7.2, pad=3); ax.grid(False)
    xt = np.arange(np.ceil(ext[0] + 0.3), ext[1] - 0.3, 2); yt = np.arange(np.ceil(ext[2] + 0.3), ext[3] - 0.3, 2)
    ax.set_xticks(xt); ax.set_xticklabels([f"{v:.0f}°E" for v in xt])
    ax.set_yticks(yt); ax.set_yticklabels([f"{v:.0f}°N" for v in yt] if k == 0 else [])
    ax.tick_params(labelsize=5.8, length=2)
cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.13, aspect=50, extend="both")
cb.set_label("PM$_{2.5}$ at 1 km, 1 day ahead (µg m$^{-3}$); arrows: tile edges", fontsize=6.3)
cb.ax.tick_params(labelsize=5.8)

# (d) seam ratio over the whole of Europe
axd = fig.add_subplot(gs[3])
pos = axd.get_position(); axd.set_position([pos.x0 + 0.075, pos.y0, pos.width - 0.075, pos.height])
bars = [("SimVP, tiled", 4.82, STYLE["simvp"]["color"]), ("Earthformer, tiled", 1.77, STYLE["earthformer"]["color"]),
        ("Same net, regression", 1.19, "#8FB2DD"), (OURS_KM, 1.01, STYLE["foehn"]["color"]),
        ("Observed (GHAP)", 1.04, "#8064A2")]
yy = np.arange(len(bars))
axd.barh(yy, [b[1] for b in bars], color=[b[2] for b in bars], height=0.62, edgecolor="white", lw=0.8, zorder=3)
for y_, (lab, v, _) in zip(yy, bars):
    axd.text(v + 0.1, y_, f"{v:.2f}", ha="left", va="center", fontsize=6.3, color="#222222",
             fontweight="bold" if lab == OURS_KM else "normal")
axd.axvline(1, color="#555555", ls="--", lw=0.7, zorder=4)
axd.text(1.08, 4.55, "no seam", ha="left", va="center", fontsize=5.8, color="#555555")
axd.set_yticks(yy); axd.set_yticklabels([b[0] for b in bars], fontsize=6.2)
axd.tick_params(axis="y", length=0, pad=2); axd.tick_params(axis="x", labelsize=5.8)
axd.set_xlim(0, 5.7); axd.set_xticks([0, 1, 2, 3, 4, 5]); axd.set_ylim(-0.55, 4.8)
axd.set_xlabel("Jump at tile edges / elsewhere", fontsize=6.3, labelpad=2)
axd.set_title("(d) Seams, whole of Europe", fontsize=7.2, pad=3)
axd.grid(axis="y", visible=False); axd.grid(axis="x", lw=0.4, color="#e3e3e3", zorder=0)
save(fig, "fig_km")
print("window lat %.2f-%.2f lon %.2f-%.2f | RMSE tiled %.2f free %.2f" % (ext[2], ext[3], ext[0], ext[1], rmse(tiled), rmse(free)))
