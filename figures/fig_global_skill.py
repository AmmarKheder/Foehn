#!/usr/bin/env python3
"""Fig. 2 of the Nature paper: skill at ~8,000 PM2.5 monitoring stations on five continents (2022, never used in
training). a, RMSE versus lead time for this work, CAMS, Aurora, persistence and GHAP (identical station-days at each
lead). b, station maps: RMSE of this work and its difference to CAMS and Aurora (leads 1-4 pooled; blue = this work
better; boxes: share of stations where this work is better). c, scorecards: RMSE difference (%) per network and lead.
    python fig_global_skill.py <run>     (data from figdata_global_skill.py)"""
import sys
from pathlib import Path

import numpy as np

from nature_style import DOUBLE, NET_NAMES, STYLE, diff_cmap, letter, plt, save, world_axes

RUN = sys.argv[1] if len(sys.argv) > 1 else "fp_flow_shift25_global_L7_fix"
z = np.load(Path(__file__).resolve().parent / "data" / f"global_skill_{RUN}.npz")
S, CSI, SE, NN, lat, lon, src = z["S"], z["CSI"], z["SE"], z["NN"], z["lat"], z["lon"], z["src"]
MODELS = [str(m) for m in z["models"]]; NETS = [int(x) for x in z["nets"]]
rmse = lambda s: np.where(s[..., 0] > 0, np.sqrt(s[..., 2] / np.maximum(s[..., 0], 1)), np.nan)
R = rmse(S)                                        # (net, lead, model)
leads = np.arange(1, 8)
order = [len(NETS) - 1, NETS.index(1), NETS.index(6), NETS.index(4), NETS.index(5), NETS.index(0), NETS.index(3), NETS.index(2)]
title = lambda g: "All stations" if NETS[g] == -1 else NET_NAMES[NETS[g]]
nst = lambda g: int(((src >= 0) if NETS[g] == -1 else (src == NETS[g])).sum())

nst = lambda g: int((((src >= 0) if NETS[g] == -1 else (src == NETS[g])) & (NN > 0)).sum())   # stations that contributed

fig = plt.figure(figsize=(DOUBLE, 6.4))
# a: RMSE versus lead, by network
W, H, X0 = 0.205, 0.145, 0.06
for k, g in enumerate(order):
    ax = fig.add_axes([X0 + (k % 4) * 0.243, [0.79, 0.578][k // 4], W, 0.152])
    for m in ("ghap", "persistence", "aurora", "cams", "ours"):
        st = STYLE[m]; y = R[g, :, MODELS.index(m)]
        ax.plot(leads, y, color=st["color"], ls=st["ls"], lw=st["lw"], marker=st["marker"], ms=st["ms"], zorder=st["z"],
                markeredgecolor="white", markeredgewidth=0.3, clip_on=False)
    ax.set_xticks(leads); ax.set_xlim(0.7, 7.3)
    lo = np.nanmin(R[g]); hi = np.nanmax(R[g])
    ax.set_ylim(max(0, lo - 0.12 * (hi - lo)), hi + 0.08 * (hi - lo))
    ax.set_title(f"{title(g)}  ·  {nst(g):,} stations", fontsize=6.3, pad=2)
    if k % 4 == 0:
        ax.set_ylabel("RMSE (µg m$^{-3}$)")
    if k // 4 == 1:
        ax.set_xlabel("Lead time (days)")
    if k == 0:
        letter(ax, "a", x=-0.2, y=1.13)
handles = [plt.Line2D([], [], color=STYLE[m]["color"], ls=STYLE[m]["ls"], lw=STYLE[m]["lw"], marker=STYLE[m]["marker"],
                      ms=STYLE[m]["ms"], markeredgecolor="white", markeredgewidth=0.3) for m in ("ours", "cams", "aurora", "persistence", "ghap")]
fig.legend(handles, [STYLE[m]["label"] for m in ("ours", "cams", "aurora", "persistence", "ghap")], loc="lower center",
           bbox_to_anchor=(0.53, 0.962), ncol=5, frameon=False)

# b: station maps (leads 1-4 pooled, same station-days)
ok = NN >= 100
r_st = np.sqrt(SE / np.maximum(NN, 1)[:, None]); r_st[~ok] = np.nan
mo, mc, ma = MODELS.index("ours"), MODELS.index("cams"), MODELS.index("aurora")
from matplotlib.colors import BoundaryNorm, ListedColormap
seq_b = [0, 4, 6, 8, 10, 12, 15, 20, 30, 40, 60]
seq = ListedColormap(plt.get_cmap("magma_r")(np.linspace(0.05, 0.92, len(seq_b) - 1)))
DB = [-10, -6, -3, -1, 1, 3, 6, 10]
panels = [("RMSE of this work (lead times 1–4 days)", r_st[:, mo], seq, BoundaryNorm(seq_b, seq.N), None, None),
          ("RMSE difference: this work − CAMS", r_st[:, mo] - r_st[:, mc], *diff_cmap(DB), "CAMS", None),
          ("RMSE difference: this work − Aurora", r_st[:, mo] - r_st[:, ma], *diff_cmap(DB), "Aurora", None)]
import cartopy.crs as ccrs
for k, (ttl, v, cmap, norm, ref, _) in enumerate(panels):
    ax = world_axes(fig, [0.005 + k * 0.333, 0.305, 0.325, 0.185])
    good = np.isfinite(v)
    o = np.argsort(np.abs(v[good]) if ref else v[good])                     # strongest signals on top
    ax.scatter(lon[good][o], lat[good][o], c=v[good][o], s=1.5, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
               linewidths=0, zorder=5, rasterized=True)
    ax.set_title(ttl, fontsize=6.5, pad=2)
    if ref is not None:
        share = np.mean(v[good] < 0) * 100
        ax.text(0.085, 0.43, f"{share:.1f}%", transform=ax.transAxes, fontsize=7.5, fontweight="bold", color="#08306b", zorder=10,
                ha="left", va="bottom")
        ax.text(0.085, 0.42, f"of stations\nbetter than\n{ref}", transform=ax.transAxes, fontsize=5.2, color="#08306b", zorder=10,
                ha="left", va="top", linespacing=1.05)
    cax = fig.add_axes([0.045 + k * 0.333, 0.29, 0.245, 0.008])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                      extend="both" if ref else "max", spacing="uniform", ticks=DB if ref else seq_b)
    cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.4, pad=1); cb.outline.set_linewidth(0.4)
    cb.ax.set_xticklabels([f"{t:g}" for t in (DB if ref else seq_b)])
    cb.set_label("µg m$^{-3}$", fontsize=5.8, labelpad=1)
    if ref:
        cax.text(0.0, -3.4, "← this work better", transform=cax.transAxes, ha="left", va="top", fontsize=5.5, color="#2166ac")
        cax.text(1.0, -3.4, "worse →", transform=cax.transAxes, ha="right", va="top", fontsize=5.5, color="#b2182b")
    if k == 0:
        letter(ax, "b", x=0.0, y=1.0)

# c: scorecards, RMSE difference (%)
PB = [-40, -25, -15, -8, -3, 3, 8, 15, 25, 40]
cm, nm = diff_cmap(PB)
rows = order
for k, (ref, nl, ttl) in enumerate((("cams", 4, "versus CAMS"), ("aurora", 4, "versus Aurora"), ("persistence", 7, "versus persistence"))):
    D = 100 * (R[rows, :nl, mo] / R[rows, :nl, MODELS.index(ref)] - 1)
    ax = fig.add_axes([[0.135, 0.355, 0.575][k], 0.035, 0.12 if nl == 4 else 0.21, 0.15])
    ax.imshow(D, cmap=cm, norm=nm, aspect="auto")
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            ax.text(j, i, f"{D[i, j]:+.0f}", ha="center", va="center", fontsize=4.8,
                    color="white" if abs(D[i, j]) >= 25 else "#111111")
    ax.set_xticks(range(nl)); ax.set_xticklabels([str(i) for i in range(1, nl + 1)])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([title(g) for g in rows] if k == 0 else [])
    ax.tick_params(length=0, pad=1.5); [sp.set_visible(False) for sp in ax.spines.values()]
    ax.set_xlabel("Lead time (days)", fontsize=6); ax.set_title(f"RMSE difference {ttl}", fontsize=6.5, pad=2)
    if k == 0:
        letter(ax, "c", x=-0.95, y=1.02)
cax = fig.add_axes([0.825, 0.05, 0.009, 0.12])
TK = [-40, -15, -3, 3, 15, 40]
cb = fig.colorbar(plt.cm.ScalarMappable(norm=nm, cmap=cm), cax=cax, extend="both", spacing="uniform", ticks=TK)
cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.4, pad=1); cb.outline.set_linewidth(0.4)
cb.ax.set_yticklabels([f"{t:+g}%" for t in TK])
cax.text(0.5, 1.13, "worse", transform=cax.transAxes, ha="center", va="bottom", fontsize=5.5, color="#b2182b")
cax.text(0.5, -0.13, "better", transform=cax.transAxes, ha="center", va="top", fontsize=5.5, color="#2166ac")
save(fig, f"fig2_skill_{RUN}")
for g in order:
    print(f"{title(g):18s} lead1: ours {R[g,0,mo]:.2f} CAMS {R[g,0,mc]:.2f} Aurora {R[g,0,ma]:.2f} | lead7 ours {R[g,6,mo]:.2f}")
