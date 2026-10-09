#!/usr/bin/env python3
"""Fig. 2 of the Nature paper: skill at ~8,000 PM2.5 monitoring stations on five continents (2022; stations never used
for training). a, RMSE and b, critical success index of polluted days (daily mean > 35 ug/m3, WHO interim target 3)
versus lead time, all stations, for this work, CAMS, Aurora, climatology and GHAP (identical station-days at each lead).
c, stations where this work is better than CAMS / Aurora (RMSE difference, lead times 1-4 days pooled). d, RMSE
difference (%) per monitoring network and lead time.     python fig2_skill.py <run>   (data: figdata_global_skill.py)"""
import sys
from pathlib import Path

import cartopy.crs as ccrs
import numpy as np

from nature_style import DOUBLE, NET_NAMES, STYLE, diff_cmap, letter, plt, save, world_axes

RUN = sys.argv[1] if len(sys.argv) > 1 else "fp_flow_shift25_global_L7_fix"
z = np.load(Path(__file__).resolve().parent / "data" / f"global_skill_{RUN}.npz")
S, CSI, SE, NN, lat, lon, src = z["S"], z["CSI"], z["SE"], z["NN"], z["lat"], z["lon"], z["src"]
MODELS = [str(m) for m in z["models"]]; NETS = [int(x) for x in z["nets"]]
R = np.where(S[..., 0] > 0, np.sqrt(S[..., 2] / np.maximum(S[..., 0], 1)), np.nan)
C = np.where(CSI[..., 1] > 0, CSI[..., 0] / np.maximum(CSI[..., 1], 1), np.nan)
leads = np.arange(1, 8); ALL = len(NETS) - 1
mo = MODELS.index("ours")
nst = int((NN > 0).sum()); nsd = int(S[ALL, 0, mo, 0])

fig = plt.figure(figsize=(DOUBLE, 6.35))
# a, b: all stations
for k, (Y, ylab, lt) in enumerate(((R, "RMSE (µg m$^{-3}$)", "a"), (C, "CSI, daily mean > 35 µg m$^{-3}$", "b"))):
    ax = fig.add_axes([0.075 + k * 0.5, 0.665, 0.4, 0.255])
    for m in ("ghap", "climatology", "aurora", "cams", "ours"):
        st = STYLE[m]
        ax.plot(leads, Y[ALL, :, MODELS.index(m)], color=st["color"], ls=st["ls"], lw=st["lw"] + 0.2, marker=st["marker"],
                ms=st["ms"] + 0.6, zorder=st["z"], markeredgecolor="white", markeredgewidth=0.4, clip_on=False)
    ax.set_xticks(leads); ax.set_xlim(0.7, 7.3); ax.set_xlabel("Lead time (days)"); ax.set_ylabel(ylab)
    ax.set_title(f"Error at {nst:,} stations ({nsd / 1e6:.1f} million station-days per lead time; lower is better)" if k == 0
                 else "Detection of polluted days (higher is better)", fontsize=6.5, pad=3)
    letter(ax, lt, x=-0.13, y=1.0)
handles = [plt.Line2D([], [], color=STYLE[m]["color"], ls=STYLE[m]["ls"], lw=STYLE[m]["lw"] + 0.2, marker=STYLE[m]["marker"],
                      ms=STYLE[m]["ms"] + 0.6, markeredgecolor="white", markeredgewidth=0.4) for m in ("ours", "cams", "aurora", "climatology", "ghap")]
fig.legend(handles, [STYLE[m]["label"] for m in ("ours", "cams", "aurora", "climatology", "ghap")], loc="lower center",
           bbox_to_anchor=(0.52, 0.955), ncol=5, frameon=False)

# c: where this work is better (leads 1-4, same station-days)
ok = NN >= 100
r_st = np.sqrt(SE / np.maximum(NN, 1)[:, None]); r_st[~ok] = np.nan
DB = [-10, -6, -3, -1, 1, 3, 6, 10]
cmap, norm = diff_cmap(DB)
for k, ref in enumerate(("cams", "aurora")):
    v = r_st[:, mo] - r_st[:, MODELS.index(ref)]
    ax = world_axes(fig, [0.01 + k * 0.5, 0.315, 0.48, 0.265])
    good = np.isfinite(v); o = np.argsort(np.abs(v[good]))
    ax.scatter(lon[good][o], lat[good][o], c=v[good][o], s=2.0, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
               linewidths=0, zorder=5, rasterized=True)
    name = "CAMS" if ref == "cams" else "Aurora"
    ax.set_title(f"RMSE difference, this work − {name}", fontsize=6.5, pad=2)
    ax.text(0.075, 0.40, f"{np.mean(v[good] < 0) * 100:.0f}%", transform=ax.transAxes, fontsize=9, fontweight="bold", color="#08306b", zorder=10)
    ax.text(0.075, 0.385, f"of stations\nbetter than\n{name}", transform=ax.transAxes, fontsize=5.5, color="#08306b", va="top", linespacing=1.05, zorder=10)
    if k == 0:
        letter(ax, "c", x=0.0, y=0.98)
cax = fig.add_axes([0.35, 0.292, 0.3, 0.009])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="both", spacing="uniform", ticks=DB)
cb.ax.set_xticklabels([f"{t:g}" for t in DB]); cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.4, pad=1); cb.outline.set_linewidth(0.4)
cb.set_label("RMSE difference (µg m$^{-3}$)", fontsize=5.8, labelpad=1)
cax.text(-0.06, 0.5, "this work better", transform=cax.transAxes, ha="right", va="center", fontsize=5.5, color="#2166ac")
cax.text(1.06, 0.5, "worse", transform=cax.transAxes, ha="left", va="center", fontsize=5.5, color="#b2182b")

# d: scorecards per network
PB = [-40, -25, -15, -8, -3, 3, 8, 15, 25, 40]
cm, nm = diff_cmap(PB)
order = [ALL, NETS.index(1), NETS.index(6), NETS.index(4), NETS.index(5), NETS.index(0), NETS.index(3), NETS.index(2)]
title = lambda g: "All stations" if NETS[g] == -1 else NET_NAMES[NETS[g]]
for k, (ref, nl, ttl) in enumerate((("cams", 4, "versus CAMS"), ("aurora", 4, "versus Aurora"), ("climatology", 7, "versus climatology"))):
    D = 100 * (R[order, :nl, mo] / R[order, :nl, MODELS.index(ref)] - 1)
    ax = fig.add_axes([[0.135, 0.355, 0.575][k], 0.03, 0.12 if nl == 4 else 0.21, 0.165])
    ax.imshow(D, cmap=cm, norm=nm, aspect="auto")
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            ax.text(j, i, f"{D[i, j]:+.0f}", ha="center", va="center", fontsize=4.8, color="white" if abs(D[i, j]) >= 25 else "#111111")
    ax.set_xticks(range(nl)); ax.set_xticklabels([str(i) for i in range(1, nl + 1)])
    ax.set_yticks(range(len(order))); ax.set_yticklabels([title(g) for g in order] if k == 0 else [])
    ax.tick_params(length=0, pad=1.5); [sp.set_visible(False) for sp in ax.spines.values()]
    ax.set_xlabel("Lead time (days)", fontsize=6); ax.set_title(f"RMSE difference {ttl}", fontsize=6.5, pad=2)
    if k == 0:
        letter(ax, "d", x=-0.95, y=1.03)
cax = fig.add_axes([0.825, 0.05, 0.009, 0.13])
TK = [-40, -15, -3, 3, 15, 40]
cb = fig.colorbar(plt.cm.ScalarMappable(norm=nm, cmap=cm), cax=cax, extend="both", spacing="uniform", ticks=TK)
cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.4, pad=1); cb.outline.set_linewidth(0.4)
cb.ax.set_yticklabels([f"{t:+g}%" for t in TK])
cax.text(0.5, 1.12, "worse", transform=cax.transAxes, ha="center", va="bottom", fontsize=5.5, color="#b2182b")
cax.text(0.5, -0.12, "better", transform=cax.transAxes, ha="center", va="top", fontsize=5.5, color="#2166ac")
save(fig, f"fig2_skill_v2_{RUN}")
print("all stations RMSE lead 1..7:", np.round(R[ALL, :, mo], 2), "| CAMS", np.round(R[ALL, :4, MODELS.index("cams")], 2))
print("CSI35 ours", np.round(C[ALL, :, mo], 3), "| CAMS", np.round(C[ALL, :4, MODELS.index("cams")], 3), "| Aurora", np.round(C[ALL, :4, MODELS.index("aurora")], 3))
