#!/usr/bin/env python3
"""Fig. 2 — skill against operational forecasts at monitoring stations (2022, identical station-days for every system):
a, scorecard of the change of RMSE and CRPS of Foehn relative to the CAMS forecast and to Aurora, by network and lead
time (blue: Foehn better); b, c, RMSE and CRPS versus lead time at all stations for Foehn, CAMS, Aurora, persistence and
climatology. Data: data/compare_baselines_<label>.npz (compare_baselines.py).
    python fig2_scorecard.py [label A60]"""
import sys

import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

from nature_style import DOUBLE, MM, STYLE, letter, plt, save

LABEL = sys.argv[1] if len(sys.argv) > 1 else "A60"
z = np.load(f"data/compare_baselines_{LABEL}.npz")
S, SYS, NETS = z["S"], [str(s) for s in z["systems"]], [str(n) for n in z["nets"]]
K = {"ours": SYS.index(LABEL), "cams": SYS.index("CAMS"), "aurora": SYS.index("Aurora"),
     "persistence": SYS.index("persistence"), "climatology": SYS.index("climatology")}
ROWS = [("all official", "All stations"), ("China (CNEMC)", "China"), ("India (CPCB)", "India"), ("Korea (AirKorea)", "Korea"),
        ("Japan (NIES)", "Japan"), ("USA (EPA)", "United States"), ("Europe (EEA)", "Europe")]


def score(g, h, k, m):
    v = S[g, h - 1, k]
    if v[0] == 0:
        return np.nan
    return np.sqrt(v[2] / v[0]) if m == "rmse" else v[8] / v[0]


# ---------------------------------------------------------------- scorecard values: % change of the error (negative = better)
METRICS = [("rmse", "RMSE"), ("crps", "CRPS")]
cols = [("cams", h) for h in range(1, 5)] + [None] + [("aurora", h) for h in range(1, 5)]
rows = [(g_name, lab, m, mlab) for g_name, lab in ROWS for m, mlab in METRICS]
V = np.full((len(rows), len(cols)), np.nan)
for i, (g_name, lab, m, mlab) in enumerate(rows):
    g = NETS.index(g_name)
    for j, c in enumerate(cols):
        if c is None:
            continue
        ref, h = c
        a, b = score(g, h, K["ours"], m), score(g, h, K[ref], m)
        V[i, j] = 100.0 * (a / b - 1.0)

fig = plt.figure(figsize=(DOUBLE, 92 * MM))
ax = fig.add_axes([0.13, 0.08, 0.40, 0.80])
bounds = [-50, -40, -30, -20, -10, -5, 5, 10, 20, 30, 40, 50]                 # symmetric, white bin = within +-5 %
cmap = LinearSegmentedColormap.from_list("bwr_nature", ["#2166ac", "#67a9cf", "#d1e5f0", "#f7f7f7", "#fddbc7", "#ef8a62", "#b2182b"])
cm = plt.cm.colors.ListedColormap(cmap(np.linspace(0, 1, len(bounds) + 1)))   # bins + the two extensions
norm = BoundaryNorm(bounds, cm.N, extend="both")
Vm = np.ma.masked_invalid(V)
ax.pcolormesh(np.arange(len(cols) + 1), np.arange(len(rows) + 1), Vm, cmap=cm, norm=norm, edgecolors="white", linewidth=0.6)
for i in range(len(rows)):
    for j in range(len(cols)):
        if np.isfinite(V[i, j]):
            v = V[i, j]
            ax.text(j + 0.5, i + 0.5, f"{v:+.0f}", ha="center", va="center", fontsize=5,
                    color="white" if abs(v) >= 30 else "#222222")
ax.set_xlim(0, len(cols)); ax.set_ylim(len(rows), 0)
ax.set_xticks([j + 0.5 for j, c in enumerate(cols) if c is not None])
ax.set_xticklabels([str(c[1]) for c in cols if c is not None], fontsize=6)
ax.set_yticks(np.arange(len(rows)) + 0.5)
ax.set_yticklabels([mlab for _, _, _, mlab in rows], fontsize=5.5)
ax.tick_params(length=0, pad=1.5)
for sp in ax.spines.values():
    sp.set_visible(False)
for k, (g_name, lab) in enumerate(ROWS):                          # network names, one per pair of rows
    ax.text(-1.55, 2 * k + 1.0, lab, ha="right", va="center", fontsize=6, transform=ax.transData)
    if k:
        ax.axhline(2 * k, color="white", lw=2.0)
ax.text(2.0, -0.55, "Against CAMS", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
ax.text(7.0, -0.55, "Against Aurora", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
ax.set_xlabel("Lead time (days)", fontsize=6, labelpad=2)
cax = fig.add_axes([0.16, 0.0, 0.34, 0.018])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), cax=cax, orientation="horizontal", ticks=bounds, extend="both")
cb.ax.tick_params(labelsize=5, length=1.5, pad=1); cb.outline.set_linewidth(0.4)
cb.set_label("Change of error relative to the baseline (%); negative: Foehn better", fontsize=5.5, labelpad=1)
letter(ax, "a", x=-0.27, y=1.04)

# ---------------------------------------------------------------- lead-time curves, all stations
g = NETS.index("all official")
for p, (m, ylab, lett) in enumerate((("rmse", "RMSE (µg m$^{-3}$)", "b"), ("crps", "CRPS (µg m$^{-3}$)", "c"))):
    axl = fig.add_axes([0.64, 0.56 - 0.48 * p, 0.34, 0.37])
    for key in ("climatology", "persistence", "aurora", "cams", "ours"):
        st = dict(STYLE[key])
        if key == "persistence":                                   # distinct from climatology (black dashes)
            st.update(color="#9a9a9a", ls=(0, (3, 1.5)))
        leads = range(1, 5) if key in ("cams", "aurora") else range(1, 8)
        y = [score(g, h, K[key], m) for h in leads]
        axl.plot(list(leads), y, color=st["color"], ls=st["ls"], marker=st["marker"] or None, lw=st["lw"], ms=st["ms"],
                 zorder=st["z"], label=st["label"].split(" (")[0])
    axl.set_xticks(range(1, 8)); axl.set_xlim(0.7, 7.3)
    axl.set_ylabel(ylab, fontsize=6)
    axl.grid(axis="y", lw=0.3, color="#dddddd"); axl.set_axisbelow(True)
    axl.spines[["top", "right"]].set_visible(False)
    axl.tick_params(labelsize=5.5, length=2)
    if p == 1:
        axl.set_xlabel("Lead time (days)", fontsize=6)
    else:
        axl.tick_params(labelbottom=False)
        hs, ls_ = axl.get_legend_handles_labels()
        order = [ls_.index(n) for n in ("Foehn", "CAMS", "Aurora", "Persistence", "Climatology")]
        axl.legend([hs[i] for i in order], [ls_[i] for i in order], fontsize=5.5, frameon=False, ncol=3, loc="lower left",
                   bbox_to_anchor=(-0.02, 1.0), handlelength=2.2, columnspacing=1.0)
    letter(axl, lett, x=-0.2, y=1.0)
save(fig, f"fig2_scorecard_{LABEL}")
