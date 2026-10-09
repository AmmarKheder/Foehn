#!/usr/bin/env python3
"""Extended Data figure: distribution of the forecasts versus the AI baselines (Europe, 0.25 deg, 2022, 61 start days).
a, b: quantile-quantile plots of forecast PM2.5 against the target (GHAP), leads 1 and 3 days, all valid land cells;
c: forecast / target ratio of the upper quantiles (95, 99, 99.9 %). Deterministic AI models trained on the same
target over Europe (ConvLSTM, Earthformer, SimVP, TopoFlow) against Foehn (global model; ensemble mean and one member).
    python figED_quantiles_ai.py <global run>       (data: figdata_quantiles_eu.py)"""
import sys

import numpy as np

from nature_style import DOUBLE, STYLE, letter, plt, save

RUN = sys.argv[1] if len(sys.argv) > 1 else "fp_flow_shift25_global_L7_fix"
z = np.load(f"data/quantiles_eu_{RUN}.npz")
P = z["p"]
names = [str(n) for n in z["names"]]
SHOW = [("foehn_member", STYLE["member"]), ("foehn_mean", STYLE["ours"]), ("topoflow", STYLE["topoflow"]),
        ("earthformer", STYLE["earthformer"]), ("convlstm", STYLE["convlstm"]), ("simvp", STYLE["simvp"])]
LAB = {"foehn_member": "Foehn, one member", "foehn_mean": "Foehn, ensemble mean", "topoflow": "TopoFlow",
       "earthformer": "Earthformer", "convlstm": "ConvLSTM", "simvp": "SimVP"}
fig = plt.figure(figsize=(DOUBLE, 2.45))
for k, h in enumerate((1, 3)):
    ax = fig.add_axes([0.065 + k * 0.315, 0.2, 0.25, 0.68])
    tq = z[f"q_target_{h}"]
    lim = [0.8 * tq[0], 1.25 * tq[-1]]
    ax.plot(lim, lim, color="#111111", lw=0.6, ls=(0, (1, 1)), zorder=1)
    for key, st in SHOW:
        if f"q_{key}_{h}" not in z.files:
            continue
        ax.plot(tq, z[f"q_{key}_{h}"], color=st["color"], lw=st["lw"], marker=st["marker"], ms=2.2, zorder=st["z"],
                markeredgecolor="white", markeredgewidth=0.3, label=LAB[key])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Target quantile, GHAP (µg m$^{-3}$)"); ax.set_ylabel("Forecast quantile (µg m$^{-3}$)")
    ax.set_title(f"Lead time {h} day{'s' if h > 1 else ''}", pad=3)
    for q in (95, 99, 99.9):
        x = tq[list(P).index(q)]
        ax.axvline(x, color="#bbbbbb", lw=0.4, zorder=0)
        ax.text(x, lim[0] * 1.05, f" {q:g}%", fontsize=5, color="#777777", rotation=90, va="bottom", ha="right")
    letter(ax, "ab"[k], x=-0.2, y=1.02)
    if k == 0:
        ax.legend(loc="upper left", fontsize=5.3, handlelength=1.6, labelspacing=0.25)
# c: ratios of the upper quantiles
ax = fig.add_axes([0.72, 0.2, 0.265, 0.68])
qs = (95, 99, 99.9)
keys = [k_ for k_, _ in SHOW if f"q_{k_}_3" in z.files]
xx = np.arange(len(keys))
for j, (h, alpha) in enumerate(((1, 0.45), (3, 1.0))):
    for i, key in enumerate(keys):
        r = [z[f"q_{key}_{h}"][list(P).index(q)] / z[f"q_target_{h}"][list(P).index(q)] for q in qs]
        st = dict(SHOW)[key]
        ax.bar(i + (j - 0.5) * 0.38, r[1], width=0.36, color=st["color"], alpha=alpha, lw=0, zorder=3)
        ax.plot([i + (j - 0.5) * 0.38] * 2, [r[0], r[2]], color="#333333", lw=0.5, zorder=4)
ax.axhline(1.0, color="#111111", lw=0.6, ls=(0, (1, 1)))
SHORT = {"foehn_member": "Foehn\nmember", "foehn_mean": "Foehn\nmean", "topoflow": "Topo-\nFlow", "earthformer": "Earth-\nformer",
         "convlstm": "Conv-\nLSTM", "simvp": "SimVP"}
ax.set_xticks(xx); ax.set_xticklabels([SHORT[k_] for k_ in keys], fontsize=5.3, linespacing=0.95)
ax.set_ylabel("Forecast / target, 99 % quantile\n(whiskers: 95 % and 99.9 %)", fontsize=6)
ax.set_ylim(0.5, 1.2)
ax.text(0.02, 0.97, "pale: 1 day, dark: 3 days", transform=ax.transAxes, fontsize=5.2, va="top")
letter(ax, "c", x=-0.22, y=1.02)
save(fig, f"figED_quantiles_ai_{RUN}")
