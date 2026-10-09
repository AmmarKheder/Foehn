#!/usr/bin/env python3
"""Extended Data figure: cost and robustness. a, measured training energy (node power counters, LUMI MI250X) and
node-hours per model; b, inference cost per forecast against the number of output pixels (one GCD); c, station CRPS
versus lead time of three trainings (seeds 42, 43, 44) of Foehn, identical station-days.
    python figED_cost.py <run>   (data: figdata_cost.py -> cost_all.json, figdata_seeds.py)"""
import json
import sys

import numpy as np

from nature_style import DOUBLE, STYLE, letter, plt, save

RUN = sys.argv[1] if len(sys.argv) > 1 else "fp_flow_shift25_global_L7_fix"
C = json.load(open("data/cost_all.json"))
S = np.load(f"data/seeds_{RUN}.npz")
fig = plt.figure(figsize=(DOUBLE, 2.35))
# a: training energy
ax = fig.add_axes([0.14, 0.2, 0.22, 0.7])
rows = [("ConvLSTM", "convlstm", "ConvLSTM, Europe 0.25°"), ("SimVP", "simvp", "SimVP, Europe 0.25°"),
        ("Earthformer", "earthformer", "Earthformer, Europe 0.25°"), ("ClimaX", "climax", "ClimaX, Europe 0.25°"),
        ("TopoFlow", "topoflow", "TopoFlow, Europe 0.25°"), ("Earthformer (1 km region)", "earthformer", "Earthformer, 1 km region"),
        ("SimVP (1 km region)", "simvp", "SimVP, 1 km region"), ("This work, Europe 1 km", "ours", "Foehn, Europe 1 km"),
        ("This work, global 1 km", "ours", "Foehn, global 1 km")]
y = np.arange(len(rows))[::-1]
for yy, (k, sty, lab) in zip(y, rows):
    e, nh = C[k]["energy_kwh"], C[k]["node_hours"]
    ax.barh(yy, e, height=0.62, color=STYLE[sty]["color"], alpha=0.95 if "Foehn" in lab else 0.75, lw=0, zorder=3)
    ax.text(e * 1.15, yy, f"{e:.3g} kWh · {nh:.3g} node-h", va="center", fontsize=5, color="#333333")
ax.set_yticks(y); ax.set_yticklabels([r[2] for r in rows], fontsize=5.5)
ax.set_xscale("log"); ax.set_xlim(0.3, 3000)
ax.set_xlabel("Training energy per run (kWh)")
letter(ax, "a", x=-0.62, y=1.02)
# b: inference cost per forecast versus output pixels
ax = fig.add_axes([0.47, 0.2, 0.2, 0.7])
pts = [("Baselines, Europe 0.25°", 168 * 280, C["Earthformer"]["forward_s"], STYLE["earthformer"]["color"], "o", (5, 0), "left"),
       ("Aurora, global 0.4°", 900 * 450, C["Aurora"]["forward_s"], STYLE["aurora"]["color"], "o", (5, 0), "left"),
       ("Foehn, Europe 1 km", 4192 * 6992, 8 * C["This work, Europe 1 km"]["forward_s"], STYLE["ours"]["color"], "s", (-6, 0), "right"),
       ("Foehn, global 1 km", 14400 * 36000, 745.0 / 4, STYLE["ours"]["color"], "D", (-6, 0), "right")]
for lab, px, sec, col, mk, off, ha in pts:
    ax.plot(px, sec, mk, color=col, ms=4.5, markeredgecolor="white", markeredgewidth=0.4, zorder=4)
    ax.annotate(lab, (px, sec), xytext=off, textcoords="offset points", fontsize=5.2, va="center", ha=ha)
ax.text(0.03, 0.97, "Foehn: one member,\n8 flow steps", transform=ax.transAxes, fontsize=5, va="top", color="#555555")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e4, 3e9); ax.set_ylim(3e-3, 3e3)
ax.set_xlabel("Output pixels per forecast"); ax.set_ylabel("GPU seconds per forecast (one GCD)")
letter(ax, "b", x=-0.3, y=1.02)
# c: seeds
ax = fig.add_axes([0.79, 0.2, 0.19, 0.7])
leads = np.arange(1, 8)
cr = np.array([S[f"crps_{h}"] for h in leads])                       # (7, 3)
for k, (lab, ls) in enumerate((("seed 42", "-"), ("seed 43", (0, (3, 1.5))), ("seed 44", (0, (1, 1.2))))):
    ax.plot(leads, cr[:, k], color=STYLE["ours"]["color"], ls=ls, lw=0.9, marker="o", ms=2, label=lab)
ax.fill_between(leads, cr.min(1), cr.max(1), color=STYLE["member"]["color"], alpha=0.35, lw=0)
ax.set_xlabel("Lead time (days)"); ax.set_ylabel("Station CRPS (µg m$^{-3}$)")
ax.set_xticks(leads)
ax.legend(loc="lower right", fontsize=5.2)
letter(ax, "c", x=-0.32, y=1.02)
save(fig, f"figED_cost_{RUN}")
