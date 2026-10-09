#!/usr/bin/env python3
"""Skill versus lead time (the MOSAIC 'WeatherBench' figure): RMSE at 0.25 deg, at 1 km and at the EEA stations, and CSI
of exceedances of 25 ug/m3, Europe 2022, real HRES forecast weather; mean over seeds, shaded +- 1 std; references
(climatology, CAMS operational forecast, Aurora zero-shot)."""
import numpy as np
import matplotlib.pyplot as plt

from fig_style import COL2, LEADS, STYLE, load, save

ORDER = ["climatology", "cams", "aurora", "climax", "topoflow", "simvp", "earthformer", "convlstm", "nofp", "foehn"]
PANELS = [("rmse_025", "pers_025", "RMSE, 0.25° (µg m$^{-3}$)", "(a) 25 km"),
          ("rmse_1km", "pers_1km", "RMSE, 1 km (µg m$^{-3}$)", "(b) 1 km"),
          ("rmse_eea", "pers_eea", "RMSE at stations (µg m$^{-3}$)", "(c) 2,971 EEA stations"),
          ("csi_025", "pcsi_025", "CSI, > 25 µg m$^{-3}$", "(d) Exceedances")]

LEARNED = ["climax", "topoflow", "simvp", "earthformer", "convlstm", "nofp", "foehn"]
fig, axes = plt.subplots(1, 4, figsize=(COL2, 2.05), constrained_layout=True)
for ax, (key, pkey, ylab, title) in zip(axes, PANELS):
    curves = {}
    for m in ORDER:
        mean, sd, n = load(m, key)
        if n == 0 or np.all(np.isnan(mean)):
            continue
        curves[m] = (mean, sd, n)
        st = STYLE[m]
        ax.plot(LEADS, mean, color=st["color"], ls=st["ls"], marker=st["marker"], ms=3.3 if m in ("foehn", "nofp") else 2.0, lw=1.9 if m == "foehn" else 0.85,
                label=st["label"], zorder=st["z"])
        if n > 1 and m == "foehn":
            ax.fill_between(LEADS, mean - sd, mean + sd, color=st["color"], alpha=0.15, lw=0, zorder=st["z"] - 1)
    ax.set_title(title)
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel(ylab)
    ax.set_xticks(LEADS)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc="outside lower center", ncol=5, frameon=False, columnspacing=1.2, handlelength=2.2)
save(fig, "fig_leadtime")
