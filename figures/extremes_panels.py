#!/usr/bin/env python3
"""The four panels on local extremes and the value of warnings (Fig. 2e-h until 15 October 2026, now under the two rows of
maps of the Extended Data figure on hidden exposure, fig_exposure.py): detection of the days above the local 99th percentile
over land, SEDI by lead time and by region one day ahead (tail_events.py: local percentiles of each 0.25 deg cell and calendar
day, 2017-2020, identical cells and days), and the retrospective potential relative economic value of warnings of days above
the local 95th and 99th percentiles at the stations (extremes_stations.py), Foehn's ensemble mean against the CAMS forecast and
Aurora issued at the same time. extremes_row(fig, fx, fy, y0) draws them into a 183 mm wide figure; the letters go before
the left-aligned titles of the returned axes (gc_style.letter_before, after fig.canvas.draw())."""
import numpy as np

import final_model as FM
from fig_gc4_extremes import rev_curves, sedi
from gc_style import AURORA, CAMS, FOEHN, clean
from nature_style import HERE


def extremes_row(fig, fx, fy, y0=12.5, h=22.0, legend_y=0.6):
    """axes bottom at y0 mm and h mm high; the legend of the row at legend_y mm; returns the four axes"""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    te = np.load(HERE / "data" / f"tail_events_{FM.LABEL}.npz"); CT = te["C"]
    TS = [str(x) for x in te["systems"]]; TR = [str(r) for r in te["regions"]]
    SYSE = [("Foehn", f"{FM.LABEL} (mean)", FOEHN, 1.65, "o", 2.4), ("CAMS", "CAMS", CAMS, 1.0, "s", 1.9),
            ("Aurora", "Aurora", AURORA, 1.0, "D", 1.8)]
    L = np.arange(1, 8); gl = TR.index("Global")
    ae = fig.add_axes([fx(11), fy(y0), fx(33), fy(h)])
    for nm, key, col, lw, mk, ms in SYSE:
        c = CT[gl, :, 2, TS.index(key)]; v = sedi(c); ok = np.isfinite(v) & (c[:, :2].sum(-1) > 0)
        ae.plot(L[ok], v[ok], color=col, lw=lw, marker=mk, ms=ms, zorder=5 if nm == "Foehn" else 3)
    ae.set_xticks(L); ae.set_xlim(0.6, 7.4); ae.set_ylim(0.25, 0.68)
    ae.set_xlabel("Lead time (days)", fontsize=6, labelpad=1); ae.set_ylabel("SEDI, all land", fontsize=6)
    clean(ae); ae.tick_params(labelsize=5.5)
    ae.set_title("Days above the local P99", fontsize=6.3, loc="left", pad=5)
    af = fig.add_axes([fx(66), fy(y0), fx(24), fy(h)])
    regs = [r for r in TR if r != "Global"]; yy = np.arange(len(regs))[::-1]
    for y_, r in zip(yy, regs):
        vals = [float(sedi(CT[TR.index(r), 0, 2, TS.index(key)])) for _, key, *_ in SYSE]
        af.plot([min(vals), max(vals)], [y_, y_], color="#D5DBE0", lw=2.2, solid_capstyle="round", zorder=1)
        for (nm, key, col, lw, mk, ms), v in zip(SYSE, vals):
            af.plot([v], [y_], marker=mk, ms=ms + (0.9 if nm == "Foehn" else 0.4), color=col, mec="white", mew=0.35,
                    ls="none", zorder=5 if nm == "Foehn" else 3)
        print(f"SEDI >P99 1 d {r:14s}", " ".join(f"{nm} {v:.2f}" for (nm, *_), v in zip(SYSE, vals)))
    af.set_yticks(yy); af.set_yticklabels(regs, fontsize=5.5); af.set_ylim(-0.6, len(regs) - 0.4)
    af.set_xlim(0.2, 0.72); af.set_xticks([0.2, 0.4, 0.6])
    af.set_xlabel("SEDI, 1 day ahead", fontsize=6, labelpad=1)
    clean(af); af.tick_params(labelsize=5.5); af.tick_params(axis="y", length=0)
    af.grid(axis="y", visible=False)
    af.set_title("By region", fontsize=6.3, loc="left", pad=5)
    es = np.load(HERE / "data" / f"extremes_stations_{FM.LABEL}.npz"); CE = es["C"]; ES = [str(x) for x in es["systems"]]
    alphas = np.logspace(np.log10(0.003), np.log10(0.6), 250)
    SYSR = [("Foehn", FM.LABEL, FOEHN, 1.65), ("CAMS", "CAMS", CAMS, 1.0), ("Aurora", "Aurora", AURORA, 1.0)]
    axs = [ae, af]
    for k, (tier, title, ymax) in enumerate(((1, "Warning value, days above P95", 0.4), (2, "Warning value, days above P99", 0.3))):
        axr = fig.add_axes([fx(104 + 44 * k), fy(y0), fx(34), fy(h)])
        for nm, key, col, lw in SYSR:
            for hh, ls in ((0, "-"), (2, (0, (4, 2)))):
                v = rev_curves(CE[0, tier, hh, ES.index(key)], alphas)
                axr.plot(alphas, v, color=col, lw=lw, ls=ls, zorder=5 if nm == "Foehn" else 3)
        axr.set_xscale("log"); axr.set_xlim(0.01, 0.6); axr.set_ylim(0, ymax)
        axr.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 0.5]))
        axr.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}")); axr.xaxis.set_minor_formatter(NullFormatter())
        axr.set_xlabel("Cost / avoided loss", fontsize=6, labelpad=1)
        if k == 0:
            axr.set_ylabel("Relative economic value", fontsize=6)
        clean(axr); axr.tick_params(labelsize=5.5)
        axr.set_title(title, fontsize=6.3, loc="left", pad=5)
        axs.append(axr)
    handles = [Line2D([0], [0], color=c, linewidth=1.6, label=nm) for nm, _, c, _ in SYSR]
    handles += [Line2D([0], [0], color="#56636D", linewidth=1.1, linestyle=ls, label=lab)
                for lab, ls in (("1 day ahead", "-"), ("3 days ahead", (0, (4, 2))))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, fy(legend_y)), ncol=5, fontsize=5.5,
               columnspacing=1.0, handlelength=1.8)
    return axs
