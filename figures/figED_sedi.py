#!/usr/bin/env python3
"""Extended Data figure: detection of local extremes over land (tail_events.py, 2022). Events are the days above the local
90th, 95th or 99th percentile of the 2017-2020 GHAP climatology of each 0.25 deg cell and calendar day (+-15 days); forecasts
and the GHAP block means of the target day are compared on identical cells and days.
a-c, symmetric extremal dependence index (SEDI) of the three tiers over all land by lead time, for Foehn's ensemble mean
     and the CAMS and Aurora forecasts issued at the same time;
d,   SEDI of the extreme tier (> P99) one day ahead by region.
    python figED_sedi.py"""
import numpy as np

from fig_gc4_extremes import sedi
from gc_style import AURORA, CAMS, FOEHN, clean, label
import final_model as FM
from nature_style import DOUBLE, HERE, MM, plt, save


def main():
    te = np.load(HERE / "data" / f"tail_events_{FM.LABEL}.npz"); CT = te["C"]
    TS = [str(s) for s in te["systems"]]; TR = [str(r) for r in te["regions"]]
    SYS = [("Foehn", f"{FM.LABEL} (mean)", FOEHN, 1.6, "o"), ("CAMS", "CAMS", CAMS, 1.0, "s"), ("Aurora", "Aurora", AURORA, 1.0, "D")]
    W, H = DOUBLE, 52 * MM
    fig = plt.figure(figsize=(W, H))
    fx = lambda mm: mm / (W / MM); fy = lambda mm: mm / (H / MM)
    L = np.arange(1, 8); g = TR.index("Global")
    for t, name in enumerate(("Days > local 90th percentile", "Days > local 95th percentile", "Days > local 99th percentile")):
        ax = fig.add_axes([fx(12 + 39 * t), fy(10), fx(32), fy(34)])
        for nm, key, col, lw, mk in SYS:
            c = CT[g, :, t, TS.index(key)]; v = sedi(c); ok = np.isfinite(v) & (c[:, :2].sum(-1) > 0)
            ax.plot(L[ok], v[ok], color=col, lw=lw, marker=mk, ms=2.2, label=nm, zorder=5 if nm == "Foehn" else 3)
        ax.set_xticks(L); ax.set_xlabel("Lead time (days)", fontsize=6, labelpad=1); clean(ax)
        if t == 0:
            ax.set_ylabel("SEDI, all land", fontsize=6)
            ax.legend(fontsize=5, loc="upper right", frameon=False, handlelength=1.6)
        ax.set_title(name, fontsize=6.2, loc="left", pad=3)
        label(fig, fx(3 + 39 * t), fy(46), "abc"[t])
    ad = fig.add_axes([fx(152), fy(10), fx(29), fy(34)])
    regs = [r for r in TR if r != "Global"]; y = np.arange(len(regs))
    for k, (nm, key, col, lw, mk) in enumerate(SYS):
        v = [float(sedi(CT[TR.index(r), 0, 2, TS.index(key)])) for r in regs]
        ad.barh(y + (k - 1) * 0.26, v, height=0.24, color=col, label=nm, zorder=3)
    ad.set_yticks(y); ad.set_yticklabels(regs, fontsize=5.5); ad.invert_yaxis(); ad.set_xlim(0, 1)
    ad.set_xlabel("SEDI, > P99, 1 day ahead", fontsize=6, labelpad=1); clean(ad)
    label(fig, fx(126), fy(46), "d")
    save(fig, "figED_sedi")
    print({nm: np.round(sedi(CT[g, :, 2, TS.index(key)]), 2) for nm, key, *_ in SYS})


if __name__ == "__main__":
    main()
