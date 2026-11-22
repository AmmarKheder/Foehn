#!/usr/bin/env python3
"""NEW figure (Supplementary): detection of local extremes under every definition of the threshold, at equal issuance.
SEDI of the days above the 99th percentile (default; argument P95 or P90 for the other tiers) by lead time, Foehn's ensemble
mean, the CAMS forecast and Aurora issued at the same time (CAMS run of 00 UTC d+1, Aurora initialised at 00 UTC d+1).
Rows: the truth. Top, the 1 km estimates over all land (0.25 deg block means, 251,310 cells, 178 start days); bottom, the
measurements at the stations of the six official networks (about one million station-days at one day).
Columns: the definition of the event and of the threshold.
  1  local climate, same threshold: the local percentile of the estimates of 2017-2020 for the cell (or station pixel) and
     calendar day, the same for the truth and every system (the paper's definition; top = Extended Data Fig. 4g);
  2  local climate, matched frequency: the same events; each system's threshold is rescaled by one factor at each cell or
     station so that it warns as often as the event occurs there;
  3  year 2022, same threshold: the percentile of the truth over the scored days of 2022 at each cell or station (the
     estimates over land, each station's own measurements at the stations), the same for every system;
  4  year 2022, matched frequency: the truth and each system have their own percentile at each cell or station.
Bands: 95 % intervals, moving-block bootstrap over start days (7-day blocks, 10,000 resamples), thresholds kept fixed.
Dotted line: the CAMS forecast 1 day ahead. Numbers and intervals are read from ../data/derived.json (derive.py).
    python -B fig_sedi_definitions.py [P99|P95|P90]   -> ../figs/fig_sedi_definitions_<tier>.{pdf,png}"""
import json
import os
import sys
from pathlib import Path

import numpy as np

FIG = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/figures"))
sys.path.append(str(FIG))
HERE = Path(__file__).resolve().parent.parent
SFX = f"_{os.environ['FOEHN_DESIGN']}" if os.environ.get("FOEHN_DESIGN") else ""   # A16 rebuild: design label (unset = unchanged)
FIGD = HERE.parents[1] / "rebuild" / f"figs{SFX}" if SFX else HERE / "figs"   # renders of a design: natcomms/rebuild/figs_<D>
TIER = sys.argv[1] if len(sys.argv) > 1 else "P99"

from gc_style import AURORA, CAMS, FOEHN, clean, letter_before  # noqa: E402
from nature_style import DOUBLE, MM, plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

R = json.loads((HERE / f"data{SFX}" / "derived.json").read_text())
COLS = ["Local climate, same threshold", "Local climate, matched frequency", "Year 2022, same threshold", "Year 2022, matched frequency"]
ROWS = [("land", "Global", ["ghap", "matched", "truth2022", "own"], "SEDI against the 1 km\nestimates, all land"),
        ("stations", "all", ["ghap", "matched", "station", "own"], "SEDI against the measure-\nments at the stations")]
SYS = [("Foehn", "Foehn", FOEHN, 1.65, "o", 2.4, 5), ("CAMS forecast", "CAMS", CAMS, 1.0, "s", 1.9, 3), ("Aurora", "Aurora", AURORA, 1.0, "D", 1.8, 3)]
W, H = DOUBLE, 86 * MM
fig = plt.figure(figsize=(W, H))
fx = lambda mm: mm / (W / MM); fy = lambda mm: mm / (H / MM)
X0, PW, GAP, PH = 13.0, 37.5, 5.5, 25.0
Y0 = [51.0, 13.0]
vals = []
for dom, reg, kinds, _ in ROWS:
    for kind in kinds:
        d = R[dom][kind][TIER][reg]
        for _, key, *_ in SYS:
            vals += [v for v in d[key]["ci_low"] + d[key]["ci_high"] if v is not None]
ylo, yhi = np.floor(min(vals) * 10) / 10, np.ceil(max(vals) * 10) / 10
axs = []
for r, (dom, reg, kinds, ylab) in enumerate(ROWS):
    for c, kind in enumerate(kinds):
        ax = fig.add_axes([fx(X0 + c * (PW + GAP)), fy(Y0[r]), fx(PW), fy(PH)])
        d = R[dom][kind][TIER][reg]
        ax.axhline(d["CAMS"]["sedi"][0], color=CAMS, lw=0.5, ls=(0, (1, 1.6)), zorder=2)
        for nm, key, col, lw, mk, ms, z in SYS:
            v = np.array([np.nan if x is None else x for x in d[key]["sedi"]], float); n = int(np.isfinite(v).sum()); L = np.arange(1, n + 1)
            lo, hi = np.array(d[key]["ci_low"][:n], float), np.array(d[key]["ci_high"][:n], float)
            ax.fill_between(L, lo, hi, color=col, alpha=0.16, lw=0, zorder=z - 1)
            ax.plot(L, v[:n], color=col, lw=lw, marker=mk, ms=ms, zorder=z)
            print(f"{dom:8s} {kind:9s} {TIER} {nm:13s}", " ".join(f"{x:.3f}" for x in v[:n]))
        ax.set_xticks(np.arange(1, 8)); ax.set_xlim(0.6, 7.4); ax.set_ylim(ylo, yhi)
        ax.set_yticks(np.arange(ylo, yhi + 1e-9, 0.1))
        clean(ax); ax.tick_params(labelsize=5.5)
        if r == 1:
            ax.set_xlabel("Lead time (days)", fontsize=6, labelpad=1)
        if c == 0:
            ax.set_ylabel(ylab, fontsize=6, linespacing=1.15)
        else:
            ax.set_yticklabels([])
        ax.set_title(COLS[c], fontsize=6.0, loc="left", pad=5)
        axs.append(ax)
handles = [Line2D([0], [0], color=c_, linewidth=1.6 if nm == "Foehn" else 1.1, marker=mk, markersize=ms, label=nm) for nm, _, c_, _, mk, ms, _ in SYS]
handles.append(Line2D([0], [0], color=CAMS, linewidth=0.6, linestyle=(0, (1, 1.6)), label="CAMS forecast, 1 day ahead"))
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, fy(0.6)), ncol=4, fontsize=5.5, columnspacing=1.2, handlelength=2.0)
fig.canvas.draw()
for ax_, s_ in zip(axs, "abcdefgh"):
    letter_before(fig, ax_._left_title, s_)
for ext in ("pdf", "png"):
    fig.savefig(FIGD / f"fig_sedi_definitions_{TIER}.{ext}")
print("saved", FIGD / f"fig_sedi_definitions_{TIER}")
