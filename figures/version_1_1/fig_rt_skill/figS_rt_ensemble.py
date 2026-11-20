#!/usr/bin/env python3
"""Supplementary figure: the eight members of the real-time 1 km forecast at the monitors, as issued and with the calibrated
noise (the panel f of fig_rt_skill.py, drawn alone; same numbers: numbers_cal.json, key f_ensemble, made by
`./render.sh --data --score <scorer directory of the calibrated reforecasts> --arm2 kmcal --name2 "Calibrated noise" --tag cal`).
a, share of station-days by the number of members below the measurement; b, RMSE of the ensemble mean by class of ensemble
spread. Dashed: a calibrated ensemble.
    python -B figS_rt_ensemble.py   -> figS_rt_ensemble.pdf, .png"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/scratch/project_462001804/ammar/eccv/cran-pm/experiments/figures")

import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from gc_style import FOEHN, clean, letter_before  # noqa: E402
from nature_style import MM, plt  # noqa: E402

ARM2, MUTED = "#8FB2DD", "#555555"
W_MM, H_MM = 120.0, 52.0


def main():
    F = json.loads((HERE / "numbers_cal.json").read_text())["f_ensemble"]
    arms = list(F["arms"]); assert len(arms) == 2, arms
    names = {arms[0]: "As issued", arms[1]: "Calibrated noise"}
    cols = {arms[0]: FOEHN, arms[1]: ARM2}
    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
    R = lambda x, y, w, h: [x / W_MM, y / H_MM, w / W_MM, h / H_MM]
    y0, h = 9.5, 35.0
    af = fig.add_axes(R(12.0, y0, 52.0, h))
    nr = len(F[arms[0]]["rank_hist_pct"]); wb = 0.74 / len(arms)
    for j, x in enumerate(arms):
        af.bar(np.arange(nr) + (j - 0.5 * (len(arms) - 1)) * wb, F[x]["rank_hist_pct"], width=wb, color=cols[x], lw=0, zorder=3)
    af.axhline(F["expected"]["rank_share_pct"], color="#555555", lw=0.7, ls=(0, (3, 2)), zorder=4)
    af.text(4.0, F["expected"]["rank_share_pct"] + 1.0, "calibrated", fontsize=5.3, color=MUTED, ha="center", va="bottom")
    af.set_xlim(-0.6, nr - 0.4); af.set_xticks(np.arange(nr)); af.set_ylim(0, 50); af.set_yticks([0, 10, 20, 30, 40, 50])
    clean(af); af.grid(axis="x", visible=False); af.tick_params(labelsize=5.5)
    af.set_xlabel("Members below the measurement", fontsize=5.8, labelpad=1.5)
    af.set_ylabel("Station-days (%)", fontsize=5.8, labelpad=2)
    af.legend([Patch(fc=cols[x], lw=0) for x in arms], [names[x] for x in arms], fontsize=5.3, loc="upper center", bbox_to_anchor=(0.52, 1.0), frameon=False,
              handlelength=1.2, handletextpad=0.5, labelspacing=0.3, borderaxespad=0.2)
    ta = af.set_title("Rank of the measurement among the members", fontsize=6.3, loc="left", pad=2.5)
    ag = fig.add_axes(R(80.0, y0, 35.0, h))
    lm = (0.35, 110.0)
    ag.plot(lm, lm, color="#555555", lw=0.7, ls=(0, (3, 2)), zorder=2)
    ag.text(1.05, 0.82, "calibrated", fontsize=5.3, color=MUTED, ha="left", va="top", rotation=np.degrees(np.arctan(h / 35.0)), rotation_mode="anchor")
    for x in arms:
        sc = F[x]["spread_classes"]
        ag.plot(sc["spread"], sc["rmse"], color=cols[x], lw=1.2, marker="o", ms=2.4, mfc=cols[x], mec=cols[x], zorder=4)
    ag.set_xscale("log"); ag.set_yscale("log"); ag.set_xlim(*lm); ag.set_ylim(*lm)
    for axis in (ag.xaxis, ag.yaxis):
        axis.set_major_locator(plt.FixedLocator([1, 10, 100])); axis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{v:g}"))
        axis.set_minor_locator(plt.FixedLocator([0.5, 2, 5, 20, 50])); axis.set_minor_formatter(plt.NullFormatter())
    clean(ag); ag.tick_params(which="minor", length=1.2, width=0.4); ag.tick_params(labelsize=5.5)
    ag.set_xlabel("Ensemble spread (µg m$^{-3}$)", fontsize=5.8, labelpad=1.5)
    ag.set_ylabel("RMSE (µg m$^{-3}$)", fontsize=5.8, labelpad=1)
    tb = ag.set_title("Spread and error", fontsize=6.3, loc="left", pad=2.5)
    fig.canvas.draw()
    for t, s_ in ((ta, "a"), (tb, "b")):
        letter_before(fig, t, s_)
    for x in arms:
        e = F[x]; print(x, {k: e[k] for k in e if k not in ("rank_hist_pct", "spread_classes")})
    fig.savefig(HERE / "figS_rt_ensemble.pdf"); fig.savefig(HERE / "figS_rt_ensemble.png", dpi=300)
    print("saved", HERE / "figS_rt_ensemble.pdf")


if __name__ == "__main__":
    main()
