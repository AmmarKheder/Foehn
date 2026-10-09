#!/usr/bin/env python3
"""Extended Data figure: the station comparison in detail (compare_fair.py, 2022, identical station-days) and calibration.
a-f, RMSE by lead time for each official network: Foehn (ensemble mean, 1 km), CAMS and Aurora issued at the same time, and
     the climatology of the 1 km estimates (GHAP 2017-2020); the bias of each system one day ahead is written in the panel.
g, all stations: Foehn read at its 1 km pixel and averaged to 0.25 and 0.5 deg first (equal resolution), with CAMS and Aurora.
h, calibration on the validation year 2021 against GHAP (four members, every eighth start day): spread-skill ratio of the
   1 km model, and of the model trained in Aurora's protocol (eight members, May-November 2022, against the CAMS analysis).
    python figED_fair.py"""
import glob
from pathlib import Path

import numpy as np

import final_model as FM
from nature_style import COLOR, DOUBLE, HERE, MM, plt, save
import gc_style as GC

COLOR = dict(COLOR, ours=GC.FOEHN, cams=GC.CAMS, aurora=GC.AURORA)   # the colours of the main figures

RUNS = Path("/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km/runs")
NETS = ["Europe (EEA)", "China (CNEMC)", "USA (EPA)", "Korea (AirKorea)", "Japan (NIES)", "India (CPCB)"]
ORDER = [1, 2, 0, 3, 4, 5]                                                  # China, USA, Europe, Korea, Japan, India
STY = {"Foehn": (COLOR["ours"], "-", "o", 1.4), "CAMS": (COLOR["cams"], "-", "s", 0.9), "Aurora": (COLOR["aurora"], "-", "D", 0.9),
       "Climatology": (GC.PERS, (0, (3, 2)), "", 0.9)}


def clean(ax):
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(labelsize=5.5, length=2)


def spread_skill_2021(ev):
    parts = sorted(glob.glob(str(ev / "part_*.npz")))
    z0 = np.load(parts[0]); C = list(z0["rec_cols"]); M = int(z0["info"][5])
    R = np.concatenate([np.load(p)["recs"] for p in parts])
    out = []
    for h in range(1, 8):
        r = R[R[:, C.index("lead")] == h]
        n, sse, var = r[:, C.index("n")].sum(), r[:, C.index("sse")].sum(), r[:, C.index("var")].sum()
        out.append(np.sqrt((M + 1) / M * var / n) / np.sqrt(sse / n))
    return np.array(out)


def main():
    fz = np.load(HERE / "data" / f"compare_fair_{FM.LABEL}.npz"); SYS = [str(s) for s in fz["systems"]]; S = fz["S"]
    I = {"Foehn": SYS.index(f"{FM.LABEL} (1 km)"), "CAMS": SYS.index("CAMS (run d+1, equal issuance)"),
         "Aurora": SYS.index("Aurora (00 UTC d+1, equal issuance)"), "Climatology": SYS.index("climatology (GHAP 2017-2020)")}
    r = lambda g, h, i: np.sqrt(S[g, h, i, 2] / S[g, h, i, 0])
    b = lambda g, h, i: S[g, h, i, 1] / S[g, h, i, 0]
    W, H = DOUBLE, 122 * MM
    fig = plt.figure(figsize=(W, H))
    fx = lambda mm: mm / (W / MM); fy = lambda mm: mm / (H / MM)
    for k, g in enumerate(ORDER):
        col_, row_ = k % 3, k // 3
        ax = fig.add_axes([fx(12 + 60 * col_), fy(83 - 38 * row_), fx(44), fy(29)])
        for nm, i in I.items():
            c, ls, mk, lw = STY[nm]
            ax.plot(range(1, 5), [r(g, h, i) for h in range(4)], color=c, ls=ls, marker=mk or None, ms=2.6, lw=lw,
                    label=nm, zorder=6 if nm == "Foehn" else 3)
        ax.set_xticks(range(1, 5)); clean(ax)
        ax.set_title(f"$\\bf{{{'abcdef'[k]}}}$  {NETS[g]}", fontsize=6.5, loc="left", pad=2)
        ax.text(0.98, 0.04, "bias day 1: " + ", ".join(f"{b(g, 0, I[n]):+.1f}" for n in ("Foehn", "CAMS", "Aurora")),
                transform=ax.transAxes, ha="right", fontsize=4.8, color="#444444")
        if col_ == 0:
            ax.set_ylabel("RMSE (µg m$^{-3}$)", fontsize=6)
        if k == 0:
            fig.legend(*ax.get_legend_handles_labels(), fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
                       handlelength=2.2, columnspacing=1.6)
    # g: equal resolution
    g = S.shape[0] - 1
    ag = fig.add_axes([fx(12), fy(8), fx(74), fy(28)])
    for nm, key, c, ls in (("Foehn, 1 km", f"{FM.LABEL} (1 km)", COLOR["ours"], "-"), ("Foehn at 0.25°", f"{FM.LABEL} at 0.25 deg", COLOR["ours"], (0, (3, 1.5))),
                           ("Foehn at 0.5°", f"{FM.LABEL} at 0.5 deg", COLOR["ours"], (0, (1, 1.2))), ("CAMS", "CAMS (run d+1, equal issuance)", COLOR["cams"], "-"),
                           ("Aurora", "Aurora (00 UTC d+1, equal issuance)", COLOR["aurora"], "-")):
        ag.plot(range(1, 5), [r(g, h, SYS.index(key)) for h in range(4)], color=c, ls=ls, lw=1.1, marker="o", ms=2.2, label=nm)
    ag.set_xticks(range(1, 5)); ag.set_xlabel("Lead time (days)", fontsize=6, labelpad=1); ag.set_ylabel("RMSE, all stations (µg m$^{-3}$)", fontsize=6)
    ag.legend(fontsize=4.8, loc="lower right", frameon=False, ncol=2, handlelength=2.0); clean(ag)
    ag.set_title("$\\bf{g}$  All stations, equal resolution", fontsize=6.5, loc="left", pad=2)
    # h: calibration
    ah = fig.add_axes([fx(108), fy(8), fx(73), fy(28)])
    ss = spread_skill_2021(FM.SEL21_DIR)
    ah.plot(range(1, 8), ss, color=COLOR["ours"], marker="o", ms=2.6, lw=1.3, label="Foehn, 1 km (2021, against GHAP)")
    if FM.LABEL != "e02":                                                   # before the CRPS training
        ah.plot(range(1, 8), spread_skill_2021(FM.RUNS / FM.MODELS["e02"]["run"] / FM.MODELS["e02"]["sel21"]), color=COLOR["ours"],
                ls=(0, (3, 2)), marker="o", ms=2.0, lw=0.9, mfc="white", label="the same before CRPS training")
    cz = np.load(HERE / "data" / "cams12h_scores_foehn_emis.npz"); gg = list(cz["regions"]).index("Global")
    ah.plot(cz["leads_h"] / 24, cz["spread_skill"][:, gg] if cz["spread_skill"].shape[0] == 10 else cz["spread_skill"][gg], color="#8FB2DD",
            marker="s", ms=2.2, lw=1.0, label="Foehn in Aurora's protocol (2022, against CAMS)")
    ah.axhline(1, color="#999999", lw=0.6, ls=(0, (3, 2))); ah.text(7.0, 1.02, "calibrated", fontsize=5, ha="right", va="bottom", color="#666666")
    ah.set_ylim(0, 1.15); ah.set_xlabel("Lead time (days)", fontsize=6, labelpad=1); ah.set_ylabel("Spread / RMSE", fontsize=6)
    ah.legend(fontsize=4.8, loc="lower left", frameon=False); clean(ah)
    ah.set_title("$\\bf{h}$  Ensemble calibration", fontsize=6.5, loc="left", pad=2)
    save(fig, "figED_fair")
    print("spread-skill 2021:", np.round(ss, 2))


if __name__ == "__main__":
    main()
