#!/usr/bin/env python3
"""Extended Data figure: Foehn trained in Aurora's protocol, by region (global1km/eval_cams12h.py, data/cams12h_scores_foehn_emis.npz).
Latitude-weighted RMSE against the CAMS analysis of the forecasts initialised at 00 UTC from 1 May to 30 November 2022
(214 initialisations), by lead time, for Foehn (mean of eight members; 9.9 million parameters), Aurora (1.3 billion), the
operational CAMS forecast and persistence of the analysis, over the whole domain and nine regions.
    python figED_aurora.py"""
import numpy as np

from nature_style import COLOR, DOUBLE, HERE, MM, plt, save
import gc_style as GC

COLOR = dict(COLOR, ours=GC.FOEHN, cams=GC.CAMS, aurora=GC.AURORA)   # the colours of the main figures

STY = {"Foehn-CAMS (mean)": (COLOR["ours"], "-", "o", 1.3, "Foehn"), "Aurora": (COLOR["aurora"], "-", "D", 0.9, "Aurora"),
       "CAMS forecast": (COLOR["cams"], "-", "s", 0.9, "CAMS forecast"), "persistence": (GC.PERS, (0, (3, 2)), "", 0.9, "persistence")}


def main():
    z = np.load(HERE / "data" / "cams12h_scores_foehn_emis.npz")
    S = [str(s) for s in z["systems"]]; R = [str(r) for r in z["regions"]]; L = z["leads_h"]
    W, H = DOUBLE, 92 * MM
    fig = plt.figure(figsize=(W, H))
    fx = lambda mm: mm / (W / MM); fy = lambda mm: mm / (H / MM)
    for k, r in enumerate(R):
        c_, r_ = k % 5, k // 5
        ax = fig.add_axes([fx(11 + 35 * c_), fy(52 - 44 * r_), fx(29), fy(32)])
        g = R.index(r)
        for key, (col, ls, mk, lw, lab) in STY.items():
            ax.plot(L, z["rmse"][S.index(key), :, g], color=col, ls=ls, marker=mk or None, ms=1.8, lw=lw, label=lab,
                    zorder=6 if key.startswith("Foehn") else 3)
        f, a = z["rmse"][S.index("Foehn-CAMS (mean)"), :, g], z["rmse"][S.index("Aurora"), :, g]
        ax.text(0.97, 0.05, f"vs Aurora at 72 h: {100 * (f[5] / a[5] - 1):+.0f}%", transform=ax.transAxes, ha="right", fontsize=4.8,
                color="#333333")
        ax.set_xticks([24, 72, 120]); ax.tick_params(labelsize=5, length=1.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.set_title(f"$\\bf{{{'abcdefghij'[k]}}}$  {r}", fontsize=6, loc="left", pad=2)
        if c_ == 0:
            ax.set_ylabel("RMSE (µg m$^{-3}$)", fontsize=5.5)
        if r_ == 1:
            ax.set_xlabel("Lead time (h)", fontsize=5.5, labelpad=1)
        if k == 0:
            ax.legend(fontsize=4.6, loc="upper left", frameon=False, handlelength=1.6)
    save(fig, "figED_aurora")
    g = R.index("Global")
    print("global Foehn/Aurora % by lead:", np.round(100 * (z["rmse"][0, :, g] / z["rmse"][S.index("Aurora"), :, g] - 1), 1))


if __name__ == "__main__":
    main()
