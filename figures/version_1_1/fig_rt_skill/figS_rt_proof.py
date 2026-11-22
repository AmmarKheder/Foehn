#!/usr/bin/env python3
"""The real-time forecasts verified at the monitors, ONE small panel:
RMSE by lead day of the real-time 1 km forecast, of its 0.4 degree input, of the CAMS forecast of the same cycle and of
the climatology of the 1 km estimates, on the identical 1,819,647 station-days at 11,121 monitors (48 issue days).
Numbers: numbers.json (keys b_lead/rmse and legend_numbers), built by rt_skill_data.py from the scorer directory
replay/score_cams3h (the CAMS daily mean is that of its eight 3-hourly steps).
    python -B figS_rt_proof.py   -> figS_rt_proof.pdf, .png"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/scratch/project_462001804/ammar/eccv/cran-pm/experiments/figures")
from gc_style import clean  # noqa: E402
from nature_style import MM, plt  # noqa: E402

BLUE, TEAL, SLATE, GREY = "#2F6FB7", "#3E8E86", "#6B7480", "#A9B1BC"     # the soft colours of the warnings panel of Fig. 2
W_MM, H_MM = 100.0, 66.0


def main():
    n = json.loads((HERE / "numbers.json").read_text())
    r = n["b_lead"]["rmse"]; lead = [1, 2, 3, 4, 5]
    pooled = n["legend_numbers"]["rmse_pooled"]; ch = n["legend_numbers"]["change_pooled"]["cams"]
    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
    ax = fig.add_axes([13.5 / W_MM, 11.5 / H_MM, 55.0 / W_MM, 47.0 / H_MM])
    series = (("cams", "CAMS forecast", SLATE, "-", "s", 1.3, 3.2), ("rt04", "Foehn in real time, 40 km", TEAL, "-", "^", 1.3, 3.4),
              ("clim", "Climatology, 1 km", GREY, (0, (3, 2)), "o", 1.2, 2.8), ("km", "Foehn in real time, 1 km", BLUE, "-", "o", 2.2, 4.2))
    for key, name, col, ls, mk, lw, ms in series:
        ax.plot(lead, r[key], color=col, ls=ls, lw=lw, marker=mk, ms=ms, mfc=col, mec=col, zorder=5 if key == "km" else 3)
        ax.text(5.18, r[key][-1], name, color=col, fontsize=5.8, va="center", ha="left", fontweight="bold" if key == "km" else "normal")
    for x, v in zip(lead, r["km"]):
        ax.text(x, v - 0.32, f"{v:.1f}", color=BLUE, fontsize=5.2, ha="center", va="top", fontweight="bold")
    for x, v in zip(lead, r["cams"]):
        ax.text(x, v + 0.28, f"{v:.1f}", color=SLATE, fontsize=5.2, ha="center", va="bottom")
    ax.set_xlim(0.7, 5.3); ax.set_xticks(lead); ax.set_ylim(14.5, 21.5); ax.set_yticks([15, 17, 19, 21])
    ax.set_xlabel("Lead day", fontsize=6.2, labelpad=1.5); ax.set_ylabel("RMSE at the monitors (µg m$^{-3}$)", fontsize=6.2, labelpad=2)
    clean(ax); ax.tick_params(labelsize=5.6); ax.grid(axis="y", lw=0.3, color="#E3E3E3", zorder=0); ax.set_axisbelow(True)
    ax.set_title("Real-time forecasts at 11,121 monitors", fontsize=6.6, loc="left", pad=4)
    print("pooled RMSE:", {k: round(v, 2) for k, v in pooled.items()}, "| 1 km against CAMS: %.1f %% [%.1f, %.1f]" % (ch["est"], ch["ci"][0], ch["ci"][1]))
    print("by lead day:", {k: [round(x, 2) for x in r[k]] for k in ("km", "rt04", "cams", "clim")})
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"figS_rt_proof.{ext}", **({"dpi": 300} if ext == "png" else {}))
    print("saved", HERE / "figS_rt_proof.pdf")


if __name__ == "__main__":
    main()
