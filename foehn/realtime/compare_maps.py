#!/usr/bin/env python3
"""RT step 4b: the ensemble mean of a real day at +24 h and +120 h next to the CAMS operational forecast of the same run
(data/<date>/cams_fc.npz, fetch_camsfc.py), to spot a gross bias of the real-time inputs. Writes out/<date>/check_maps.png
and prints the global means, the RMSE between the two and the correlation.
    python compare_maps.py 2026-10-10"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

import common as C

PM_COLORS = ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#ffffbf", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"]
PM_BOUNDS = [0, 5, 10, 15, 25, 37.5, 50, 75, 100, 150, 250]


def main():
    d = C.parse_date(sys.argv[1]); D, O = C.day_dir(d), C.out_dir(d)
    P = np.load(O / "pred.npy").astype(np.float32).mean(0)                              # (10, 361, 900)
    cf = np.load(D / "cams_fc.npz"); lds = list(cf["leads"]); cams = cf["pm25"][[lds.index(24), lds.index(120)]].astype(np.float32)   # lat 90..-90, lon 0..
    cams = np.roll(cams[:, 15:376], 450, axis=-1)                                        # the model's rows and longitudes
    cmap = ListedColormap(PM_COLORS[:-1]); cmap.set_over(PM_COLORS[-1]); norm = BoundaryNorm(PM_BOUNDS, cmap.N)
    lat = 84.0 - 0.4 * np.arange(361); w = np.cos(np.deg2rad(lat))[:, None]
    fig, axs = plt.subplots(2, 2, figsize=(14, 7.6))
    rows = [("+24 h", P[1], cams[0]), ("+120 h", P[9], cams[1])]
    for i, (lab, a, b) in enumerate(rows):
        for j, (name, x) in enumerate((("RT (ensemble mean)", a), ("CAMS operational forecast", b))):
            ax = axs[i, j]; ax.imshow(x, cmap=cmap, norm=norm, extent=(-180, 180, -60, 84), interpolation="nearest", aspect="auto")
            ax.set_title(f"{name}, {lab}, {d}: mean {float((w * x).sum() / (w * np.ones_like(x)).sum()):.2f} µg m⁻³", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        e = a - b
        rm = float(np.sqrt((w * e ** 2).sum() / (w * np.ones_like(e)).sum())); r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        bias = float((w * e).sum() / (w * np.ones_like(e)).sum())
        print(f"{lab}: RT mean {float((w * a).sum() / (w * np.ones_like(a)).sum()):.2f}, CAMS mean {float((w * b).sum() / (w * np.ones_like(b)).sum()):.2f}, "
              f"bias RT-CAMS {bias:+.2f}, RMSE between them {rm:.2f}, correlation {r:.3f}")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, orientation="horizontal", fraction=0.03, pad=0.03, extend="max")
    cb.set_label("PM$_{2.5}$ (µg m$^{-3}$)")
    fig.savefig(O / "check_maps.png", dpi=110, bbox_inches="tight"); print("written", O / "check_maps.png")


if __name__ == "__main__":
    main()
