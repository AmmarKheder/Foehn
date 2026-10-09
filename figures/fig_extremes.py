#!/usr/bin/env python3
"""The MOSAIC 'scores vs lead time' grid, focused on pollution extremes: RMSE, CRPS and CSI of exceedances of 25 ug/m3
(rows) on all days, on the 10 % most polluted days (largest land fraction above 25 ug/m3 on the target day) and on the
documented transported episodes of 2022 (columns), 0.25 deg, real HRES forecast weather; mean over 3 seeds, shaded
+- 1 std. CRPS of point forecasts = MAE. All models scored on the same (start day, lead) samples."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fig_style import COL2, EXP, LEADS, STYLE, save

SEEDS = ("", "_s43", "_s44")
EPIS = [(69, 79), (83, 89), (187, 201), (217, 225)]
meta = np.load(EXP / "dejavu/data/meta.npz")
land = meta["land"].astype(bool)
i0 = int(np.where(meta["year"] == 2022)[0][0])
pm = np.load(EXP / "dejavu/data/pm.npy", mmap_mode="r")
TRUTH = {d: np.asarray(pm[i0 + d], np.float32) for d in range(365)}
FRAC = {d: float(np.nanmean(np.where(land, TRUTH[d], np.nan) > 25)) for d in range(365)}


def rows_flow(run):
    """-> {(start day, lead): [n, sse, crps, hit, uni]} (dejavu records, 8 members, 8 steps, ensemble-mean CSI)."""
    r = np.load(EXP / "dejavu/runs" / run / "records_2022_hres_rec8.npy")
    return {(int(x[1]), int(x[2])): np.array([x[3], x[4], x[5], x[10], x[11]]) for x in r}


def rows_base(run):
    r = np.load(EXP / "retrain/runs" / run / "records_coarse_2022_hres.npy")
    return {(int(x[0]), int(x[1])): np.array([x[2], x[3], x[4], x[5], x[6]]) for x in r}


def rows_grid(npy):
    arr = np.load(npy, mmap_mode="r")
    out = {}
    for d in range(365):
        for h in LEADS:
            if d + h >= 365 or not np.isfinite(arr[d, h - 1]).all():
                continue
            f, t = np.asarray(arr[d, h - 1], np.float32), TRUTH[d + h]
            m = land & np.isfinite(t) & (t > 0)
            e, g = t[m] > 25, f[m] > 25
            out[d, h] = np.array([m.sum(), ((f[m] - t[m]) ** 2).sum(), np.abs(f[m] - t[m]).sum(), (e & g).sum(), (e | g).sum()])
    return out


MODELS = {"foehn": [rows_flow(f"noise_t31_fpstab{s}") for s in SEEDS], "nofp": [rows_flow(f"noise_t31{s}") for s in SEEDS],
          **{m: [rows_base(f"B_{m}_mettarget31{s}") for s in SEEDS] for m in ("convlstm", "earthformer", "simvp", "topoflow", "climax")},
          "aurora": [rows_grid(EXP / "benchmark/aurora/out/aurora_2022.npy")], "cams": [rows_grid(EXP / "benchmark/cams_forecast_2022.npy")],
          "climatology": [rows_grid(EXP / "benchmark/climatology_2022.npy")]}
common = set.intersection(*[set(r) for rs in MODELS.values() for r in rs])
print("common samples", len(common))


def dayset(name, h):
    keys = sorted(k for k in common if k[1] == h)
    fr = np.array([FRAC[d + h] for d, _ in keys])
    if name == "all":
        return keys
    if name == "top10":
        return [k for k, f in zip(keys, fr) if f >= np.quantile(fr, 0.9)]
    return [k for k in keys if any(a <= k[0] <= b for a, b in EPIS)]


def score(rows, keys, metric):
    s = np.sum([rows[k] for k in keys], 0)
    return {"rmse": np.sqrt(s[1] / s[0]), "crps": s[2] / s[0], "csi": s[3] / max(s[4], 1)}[metric]


ORDER = ["climatology", "cams", "aurora", "climax", "topoflow", "simvp", "earthformer", "convlstm", "nofp", "foehn"]
COLS = [("all", "All days"), ("top10", "10 % most polluted days"), ("episodes", "Transported episodes")]
ROWS = [("rmse", "RMSE (µg m$^{-3}$)"), ("crps", "CRPS (µg m$^{-3}$)"), ("csi", "CSI, > 25 µg m$^{-3}$")]


def soften(hexcol, f=0.22):
    """Mix a colour with white (quieter baselines, same hue as in the other figures)."""
    c = np.array([int(hexcol[k:k + 2], 16) for k in (1, 3, 5)]) / 255.0
    return tuple(c + (1 - c) * f)


# scorecard (GenCast style): change of Foehn's score relative to each system, blue = Foehn better
from gc_style import sc_cmap  # noqa: E402
summary = {}
for dset, _ in COLS:
    for metric, _ in ROWS:
        for m in ORDER:
            vals = np.array([[score(r, dayset(dset, h), metric) for h in LEADS] for r in MODELS[m]])
            summary[m, dset, metric] = vals.mean(0)
NAMES = {"nofp": "Foehn, no transport layer", "convlstm": "ConvLSTM", "earthformer": "Earthformer", "simvp": "SimVP", "topoflow": "TopoFlow",
         "climax": "ClimaX", "aurora": "Aurora", "cams": "CAMS (operational)", "climatology": "Climatology"}
SYSR = ["nofp", "convlstm", "earthformer", "simvp", "topoflow", "climax", "aurora", "cams", "climatology"]
MNAME = {"rmse": "RMSE", "crps": "CRPS", "csi": "CSI > 25"}
cols = [(dset, metric, h) for dset, _ in COLS for metric, _ in ROWS for h in LEADS]
D = np.full((len(SYSR), len(cols)), np.nan)
for i, m in enumerate(SYSR):
    for j, (dset, metric, h) in enumerate(cols):
        f, o = summary["foehn", dset, metric][h - 1], summary[m, dset, metric][h - 1]
        D[i, j] = 100 * (f / o - 1) * (-1 if metric == "csi" else 1)          # negative = Foehn better, for every score
cm, nm = sc_cmap()
fig = plt.figure(figsize=(COL2, 2.45))
ax = fig.add_axes([0.17, 0.12, 0.74, 0.62])
gap = np.concatenate([[0], np.cumsum([0.35 if (j % 12 == 0) else (0.15 if j % 4 == 0 else 0) for j in range(1, len(cols))])])
x0 = np.arange(len(cols)) + gap
for i in range(len(SYSR)):
    for j in range(len(cols)):
        ax.add_patch(plt.Rectangle((x0[j], i), 0.94, 0.9, color=cm(nm(D[i, j])), lw=0))
ax.set_xlim(-0.2, x0[-1] + 1.1); ax.set_ylim(len(SYSR), -0.1)
ax.set_yticks(np.arange(len(SYSR)) + 0.45); ax.set_yticklabels([NAMES[m] for m in SYSR], fontsize=6)
ax.set_xticks(x0 + 0.47); ax.set_xticklabels([str(h) for _, _, h in cols], fontsize=4.6)
ax.tick_params(length=0, pad=1.5)
for s_ in ax.spines.values():
    s_.set_visible(False)
for g, (dset, dname) in enumerate(COLS):
    xa, xb = x0[12 * g], x0[12 * g + 11] + 0.94
    ax.text((xa + xb) / 2, -1.55, dname, ha="center", va="bottom", fontsize=6.3, fontweight="bold")
    for k, (metric, _) in enumerate(ROWS):
        xa2, xb2 = x0[12 * g + 4 * k], x0[12 * g + 4 * k + 3] + 0.94
        ax.text((xa2 + xb2) / 2, -0.35, MNAME[metric], ha="center", va="bottom", fontsize=5.4)
ax.text((x0[0] + x0[-1] + 0.94) / 2, len(SYSR) + 1.0, "Lead time (days)", ha="center", va="top", fontsize=5.6)
cax = fig.add_axes([0.93, 0.12, 0.012, 0.62])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=nm, cmap=cm), cax=cax, extend="both"); cb.outline.set_linewidth(0.4)
cb.ax.tick_params(labelsize=5, length=1.5); cb.set_label("Change of Foehn's score (%), blue: Foehn better", fontsize=5.4)
save(fig, "fig_extremes")
for dset, _ in COLS:
    for metric, _ in ROWS:
        best = min((summary[m, dset, metric][2], m) for m in ORDER if m not in ("foehn", "nofp", "cams", "aurora", "climatology")) if metric != "csi" else \
               max((summary[m, dset, metric][2], m) for m in ORDER if m not in ("foehn", "nofp", "cams", "aurora", "climatology"))
        f = summary["foehn", dset, metric][2]
        print(f"{dset:9s} {metric:5s} T+3: FOEHN {f:.3f} | best baseline {best[1]} {best[0]:.3f} | change {100 * (f / best[0] - 1):+.1f}%")
print("cells where Foehn is better:", int((D < 0).sum()), "of", D.size)
