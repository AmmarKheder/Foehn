#!/usr/bin/env python3
"""Equal-resolution study (appendix). Every model trained natively at 1 km on the regional window (49.5-41.8 N,
0-15.4 E; same inputs, HRES weather, 2022), plus FOEHN trained at 0.25 deg over Europe and interpolated as a reference.
(a-e) 1 km PM2.5 one day ahead over the Po valley (the part of the window with most stations) on the most polluted of
the saved days, with the EEA measurements of that day (circles) and the RMSE at those stations; (f-h) cover all of 2022;
(f) RMSE at the EEA stations of the window (never used in training) versus lead time; (g) skill of the sub-0.25 deg
detail, 1 - |a_pred - a_obs|^2 / |a_obs|^2 with a the anomaly from the 0.25 deg block mean (0 = no detail, < 0 = wrong
detail); (h) per-station RMSE over all days and leads, our 1 km network (regression) against Earthformer trained at 1 km."""
import glob
import json

import matplotlib.pyplot as plt
import numpy as np

from fig_style import COL2, EXP, STYLE, save

REG = EXP / "regional/runs"
MODELS = {  # key -> (label, eval dir, colour, line style, marker)
    "earthformer": ("Earthformer, 1 km", REG / "earthformer/eval_region", STYLE["earthformer"]["color"], "-", "^"),
    "simvp": ("SimVP, 1 km", REG / "simvp/eval_region", STYLE["simvp"]["color"], "-", "v"),
    "flowreg": ("Ours, regression, 1 km", REG / "flowreg/eval_region", "#e8998d", "--", "o"),
    "flow": ("Ours, flow matching, 1 km", REG / "flow/eval_region", STYLE["foehn"]["color"], "-", "o"),
    "foehn": ("FOEHN, 0.25$^\\circ$ interpolated", EXP / "dejavu/runs/noise_t31_fpstab/eval_region", "#111111", ":", ""),
}
LEADS = (1, 2, 3, 4)

# ------------------------------------------------------------------ maps: Po valley and Alps, 1 km
FINE = EXP / "fullimage/data/ghap1km.npy"
with open(FINE, "rb") as f:
    np.lib.format.read_magic(f); np.lib.format.read_array_header_1_0(f); HDR = f.tell()
OFF22 = 365 * 4 + 1 + 365
R0, R1, C0, C1 = 2530, 2830, 3230, 3780                    # 46.7-43.7 N, 7.3-12.8 E (Po valley, most stations)
land = np.load(EXP / "fullimage/data/static1km.npz")["land"][R0:R1, C0:C1] > 0


def truth(d):
    a = np.memmap(FINE, dtype=np.float16, mode="r", offset=HDR + (OFF22 + d) * 4192 * 6992 * 2, shape=(4192, 6992))
    return np.asarray(a[R0:R1, C0:C1], np.float32)


def pred(key, d, h=1):
    for p in sorted(glob.glob(str(MODELS[key][1] / "part_*.npz"))):
        z = np.load(p)
        if f"pred_d{d}_h{h}" in z.files:
            return z[f"pred_d{d}_h{h}"][R0:R1, C0:C1].astype(np.float32)


D = max((40, 100, 200, 300), key=lambda d: np.nanmean(np.where(land, truth(d + 1), np.nan)))   # most polluted saved day
# EEA stations of the crop with their measurement on the target day (from the evaluation records of the window)
import zarr  # noqa: E402
xy = np.asarray(zarr.open("/scratch/project_462001140/ammar/eccv/data/zarr/eea_stations/eea_pm25_daily.zarr", mode="r")["station_coords"])
_rec = np.concatenate([np.load(p)["stations"] for p in sorted(glob.glob(str(MODELS["earthformer"][1] / "part_*.npz")))])
_rec = _rec[(_rec[:, 1] == D) & (_rec[:, 2] == 1)]
_lat, _lon = xy[_rec[:, 0].astype(int), 0], xy[_rec[:, 0].astype(int), 1]
_in = (_lat < 72 - 0.01 * R0) & (_lat > 72 - 0.01 * R1) & (_lon > -25 + 0.01 * C0) & (_lon < -25 + 0.01 * C1)
ST_LAT, ST_LON, ST_OBS, ST_IDX = _lat[_in], _lon[_in], _rec[_in, 6], _rec[_in, 0].astype(int)


def station_err(key):
    """RMSE at the stations of the crop on the target day (key None: the satellite product itself)."""
    r = np.concatenate([np.load(p)["stations"] for p in sorted(glob.glob(str(MODELS[key or "earthformer"][1] / "part_*.npz")))])
    r = r[(r[:, 1] == D) & (r[:, 2] == 1) & np.isin(r[:, 0].astype(int), ST_IDX)]
    return float(np.sqrt(np.mean((r[:, 5 if key is None else 3] - r[:, 6]) ** 2)))
obs = np.where(land, truth(D + 1), np.nan)
vmax = float(np.nanpercentile(obs, 99))
ext = [-25 + 0.01 * C0, -25 + 0.01 * C1, 72 - 0.01 * R1, 72 - 0.01 * R0]

fig = plt.figure(figsize=(COL2, 3.9))
top = fig.add_gridspec(1, 5, left=0.06, right=0.99, top=0.965, bottom=0.71, wspace=0.06)
maps = [("(a) Satellite product", obs, None)] + [(f"({'bcde'[k]}) {MODELS[m][0].replace(', 1 km', '')}",
                                                    np.where(land, pred(m, D), np.nan), m)
                                                   for k, m in enumerate(("earthformer", "simvp", "flowreg", "flow"))]
for k, (title, a, key) in enumerate(maps):
    ax = fig.add_subplot(top[k])
    ax.set_facecolor("#dfe7f1")
    im = ax.imshow(a, extent=ext, cmap="RdYlBu_r", vmin=0, vmax=vmax, interpolation="none")
    ax.scatter(ST_LON, ST_LAT, c=ST_OBS, cmap="RdYlBu_r", vmin=0, vmax=vmax, s=7, edgecolor="#111111", lw=0.35, zorder=4)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_title(title.replace("Ours, ", "Ours, "), fontsize=6.6, pad=2); ax.grid(False)
    ax.text(0.03, 0.04, f"stations: RMSE {station_err(key):.1f}", transform=ax.transAxes, fontsize=5.6, zorder=6,
            fontweight="bold" if key == "flowreg" else "normal", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
    ax.set_xticks([8, 10, 12]); ax.set_xticklabels(["8°E", "10°E", "12°E"])
    ax.set_yticks([44, 45, 46]); ax.set_yticklabels(["44°N", "45°N", "46°N"] if k == 0 else [])
    if False:
        for name, (la, lo) in {"Milan": (45.46, 9.19), "Bologna": (44.49, 11.34)}.items():
            ax.plot(lo, la, "o", ms=2.6, mfc="white", mec="#111111", mew=0.6)
            ax.annotate(name, (lo, la), xytext=(0, 5), textcoords="offset points", fontsize=5.4, ha="center",
                        path_effects=[__import__("matplotlib.patheffects", fromlist=["x"]).withStroke(linewidth=1.3, foreground="white")])
    ax.tick_params(labelsize=5.6, length=2)
date = (np.datetime64("2022-01-01") + np.timedelta64(D + 1, "D")).astype(str)
cax = fig.add_axes([0.3, 0.665, 0.4, 0.016])
cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="max", ticks=np.arange(0, vmax + 1e-6, 5))
cb.set_label(f"PM$_{{2.5}}$ at 1 km (µg m$^{{-3}}$), {date}, 1 day ahead; circles: EEA stations (measured)", fontsize=6.2); cb.ax.tick_params(labelsize=5.6)

# ------------------------------------------------------------------ scores
bot = fig.add_gridspec(1, 3, left=0.075, right=0.99, top=0.5, bottom=0.17, wspace=0.34)
metrics = {m: json.loads((v[1] / "metrics.json").read_text()) for m, v in MODELS.items()}
axf, axg = fig.add_subplot(bot[0]), fig.add_subplot(bot[1])
for m, (lab, _, col, ls, mk) in MODELS.items():
    st = [metrics[m][f"T+{h}"]["stations"]["model_rmse"] for h in LEADS]
    ds = [metrics[m][f"T+{h}"].get("scale_aware_region", {}).get("fine_detail_skill", np.nan) for h in LEADS]
    kw = dict(color=col, ls=ls, marker=mk or None, ms=3.2, lw=1.8 if m == "flow" else 1.2, label=lab)
    axf.plot(LEADS, st, **kw)
    if m != "foehn":
        axg.plot(LEADS, ds, **kw)
axf.set_title("(f) Error at stations", fontsize=7.2); axf.set_ylabel("RMSE at stations (µg m$^{-3}$)", fontsize=6.6)
axg.set_title("(g) Sub-grid detail", fontsize=7.2); axg.set_ylabel("Detail skill", fontsize=6.6)
axg.axhline(0, color="#555555", ls="--", lw=0.7)
axg.text(1.0, 0.012, "no detail", fontsize=5.6, color="#555555", va="bottom")
for ax in (axf, axg):
    ax.set_xticks(LEADS); ax.set_xlabel("Lead time (days)", fontsize=6.6); ax.tick_params(labelsize=5.8)


def station_rmse(key):
    rows = np.concatenate([np.load(p)["stations"] for p in sorted(glob.glob(str(MODELS[key][1] / "part_*.npz")))])
    out = {}
    for idx, d, h, p, _, _, o in rows:
        out.setdefault(int(idx), {})[(int(d), int(h))] = (p, o)
    return out


a, b = station_rmse("flowreg"), station_rmse("earthformer")
xs, ys = [], []
for s in set(a) & set(b):
    keys = set(a[s]) & set(b[s])
    if len(keys) < 100:
        continue
    xs.append(np.sqrt(np.mean([(b[s][k][0] - b[s][k][1]) ** 2 for k in keys])))
    ys.append(np.sqrt(np.mean([(a[s][k][0] - a[s][k][1]) ** 2 for k in keys])))
xs, ys = np.array(xs), np.array(ys)
axh = fig.add_subplot(bot[2])
lim = float(np.percentile(np.r_[xs, ys], 99)) * 1.05
axh.scatter(xs, ys, s=4, color=MODELS["flowreg"][2], edgecolor="#8c3b33", lw=0.2, alpha=0.8, zorder=3)
axh.plot([0, lim], [0, lim], color="#555555", ls="--", lw=0.7)
axh.set_xlim(0, lim); axh.set_ylim(0, lim); axh.set_aspect("equal")
axh.text(0.04, 0.95, f"ours better at {100 * np.mean(ys < xs):.0f}% of\n{len(xs)} stations", transform=axh.transAxes,
         fontsize=6.0, va="top")
axh.set_xlabel("Earthformer, 1 km", fontsize=6.6); axh.set_ylabel("Ours, 1 km", fontsize=6.6)
axh.set_title("(h) RMSE per station", fontsize=7.2); axh.tick_params(labelsize=5.8)
h_, l_ = axf.get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=5, frameon=False, fontsize=6.2, bbox_to_anchor=(0.5, -0.005), handlelength=2.2,
           columnspacing=1.2)
save(fig, "fig_regional")
print("stations on the maps", len(ST_OBS), "| day", D, date, "| stations", len(xs), "ours better at", round(100 * np.mean(ys < xs), 1), "% | median ratio",
      round(float(np.median(ys / xs)), 3))
