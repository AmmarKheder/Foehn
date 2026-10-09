#!/usr/bin/env python3
"""Station forecast accuracy and retrospective potential warning value (main Fig. 5).

Run without arguments to update paper/nature/figs. Use --output-dir for a preview.
All panels read the existing final-model evaluation summaries; no inference is run.
The bootstrap resamples seven consecutive evaluated starts, spaced two days apart.
boot_inits and block_bootstrap remain importable by the other paper scripts.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

import final_model as FM
from gc_style import AURORA, CAMS, FOEHN, block_bootstrap
from nature_style import DOUBLE, HERE, MM, OUT, plt

NETWORKS = ["Europe", "China", "United States", "Korea", "Japan", "India", "All stations"]
PERSISTENCE, CLIMATOLOGY = "#4C807A", "#9A9FA6"
BOOTSTRAP_REPLICATES, BLOCK_STARTS, BOOTSTRAP_SEED = 10000, 7, 0


def boot_inits(a, b, B=7, n=10000, seed=0):
    """Relative change in mean per-initialization RMSE; retained for ED imports."""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    nd = len(a)
    rng = np.random.default_rng(seed)
    idx = (rng.integers(0, nd - B + 1, size=(n, -(-nd // B)))[:, :, None]
           + np.arange(B)).reshape(n, -1)[:, :nd]
    r = a[idx].mean(1) / b[idx].mean(1) - 1
    return 100 * (a.mean() / b.mean() - 1), 100 * np.percentile(r, 2.5), 100 * np.percentile(r, 97.5)


def paired_rmse_intervals(pdn, indices):
    """Pointwise absolute RMSE intervals, with shared block draws across systems/leads."""
    a = pdn[indices, -1]
    valid = (np.all(np.isfinite(a), axis=(0, 2, 3))
             & np.all(a[..., 1] > 0, axis=(0, 2)))
    a = a[:, valid]
    if not np.all(a[..., 1] == a[:1, ..., 1]):
        raise ValueError("The systems must share the same station-day counts.")
    nd = a.shape[1]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = (rng.integers(0, nd - BLOCK_STARTS + 1,
                       size=(BOOTSTRAP_REPLICATES, -(-nd // BLOCK_STARTS)))[:, :, None]
           + np.arange(BLOCK_STARTS)).reshape(BOOTSTRAP_REPLICATES, -1)[:, :nd]
    point = np.sqrt(a[..., 0].sum(1) / a[..., 1].sum(1))
    samples = np.empty((len(indices), 4, BOOTSTRAP_REPLICATES))
    for k in range(len(indices)):
        for h in range(4):
            samples[k, h] = np.sqrt(a[k, idx, h, 0].sum(1) / a[k, idx, h, 1].sum(1))
    low, high = np.percentile(samples, [2.5, 97.5], axis=-1)
    return point, low, high, int(nd)


def panel_title(fig, fx, fy, x, y, letter, title):
    fig.text(fx(x), fy(y), letter, fontsize=10, weight="bold", va="bottom")
    fig.text(fx(x + 5.5), fy(y + 0.1), title, fontsize=8.5, weight="medium", va="bottom")


def clean_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#7C8994")
    ax.spines[["left", "bottom"]].set_linewidth(0.5)
    ax.tick_params(labelsize=7.1, length=2, width=0.5, color="#7C8994")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E7EDF1", linewidth=0.45)


def draw_scorecard(ax, matrix, low, high, cmap, norm):
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    for (i, j), value in np.ndenumerate(matrix):
        if not np.isfinite(value):
            continue
        rgba = cmap(norm(value))
        luminance = np.dot(rgba[:3], [0.2126, 0.7152, 0.0722])
        color = "white" if luminance < 0.52 else "#1F2933"
        text = "0" if round(value) == 0 else f"{value:+.0f}"
        ax.text(j, i, text, ha="center", va="center", fontsize=8.4, color=color,
                weight="bold" if i == len(NETWORKS) - 1 else "normal")
        if low[i, j] <= 0 <= high[i, j]:
            ax.scatter(j + 0.32, i - 0.27, marker="o", s=10, facecolors="none",
                       edgecolors=color, linewidths=0.65, zorder=5)
    ax.set_xticks(range(4), ["1 d", "2 d", "3 d", "4 d"], fontsize=7.4)
    ax.set_yticks(range(len(NETWORKS)), NETWORKS, fontsize=7.5)
    ax.tick_params(length=0, pad=4)
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(-0.5, 4), minor=True)
    ax.set_yticks(np.arange(-0.5, len(NETWORKS)), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    ax.axhline(len(NETWORKS) - 1.5, color="#52616C", linewidth=0.7)
    ax.get_yticklabels()[-1].set_weight("bold")


def main(output_dir=OUT):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = HERE / "data" / f"compare_fair_{FM.LABEL}.npz"
    event_path = HERE / "data" / f"extremes_stations_{FM.LABEL}.npz"
    fz, events = np.load(data_path), np.load(event_path)
    systems = [str(s) for s in fz["systems"]]
    pdn, sst = fz["PDN"], fz["SST"]
    foehn_key = f"{FM.LABEL} (1 km)"
    cams_key = "CAMS (run d+1, equal issuance)"
    aurora_key = "Aurora (00 UTC d+1, equal issuance)"
    i_foehn, i_cams = systems.index(foehn_key), systems.index(cams_key)

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 7.6,
                         "xtick.labelsize": 7.1, "ytick.labelsize": 7.1})
    W, H = DOUBLE, 124 * MM
    fig = plt.figure(figsize=(W, H), facecolor="white")
    fx, fy = lambda mm: mm / (W / MM), lambda mm: mm / (H / MM)
    rect = lambda x, y, w, h: [fx(x), fy(y), fx(w), fy(h)]

    # a: the verdict at the stations, the planet from three sides. Every official station with at least 30 common
    # forecast days, coloured by its change in RMSE versus the CAMS forecast at one day ahead (identical station-days);
    # the small curve is the cumulative share of stations below each change, and doubles as the colour scale.
    ccmap = LinearSegmentedColormap.from_list(
        "station_rmse", ["#143D70", "#3479B1", "#8AB7D7", "#D7E6F0", "#FFFFFF",
                         "#F4DDD5", "#DDA493", "#BA6655", "#803C37"], N=256)
    cnorm = TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40)
    cnt = sst[i_foehn, 0, :, 1]
    valid = cnt >= 30
    with np.errstate(all="ignore"):
        chg = 100 * (np.sqrt(sst[i_foehn, 0, :, 0] / np.maximum(cnt, 1))
                     / np.sqrt(sst[i_cams, 0, :, 0] / np.maximum(sst[i_cams, 0, :, 1], 1)) - 1)
    win = float(np.mean(chg[valid] < 0))
    svb = np.argsort(np.abs(chg[valid]))
    vlon, vlat, vchg = fz["lon"][valid][svb], fz["lat"][valid][svb], chg[valid][svb]
    for (clon, clat), rct in [((-95, 22), rect(39, 67, 44, 44)), ((12, 48), rect(84.5, 64, 50, 50)),
                              ((110, 25), rect(136, 67, 44, 44))]:
        axg = fig.add_axes(rct, projection=ccrs.Orthographic(clon, clat))
        axg.set_global()
        axg.set_facecolor("white")
        axg.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#F2F1EE", edgecolor="none", zorder=0)
        axg.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.2, edgecolor="#B4BCC4", zorder=1)
        axg.gridlines(xlocs=range(-180, 180, 30), ylocs=range(-60, 90, 30), linewidth=0.15,
                      color="#DCE1E6", zorder=2)
        cosd = (np.sin(np.deg2rad(vlat)) * np.sin(np.deg2rad(clat))
                + np.cos(np.deg2rad(vlat)) * np.cos(np.deg2rad(clat)) * np.cos(np.deg2rad(vlon - clon)))
        near = cosd > np.cos(np.deg2rad(70.0))              # the view's own continent, not the smeared limb
        axg.scatter(vlon[near], vlat[near], c=vchg[near], cmap=ccmap, norm=cnorm, s=2.2, edgecolors="none",
                    transform=ccrs.PlateCarree(), rasterized=True, zorder=5)
        xl, yl = axg.projection.x_limits, axg.projection.y_limits
        gx = np.linspace(-1, 1, 420)
        rr = np.hypot(gx[None, :], gx[:, None])
        sh = np.zeros((420, 420, 4), np.float32)
        sh[..., :3] = 0.18
        sh[..., 3] = 0.13 * np.clip(rr, 0, 1) ** 6
        sh[rr > 1.0, 3] = 0.0
        axg.imshow(sh, extent=(xl[0], xl[1], yl[0], yl[1]), transform=axg.projection, origin="lower",
                   interpolation="bilinear", zorder=6)
        axg.spines["geo"].set_linewidth(0.5)
        axg.spines["geo"].set_edgecolor("#C9D1D8")
        for rk, al in ((0.505, 0.12), (0.516, 0.07), (0.528, 0.035)):
            axg.add_patch(Circle((0.5, 0.5), rk, transform=axg.transAxes, facecolor="none",
                                 edgecolor="#5B8FD4", lw=1.1, alpha=al, clip_on=False, zorder=0))
    so = np.argsort(chg[valid])
    xs, ysf = chg[valid][so], 100.0 * (np.arange(int(valid.sum())) + 1) / int(valid.sum())
    X0, X1 = -60.0, 40.0
    axc = fig.add_axes(rect(7, 68, 25, 18))
    im = axc.imshow(np.linspace(X0, X1, 1024)[None, :], extent=(X0, X1, 0, 100), origin="lower", aspect="auto",
                    cmap=ccmap, norm=cnorm, zorder=1, alpha=0.9, interpolation="bilinear")
    xc = np.clip(xs, X0, X1)
    verts = np.column_stack([np.concatenate([[X0], xc, [X1, X1, X0]]),
                             np.concatenate([[0.0], ysf, [100.0, 0.0, 0.0]])])
    clipper = Polygon(verts, closed=True, facecolor="none", edgecolor="none", transform=axc.transData)
    axc.add_patch(clipper)
    im.set_clip_path(clipper)
    axc.plot(np.concatenate([[X0], xc]), np.concatenate([[0.0], ysf]), color="#17457F", lw=1.1, zorder=3,
             solid_capstyle="round")
    axc.axvline(0, color="#999999", lw=0.4, zorder=2)
    axc.plot([X0, 0], [100 * win, 100 * win], color="#777777", lw=0.45, ls=(0, (3, 2)), zorder=2)
    axc.plot([0], [100 * win], marker="o", ms=2.8, color="#1F5AA6", mec="white", mew=0.45, zorder=4)
    axc.set_xlim(X0, X1)
    axc.set_ylim(0, 100)
    axc.set_xticks([-40, 0, 40])
    axc.set_yticks([0, 50, 100])
    axc.tick_params(labelsize=6.0, length=1.5, width=0.4, pad=1)
    axc.set_xlabel("RMSE change versus CAMS (%)", fontsize=6.4, labelpad=1.5)
    axc.spines[["top", "right"]].set_visible(False)
    axc.spines[["left", "bottom"]].set_linewidth(0.4)
    fig.text(fx(7), fy(106), f"{win:.0%}", color="#1F5AA6", fontsize=18, weight="bold")
    fig.text(fx(7), fy(102.5), f"of the {int(valid.sum()):,} official stations\nwith at least 30 common\nforecast days have lower\nerror than CAMS",
             fontsize=6.6, color="#384B5A", va="top", linespacing=1.4)
    panel_title(fig, fx, fy, 2, 116, "a", "The verdict at the stations")
    print(f"verdict: {win:.0%} of {int(valid.sum())} stations better than CAMS")

    # b: same station-days, with paired temporal bootstrap uncertainty.
    rmse_styles = [
        ("Foehn", foehn_key, FOEHN, "-", 1.7),
        ("CAMS", cams_key, CAMS, "-", 1.0),
        ("Aurora", aurora_key, AURORA, "-", 1.0),
        ("Climatology", "climatology (GHAP 2017-2020)", CLIMATOLOGY, (0, (1.4, 1.7)), 1.0),
    ]
    point, low, high, n_starts = paired_rmse_intervals(
        pdn, [systems.index(style[1]) for style in rmse_styles])
    ax_rmse = fig.add_axes(rect(16, 14, 46, 40))
    lead = np.arange(1, 5)
    offsets = np.linspace(-0.11, 0.11, len(rmse_styles))
    for k, (name, key, color, ls, lw) in enumerate(rmse_styles):
        ax_rmse.errorbar(lead + offsets[k], point[k],
                        yerr=np.stack([point[k] - low[k], high[k] - point[k]]),
                        fmt="none", ecolor=color, alpha=0.9, elinewidth=0.9,
                        capsize=1.4, capthick=0.8, zorder=1)
        ax_rmse.plot(lead, point[k], color=color, linestyle=ls, linewidth=lw,
                     marker="o" if name != "Climatology" else None, markersize=2.7,
                     label=name, zorder=5 if name == "Foehn" else 3)
    ax_rmse.set(xlim=(0.7, 4.3), ylim=(9, 20.3), xticks=lead, yticks=[10, 12, 14, 16, 18, 20])
    ax_rmse.set_xlabel("Lead time (days)", labelpad=2)
    ax_rmse.set_ylabel("RMSE (µg m$^{-3}$)", labelpad=2)
    clean_axes(ax_rmse)
    ax_rmse.legend(loc="upper left", bbox_to_anchor=(0.02, 1.02), ncol=1,
                   fontsize=6.6, handlelength=1.6, columnspacing=0.7, labelspacing=0.3)
    panel_title(fig, fx, fy, 2, 58, "b", "Pooled station error")

    # c: the CRPS test, one day to one week (figdata_station_crps.py; identical station-days at each lead)
    sc = np.load(HERE / "data" / f"station_crps_{FM.LABEL}.npz")
    sc_sys = [str(x) for x in sc["systems"]]; sc_mean = sc["mean"]; sc_leads = sc["leads"]
    ax_crps = fig.add_axes(rect(92, 14, 88, 40))
    crps_styles = [("Climatology", "climatology (GHAP 2017-2020)", CLIMATOLOGY, (0, (1.4, 1.7)), 1.0),
                   ("CAMS", "CAMS (run d+1, equal issuance)", CAMS, "-", 1.0),
                   ("Aurora", "Aurora (00 UTC d+1, equal issuance)", AURORA, "-", 1.0),
                   ("Foehn, 4 members", "Foehn (fair CRPS, 4 members)", FOEHN, "-", 1.7)]
    for name, key, color, ls, lw in crps_styles:
        v = sc_mean[:, sc_sys.index(key)]; okl = np.isfinite(v)
        ax_crps.plot(sc_leads[okl], v[okl], color=color, linestyle=ls, linewidth=lw,
                     marker="o" if name.startswith("Foehn") else None, markersize=2.7,
                     label=name, zorder=5 if name.startswith("Foehn") else 3)
    ax_crps.set(xlim=(0.7, 7.3), ylim=(0, 10.4), xticks=sc_leads, yticks=[0, 2, 4, 6, 8, 10])
    ax_crps.set_xlabel("Lead time (days)", labelpad=2)
    ax_crps.set_ylabel("CRPS (µg m$^{-3}$)", labelpad=2)
    clean_axes(ax_crps)
    handles, labels = ax_crps.get_legend_handles_labels()
    ax_crps.legend(handles[::-1], labels[::-1], loc="lower right", bbox_to_anchor=(0.985, 0.05), fontsize=6.9,
                   handlelength=1.7, labelspacing=0.35, frameon=False)
    panel_title(fig, fx, fy, 80, 58, "c", "The CRPS test, to one week")
    crps_test = {str(sc_sys[j]): sc_mean[:, j].tolist() for j in range(len(sc_sys))}

    summary = {"model": FM.LABEL, "run": FM.M["run"], "figure_size_mm": [183, 124],
               "bootstrap": {"resamples": BOOTSTRAP_REPLICATES, "block_starts": BLOCK_STARTS,
                             "start_spacing_days": 2, "valid_starts": n_starts, "seed": BOOTSTRAP_SEED},
               "pooled_rmse": [{"system": style[0], "rmse": point[k].tolist(),
                                "ci95_low": low[k].tolist(), "ci95_high": high[k].tolist()}
                               for k, style in enumerate(rmse_styles)],
               "crps_test": crps_test,
               "station_verdict": {"stations": int(valid.sum()), "share_better_than_cams": win,
                                   "lead_days": 1, "min_common_days": 30},
               "input_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (data_path, event_path)}}
    name = "fig_gc3_scorecards"
    for ext in ("pdf", "png"):
        path = output_dir / f"{name}.{ext}"
        temporary = path.with_name(path.stem + ".tmp." + ext)
        fig.savefig(temporary, format=ext, dpi=450 if ext == "png" else 600,
                    bbox_inches="tight", pad_inches=0.02, facecolor="white")
        temporary.replace(path)
    (output_dir / f"{name}_data.json").write_text(json.dumps(summary, indent=2) + "\n")
    plt.close(fig)
    print(f"Saved {output_dir / name}; model {FM.LABEL}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    main(parser.parse_args().output_dir)
