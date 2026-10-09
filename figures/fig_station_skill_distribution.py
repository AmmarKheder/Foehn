#!/usr/bin/env python3
"""Reviewable station-skill alternatives to the ranked-variable Aurora plot.

Reads saved matched-observation SSE/count summaries; does not run inference.
Outputs two vector PDFs, PNG previews, a comparison sheet and auditable data.
Stations must have >=30 common days at BOTH 1-day and 3-day lead time.
Every station receives equal weight. Curves are ranked independently.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

import final_model as FM
from gc_style import AURORA, FOEHN
from nature_style import DOUBLE, HERE, MM, plt

DEFAULT_OUT = (HERE.parents[1] / "paper" / "nature" /
               "fig5_redesign_20261009" / "alternatives")
LEADS = (0, 2)
COLORS = {"Foehn": FOEHN, "Aurora": AURORA}
ERROR_TICKS = [-100, -50, 0, 50, 100, 200, 400]


def load_station_changes():
    path = HERE / "data" / f"compare_fair_{FM.LABEL}.npz"
    with np.load(path) as z:
        systems = [str(s) for s in z["systems"]]
        indices = [systems.index(key) for key in (
            f"{FM.LABEL} (1 km)", "Aurora (00 UTC d+1, equal issuance)",
            "CAMS (run d+1, equal issuance)")]
        sst = z["SST"][indices][:, LEADS]
        # Check the paired comparison before computing a common station mask.
        if not np.array_equal(sst[..., 1], np.broadcast_to(sst[:1, ..., 1], sst[..., 1].shape)):
            raise ValueError("Systems do not share identical station-day counts.")
        with np.errstate(divide="ignore", invalid="ignore"):
            rmse = np.sqrt(sst[..., 0] / sst[..., 1])
        valid = (np.all(sst[..., 1] >= 30, axis=(0, 1))
                 & np.all(np.isfinite(rmse), axis=(0, 1))
                 & np.all(rmse[-1] > 0, axis=0))
        changes = 100 * (rmse[:2, :, valid] / rmse[-1:, :, valid] - 1)
        # Avoid ambiguous advanced indexing: model, lead, station.
        assert changes.shape == (2, 2, int(valid.sum()))
        station_indices = np.flatnonzero(valid)
    return path, changes, station_indices


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#7C8994")
    ax.spines[["left", "bottom"]].set_linewidth(0.5)
    ax.tick_params(labelsize=7.3, length=2.5, width=0.5, color="#7C8994")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E7EDF1", linewidth=0.5)


def error_axis(ax, which):
    # All observations, including long adverse tails, remain visible.
    setter = ax.set_xscale if which == "x" else ax.set_yscale
    setter("symlog", base=10, linthresh=50, linscale=1)
    axis = ax.xaxis if which == "x" else ax.yaxis
    axis.set_major_locator(FixedLocator(ERROR_TICKS))
    axis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:+.0f}"))
    axis.set_minor_locator(NullLocator())
    (ax.set_xlim if which == "x" else ax.set_ylim)(-100, 470)


def draw_panel(ax, changes, lead_idx, kind, letter, show_ylabel=True):
    n = changes.shape[-1]
    clean(ax)
    ax.set_title(f"{LEADS[lead_idx] + 1} day{'s' if LEADS[lead_idx] else ''} ahead",
                 loc="left", fontsize=9, weight="normal", pad=10)
    ax.text(-0.13, 1.065, letter, transform=ax.transAxes, fontsize=10,
            weight="bold", va="bottom")
    if kind == "cdf":
        error_axis(ax, "x")
        ax.set(ylim=(0, 101), yticks=[0, 25, 50, 75, 100])
        ax.axvline(0, color="#66727D", linestyle=(0, (3, 2)), linewidth=0.85, zorder=1)
        ax.set_xlabel("Change in station RMSE vs CAMS (%)", fontsize=7.8, labelpad=5)
        if show_ylabel:
            ax.set_ylabel("Cumulative share of stations (%)", fontsize=7.8, labelpad=5)
        ax.text(0, 104, "CAMS", ha="center", va="bottom", fontsize=7, color="#52616C")
    else:
        error_axis(ax, "y")
        ax.set(xlim=(0, 100), xticks=[0, 25, 50, 75, 100])
        ax.axhline(0, color="#66727D", linestyle=(0, (3, 2)), linewidth=0.85, zorder=1)
        ax.set_xlabel("Station percentile (independently ranked)", fontsize=7.8, labelpad=5)
        if show_ylabel:
            ax.set_ylabel("Change in station RMSE vs CAMS (%)", fontsize=7.8, labelpad=5)
        ax.text(99, 4, "CAMS", ha="right", va="bottom", fontsize=7, color="#52616C")
    for model_idx, (name, color) in enumerate(COLORS.items()):
        values = changes[model_idx, lead_idx]
        sorted_values = np.sort(values)
        fraction = float(np.mean(values < 0) * 100)
        if kind == "cdf":
            ax.step(sorted_values, 100 * np.arange(1, n + 1) / n,
                    where="post", color=color, linewidth=1.5, label=name, zorder=3)
            ax.hlines(fraction, -100, 0, color=color, linewidth=0.65,
                      alpha=0.45, linestyle=(0, (2, 2)), zorder=2)
            ax.plot(0, fraction, "o", color=color, markersize=3.8, zorder=5)
            ax.text(-98, fraction + 2.5, f"{fraction:.1f}% of stations\nhave lower RMSE",
                    fontsize=7.1, color=color, va="bottom", zorder=6,
                    linespacing=1.1)
        else:
            ax.plot(100 * (np.arange(n) + 0.5) / n, sorted_values,
                    color=color, linewidth=1.5, label=name, zorder=3)
            ax.plot(fraction, 0, "o", color=color, markersize=3.8, zorder=5)
            ax.annotate(f"{fraction:.1f}%", xy=(fraction, 0),
                        xytext=(-3, -11 if model_idx == 0 else 6),
                        textcoords="offset points", fontsize=7.4, color=color,
                        ha="right", va="top" if model_idx == 0 else "bottom")


def decoration(fig, n, kind, top=0.96, bottom=0.04):
    handles = [Line2D([0], [0], color=color, linewidth=1.8, label=name)
               for name, color in COLORS.items()]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, top),
               ncol=2, fontsize=8, handlelength=2.2, columnspacing=1.5)
    fig.text(0.10, top - 0.025, f"{n:,} matched stations · equal station weighting",
             fontsize=7.7, color="#384B5A")
    axis = "Horizontal" if kind == "cdf" else "Vertical"
    fig.text(0.10, bottom,
             f"{axis} error axis: linear within ±50%; logarithmic outside. All stations are shown.",
             fontsize=7, color="#52616C")


def save(fig, out, stem):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=350, bbox_inches=None, facecolor="white")
    plt.close(fig)


def main(output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path, changes, station_indices = load_station_changes()
    n = changes.shape[-1]
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 7.8,
                         "xtick.labelsize": 7.3, "ytick.labelsize": 7.3})
    for kind, stem in (("cdf", "station_skill_cdf"), ("ranked", "station_skill_ranked")):
        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 85 * MM))
        fig.subplots_adjust(left=0.10, right=0.985, bottom=0.24, top=0.76, wspace=0.24)
        for h, ax in enumerate(axes):
            draw_panel(ax, changes, h, kind, "ab"[h], show_ylabel=(h == 0))
        decoration(fig, n, kind)
        save(fig, out, stem)
    # A single review sheet shows both encodings of the exact same data.
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 169 * MM))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.12, top=0.88,
                        wspace=0.24, hspace=0.85)
    for row, kind in enumerate(("cdf", "ranked")):
        for h in range(2):
            draw_panel(axes[row, h], changes, h, kind, "abcd"[2 * row + h], h == 0)
    decoration(fig, n, "cdf", top=0.985, bottom=0.035)
    fig.text(0.10, 0.015,
             "Bottom row: the same error scaling is vertical; each model is ranked independently.",
             fontsize=7, color="#52616C")
    save(fig, out, "station_skill_comparison")
    q = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    summary = {
        "source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model": FM.LABEL, "metric": "100 * (station_RMSE / CAMS_station_RMSE - 1)",
        "comparison": "equal forecast issuance; identical station-days within each lead",
        "eligibility": ">=30 common days at both 1-day and 3-day leads; positive CAMS RMSE",
        "n_stations": n, "weighting": "equal weight per station (not pooled station-days)",
        "axis": "symlog, linthresh=50, linscale=1, base=10; full observed range",
        "quantile_percentiles": q, "statistics": {},
    }
    for m, name in enumerate(COLORS):
        summary["statistics"][name] = {}
        for h, lead in enumerate(LEADS):
            values = changes[m, h]
            summary["statistics"][name][f"day_{lead + 1}"] = {
                "quantiles_percent": np.percentile(values, q).tolist(),
                "fraction_lower_error": float(np.mean(values < 0)),
                "fraction_more_than_100pct_higher_error": float(np.mean(values > 100)),
            }
    (out / "station_skill_distribution_data.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(out / "station_skill_distribution_values.npz",
                        station_indices=station_indices, leads_days=np.array([1, 3]),
                        models=np.array(list(COLORS)), relative_rmse_change_percent=changes)
    print(json.dumps({"output_dir": str(out), "n_stations": n,
                      "statistics": summary["statistics"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    main(parser.parse_args().output_dir)
