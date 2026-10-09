"""Shared style of the paper figures: AISTATS widths, sans-serif 7-8 pt, thin marks, one fixed colour per model
(validated categorical palette, fixed order), references (climatology, CAMS, Aurora)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os

EXP = Path(__file__).resolve().parents[1]
NATURE = os.environ.get("NATURE", "0") == "1"      # NATURE=1: Nature Extended Data style (nature_style.py), paper/nature/figs
OUT = Path(__file__).resolve().parents[2] / "paper" / ("nature/figs" if NATURE else "aistats2027/figures")
OUT.mkdir(parents=True, exist_ok=True)
COL1, COL2 = (89 / 25.4, 183 / 25.4) if NATURE else (3.25, 6.75)   # Nature single / double column, AISTATS column / full

plt.rcParams.update({   # MOSAIC look: serif (Times-like), boxed axes, light grid, inward ticks
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "lines.linewidth": 1.3, "lines.markersize": 3.5,
    "axes.spines.top": True, "axes.spines.right": True, "axes.grid": True, "grid.color": "#e8e8e8",
    "grid.linewidth": 0.5, "axes.edgecolor": "#222222", "text.color": "#0b0b0b", "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#222222", "ytick.color": "#222222", "legend.frameon": True, "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc", "legend.fancybox": False, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42, "image.interpolation": "none",
})

# categorical slots in fixed order (reference palette, light mode); the colour follows the model in every figure
STYLE = {   # our model in brick red (as MOSAIC), everything else in the soft Tableau-10 tones, thin, so FOEHN reads first
    "foehn":       dict(label="FOEHN",                  color="#c8453c", ls="-",  marker="o", z=10),
    "nofp":        dict(label="FOEHN w/o footprint",    color="#e8998d", ls="--", marker="o", z=9),
    "convlstm":    dict(label="ConvLSTM",               color="#8c6d31", ls="-",  marker="s", z=5),
    "earthformer": dict(label="Earthformer",            color="#76b7b2", ls="-",  marker="^", z=5),
    "simvp":       dict(label="SimVP",                  color="#c9a227", ls="-",  marker="v", z=5),
    "climax":      dict(label="ClimaX",                 color="#b07aa1", ls="-",  marker="D", z=5),
    "topoflow":    dict(label="TopoFlow",               color="#59a14f", ls="-",  marker="P", z=5),
    "aurora":      dict(label="Aurora (zero-shot)",     color="#f28e2b", ls="-",  marker="h", z=4),
    "cams":        dict(label="CAMS operational",       color="#9c755f", ls="-",  marker="*", z=4),
    "climatology": dict(label="Climatology",            color="#111111", ls=":",  marker="", z=1),
    "persistence": dict(label="Persistence",            color="#111111", ls=":",  marker="", z=1),
}
if NATURE:                                          # Nature look (Liberation Sans, open axes) and the paper's names
    import nature_style  # noqa: F401,E402  (sets the rcParams)
    STYLE["foehn"]["label"] = "Foehn"
    STYLE["foehn"].update(color="#1F5AA6"); STYLE["nofp"].update(color="#8FB2DD")     # the paper's colours: Foehn blue,
    STYLE["cams"].update(color="#111111"); STYLE["aurora"].update(color="#D55E00")    # CAMS black, Aurora orange
    STYLE["climatology"].update(color="#9A9A9A", ls=(0, (3, 2)))
    for k_, c_ in (("convlstm", "#8C6D31"), ("earthformer", "#2E9E9E"), ("simvp", "#C9A227"), ("topoflow", "#59A14F"), ("climax", "#B07AA1")):
        STYLE[k_].update(color=c_)                  # other networks: distinct muted hues; Foehn blue still reads first
    STYLE["nofp"]["label"] = "Foehn, no transport layer"
    STYLE["aurora"]["label"] = "Aurora"
    STYLE["cams"]["label"] = "CAMS (operational)"
    plt.rcParams.update({"xtick.top": False, "ytick.right": False, "axes.grid": False})
    import re
    from matplotlib.axes import Axes
    _set_title = Axes.set_title

    def _nature_title(self, label, *args, **kw):            # "(a) text" -> bold panel letter, left-aligned (Nature)
        m = re.match(r"^\(([a-z])\)\s*(.*)$", str(label), re.S)
        if m:
            kw["loc"] = "left"
            label = r"$\bf{" + m.group(1) + "}$  " + m.group(2)
        return _set_title(self, label, *args, **kw)
    Axes.set_title = _nature_title
OURS_KM = "Foehn, tile-free" if NATURE else "FOEHN-km"
SEEDS = ("", "_s43", "_s44")
RUNS = {  # model -> list of evaluate.py output dirs (one per seed), Europe, real HRES weather, 2022
    "foehn": [EXP / f"dejavu/runs/noise_t31_fpstab{s}/eval1km_hres_none" for s in SEEDS],
    "nofp": [EXP / f"dejavu/runs/noise_t31{s}/eval1km_hres_none" for s in SEEDS],
    **{m: [EXP / f"retrain/runs/B_{m}_mettarget31{s}/eval1km_hres_none" for s in SEEDS]
       for m in ("convlstm", "earthformer", "simvp", "climax", "topoflow")},
    "cams": [EXP / "benchmark/eval1km_cams"],
    "aurora": [EXP / "benchmark/eval1km_aurora"],
    "climatology": [EXP / "benchmark/eval1km_climatology"],
}
LEADS = (1, 2, 3, 4)


def metric(res, key):
    """res = metrics.json dict; key -> list over leads."""
    get = {
        "rmse_025": lambda v: v["coarse_025deg"]["rmse"], "rmse_1km": lambda v: v["grid"]["model"]["rmse"],
        "rmse_eea": lambda v: v["stations"]["model_rmse"], "csi_025": lambda v: v["coarse_025deg"]["csi25"],
        "pers_025": lambda v: v["coarse_025deg"]["persistence_rmse"], "pers_1km": lambda v: v["grid"]["persistence"]["rmse"],
        "pers_eea": lambda v: v["stations"]["persistence_rmse"], "pcsi_025": lambda v: v["coarse_025deg"]["csi25_persistence"],
        "crps_1km": lambda v: v.get("ensemble", {}).get("crps", np.nan),
        "fss5": lambda v: v.get("scale_aware_region", {}).get("fss25_5km", np.nan),
    }[key]
    return [get(res[f"T+{h}"]) for h in LEADS]


def load(model, key):
    """-> (mean over seeds, std over seeds, n seeds) per lead; NaN if nothing evaluated yet."""
    vals = []
    for d in RUNS[model]:
        f = Path(d) / "metrics.json"
        if f.exists():
            vals.append(metric(json.loads(f.read_text()), key))
    if not vals:
        return np.full(len(LEADS), np.nan), np.full(len(LEADS), np.nan), 0
    v = np.array(vals, float)
    return v.mean(0), v.std(0), len(vals)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    print("saved", OUT / name)
