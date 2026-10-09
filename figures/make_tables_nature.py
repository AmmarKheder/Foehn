#!/usr/bin/env python3
"""Extended Data Tables 1 and 2 of the Nature paper, from the same evaluation outputs as make_tables.py and make_ablation.py
(Europe, test year 2022, real ECMWF HRES forecast weather, three seeds), with the size and inference cost of every network
(bench_cost.py, data/cost_*.json). Writes paper/nature/tables_nature/{main,ablation}.tex.
    python make_tables_nature.py"""
import json
from pathlib import Path

import numpy as np

from fig_style import RUNS

HERE = Path(__file__).resolve().parent
TAB = HERE.parents[1] / "paper" / "nature" / "tables_nature"
DV = HERE.parent / "dejavu" / "runs"
LEADS = (1, 4)

# ---------------------------------------------------------------------------------------------- Extended Data Table 1
crps = lambda v: v.get("ensemble", {}).get("crps", v["grid"]["model"]["mae"])      # CRPS of a single forecast = MAE
METRICS = [("RMSE, 0.25$^\\circ$", lambda v: v["coarse_025deg"]["rmse"], False, 2),
           ("RMSE, 1\\,km", lambda v: v["grid"]["model"]["rmse"], False, 2),
           ("RMSE, stations", lambda v: v["stations"]["model_rmse"], False, 2),
           ("CRPS, 1\\,km", crps, False, 2),
           ("CSI, $>$25", lambda v: v["coarse_025deg"]["csi25"], True, 3)]
COST = {k: json.loads((HERE / "data" / f"cost_{k}.json").read_text()) for k in
        ("aurora", "convlstm", "simvp", "earthformer", "climax", "topoflow", "nofp", "foehn")}
MODELS = [("Reference", "Climatology", "climatology", "--", None),
          ("Reference", "CAMS (operational)", "cams", "Physics, 0.4$^\\circ$", None),
          ("Reference", "Aurora (zero-shot)", "aurora", "Single", "aurora"),
          ("Baselines", "ConvLSTM", "convlstm", "Single", "convlstm"),
          ("Baselines", "SimVP", "simvp", "Single", "simvp"),
          ("Baselines", "Earthformer", "earthformer", "Single", "earthformer"),
          ("Baselines", "ClimaX", "climax", "Single", "climax"),
          ("Baselines", "TopoFlow", "topoflow", "Single", "topoflow"),
          ("Foehn-0.25$^\\circ$", "Foehn, no transport layer", "nofp", "Ensemble", "nofp"),
          ("Foehn-0.25$^\\circ$", "Foehn", "foehn", "Ensemble", "foehn")]


def run_values(key):
    out = []
    for d in RUNS[key]:
        f = Path(d) / "metrics.json"
        if f.exists():
            r = json.loads(f.read_text())
            out.append([[fn(r[f"T+{h}"]) for h in LEADS] for _, fn, _, _ in METRICS])
    return np.array(out, float)                                  # (seeds, metrics, leads)


def cell(m, s, prec, rank):
    t = f"{m:.{prec}f}" + (f"\\,{{\\scriptsize$\\pm${s:.{prec}f}}}" if s is not None and s > 0 else "")
    return "\\textbf{" + t + "}" if rank == 0 else "\\underline{" + t + "}" if rank == 1 else t


def table1():
    vals = {key: run_values(key) for _, _, key, _, _ in MODELS}
    mean = {k: v.mean(0) for k, v in vals.items()}
    sd = {k: (v.std(0) if len(v) > 1 else None) for k, v in vals.items()}
    ranked = [key for g, _, key, _, _ in MODELS if g != "Reference"]
    rank = {}
    for j, (_, _, hib, _) in enumerate(METRICS):
        for k in range(len(LEADS)):
            order = sorted(ranked, key=lambda m_: -mean[m_][j, k] if hib else mean[m_][j, k])
            for r, m_ in enumerate(order[:2]):
                rank[m_, j, k] = r
    base = [key for g, _, key, _, _ in MODELS if g == "Baselines"]
    short = {"RMSE, 0.25$^\\circ$": "RMSE\\\\0.25$^\\circ$", "RMSE, 1\\,km": "RMSE\\\\1\\,km", "RMSE, stations": "RMSE\\\\stations",
             "CRPS, 1\\,km": "CRPS\\\\1\\,km", "CSI, $>$25": "CSI\\\\$>$25"}
    head = " & ".join("\\begin{tabular}[b]{@{}c@{}}" + short[h] + "\\end{tabular}" for h, *_ in METRICS)
    ncol = 2 + len(METRICS) * len(LEADS)
    lines = [
        "\\begin{table}[p]",
        "\\centering",
        "\\caption{\\textbf{European benchmark: all models on the same test year, weather and targets.} Europe, 2022, with the "
        "real ECMWF HRES forecast weather as input for every network and no station data in training. Root-mean-square error "
        "(RMSE, \\textmu g\\,m$^{-3}$) on 0.25$^\\circ$ block means, on all 1\\,km land pixels and at the 2,971 EEA stations; "
        "continuous ranked probability score (CRPS, \\textmu g\\,m$^{-3}$) at 1\\,km, which for a single forecast is its mean "
        "absolute error; critical success index (CSI) of the exceedances of 25\\,\\textmu g\\,m$^{-3}$ at 0.25$^\\circ$ (higher is "
        "better), one and four days ahead. Trained networks: mean over three training seeds, with the standard deviation in "
        "grey below; best of the trained networks in bold, second underlined. Params, millions of parameters (bench\\_cost); "
        "Foehn-0.25$^\\circ$ and its variant produce ensembles, the other networks a single forecast. Last row: "
        "Foehn-0.25$^\\circ$ against the best baseline network of each column (negative is better for RMSE and CRPS, positive "
        "for CSI). Climatology: 2017--2020 mean of the 1\\,km estimates for the same calendar days ($\\pm$7 days).}",
        "\\label{tab:main}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2.7pt}",
        "\\begin{tabular}{l r " + "c" * len(METRICS) + " " + "c" * len(METRICS) + "}",
        "\\toprule",
        f" & & \\multicolumn{{{len(METRICS)}}}{{c}}{{One day ahead}} & \\multicolumn{{{len(METRICS)}}}{{c}}{{Four days ahead}} \\\\",
        f"\\cmidrule(lr){{3-{2 + len(METRICS)}}} \\cmidrule(lr){{{3 + len(METRICS)}-{2 + 2 * len(METRICS)}}}",
        "Model & Params & " + head + " & " + head + " \\\\",
        "\\midrule"]
    prev = None
    for g, name, key, out, ck in MODELS:
        if g != prev:
            if prev is not None:
                lines.append("\\midrule")
            lines.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{g}}}}} \\\\")
        c = COST.get(ck) if ck else None
        par = "--" if c is None else (f"{c['params_m']:,.0f}" if c["params_m"] >= 100 else f"{c['params_m']:.1f}")
        row, srow = ["\\quad " + name, par], ["", ""]
        for k in range(len(LEADS)):
            for j, (_, _, hib, prec) in enumerate(METRICS):
                row.append(cell(mean[key][j, k], None, prec, rank.get((key, j, k), 9)))
                if sd[key] is not None:
                    srow.append(f"{{\\color{{gray}}\\scriptsize$\\pm${sd[key][j, k]:.{prec}f}}}")
        lines.append(" & ".join(row) + " \\\\")
        if sd[key] is not None:
            lines.append(" & ".join(srow) + " \\\\[1pt]")
        prev = g
    lines.append("\\midrule")
    gain = []
    for k in range(len(LEADS)):
        for j, (_, _, hib, _) in enumerate(METRICS):
            best = max(mean[b][j, k] for b in base) if hib else min(mean[b][j, k] for b in base)
            gain.append(f"{100 * (mean['foehn'][j, k] / best - 1):+.1f}\\%")
    lines.append("\\multicolumn{2}{l}{\\textit{Foehn vs best baseline}} & " + " & ".join(gain) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "main.tex").write_text("\n".join(lines) + "\n")
    print("wrote main.tex; Foehn vs best baseline:", " ".join(gain))


# ---------------------------------------------------------------------------------------------- Extended Data Table 2
SEEDS = ("", "_s43", "_s44")
ABL = [  # group, label, what changes, run prefix, json file, key
    ("", "Foehn-0.25$^\\circ$", "Full model: flow matching, transport layer, 8 flow steps", "noise_t31_fpstab", "eval_2022_hres.json", "noise S=8"),
    ("Transport layer", "No transport layer", "FLEXGREEN footprint inputs removed", "noise_t31", "eval_2022_hres.json", "noise S=8"),
    ("", "Wrong wind", "Footprint computed with the forecast wind of a day 15 days away", "noise_t31_fpshift15stab", "eval_2022_hres_rec8.json", "noise S=8"),
    ("", "BLH weights", "Path points weighted by the start-day boundary-layer height, not the forecast stability", "noise_t31_fpblhp", "eval_2022_hres.json", "noise S=8"),
    ("Training objective", "Regression", "Same network and inputs trained with the squared error (best validation epoch)", "reg_t31_fpstab", "eval_2022_hres_rec8.json", "reg S=8"),
    ("Sampler", "4 flow steps", "Same weights as the full model, 4 integration steps", "noise_t31_fpstab", "eval_2022_hres.json", "noise S=4"),
    ("", "1 flow step", "Same weights as the full model, a single step", "noise_t31_fpstab", "eval_2022_hres.json", "noise S=1"),
]
ACOLS = [(1, "rmse", 2), (1, "crps", 3), (1, "csi_mean", 3), (3, "rmse", 2), (3, "crps", 3), (3, "csi_mean", 3)]


def abl_values(prefix, fn, key):
    vals = []
    for s in SEEDS:
        p = DV / f"{prefix}{s}" / fn
        if p.exists():
            m = json.loads(p.read_text())
            if all(f"{key}|T+{h}" in m for h, _, _ in ACOLS):
                vals.append([m[f"{key}|T+{h}"][k] for h, k, _ in ACOLS])
    return np.array(vals, float)


def table2():
    rows = [(g, lab, what, abl_values(pre, fn, key)) for g, lab, what, pre, fn, key in ABL]
    ref = rows[0][3].mean(0)
    best = {}
    for j, (_, k, _) in enumerate(ACOLS):
        xs = [r[3].mean(0)[j] for r in rows]
        best[j] = max(xs) if k == "csi_mean" else min(xs)
    lines = [
        "\\begin{table}[p]",
        "\\centering",
        "\\caption{\\textbf{What each part of Foehn contributes.} Ablations of Foehn-0.25$^\\circ$ on the European benchmark "
        "(0.25$^\\circ$, Europe, 2022, real HRES forecast weather, ensembles of 8 members). Each variant is retrained from "
        "scratch with three seeds, except the sampler variants, which reuse the weights of the full model; means over the seeds, "
        "with the standard deviation in grey below. RMSE and CRPS in \\textmu g\\,m$^{-3}$ (lower is better), CSI of the "
        "exceedances of 25\\,\\textmu g\\,m$^{-3}$ by the ensemble mean (higher is better); best in bold. $\\Delta$CRPS: change of "
        "the CRPS against the full model (positive is worse).}",
        "\\label{tab:ablation}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2.8pt}",
        "\\begin{tabular}{p{2.5cm} p{4.1cm} ccc ccc cc}",
        "\\toprule",
        " & & \\multicolumn{3}{c}{One day ahead} & \\multicolumn{3}{c}{Three days ahead} & \\multicolumn{2}{c}{$\\Delta$CRPS} \\\\",
        "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-10}",
        "Variant & What changes & RMSE & CRPS & CSI & RMSE & CRPS & CSI & 1\\,d & 3\\,d \\\\",
        "\\midrule"]
    for i, (g, lab, what, v) in enumerate(rows):
        if g:
            lines.append("\\midrule")
            lines.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{{g}}}}} \\\\")
        m, s = v.mean(0), (v.std(0) if len(v) > 1 else np.zeros(len(ACOLS)))
        cells, scells = [], []
        for j, (_, k, prec) in enumerate(ACOLS):
            t = f"{m[j]:.{prec}f}"
            cells.append(f"\\textbf{{{t}}}" if t == f"{best[j]:.{prec}f}" else t)
            scells.append(f"{{\\color{{gray}}\\scriptsize$\\pm${s[j]:.{prec}f}}}")
        d = ["--", "--"] if i == 0 else [f"{100 * (m[1] / ref[1] - 1):+.1f}\\%", f"{100 * (m[4] / ref[4] - 1):+.1f}\\%"]
        lines.append(" & ".join([("" if i == 0 else "\\quad ") + lab, what] + cells + d) + " \\\\")
        lines.append(" & ".join(["", ""] + scells + ["", ""]) + " \\\\[1pt]")
        print(f"{lab:22s} n={len(v)} dCRPS " + " ".join(d))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "ablation.tex").write_text("\n".join(lines) + "\n")
    print("wrote ablation.tex")


if __name__ == "__main__":
    table1()
    table2()
