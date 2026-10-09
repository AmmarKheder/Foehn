#!/usr/bin/env python3
"""LaTeX tables of the paper from the evaluate.py outputs (MOSAIC conventions: best in bold, second underlined,
references shaded, mean +- std over seeds). Writes paper/aistats2027/tables/{main,regional}.tex."""
import json
from pathlib import Path

import numpy as np

from fig_style import EXP, RUNS, SEEDS

TAB = Path(__file__).resolve().parents[2] / "paper" / "aistats2027" / "tables"
TAB.mkdir(exist_ok=True)


def values(dirs, fn):
    out = []
    for d in dirs:
        f = Path(d) / "metrics.json"
        if f.exists():
            r = json.loads(f.read_text())
            out.append([fn(r[f"T+{h}"]) for h in (1, 4)])
    return np.array(out, float) if out else None


def fmt(v, sd, rank, prec=2):
    s = f"{v:.{prec}f}" + (f"\\,{{\\scriptsize$\\pm${sd:.{prec}f}}}" if sd is not None and sd > 0 else "")
    return "\\textbf{" + s + "}" if rank == 0 else "\\underline{" + s + "}" if rank == 1 else s


def table(rows, cols, caption, label, path, ranked_groups):
    """rows: [(group, name, dirs)], cols: [(header, fn, higher_is_better)] each with T+1 and T+4 sub-columns."""
    cells = {}
    for g, name, dirs in rows:
        for j, (_, fn, _) in enumerate(cols):
            v = values(dirs, fn)
            cells[name, j] = None if v is None else (v.mean(0), v.std(0) if len(v) > 1 else None)
    ranks = {}
    for j, (_, _, hib) in enumerate(cols):
        for k in (0, 1):
            cand = [(c[0][k], n) for (n, jj), c in cells.items() if jj == j and c is not None and np.isfinite(c[0][k])
                    and any(r[1] == n and r[0] in ranked_groups for r in rows)]
            cand.sort(key=lambda t: -t[0] if hib else t[0])
            for r, (_, n) in enumerate(cand[:2]):
                ranks[n, j, k] = r
    lines = ["\\begin{table*}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\small",
             "\\setlength{\\tabcolsep}{3.2pt}",
             "\\resizebox{\\textwidth}{!}{\\begin{tabular}{ll" + "cc" * len(cols) + "}", "\\toprule",
             " & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{h}}}" for h, _, _ in cols) + " \\\\",
             " ".join(f"\\cmidrule(lr){{{3 + 2 * j}-{4 + 2 * j}}}" for j in range(len(cols))),
             "Group & Model & " + " & ".join("T+1 & T+4" for _ in cols) + " \\\\", "\\midrule"]
    prev = None
    for g, name, dirs in rows:
        if prev is not None and g != prev:
            lines.append("\\midrule")
        prev = g
        row = [g if g != getattr(table, "_last", None) else "", name]
        table._last = g
        for j in range(len(cols)):
            c = cells[name, j]
            for k in (0, 1):
                if c is None or not np.isfinite(c[0][k]):
                    row.append("--")
                else:
                    row.append(fmt(c[0][k], None if c[1] is None else c[1][k], ranks.get((name, j, k), 9),
                                   3 if cols[j][2] else 2))
        shade = "\\rowcolor{gray!12}" if g == "Reference" else ""
        lines.append(shade + " & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    path.write_text("\n".join(lines) + "\n")
    print("wrote", path)


crps = lambda v: v.get("ensemble", {}).get("crps", v["grid"]["model"]["mae"])   # CRPS of a point forecast = MAE
COLS = [("RMSE 25\\,km", lambda v: v["coarse_025deg"]["rmse"], False),
        ("RMSE 1\\,km", lambda v: v["grid"]["model"]["rmse"], False),
        ("RMSE stations", lambda v: v["stations"]["model_rmse"], False),
        ("CRPS 1\\,km", crps, False),
        ("CSI $>$25", lambda v: v["coarse_025deg"]["csi25"], True)]
pers = [EXP / "retrain/runs/B_convlstm_mettarget31/eval1km_hres_none"]
PERS_COLS = [("RMSE 25\\,km", lambda v: v["coarse_025deg"]["persistence_rmse"], False),
             ("RMSE 1\\,km", lambda v: v["grid"]["persistence"]["rmse"], False),
             ("RMSE stations", lambda v: v["stations"]["persistence_rmse"], False),
             ("CRPS 1\\,km", lambda v: v["grid"]["persistence"]["mae"], False),
             ("CSI $>$25", lambda v: v["coarse_025deg"]["csi25_persistence"], True)]

MAIN_ROWS = [("Reference", "Climatology", RUNS["climatology"]), ("Reference", "CAMS operational", RUNS["cams"]), ("Reference", "Aurora (zero-shot)", RUNS["aurora"]),
             ("Baselines", "ConvLSTM", RUNS["convlstm"]), ("Baselines", "SimVP", RUNS["simvp"]),
             ("Baselines", "Earthformer", RUNS["earthformer"]), ("Baselines", "ClimaX", RUNS["climax"]),
             ("Baselines", "TopoFlow", RUNS["topoflow"]),
             ("Ours", "FOEHN w/o footprint", RUNS["nofp"]), ("Ours", "FOEHN", RUNS["foehn"])]

if __name__ == "__main__":
    table(MAIN_ROWS, COLS,
          "Europe, test year 2022, real ECMWF HRES forecast weather for every model, no station supervision. RMSE in "
          "\\textmu g\\,m$^{-3}$ on 0.25$^\\circ$ block means, on all 1\\,km land pixels and at 2,971 EEA stations; CRPS at "
          "1\\,km (MAE for point forecasts); CSI of exceedances of 25\\,\\textmu g\\,m$^{-3}$ at 0.25$^\\circ$. Mean "
          "$\\pm$ std over 3 seeds. Best in bold, second underlined (references excluded). Climatology: 2017--2020 mean "
          "of the satellite product over the same calendar days ($\\pm$7 days).",
          "tab:main", TAB / "main.tex", ranked_groups=("Baselines", "Ours"))
    REG = EXP / "regional/runs"
    REG_ROWS = [("Europe-trained", "ConvLSTM", [EXP / "retrain/runs/B_convlstm_mettarget31/eval_region"]),
                ("Europe-trained", "Earthformer", [EXP / "retrain/runs/B_earthformer_mettarget31/eval_region"]),
                ("Europe-trained", "SimVP", [EXP / "retrain/runs/B_simvp_mettarget31/eval_region"]),
                ("Europe-trained", "ClimaX", [EXP / "retrain/runs/B_climax_mettarget31/eval_region"]),
                ("Europe-trained", "TopoFlow", [EXP / "retrain/runs/B_topoflow_mettarget31/eval_region"]),
                ("Europe-trained", "FOEHN", [EXP / "dejavu/runs/noise_t31_fpstab/eval_region"]),
                ("Native 1\\,km", "Earthformer", [REG / "earthformer/eval_region"]),
                ("Native 1\\,km", "SimVP", [REG / "simvp/eval_region"]),
                ("Native 1\\,km", "ClimaX", [REG / "climax/eval_region"]),
                ("Native 1\\,km", "Flow, regression (ours)", [REG / "flowreg/eval_region"]),
                ("Native 1\\,km", "Flow matching (ours)", [REG / "flow/eval_region"])]
    sa = lambda key: (lambda v: v.get("scale_aware_region", {}).get(key, np.nan))
    REG_COLS = [("RMSE 1\\,km", lambda v: v["grid"]["model"]["rmse"], False),
                ("RMSE stations", lambda v: v["stations"]["model_rmse"], False),
                ("FSS 5\\,km", sa("fss25_5km"), True),
                ("CSI 1\\,km", sa("csi25_1km"), True),
                ("Detail skill", sa("fine_detail_skill"), True)]
    # rows share names across groups: make them unique for the cell dictionary
    REG_ROWS = [(g, n if g == "Europe-trained" else n + "\\,", d) for g, n, d in REG_ROWS]
    table._last = None
    table(REG_ROWS, REG_COLS,
          "Regional window at 1\\,km (49.5--41.8$^\\circ$N, 0--15.4$^\\circ$E; Paris, Lyon, Alps, Po valley), 2022, HRES "
          "weather. Europe-trained models are evaluated on the window (0.25$^\\circ$ models interpolated bilinearly); "
          "native models are trained on the window only, at 1\\,km, with the same inputs and 30 epochs. FSS and CSI of "
          "exceedances of 25\\,\\textmu g\\,m$^{-3}$; detail skill: error of the sub-0.25$^\\circ$ anomaly relative to "
          "predicting no detail (0 = no skill). TopoFlow and ConvLSTM do not fit a 64\\,GB GPU natively at this size.",
          "tab:regional", TAB / "regional.tex", ranked_groups=("Europe-trained", "Native 1\\,km"))
