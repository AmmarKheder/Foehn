#!/usr/bin/env python3
"""Ablation table (0.25 deg, Europe, test year 2022, real ECMWF HRES forecast weather, 3 seeds, M = 8 members):
RMSE / CRPS / CSI (> 25 ug/m3, ensemble mean) at T+1 and T+3. Writes paper/aistats2027/tables/ablation.tex."""
import json
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
RUNS = EXP / "dejavu/runs"
TAB = EXP.parent / "paper/aistats2027/tables"
SEEDS = ("", "_s43", "_s44")
ROWS = [  # label, run prefix, json file, key
    ("\\foehn{}", "noise_t31_fpstab", "eval_2022_hres.json", "noise S=8"),
    ("\\quad BLH weights", "noise_t31_fpblhp", "eval_2022_hres.json", "noise S=8"),
    ("\\quad wrong wind", "noise_t31_fpshift15stab", "eval_2022_hres_rec8.json", "noise S=8"),
    ("\\quad no footprint", "noise_t31", "eval_2022_hres.json", "noise S=8"),
    ("\\quad regression", "reg_t31_fpstab", "eval_2022_hres_rec8.json", "reg S=8"),
    ("\\quad $S=4$", "noise_t31_fpstab", "eval_2022_hres.json", "noise S=4"),
    ("\\quad $S=1$", "noise_t31_fpstab", "eval_2022_hres.json", "noise S=1"),
]
COLS = [(1, "rmse"), (1, "crps"), (1, "csi_mean"), (3, "rmse"), (3, "crps"), (3, "csi_mean")]


def cell(prefix, fn, key):
    vals = []
    for s in SEEDS:
        p = RUNS / f"{prefix}{s}" / fn
        if p.exists():
            m = json.loads(p.read_text())
            if all(f"{key}|T+{h}" in m for h in (1, 3)):
                vals.append([m[f"{key}|T+{h}"][k] for h, k in COLS])
    return np.array(vals)


rows, sds = [], []
for lab, prefix, fn, key in ROWS:
    v = cell(prefix, fn, key)
    rows.append((lab, v.mean(0) if len(v) else None, len(v)))
    if len(v) > 1:
        sds.append(v.std(0))
best = {}
for j, (_, k) in enumerate(COLS):
    xs = [r[1][j] for r in rows if r[1] is not None]
    best[j] = max(xs) if k == "csi_mean" else min(xs)
fmt = lambda k, x: f"{x:.2f}" if k == "rmse" else (f"{x:.3f}"[1:] if k == "csi_mean" else f"{x:.3f}")
sd = np.max(sds, 0) if sds else np.zeros(len(COLS))
lines = ["\\begin{table}[t]", "\\centering",
         "\\caption{Ablations of \\foehn{} (0.25$^\\circ$, Europe, 2022, real HRES forecast weather, $M=8$ members; mean over "
         f"3 seeds, standard deviations at most {sd[[0, 3]].max():.2f} (RMSE), {sd[[1, 4]].max():.3f} (CRPS) and "
         f"{sd[[2, 5]].max():.3f} (CSI)). The wrong-wind model is trained and tested with the forecast wind of a day "
         "15 days away in the footprint only (\\emph{wrong wind}); \\emph{BLH weights}: path points weighted by the "
         "boundary-layer height of the start day instead of the forecast stability; \\emph{regression}: same network "
         "and inputs trained with the squared loss, stopped at its best validation epoch; $S$: flow steps (8 otherwise).}",
         "\\label{tab:ablation}", "\\footnotesize", "\\setlength{\\tabcolsep}{1.9pt}",
         "\\begin{tabular}{lcccccc}", "\\toprule",
         " & \\multicolumn{3}{c}{T+1} & \\multicolumn{3}{c}{T+3} \\\\",
         "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
         " & RMSE & CRPS & CSI & RMSE & CRPS & CSI \\\\", "\\midrule"]
for lab, v, n in rows:
    if v is None:
        lines.append(lab + " & " + " & ".join(["--"] * len(COLS)) + " \\\\")
        continue
    cells = []
    for j, (_, k) in enumerate(COLS):
        t = fmt(k, v[j])
        cells.append(f"\\textbf{{{t}}}" if fmt(k, v[j]) == fmt(k, best[j]) else t)
    lines.append(lab + (f"$^{{{n}}}$" if n < 3 else "") + " & " + " & ".join(cells) + " \\\\")
    if lab.startswith("\\quad no footprint") or lab.startswith("\\quad regression"):
        pass
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
(TAB / "ablation.tex").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
