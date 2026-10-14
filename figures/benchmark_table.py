#!/usr/bin/env python3
"""The table of the European benchmark (stage 1, Extended Data Figs. 2-4): Europe, 2022, the same inputs for every learned
model, scores averaged over the lead times of 1 to 4 days and over the three training seeds (one run for Aurora, the CAMS
forecast and climatology), and the inference cost of one member of one forecast on one MI250X compute die.
    python benchmark_table.py [md|html]   -> the table on stdout (Markdown for the README, rows for the website)"""
import json
import sys
from pathlib import Path

import numpy as np

from fig_style import load

DATA = Path(__file__).resolve().parent / "data"
ROWS = [("foehn", "Foehn (flow matching, European 0.25° version)", "probabilistic"),
        ("nofp", "Foehn without the learned transport layer", "probabilistic"),
        ("convlstm", "ConvLSTM", "deterministic"), ("earthformer", "Earthformer", "deterministic"),
        ("simvp", "SimVP", "deterministic"), ("topoflow", "TopoFlow", "deterministic"), ("climax", "ClimaX", "deterministic"),
        ("aurora", "Aurora", "deterministic"), ("cams", "CAMS operational forecast", "deterministic"),
        ("climatology", "Climatology", "reference")]
KEYS = ("rmse_025", "rmse_1km", "rmse_eea", "csi_025")


def table():
    out = []
    for key, name, kind in ROWS:
        vals = [float(np.nanmean(load(key, k)[0])) for k in KEYS]
        c = DATA / f"cost_{key}.json"
        cost = json.loads(c.read_text()) if c.exists() else {}
        out.append((name, kind, vals, cost.get("seconds"), cost.get("peak_gb")))
    return out


def fmt_s(s):
    return "–" if s is None else (f"{s:.1f}" if s >= 1 else f"{s:.3f}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "md"
    rows = table()
    best = [min(r[2][i] for r in rows) if i < 3 else max(r[2][i] for r in rows) for i in range(4)]
    if mode == "md":
        print("| Model | RMSE, 0.25° | RMSE, 1 km | RMSE, 2,971 EEA stations | CSI, > 25 µg m⁻³ | Inference (s) | GPU memory (GB) |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for name, kind, v, s, gb in rows:
            cells = [(f"**{x:.2f}**" if abs(x - best[i]) < 1e-9 else f"{x:.2f}") if i < 3 else
                     (f"**{x:.3f}**" if abs(x - best[i]) < 1e-9 else f"{x:.3f}") for i, x in enumerate(v)]
            print(f"| {name} | " + " | ".join(cells) + f" | {fmt_s(s)} | {'–' if gb is None else f'{gb:.2f}'} |")
    else:
        for name, kind, v, s, gb in rows:
            cls = ' class="best"' if name.startswith("Foehn (") else ""
            cells = "".join(f"<td>{x:.2f}</td>" if i < 3 else f"<td>{x:.3f}</td>" for i, x in enumerate(v))
            print(f"<tr{cls}><td>{name}</td>{cells}<td>{fmt_s(s)}</td><td>{'–' if gb is None else f'{gb:.2f}'}</td></tr>")


if __name__ == "__main__":
    main()
