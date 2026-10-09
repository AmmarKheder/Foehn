#!/usr/bin/env python3
"""Data of the cost figure: for every model, parameters, time and peak memory of one forward pass (bench_cost.py,
one MI250X GCD), and the measured training cost on LUMI-G (Slurm accounting: node-hours and energy from the node power
counters; seeds averaged; accounting failures (energy < 1 kJ) reported as missing, never estimated).
    python figdata_cost.py   -> data/cost_all.json"""
import json
import subprocess
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent / "data"
JOBS = {   # model: (setting, [training job ids])
    "ConvLSTM": ("Europe 0.25°", [22354264, 22376690, 22376692]),
    "Earthformer": ("Europe 0.25°", [22354265, 22376694, 22376696]),
    "SimVP": ("Europe 0.25°", [22376634, 22376698, 22376700]),
    "ClimaX": ("Europe 0.25°", [22376636, 22376702, 22376704]),
    "TopoFlow": ("Europe 0.25°", [22376638, 22376706, 22376708]),
    "Earthformer (1 km region)": ("Europe 1 km region", [22409419]),
    "SimVP (1 km region)": ("Europe 1 km region", [22409421]),
    "This work, Europe 1 km": ("Europe 1 km", [22402377]),
    "This work, global 1 km": ("Global 1 km", [22447689, 22447690, 22447691]),
}
COST = {"ConvLSTM": "convlstm", "Earthformer": "earthformer", "SimVP": "simvp", "ClimaX": "climax", "TopoFlow": "topoflow",
        "This work, Europe 1 km": "fullimage", "Aurora": "aurora"}
out = {}
for name, (setting, ids) in JOBS.items():
    q = subprocess.run(["sacct", "-j", ",".join(map(str, ids)), "-X", "-n", "-P", "-o", "JobID,NNodes,ElapsedRaw,ConsumedEnergyRaw,State"],
                       capture_output=True, text=True).stdout.strip().splitlines()
    nh, kwh, bad = [], [], 0
    for line in q:
        jid, nn, el, en, stt = line.split("|")
        nh.append(int(nn) * int(el) / 3600)
        e = float(en) if en else 0.0
        if e < 1e3:
            bad += 1
        else:
            kwh.append(e / 3.6e6)
    out[name] = dict(setting=setting, runs=len(q), node_hours=float(np.mean(nh)), gpu_hours=float(np.mean(nh)) * 8 / 2,
                     energy_kwh=float(np.mean(kwh)) if kwh else None, energy_runs=len(kwh), energy_missing=bad,
                     kw_per_node=(float(np.mean(kwh)) / float(np.mean(nh))) if kwh else None)
for name, f in COST.items():
    p = D / f"cost_{f}.json"
    if p.exists():
        out.setdefault(name, {}).update({"forward_s": json.loads(p.read_text())["seconds"], "peak_gb": json.loads(p.read_text())["peak_gb"],
                                         "params_m": json.loads(p.read_text())["params_m"]})
(D / "cost_all.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(f"{k:28s} " + "  ".join(f"{a}={(round(b, 3) if isinstance(b, float) else b)}" for a, b in v.items()))
