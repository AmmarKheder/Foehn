#!/usr/bin/env python3
"""Recompute the 324 cells of the European scorecard (Supplementary Fig. 3 of the working manuscript) exactly as
experiments/figures/fig_extremes.py does (same files, same day sets, same scores), without drawing, and list the cells
that the benchmarked network does not win. Read-only on the project; writes out/scorecard_cells.json and prints a summary.
    python3 scorecard_cells.py
"""
import json
from pathlib import Path

import numpy as np

EXP = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments")
OUT = Path(__file__).resolve().parents[1] / "out"
LEADS = (1, 2, 3, 4)
SEEDS = ("", "_s43", "_s44")
EPIS = [(69, 79), (83, 89), (187, 201), (217, 225)]          # fig_extremes.py line 16 (0-based day of year of the START day)
meta = np.load(EXP / "dejavu/data/meta.npz")
land = meta["land"].astype(bool)
i0 = int(np.where(meta["year"] == 2022)[0][0])
pm = np.load(EXP / "dejavu/data/pm.npy", mmap_mode="r")
TRUTH = {d: np.asarray(pm[i0 + d], np.float32) for d in range(365)}
with np.errstate(invalid="ignore"):
    FRAC = {d: float(np.nanmean(np.where(land, TRUTH[d], np.nan) > 25)) for d in range(365)}


def rows_flow(run):
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
            with np.errstate(invalid="ignore"):
                m = land & np.isfinite(t) & (t > 0)
            e, g = t[m] > 25, f[m] > 25
            out[d, h] = np.array([m.sum(), ((f[m] - t[m]) ** 2).sum(), np.abs(f[m] - t[m]).sum(), (e & g).sum(), (e | g).sum()])
    return out


MODELS = {"foehn": [rows_flow("noise_t31_fpstab" + s) for s in SEEDS], "nofp": [rows_flow("noise_t31" + s) for s in SEEDS]}
for m in ("convlstm", "earthformer", "simvp", "topoflow", "climax"):
    MODELS[m] = [rows_base("B_%s_mettarget31%s" % (m, s)) for s in SEEDS]
MODELS["aurora"] = [rows_grid(EXP / "benchmark/aurora/out/aurora_2022.npy")]
MODELS["cams"] = [rows_grid(EXP / "benchmark/cams_forecast_2022.npy")]
MODELS["climatology"] = [rows_grid(EXP / "benchmark/climatology_2022.npy")]
common = set.intersection(*[set(r) for rs in MODELS.values() for r in rs])
print("samples per system:", {k: [len(r) for r in v] for k, v in MODELS.items()})
print("common samples", len(common), "| start days", len({k[0] for k in common}))


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


COLS = ["all", "top10", "episodes"]
ROWS = ["rmse", "crps", "csi"]
SYSR = ["nofp", "convlstm", "earthformer", "simvp", "topoflow", "climax", "aurora", "cams", "climatology"]
summary = {}
for dset in COLS:
    for metric in ROWS:
        for m in ["foehn"] + SYSR:
            vals = np.array([[score(r, dayset(dset, h), metric) for h in LEADS] for r in MODELS[m]])
            summary[m, dset, metric] = vals.mean(0)
            summary[m, dset, metric, "sd"] = vals.std(0)
cells, notwon = [], []
for m in SYSR:
    for dset in COLS:
        for metric in ROWS:
            for h in LEADS:
                f, o = float(summary["foehn", dset, metric][h - 1]), float(summary[m, dset, metric][h - 1])
                d = 100 * (f / o - 1) * (-1 if metric == "csi" else 1)        # negative = benchmarked network better
                c = dict(system=m, days=dset, score=metric, lead=h, network=f, other=o, change_pct=d)
                cells.append(c)
                if not d < 0:
                    notwon.append(c)
print("cells", len(cells), "| won", sum(c["change_pct"] < 0 for c in cells), "| not won", len(notwon))
for c in notwon:
    print("  NOT WON: %-12s %-8s %-4s lead %d  network %.4f  other %.4f  change %+.2f %%" %
          (c["system"], c["days"], c["score"], c["lead"], c["network"], c["other"], c["change_pct"]))
n_days = {dset: {h: len(dayset(dset, h)) for h in LEADS} for dset in COLS}
print("samples per day set and lead:", n_days)
print("start days inside the four episode windows:", sorted({k[0] for k in common if any(a <= k[0] <= b for a, b in EPIS)}).__len__())
# how the wins are distributed (D18: many cells within 2 %; 108 cells set a CRPS against an absolute error)
ch = np.array([c["change_pct"] for c in cells])
print("cells with |change| < 2 %%: %d ; won by less than 2 %%: %d ; won by more than 10 %%: %d" %
      (int((np.abs(ch) < 2).sum()), int(((ch < 0) & (ch > -2)).sum()), int((ch < -10).sum())))
for m in SYSR:
    sub = np.array([c["change_pct"] for c in cells if c["system"] == m])
    print("  %-12s change min %+.1f max %+.1f median %+.1f  | cells not won %d" % (m, sub.min(), sub.max(), np.median(sub), int((sub >= 0).sum())))
print("CRPS cells:", sum(c["score"] == "crps" for c in cells))
# station panel and 0.25 deg panel of the lead-time figure (metrics.json, mean over seeds)
def mj(d):
    return json.loads((Path(d) / "metrics.json").read_text())
runs = {"foehn": [EXP / ("dejavu/runs/noise_t31_fpstab%s/eval1km_hres_none" % s) for s in SEEDS],
        "nofp": [EXP / ("dejavu/runs/noise_t31%s/eval1km_hres_none" % s) for s in SEEDS],
        "cams": [EXP / "benchmark/eval1km_cams"], "aurora": [EXP / "benchmark/eval1km_aurora"],
        "climatology": [EXP / "benchmark/eval1km_climatology"]}
for m in ("convlstm", "earthformer", "simvp", "climax", "topoflow"):
    runs[m] = [EXP / ("retrain/runs/B_%s_mettarget31%s/eval1km_hres_none" % (m, s)) for s in SEEDS]
lead = {}
for m, ds in runs.items():
    js = [mj(d) for d in ds if (Path(d) / "metrics.json").exists()]
    lead[m] = dict(n_seeds=len(js),
                   rmse_025=[float(np.mean([j["T+%d" % h]["coarse_025deg"]["rmse"] for j in js])) for h in LEADS],
                   rmse_1km=[float(np.mean([j["T+%d" % h]["grid"]["model"]["rmse"] for j in js])) for h in LEADS],
                   rmse_eea=[float(np.mean([j["T+%d" % h]["stations"]["model_rmse"] for j in js])) for h in LEADS],
                   csi_025=[float(np.mean([j["T+%d" % h]["coarse_025deg"]["csi25"] for j in js])) for h in LEADS],
                   station_obs=[int(js[0]["T+%d" % h]["stations"]["obs"]) for h in LEADS],
                   pixels=[int(js[0]["T+%d" % h]["grid"]["pixels"]) for h in LEADS])
    print("%-12s seeds %d | RMSE 0.25 %s | 1 km %s | EEA %s | CSI %s" % (m, len(js), np.round(lead[m]["rmse_025"], 2), np.round(lead[m]["rmse_1km"], 2),
                                                                       np.round(lead[m]["rmse_eea"], 2), np.round(lead[m]["csi_025"], 3)))
print("station-days per lead (network):", lead["foehn"]["station_obs"], "| GHAP at the stations, RMSE:",
      [round(mj(runs["foehn"][0])["T+%d" % h]["stations"]["ghap_target_rmse"], 2) for h in LEADS])
f4, a4 = float(np.mean(lead["foehn"]["rmse_1km"])), float(np.mean(lead["aurora"]["rmse_1km"]))
cost = {n: json.loads((EXP / "figures/data" / ("cost_%s.json" % n)).read_text()) for n in
        ("foehn", "nofp", "aurora", "convlstm", "earthformer", "simvp", "climax", "topoflow", "fullimage")}
print("cost figure: network %.3f s, Aurora %.1f s, ratio %.1f ; RMSE at 1 km (mean of 4 leads) network %.3f, Aurora %.3f, change %.1f %%" %
      (cost["foehn"]["seconds"], cost["aurora"]["seconds"], cost["aurora"]["seconds"] / cost["foehn"]["seconds"], f4, a4, 100 * (1 - f4 / a4)))
print("baselines faster than the network by:", {n: round(cost["foehn"]["seconds"] / cost[n]["seconds"], 1) for n in ("convlstm", "earthformer", "simvp", "climax", "topoflow", "nofp")})
print("1 km RMSE (mean of 4 leads):", {m: round(float(np.mean(v["rmse_1km"])), 3) for m, v in lead.items()})
OUT.mkdir(exist_ok=True)
(OUT / "scorecard_cells.json").write_text(json.dumps(dict(
    common_samples=len(common), start_days=len({k[0] for k in common}), samples_per_set_and_lead=n_days, cells=cells, not_won=notwon,
    leadtime_panels=lead, cost=cost, cost_ratio_aurora_over_network=cost["aurora"]["seconds"] / cost["foehn"]["seconds"],
    rmse1km_mean4_network=f4, rmse1km_mean4_aurora=a4), indent=1))
print("written", OUT / "scorecard_cells.json")
