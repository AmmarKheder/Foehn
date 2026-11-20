#!/usr/bin/env python3
"""Independent check of the rebuilt figure data with plain numpy (no scorer code): from the station-day table of
score_cams3h, recompute the per-cell RMSE and the share of cells of panel a, the per-issue-day changes of panel e, the
RMSE by lead day of panel b and the regional changes of panel c, and compare with numbers.json / data/*.csv of the figure
folder. Read-only.
    python -B -W ignore check_plain_numpy.py          (login node, about 15 s, 2 GB)"""
import csv
import io
import json
from pathlib import Path

import numpy as np

R = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay")
F = Path("/scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/figures_new/fig_rt_skill")
z = np.load(R / "score_cams3h/work/pairs.npz")
num = json.loads((F / "numbers.json").read_text())
issue_days = [str(x) for x in z["issue_days"]]
issue, lead, mon, obs = z["issue"].astype(int), z["lead"].astype(int), z["mon"].astype(int), z["obs"].astype(np.float64)
arms = {a: z[f"m_{a}"].astype(np.float64) for a in ("km", "rt04", "cams", "clim", "bcsd", "cams_old")}
day = np.array(issue_days)[issue]
ok = np.isfinite(obs) & (day >= "2026-09-06") & (day <= "2026-11-11")
for a in ("km", "rt04", "cams", "clim", "bcsd"):
    ok &= np.isfinite(arms[a])
print("station-days of the primary set:", int(ok.sum()), "| numbers.json:", num["sample"]["station_days"])
res = {}
res["station_days"] = int(ok.sum()) == num["sample"]["station_days"]
lat, lon = z["lat"].astype(np.float64)[mon], z["lon"].astype(np.float64)[mon]
cell = (np.floor(lat / 2) + 90).astype(int) * 1000 + (np.floor(lon / 2) + 180).astype(int)
se = {a: (arms[a] - obs) ** 2 for a in arms}
rm = lambda a, m: float(np.sqrt(se[a][m].mean()))  # noqa: E731
print("pooled RMSE km / CAMS 3-hourly / CAMS three instants:", round(rm("km", ok), 4), round(rm("cams", ok), 4), round(rm("cams_old", ok), 4),
      "| change %:", round(100 * (rm("km", ok) / rm("cams", ok) - 1), 3), round(100 * (rm("km", ok) / rm("cams_old", ok) - 1), 3))
res["pooled"] = abs(100 * (rm("km", ok) / rm("cams", ok) - 1) - num["legend_numbers"]["change_pooled"]["cams"]["est"]) < 1e-9
# ---- panel a: cells
uc, ci = np.unique(cell[ok], return_inverse=True)
n_c = np.bincount(ci)
r_km = np.sqrt(np.bincount(ci, weights=se["km"][ok]) / n_c)
r_ca = np.sqrt(np.bincount(ci, weights=se["cams"][ok]) / n_c)
col = n_c >= 100
low = int((r_km[col] < r_ca[col]).sum())
print(f"cells: {uc.size} with monitors, {int(col.sum())} coloured, {low} with the lower RMSE = {100 * low / col.sum():.2f} % | numbers.json:",
      num["a_map"]["cells_with_monitors"], num["a_map"]["cells_coloured"], num["a_map"]["share_lower_rmse_pct"]["cells_vs_cams"]["units_lower"])
res["cells"] = (uc.size, int(col.sum()), low) == (num["a_map"]["cells_with_monitors"], num["a_map"]["cells_coloured"], num["a_map"]["share_lower_rmse_pct"]["cells_vs_cams"]["units_lower"])
chg = 100 * (r_km / r_ca - 1)
B = num["a_map"]["class_bounds"]
hist = np.histogram(chg[col], bins=[-np.inf] + B + [np.inf])[0]
print("cells per class (13 classes):", hist.tolist(), "| numbers.json:", num["a_map"]["cells_per_class"])
res["classes"] = hist.tolist() == num["a_map"]["cells_per_class"]
rows = {int(r["cell"]): r for r in csv.DictReader(io.StringIO((F / "data/cells.csv").read_text()))}
d = max(abs(float(rows[int(c)]["rmse_cams"]) - v) for c, v in zip(uc, r_ca))
print("cells.csv, RMSE of CAMS per cell: largest difference", f"{d:.4f}", "(3 decimals in the file)")
res["cells_csv"] = d < 6e-4
# monitors
um, mi = np.unique(mon[ok], return_inverse=True)
n_m = np.bincount(mi)
km_m = np.bincount(mi, weights=se["km"][ok]); ca_m = np.bincount(mi, weights=se["cams"][ok])
cm = n_m >= 30
print(f"monitors: {um.size}; with at least 30 station-days {int(cm.sum())}; lower RMSE at {int((km_m[cm] < ca_m[cm]).sum())} = {100 * (km_m[cm] < ca_m[cm]).mean():.2f} % | numbers.json:",
      num["sample"]["monitors"], num["a_map"]["share_lower_rmse_pct"]["monitors_vs_cams"]["units"], num["a_map"]["share_lower_rmse_pct"]["monitors_vs_cams"]["units_lower"])
res["monitors"] = (um.size, int(cm.sum()), int((km_m[cm] < ca_m[cm]).sum())) == (num["sample"]["monitors"], num["a_map"]["share_lower_rmse_pct"]["monitors_vs_cams"]["units"], num["a_map"]["share_lower_rmse_pct"]["monitors_vs_cams"]["units_lower"])
# ---- panel b: RMSE by lead day
for a in ("km", "rt04", "cams", "clim"):
    mine = [rm(a, ok & (lead == k)) for k in range(1, 6)]
    res[f"lead_{a}"] = bool(np.allclose(mine, num["b_lead"]["rmse"][a], atol=1e-9, rtol=0))
print("RMSE of the CAMS forecast by lead day:", [round(rm("cams", ok & (lead == k)), 2) for k in range(1, 6)], "| numbers.json:", [round(v, 2) for v in num["b_lead"]["rmse"]["cams"]])
print("change against CAMS by lead day:", [round(100 * (rm("km", ok & (lead == k)) / rm("cams", ok & (lead == k)) - 1), 2) for k in range(1, 6)],
      "| numbers.json:", [round(v["est"], 2) for v in num["b_lead"]["change_pooled"]["cams"]])
res["lead_change"] = bool(np.allclose([100 * (rm("km", ok & (lead == k)) / rm("cams", ok & (lead == k)) - 1) for k in range(1, 6)], [v["est"] for v in num["b_lead"]["change_pooled"]["cams"]], atol=1e-9, rtol=0))
# ---- panel c: regions (catalogue region of each monitor)
reg = z["region"].astype(str)[mon]
mine = {g: 100 * (rm("km", ok & (reg == g)) / rm("cams", ok & (reg == g)) - 1) for g in np.unique(reg)}
print("change against CAMS by region:", {g: round(v, 2) for g, v in mine.items()})
res["regions"] = all(abs(mine[g] - num["c_regions"]["change"]["cams"][g]["est"]) < 1e-9 for g in mine)
print("mean over the regions:", round(float(np.mean(list(mine.values()))), 3), "| numbers.json:", round(num["c_regions"]["change"]["cams"]["regmean"]["est"], 3))
res["regmean"] = abs(float(np.mean(list(mine.values()))) - num["c_regions"]["change"]["cams"]["regmean"]["est"]) < 1e-9
# ---- panel e: issue days
E = num["e_days"]
mine = []
for dd in E["issue_day"]:
    m = ok & (day == dd)
    mine.append(100 * (rm("km", m) / rm("cams", m) - 1))
mine = np.array(mine)
print(f"issue days: {len(mine)}; lower on {int((mine < 0).sum())}; range {mine.min():.2f} .. {mine.max():.2f}, median {np.median(mine):.2f} | numbers.json:",
      E["counts"]["cams"]["pooled_lower"], round(E["counts"]["cams"]["pooled_min"], 2), round(E["counts"]["cams"]["pooled_max"], 2), round(E["counts"]["cams"]["pooled_median"], 2))
res["issue_days"] = bool(np.allclose(mine, [v["est"] for v in E["change_pooled"]["cams"]], atol=1e-9, rtol=0)) and int((mine < 0).sum()) == E["counts"]["cams"]["pooled_lower"]
mc = np.array([100 * (rm("km", ok & (day == dd)) / rm("clim", ok & (day == dd)) - 1) for dd in E["issue_day"]])
print("issue days below the climatology:", int((mc < 0).sum()), "| both below:", int(((mc < 0) & (mine < 0)).sum()))
res["issue_days_clim"] = int(((mc < 0) & (mine < 0)).sum()) == 48
ln_ = [x.split(",") for x in (F / "data/issue_days.csv").read_text().splitlines()[1:]]
hdr = (F / "data/issue_days.csv").read_text().splitlines()[0].split(",")
j = hdr.index("change_vs_cams_pct")
res["issue_days_csv"] = bool(np.allclose([float(r[j]) for r in ln_], mine, atol=6e-3, rtol=0))
print({k: v for k, v in res.items()})
print("ALL IDENTICAL" if all(res.values()) else "SOME DIFFER")
