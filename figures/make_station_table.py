#!/usr/bin/env python3
"""Evaluation against the EEA monitoring stations (never used in training), Europe, test year 2022, real HRES weather,
leads 1-4 pooled, identical (station, start day, lead) samples for every model: RMSE on all days, on the 10 % most
polluted days (largest land fraction above 25 ug/m3 on the target day), on the documented transported episodes, and on
the station-days that exceed 25 ug/m3 ("haze"), and the CSI of exceedances of 25 ug/m3 at the stations.
Writes paper/aistats2027/tables/stations.tex."""
from pathlib import Path

import numpy as np

from fig_style import EXP, SEEDS

TAB = EXP.parent / "paper/aistats2027/tables"
EPIS = [(69, 79), (83, 89), (187, 201), (217, 225)]          # start days of the documented episodes (as Figure 5)
ROWS = [  # group, label, list of evaluate.py output dirs (one per seed)
    ("Reference", "Satellite product (target)", None),
    ("Reference", "Climatology", [EXP / "benchmark/eval1km_climatology"]),
    ("Reference", "CAMS operational", [EXP / "benchmark/eval1km_cams"]),
    ("Reference", "Aurora (zero-shot)", [EXP / "benchmark/eval1km_aurora"]),
    *[("Baselines", lab, [EXP / f"retrain/runs/B_{m}_mettarget31{s}/eval1km_hres_none" for s in SEEDS])
      for m, lab in (("convlstm", "ConvLSTM"), ("simvp", "SimVP"), ("earthformer", "Earthformer"), ("climax", "ClimaX"),
                     ("topoflow", "TopoFlow"))],
    ("Ours", "FOEHN w/o footprint", [EXP / f"dejavu/runs/noise_t31{s}/eval1km_hres_none" for s in SEEDS]),
    ("Ours", "FOEHN", [EXP / f"dejavu/runs/noise_t31_fpstab{s}/eval1km_hres_none" for s in SEEDS]),
]

# polluted days: land fraction above 25 ug/m3 on the target day, 0.25 deg (same definition as Figure 5)
meta = np.load(EXP / "dejavu/data/meta.npz")
land = meta["land"].astype(bool)
i0 = int(np.where(meta["year"] == 2022)[0][0])
pm = np.load(EXP / "dejavu/data/pm.npy", mmap_mode="r")
WIN = np.zeros_like(land); WIN[90:121, 100:162] = True           # the regional window (rows 2250:3018, cols 2500:4036 at 1 km)
FRAC_EU = np.array([np.nanmean(np.where(land, np.asarray(pm[i0 + d], np.float32), np.nan) > 25) for d in range(365)])
FRAC_WIN = np.array([np.nanmean(np.where(land & WIN, np.asarray(pm[i0 + d], np.float32), np.nan) > 25) for d in range(365)])


def load(dirs):
    out = []
    for d in dirs:
        p = Path(d) / "records.npz"
        if p.exists():
            out.append(np.load(p)["stations"][:, [0, 1, 2, 3, 5, 6]].astype(np.float64))
    return out


def key(a):
    return (a[:, 0].astype(np.int64) * 1000 + a[:, 1].astype(np.int64)) * 10 + a[:, 2].astype(np.int64)


def compute(rows, frac):
    data = {lab: load(dirs) for _, lab, dirs in rows if dirs}
    data = {k: v for k, v in data.items() if v}
    common = None
    for runs in data.values():
        for x in runs:
            common = key(x) if common is None else np.intersect1d(common, key(x))

    def align(x):
        k = key(x); o = np.argsort(k)
        return x[o][np.isin(k[o], common)]
    anchor = next(lab for _, lab, d in rows if d and lab in data)
    ref = align(data[anchor][0])
    day, lead, obs = ref[:, 1].astype(int), ref[:, 2].astype(int), ref[:, 5]
    tday = np.clip(day + lead, 0, 364)
    epi = np.zeros(len(day), bool)
    for lo, hi in EPIS:
        epi |= (day >= lo) & (day <= hi)
    sets = [np.ones(len(day), bool), frac[tday] >= np.quantile(frac, 0.9), epi, obs > 25]
    print("samples", len(common), [int(m.sum()) for m in sets])

    def scores(pred):
        r = [float(np.sqrt(np.mean((pred[m] - obs[m]) ** 2))) for m in sets]
        e, g = obs > 25, pred > 25
        return np.array(r + [float((e & g).sum() / max((e | g).sum(), 1))])
    out = {}
    for grp, lab, dirs in rows:
        if dirs is None:
            out[lab] = (grp, np.array([scores(ref[:, 4])]))
        elif lab in data:
            out[lab] = (grp, np.array([scores(align(x)[:, 3]) for x in data[lab]]))
    for lab, (grp, v) in out.items():
        print(f"  {lab:34s} n={len(v)} " + " ".join(f"{x:.3f}" for x in v.mean(0)))
    return out


REG = EXP / "regional/runs"
ROWS_REG = [
    ("Reference", "Satellite product (target)", None),
    ("Europe-trained", "ConvLSTM", [EXP / "retrain/runs/B_convlstm_mettarget31/eval_region"]),
    ("Europe-trained", "Earthformer", [EXP / "retrain/runs/B_earthformer_mettarget31/eval_region"]),
    ("Europe-trained", "FOEHN", [EXP / "dejavu/runs/noise_t31_fpstab/eval_region"]),
    ("Native 1 km", "Earthformer, 1 km", [REG / "earthformer/eval_region"]),
    ("Native 1 km", "SimVP, 1 km", [REG / "simvp/eval_region"]),
    ("Native 1 km", "Ours, regression, 1 km", [REG / "flowreg/eval_region"]),
    ("Native 1 km", "Ours, flow matching, 1 km", [REG / "flow/eval_region"]),
]
print("== Europe"); EU = compute(ROWS, FRAC_EU)
print("== regional window"); RG = compute(ROWS_REG, FRAC_WIN)

COLS = ["All", "Top 10\\%", "Episodes", "Obs.\\,$>$25", "CSI"]


SHORT = {"Satellite product (target)": "Product (target)", "CAMS operational": "CAMS", "Aurora (zero-shot)": "Aurora",
         "FOEHN w/o footprint": "FOEHN w/o fp.", "Ours, regression, 1 km": "Ours, reg., 1 km",
         "Ours, flow matching, 1 km": "Ours, flow, 1 km"}


def block(res, ranked, title):
    learned = [lab for lab, (grp, _) in res.items() if grp in ranked]
    best, second = {}, {}
    for j in range(len(COLS)):
        vals = sorted((res[l][1].mean(0)[j], l) for l in learned)
        if j == len(COLS) - 1:
            vals = vals[::-1]
        best[j], second[j] = vals[0][1], vals[1][1]
    out, last = [f"\\multicolumn{{6}}{{l}}{{\\emph{{{title}}}}} \\\\", "\\midrule"], None
    for lab, (grp, v) in res.items():
        if last is not None and grp != last:
            out.append("\\cmidrule(lr){1-6}")
        last = grp
        v = v.mean(0); cells = []
        for j, x in enumerate(v):
            t = f"{x:.3f}"[1:] if j == len(COLS) - 1 else f"{x:.2f}"
            t = f"\\textbf{{{t}}}" if lab == best[j] else (f"\\underline{{{t}}}" if lab == second[j] else t)
            cells.append(t)
        name = SHORT.get(lab, lab).replace("FOEHN", "\\foehn{}")
        out.append(("\\rowcolor{gray!12}" if grp == "Reference" else "") + name + " & " + " & ".join(cells) + " \\\\")
    return out


lines = ["\\begin{table}[t]", "\\centering",
         "\\caption{Against the EEA monitoring stations, never used in training (2022, real HRES weather, leads 1--4 "
         "pooled, identical station-day samples within each block): RMSE (\\textmu g\\,m$^{-3}$) on all days, on the 10\\% "
         "most polluted days, on the documented transported episodes and on station-days above 25\\,\\textmu g\\,m$^{-3}$, "
         "and CSI of these exceedances (\\emph{fp.}: footprint; \\emph{reg.}: regression; Aurora zero-shot). Top: Europe, 2,971 stations, mean over 3 seeds. Bottom: the regional window of "
         "Table~\\ref{tab:regional}, where the native models are trained at 1\\,km. Best in bold, second underlined, among "
         "learned models.}",
         "\\label{tab:stations}", "\\footnotesize", "\\setlength{\\tabcolsep}{1.2pt}",
         "\\begin{tabular}{l" + "c" * len(COLS) + "}", "\\toprule",
         " & \\multicolumn{4}{c}{RMSE at stations} & \\\\", "\\cmidrule(lr){2-5}",
         " & " + " & ".join(COLS) + " \\\\", "\\midrule"]
lines += block(EU, ("Baselines", "Ours"), "Europe")
lines += ["\\midrule"] + block(RG, ("Europe-trained", "Native 1 km"), "Regional window, 1\\,km")
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
(TAB / "stations.tex").write_text("\n".join(lines) + "\n")
print("wrote", TAB / "stations.tex")
