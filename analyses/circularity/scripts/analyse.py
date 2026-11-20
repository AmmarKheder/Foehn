#!/usr/bin/env python3
"""The defence against circularity, in numbers (test year 2022, final model crps_e01). The stations only score: nothing
here trains, selects or corrects a model.

Inputs (all written by the extract_*.py scripts of this directory from stored forecasts, each checked against the
paper's own result file; no network is run):
  REF                              experiments/figures/data/compare_fair_crps_e01.npz   (the paper's station comparison)
  ../data/station_day.npz          every station-day value behind REF                    (extract_station_day.py)
  ../data/rings.npz                Foehn read on square rings around the stations        (extract_rings.py)
  ../data/foehn_cams_station_day.npz   the model of Aurora's protocol at the stations    (extract_foehn_cams.py)

Parts
  A  by network, from REF (arrays PDN, SST): RMSE of Foehn at 1 km, averaged to 0.25 and 0.5 deg, the CAMS forecast and
     Aurora at equal issuance, the climatology and the two persistences; gain of Foehn with 95 % intervals.
  B  the dependence of the 1 km estimates on each station: share of the days of 2022 on which the estimate of the day at
     the station pixel is within 0.5 ug m-3 of the measurement ("matched share"); classes >= 0.8, 0.2-0.8, < 0.2.
  C  Foehn read AWAY from the station pixel: one populated pixel and the mean of the populated pixels of the square ring
     3 pixels (0.03 deg, about 3 km) and 9 pixels (0.09 deg, about 10 km) from the station pixel; no ring pixel holds a
     monitor of our files. Leads 1 and 3 days (the stored ones), identical station-days for every column.
  D  the model of Aurora's protocol (no 1 km estimate, no station in training, selection or inputs) at the stations.
  E  sample check: the paper's comparison without the coordinate-key rule of compare_fair.py.

Intervals: the paper's paired moving-block bootstrap over start dates (blocks of seven start dates, 10,000 resamples,
seed 0, percentiles 2.5 and 97.5), as experiments/figures/compare_fair.py. For the main rows of C a second interval
resamples 2-degree cells instead (space instead of time).

    python analyse.py [R]      R = number of resamples (default 10000)   -> ../results/numbers.json, ../results/tables.txt
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE.parent / "data", HERE.parent / "results"
REF = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/figures/data/compare_fair_crps_e01.npz")
R = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
B = 7
NETS = {0: "Europe", 1: "China", 3: "USA", 4: "Korea", 5: "Japan", 6: "India"}
NCODE = list(NETS)                                              # source codes in the order of REF's networks
GROUPS = {"Europe": [0], "China": [1], "USA": [2], "Korea": [3], "Japan": [4], "India": [5],
          "named sources (Europe, China, USA)": [0, 1, 2], "not named (Korea, Japan, India)": [3, 4, 5],
          "Korea and Japan": [3, 4], "outside Europe (five networks)": [1, 2, 3, 4, 5],
          "without China": [0, 2, 3, 4, 5], "all six networks": [0, 1, 2, 3, 4, 5]}
out, lines = {}, []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); lines.append(s)


def block_index(nd, seed=0):
    """the resampling of compare_fair.py: moving blocks of B start dates"""
    rng = np.random.default_rng(seed)
    return (rng.integers(0, nd - B + 1, size=(R, -(-nd // B)))[:, :, None] + np.arange(B)).reshape(R, -1)[:, :nd]


def rmse(a):
    return float(np.sqrt(a[..., 0].sum() / a[..., 1].sum()))


def gain(a, b, idx):
    """a, b: (units, 2) sums of squared errors and counts per resampling unit, paired. Gain = 1 - RMSE_a / RMSE_b, in %.
    Returns point, low, high (95 %)."""
    pt = 1 - rmse(a) / rmse(b)
    A, Bb = a[idx], b[idx]
    with np.errstate(all="ignore"):
        bs = 1 - np.sqrt(A[..., 0].sum(1) / A[..., 1].sum(1)) / np.sqrt(Bb[..., 0].sum(1) / Bb[..., 1].sum(1))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return [round(100 * pt, 2), round(100 * float(lo), 2), round(100 * float(hi), 2)]


def fmt(g):
    return f"{g[0]:+5.1f} [{g[1]:+5.1f},{g[2]:+5.1f}]"


def day_sums(err2, mask):
    """(days, stations) squared errors and mask -> (days, 2)"""
    return np.stack([np.where(mask, err2, 0.0).sum(1), mask.sum(1)], -1).astype(np.float64)


# =====================================================================================================================
# A. by network, from the paper's file
# =====================================================================================================================
ref = np.load(REF)
PDN, PD, SST, S = ref["PDN"], ref["PD"], ref["SST"], ref["S"]
SYSN = [str(s) for s in ref["systems"]]
iF, iF25, iF05, iC, iA, iPo, iPg, iCl = 0, 1, 2, 4, 6, 7, 8, 9
ok_d = np.isfinite(PD[0, :, 0, 0])
nd = int(ok_d.sum())
IDX = block_index(nd)
assert np.allclose(PDN[:, :6].sum(1), PDN[:, 6])
src_ref = ref["src"]
P(f"# A. by network: {REF}")
P(f"systems: Foehn 1 km = '{SYSN[iF]}', CAMS = '{SYSN[iC]}', Aurora = '{SYSN[iA]}'; start dates {nd}; resamples {R}, blocks of {B}")
out["A"] = {"file": str(REF), "start_dates": nd, "resamples": R, "groups": {}}
for name, gl in GROUPS.items():
    codes = [NCODE[g] for g in gl]
    st_m = np.isin(src_ref, codes)
    row = {"stations_scored": int((SST[iF, 0, st_m, 1] > 0).sum()), "leads": {}}
    n30 = st_m & (SST[iF, 0, :, 1] >= 30)
    for h in range(4):
        a = {k: PDN[k][gl][:, ok_d, h].sum(0) for k in range(len(SYSN))}
        d = {"station_days": int(a[iF][:, 1].sum()),
             "rmse": {"foehn_1km": rmse(a[iF]), "foehn_025": rmse(a[iF25]), "foehn_05": rmse(a[iF05]), "cams": rmse(a[iC]),
                      "aurora": rmse(a[iA]), "climatology": rmse(a[iCl]), "persistence_station": rmse(a[iPo]),
                      "persistence_estimate_pixel": rmse(a[iPg])},
             "gain_vs_cams": gain(a[iF], a[iC], IDX), "gain_vs_aurora": gain(a[iF], a[iA], IDX),
             "gain_025_vs_cams": gain(a[iF25], a[iC], IDX), "gain_05_vs_cams": gain(a[iF05], a[iC], IDX),
             "gain_025_vs_aurora": gain(a[iF25], a[iA], IDX), "gain_05_vs_aurora": gain(a[iF05], a[iA], IDX),
             "gain_vs_climatology": gain(a[iF], a[iCl], IDX), "gain_vs_persistence_estimate_pixel": gain(a[iF], a[iPg], IDX),
             "gain_1km_vs_own_025": gain(a[iF], a[iF25], IDX)}
        m30 = st_m & (SST[iF, h, :, 1] >= 30)
        with np.errstate(all="ignore"):
            rF, rC, rA = (np.sqrt(SST[k, h, m30, 0] / SST[k, h, m30, 1]) for k in (iF, iC, iA))
        d["stations_30"] = int(m30.sum())
        d["share_stations_better_than_cams"] = round(100 * float((rF < rC).mean()), 1)
        d["share_stations_better_than_aurora"] = round(100 * float((rF < rA).mean()), 1)
        d["median_station_change_vs_cams"] = round(100 * float(np.median(rF / rC - 1)), 1)
        row["leads"][h + 1] = d
    out["A"]["groups"][name] = row
    P(f"\n{name}: stations scored {row['stations_scored']}")
    for h in range(4):
        d = row["leads"][h + 1]; r_ = d["rmse"]
        P(f"  {h + 1} d  n={d['station_days']:>8d}  RMSE F {r_['foehn_1km']:6.2f}  F.25 {r_['foehn_025']:6.2f}  F.5 {r_['foehn_05']:6.2f}  CAMS {r_['cams']:6.2f}  "
          f"Aur {r_['aurora']:6.2f}  clim {r_['climatology']:6.2f}  pers(est. pixel) {r_['persistence_estimate_pixel']:6.2f} | gain vs CAMS {fmt(d['gain_vs_cams'])}  "
          f"vs Aurora {fmt(d['gain_vs_aurora'])} | 0.5 deg vs CAMS {fmt(d['gain_05_vs_cams'])} vs Aurora {fmt(d['gain_05_vs_aurora'])} | "
          f"stations better than CAMS {d['share_stations_better_than_cams']} % of {d['stations_30']}, Aurora {d['share_stations_better_than_aurora']} %; "
          f"median station change vs CAMS {d['median_station_change_vs_cams']:+.1f} % | vs climatology {fmt(d['gain_vs_climatology'])} | vs start-day estimate at the pixel "
          f"{fmt(d['gain_vs_persistence_estimate_pixel'])} | 1 km vs own 0.25 deg {fmt(d['gain_1km_vs_own_025'])}")

# =====================================================================================================================
# B. dependence of the estimates on each station
# =====================================================================================================================
sd = np.load(DATA / "station_day.npz")
days, used, paper, src, lat, lon = sd["days"], sd["used"], sd["paper"], sd["src"], sd["lat"], sd["lon"]
NS = len(src)
obs = sd["obs"]
di = {int(d): i for i, d in enumerate(days)}
used_d = used.any(1)
assert int(used_d.sum()) == nd and np.array_equal(np.flatnonzero(used_d), np.flatnonzero(ok_d))
GT, OT = np.full((366, NS), np.nan, np.float32), np.full((366, NS), np.nan, np.float32)   # per target day
for i, d in enumerate(days):
    for h in range(4):
        if used[i, h]:
            GT[d + h + 1], OT[d + h + 1] = sd["ghapt"][i, h], obs[i, h]
both = np.isfinite(GT) & np.isfinite(OT)
nb = both.sum(0)
with np.errstate(all="ignore"):
    share = np.where(nb >= 30, (both & (np.abs(GT - OT) < 0.5)).sum(0) / np.maximum(nb, 1), np.nan)
cls = np.where(share >= 0.8, 2, np.where(share < 0.2, 0, 1)); cls[~np.isfinite(share)] = -1   # 2 matched, 1 between, 0 not matched
CLS = {2: "matched (>= 80 % of days)", 1: "between", 0: "not matched (< 20 % of days)"}
P("\n# B. dependence of the 1 km estimates on each station (estimate of the day at the station pixel within 0.5 ug m-3 of the measurement)")
out["B"] = {"definition": "share of the days of 2022 with |estimate at the station pixel - measurement| < 0.5 ug m-3, at least 30 days; "
            "classes >= 0.8 (matched), 0.2-0.8, < 0.2 (not matched)", "networks": {}}
st_paper = paper.any((0, 1))
for c, nm in NETS.items():
    m = st_paper & (src == c) & (cls >= 0)
    e = (GT - OT)[:, st_paper & (src == c)]; v = np.isfinite(e)
    d = {"stations": int(m.sum()), "matched": int((cls[m] == 2).sum()), "between": int((cls[m] == 1).sum()), "not_matched": int((cls[m] == 0).sum()),
         "share_station_days_within_0.5": round(100 * float((np.abs(e[v]) < 0.5).mean()), 1),
         "rmse_estimate_same_day": float(np.sqrt((e[v] ** 2).mean())), "bias_estimate_same_day": float(e[v].mean()),
         "median_station_share": round(100 * float(np.median(share[m])), 1)}
    out["B"]["networks"][nm] = d
    P(f"  {nm:7s} stations {d['stations']:5d}: matched {d['matched']:5d} ({100 * d['matched'] / d['stations']:4.1f} %), between {d['between']:5d}, not matched {d['not_matched']:5d} "
      f"({100 * d['not_matched'] / d['stations']:4.1f} %); station-days within 0.5: {d['share_station_days_within_0.5']} %; estimate of the day: RMSE "
      f"{d['rmse_estimate_same_day']:.2f}, bias {d['bias_estimate_same_day']:+.2f}")

# scores by class at the station pixel (all four leads), from the station-day values
E2 = {k: (sd[k].astype(np.float64) - obs) ** 2 for k in ("foehn", "foehn025", "foehn05", "cams1", "aur1", "clim")}
out["B"]["by_class"] = {}
P("\n  gain of Foehn at the station pixel by class (identical station-days of the paper; day-block intervals)")
SUBSETS = {"all six networks": np.ones(NS, bool), "Europe": src == 0, "China": src == 1, "USA": src == 3,
           "all except China": np.isin(src, [0, 3, 4, 5, 6])}
for sn, sm in SUBSETS.items():
    for c in (2, 1, 0):
        stm = sm & (cls == c)
        if (stm & st_paper).sum() < 20:
            continue
        row = {"stations": int((stm & st_paper).sum()), "leads": {}}
        for h in range(4):
            m = paper[used_d, h] & stm
            a = {k: day_sums(E2[k][used_d, h], m) for k in E2}
            row["leads"][h + 1] = {"station_days": int(m.sum()), "rmse": {k: rmse(a[k]) for k in a},
                                   "gain_vs_cams": gain(a["foehn"], a["cams1"], IDX), "gain_vs_aurora": gain(a["foehn"], a["aur1"], IDX),
                                   "gain_05_vs_cams": gain(a["foehn05"], a["cams1"], IDX), "gain_05_vs_aurora": gain(a["foehn05"], a["aur1"], IDX),
                                   "gain_vs_climatology": gain(a["foehn"], a["clim"], IDX)}
        out["B"]["by_class"][f"{sn} | {CLS[c]}"] = row
        P(f"  {sn:18s} {CLS[c]:28s} stations {row['stations']:5d}: " + "  ".join(
            f"{h} d: F {row['leads'][h]['rmse']['foehn']:.2f} C {row['leads'][h]['rmse']['cams1']:.2f} A {row['leads'][h]['rmse']['aur1']:.2f} "
            f"vsC {fmt(row['leads'][h]['gain_vs_cams'])} vsA {fmt(row['leads'][h]['gain_vs_aurora'])}" for h in (1, 3)))

# the forecast against the 1 km estimates of the SAME day, on identical station-days (the estimates are not a forecast)
E2g = (sd["ghapt"].astype(np.float64) - obs) ** 2
out["B"]["forecast_vs_estimates_same_day"] = {}
P("\n  Foehn's forecast against the 1 km estimates of the target day, identical station-days of the paper (gain > 0: forecast closer)")
for nm, sm in [("Europe", src == 0), ("China", src == 1), ("USA", src == 3), ("Korea", src == 4), ("Japan", src == 5), ("India", src == 6),
               ("all six networks", np.isin(src, NCODE))]:
    row = {}
    for h in range(4):
        m = paper[used_d, h] & sm & np.isfinite(sd["ghapt"][used_d, h])
        a, g_, c_ = day_sums(E2["foehn"][used_d, h], m), day_sums(E2g[used_d, h], m), day_sums(E2["clim"][used_d, h], m)
        row[h + 1] = {"station_days": int(m.sum()), "station_days_paper": int((paper[used_d, h] & sm).sum()), "rmse_foehn": rmse(a),
                      "rmse_estimates_same_day": rmse(g_), "rmse_climatology": rmse(c_), "gain_foehn_vs_estimates": gain(a, g_, IDX),
                      "gain_estimates_vs_climatology": gain(g_, c_, IDX)}
    out["B"]["forecast_vs_estimates_same_day"][nm] = row
    P(f"  {nm:17s} " + "  ".join(f"{h} d: F {row[h]['rmse_foehn']:.2f} est {row[h]['rmse_estimates_same_day']:.2f} clim {row[h]['rmse_climatology']:.2f} "
                                  f"F vs est {fmt(row[h]['gain_foehn_vs_estimates'])}" for h in (1, 2, 4)))

# the same comparison without the coordinate-key rule of compare_fair.py (every official station that has its forecasts)
pall = sd["paper_all"]
out["E"] = {"definition": "mask of compare_fair.py without its coordinate key (stations that share a float32 key are kept)", "groups": {}}
P("\n# E. sample check: the paper's comparison without the coordinate-key rule (stations dropped by the key are kept)")
for name, gl in GROUPS.items():
    stm = np.isin(src, [NCODE[g] for g in gl])
    row = {"stations": int((pall & stm).any((0, 1)).sum()), "stations_paper_mask": int((paper & stm).any((0, 1)).sum()), "leads": {}}
    for h in range(4):
        m = pall[used_d, h] & stm
        a = {k: day_sums(E2[k][used_d, h], m) for k in ("foehn", "cams1", "aur1", "clim")}
        row["leads"][h + 1] = {"station_days": int(m.sum()), "rmse": {k: rmse(a[k]) for k in a},
                               "gain_vs_cams": gain(a["foehn"], a["cams1"], IDX), "gain_vs_aurora": gain(a["foehn"], a["aur1"], IDX)}
    out["E"]["groups"][name] = row
    P(f"  {name:36s} stations {row['stations']:5d} (paper's mask {row['stations_paper_mask']:5d}): " + "  ".join(
        f"{h} d: F {row['leads'][h]['rmse']['foehn']:.2f} C {row['leads'][h]['rmse']['cams1']:.2f} A {row['leads'][h]['rmse']['aur1']:.2f} "
        f"vsC {fmt(row['leads'][h]['gain_vs_cams'])} vsA {fmt(row['leads'][h]['gain_vs_aurora'])}" for h in (1, 4)))

# =====================================================================================================================
# C. Foehn read away from the station pixel
# =====================================================================================================================
rg = np.load(DATA / "rings.npz")
keys, n_el = rg["keys"], rg["n_el"]
K3, K10, MINPIX = 3, 9, 4
LEADS = (1, 3)


def place(a):
    """(forecasts, stations) -> (start dates, 2 leads, stations)"""
    o = np.full((len(days), 2, NS), np.nan, np.float32)
    for n, (d, h, t) in enumerate(keys):
        o[di[int(d)], LEADS.index(int(h))] = a[n]
    return o


with np.errstate(all="ignore"):
    V = {"pix_pop": place(rg["st_mean"]),                                          # station pixel, the run that stored the pixels
         "one3": place(rg["f_one"][:, :, K3]), "one10": place(rg["f_one"][:, :, K10]),
         "ring3": place(rg["f_sum"][:, :, K3] / np.where(n_el[:, K3] > 0, n_el[:, K3], np.nan)),
         "ring10": place(rg["f_sum"][:, :, K10] / np.where(n_el[:, K10] > 0, n_el[:, K10], np.nan)),
         "band3": place(rg["f_sum"][:, :, 2:5].sum(-1) / np.where(n_el[:, 2:5].sum(1) > 0, n_el[:, 2:5].sum(1), np.nan)),
         "band10": place(rg["f_sum"][:, :, 8:11].sum(-1) / np.where(n_el[:, 8:11].sum(1) > 0, n_el[:, 8:11].sum(1), np.nan)),
         "g_ring3": place(rg["g_sum"][:, :, K3] / np.where(rg["g_n"][:, :, K3] > 0, rg["g_n"][:, :, K3], np.nan)),
         "g_ring10": place(rg["g_sum"][:, :, K10] / np.where(rg["g_n"][:, :, K10] > 0, rg["g_n"][:, :, K10], np.nan)),
         "g_one3": place(rg["g_one"][:, :, K3]), "g_one10": place(rg["g_one"][:, :, K10])}
    PROF_F = [place(rg["f_sum"][:, :, k] / np.where(n_el[:, k] > 0, n_el[:, k], np.nan)) for k in range(1, 13)]
    PROF_G = [place(rg["g_sum"][:, :, k] / np.where(rg["g_n"][:, :, k] > 0, rg["g_n"][:, :, k], np.nan)) for k in range(1, 13)]
hsel = [h - 1 for h in LEADS]
for k in ("foehn", "foehn025", "foehn05", "cams1", "aur1", "clim", "ghapt"):
    V[k] = sd[k][:, hsel]
Y = obs[:, hsel]
PAP = paper[:, hsel]
dpix = np.abs(V["pix_pop"] - V["foehn"])[PAP & np.isfinite(V["pix_pop"])]
P(f"\n# C. rings. Station pixel: the run that stored the populated pixels against the paper's run: mean |difference| {dpix.mean():.4f}, "
  f"max {dpix.max():.3f} ug m-3 (same forecasts, stored twice)")
avail = (n_el[:, K3] >= MINPIX) & (n_el[:, K10] >= MINPIX)
FCOLS = ["foehn", "one3", "ring3", "one10", "ring10", "foehn025", "foehn05", "cams1", "aur1", "clim"]
fin = np.all([np.isfinite(V[k]) for k in FCOLS], 0)
MASK = PAP & avail & fin
GCOLS = ["ghapt", "g_one3", "g_ring3", "g_one10", "g_ring10"]
gfin = MASK & np.all([np.isfinite(V[k]) for k in GCOLS], 0)
E2r = {k: (V[k].astype(np.float64) - Y) ** 2 for k in FCOLS + GCOLS + ["band3", "band10"]}
cell = (np.floor((lat + 90) / 2).astype(int) * 1000 + np.floor((lon + 180) / 2).astype(int))
ucell, cinv = np.unique(cell, return_inverse=True)
rngc = np.random.default_rng(1)
CIDX = rngc.integers(0, len(ucell), size=(R, len(ucell)))


def cell_sums(err2, mask):
    """(start dates, stations) -> (2-degree cells, 2)"""
    o = np.zeros((len(ucell), 2))
    np.add.at(o[:, 0], cinv, np.where(mask, err2, 0.0).sum(0)); np.add.at(o[:, 1], cinv, mask.sum(0))
    return o


out["C"] = {"definition": f"square rings at Chebyshev distance {K3} and {K10} pixels of 0.01 deg from the station pixel (24 and 72 pixels); "
            f"ring pixels = stored populated pixels (>= 100 residents) that hold no monitor of stations_2022.npz; a station needs at least {MINPIX} "
            f"such pixels in each ring; 'one' = one ring pixel drawn once per station (seed 0); leads 1 and 3 days; identical station-days",
            "pixel_check_mean_abs": float(dpix.mean()), "groups": {}}
CSUB = {"all six networks": np.isin(src, NCODE), "Europe": src == 0, "China": src == 1, "USA": src == 3, "Korea": src == 4, "Japan": src == 5,
        "India": src == 6, "named sources (Europe, China, USA)": np.isin(src, [0, 1, 3]), "not named (Korea, Japan, India)": np.isin(src, [4, 5, 6]),
        "Korea and Japan": np.isin(src, [4, 5]), "without China": np.isin(src, [0, 3, 4, 5, 6]),
        "matched stations (all networks)": np.isin(src, NCODE) & (cls == 2), "between (all networks)": np.isin(src, NCODE) & (cls == 1),
        "not matched stations (all networks)": np.isin(src, NCODE) & (cls == 0),
        "China, matched": (src == 1) & (cls == 2), "USA, matched": (src == 3) & (cls == 2), "Europe, matched": (src == 0) & (cls == 2),
        "Europe, not matched": (src == 0) & (cls == 0), "USA, not matched": (src == 3) & (cls == 0)}
P(f"  rings: {K3} px (about 3 km) and {K10} px (about 10 km); at least {MINPIX} ring pixels each; stations {int(MASK.any((0, 1)).sum())} of "
  f"{int(PAP.any((0, 1)).sum())}; station-days lead 1: {int(MASK[:, 0].sum())} of {int(PAP[:, 0].sum())}, lead 3: {int(MASK[:, 1].sum())} of {int(PAP[:, 1].sum())}")
for sn, sm in CSUB.items():
    row = {"stations": int((MASK & sm).any((0, 1)).sum()), "stations_paper": int((PAP & sm).any((0, 1)).sum()), "leads": {}}
    if row["stations"] < 20:
        continue
    for j, h in enumerate(LEADS):
        m = MASK[used_d, j] & sm
        a = {k: day_sums(E2r[k][used_d, j], m) for k in FCOLS}
        mg = gfin[used_d, j] & sm
        g = {k: day_sums(E2r[k][used_d, j], mg) for k in GCOLS}
        c_ = {k: cell_sums(E2r[k][used_d, j], m) for k in FCOLS}
        d = {"station_days": int(m.sum()), "rmse": {k: rmse(a[k]) for k in a}, "rmse_estimates_same_day": {k: rmse(g[k]) for k in g},
             "station_days_estimates": int(mg.sum())}
        for k in ("foehn", "one3", "ring3", "one10", "ring10", "foehn025", "foehn05"):
            d[f"gain_{k}_vs_cams"] = gain(a[k], a["cams1"], IDX); d[f"gain_{k}_vs_aurora"] = gain(a[k], a["aur1"], IDX)
            d[f"gain_{k}_vs_cams_cells"] = gain(c_[k], c_["cams1"], CIDX); d[f"gain_{k}_vs_aurora_cells"] = gain(c_[k], c_["aur1"], CIDX)
        for k in ("one3", "ring3", "one10", "ring10"):
            d[f"gain_{k}_vs_pixel"] = gain(a[k], a["foehn"], IDX)
        # per station: share better than the CAMS forecast (stations with at least 30 forecast days)
        ns_ = (MASK[used_d, j] & sm).sum(0); s30 = ns_ >= 30
        for k in ("foehn", "ring3", "ring10"):
            with np.errstate(all="ignore"):
                rf = np.where(MASK[used_d, j] & sm, E2r[k][used_d, j], 0).sum(0)[s30]; rc = np.where(MASK[used_d, j] & sm, E2r["cams1"][used_d, j], 0).sum(0)[s30]
            d[f"share_stations_{k}_better_than_cams"] = round(100 * float((rf < rc).mean()), 1)
        d["stations_30"] = int(s30.sum())
        mm_ = gfin[:, j] & sm
        d["estimates_share_within_0.5"] = {k: round(100 * float((np.abs(V[k][:, j] - Y[:, j])[mm_] < 0.5).mean()), 1) for k in ("ghapt", "g_one3", "g_one10")}
        d["foehn_share_within_0.5"] = round(100 * float((np.abs(V["foehn"][:, j] - Y[:, j])[mm_] < 0.5).mean()), 1)
        row["leads"][h] = d
    out["C"]["groups"][sn] = row
    P(f"\n  {sn}: stations {row['stations']} of {row['stations_paper']}")
    for h in LEADS:
        d = row["leads"][h]; r_ = d["rmse"]; g_ = d["rmse_estimates_same_day"]
        P(f"    {h} d n={d['station_days']:>7d} RMSE pixel {r_['foehn']:.2f} | one px 3 km {r_['one3']:.2f} | ring 3 km {r_['ring3']:.2f} | one px 10 km {r_['one10']:.2f} | "
          f"ring 10 km {r_['ring10']:.2f} | 0.25 {r_['foehn025']:.2f} | 0.5 {r_['foehn05']:.2f} | CAMS {r_['cams1']:.2f} | Aurora {r_['aur1']:.2f} | clim {r_['clim']:.2f}")
        P(f"         gain vs CAMS: pixel {fmt(d['gain_foehn_vs_cams'])}  one3 {fmt(d['gain_one3_vs_cams'])}  ring3 {fmt(d['gain_ring3_vs_cams'])}  "
          f"one10 {fmt(d['gain_one10_vs_cams'])}  ring10 {fmt(d['gain_ring10_vs_cams'])}  0.5deg {fmt(d['gain_foehn05_vs_cams'])}")
        P(f"         gain vs Aurora: pixel {fmt(d['gain_foehn_vs_aurora'])}  one3 {fmt(d['gain_one3_vs_aurora'])}  ring3 {fmt(d['gain_ring3_vs_aurora'])}  "
          f"one10 {fmt(d['gain_one10_vs_aurora'])}  ring10 {fmt(d['gain_ring10_vs_aurora'])}  0.5deg {fmt(d['gain_foehn05_vs_aurora'])}")
        P(f"         2-degree cells: vs CAMS pixel {fmt(d['gain_foehn_vs_cams_cells'])} ring3 {fmt(d['gain_ring3_vs_cams_cells'])} ring10 {fmt(d['gain_ring10_vs_cams_cells'])}; "
          f"vs Aurora pixel {fmt(d['gain_foehn_vs_aurora_cells'])} ring10 {fmt(d['gain_ring10_vs_aurora_cells'])}")
        P(f"         ring against pixel (gain of the ring): ring3 {fmt(d['gain_ring3_vs_pixel'])} ring10 {fmt(d['gain_ring10_vs_pixel'])} one3 {fmt(d['gain_one3_vs_pixel'])} "
          f"one10 {fmt(d['gain_one10_vs_pixel'])}; stations better than CAMS: pixel {d['share_stations_foehn_better_than_cams']} %, ring3 "
          f"{d['share_stations_ring3_better_than_cams']} %, ring10 {d['share_stations_ring10_better_than_cams']} % of {d['stations_30']}")
        w_ = d["estimates_share_within_0.5"]
        P(f"         the estimates of the same day (not a forecast), n={d['station_days_estimates']}: pixel {g_['ghapt']:.2f} | one px 3 km {g_['g_one3']:.2f} | ring 3 km "
          f"{g_['g_ring3']:.2f} | one px 10 km {g_['g_one10']:.2f} | ring 10 km {g_['g_ring10']:.2f}; within 0.5 ug m-3 of the measurement: pixel {w_['ghapt']} %, "
          f"one px 3 km {w_['g_one3']} %, one px 10 km {w_['g_one10']} % (Foehn's forecast at the pixel {d['foehn_share_within_0.5']} %)")

# sensitivity: bands 2-4 and 8-10 pixels, other minimum counts; the profile with distance
P("\n  sensitivity (all six networks, lead 1 and 3): gain vs CAMS / vs Aurora")
out["C"]["sensitivity"] = {}
allm = np.isin(src, NCODE)
for label, kA, kB, nA, nB, mn in [("rings 3 and 9 px, >= 1 pixel", "ring3", "ring10", n_el[:, K3], n_el[:, K10], 1),
                                  ("rings 3 and 9 px, >= 4 pixels", "ring3", "ring10", n_el[:, K3], n_el[:, K10], 4),
                                  ("rings 3 and 9 px, >= one sixth of the ring", "ring3", "ring10", n_el[:, K3] / 4, n_el[:, K10] / 12, 1),
                                  ("bands 2-4 and 8-10 px, >= 8 pixels", "band3", "band10", n_el[:, 2:5].sum(1), n_el[:, 8:11].sum(1), 8),
                                  ("bands 2-4 and 8-10 px, >= one sixth of the band", "band3", "band10", n_el[:, 2:5].sum(1) / 12, n_el[:, 8:11].sum(1) / 36, 1)]:
    av = (nA >= mn) & (nB >= mn)
    mk = PAP & av & allm & np.all([np.isfinite(V[k]) for k in ("foehn", kA, kB, "cams1", "aur1")], 0)
    d = {"stations": int(mk.any((0, 1)).sum())}
    for j, h in enumerate(LEADS):
        m = mk[used_d, j]
        a = {k: day_sums(E2r[k][used_d, j], m) for k in ("foehn", kA, kB, "cams1", "aur1")}
        d[h] = {"station_days": int(m.sum()), "rmse": {k: rmse(a[k]) for k in a},
                "pixel_vs_cams": gain(a["foehn"], a["cams1"], IDX), "3km_vs_cams": gain(a[kA], a["cams1"], IDX), "10km_vs_cams": gain(a[kB], a["cams1"], IDX),
                "pixel_vs_aurora": gain(a["foehn"], a["aur1"], IDX), "3km_vs_aurora": gain(a[kA], a["aur1"], IDX), "10km_vs_aurora": gain(a[kB], a["aur1"], IDX)}
        P(f"    {label:48s} {h} d stations {d['stations']} n={d[h]['station_days']}: vs CAMS pixel {fmt(d[h]['pixel_vs_cams'])} 3 km {fmt(d[h]['3km_vs_cams'])} "
          f"10 km {fmt(d[h]['10km_vs_cams'])} | vs Aurora pixel {fmt(d[h]['pixel_vs_aurora'])} 3 km {fmt(d[h]['3km_vs_aurora'])} 10 km {fmt(d[h]['10km_vs_aurora'])}")
    out["C"]["sensitivity"][label] = d

P("\n  profile with distance (stations with at least 4 pixels in every ring 1..12; RMSE of the ring mean; lead 1 day)")
avp = np.all(n_el[:, 1:13] >= MINPIX, 1)
out["C"]["profile"] = {}
for sn in ("all six networks", "China", "USA", "Europe", "Korea", "Japan", "India", "matched stations (all networks)", "not matched stations (all networks)"):
    sm = CSUB[sn]
    for j, h in enumerate(LEADS):
        mk = PAP[:, j] & avp & sm & np.isfinite(V["foehn"][:, j]) & np.isfinite(V["cams1"][:, j]) & np.all([np.isfinite(p[:, j]) for p in PROF_F], 0)
        mg = mk & np.isfinite(V["ghapt"][:, j]) & np.all([np.isfinite(p[:, j]) for p in PROF_G], 0)
        y = Y[:, j]
        prof = [float(np.sqrt(((V["foehn"][:, j] - y)[mk] ** 2).mean()))] + [float(np.sqrt(((p[:, j] - y)[mk] ** 2).mean())) for p in PROF_F]
        profg = [float(np.sqrt(((V["ghapt"][:, j] - y)[mg] ** 2).mean()))] + [float(np.sqrt(((p[:, j] - y)[mg] ** 2).mean())) for p in PROF_G]
        cam = float(np.sqrt(((V["cams1"][:, j] - y)[mk] ** 2).mean())); au = float(np.sqrt(((V["aur1"][:, j] - y)[mk] ** 2).mean()))
        out["C"]["profile"][f"{sn} | {h} d"] = {"stations": int(mk.any(0).sum()), "station_days": int(mk.sum()), "foehn_k0_to_12": prof,
                                                 "estimates_same_day_k0_to_12": profg, "cams": cam, "aurora": au}
        if h == 1:
            P(f"    {sn:36s} st {int(mk.any(0).sum()):5d} Foehn k=0..12: " + " ".join(f"{v:.2f}" for v in prof) + f" | CAMS {cam:.2f} Aurora {au:.2f}")
            P(f"    {'':36s}    estimates k=0..12: " + " ".join(f"{v:.2f}" for v in profg))

# =====================================================================================================================
# D. the model of Aurora's protocol at the stations
# =====================================================================================================================
fc = np.load(DATA / "foehn_cams_station_day.npz")
pm, fsrc, fobs = fc["paper"], fc["src"], fc["obs"]
ndf = pm.shape[0]
IDXF = block_index(ndf)
E2f = {k: (fc[k].astype(np.float64) - fobs) ** 2 for k in ("fc", "fc0", "aur", "cams")}
P(f"\n# D. the model of Aurora's protocol at the stations: {ndf} initialisations (00 UTC, 1 May - 30 Nov 2022), ensemble mean of {int(fc['members'])} members")
out["D"] = {"initialisations": ndf, "members": int(fc["members"]), "groups": {}}
for name, gl in GROUPS.items():
    stm = np.isin(fsrc, [NCODE[g] for g in gl])
    row = {"stations": int((pm & stm).any((0, 1)).sum()), "leads": {}}
    for h in range(4):
        m = pm[:, h] & stm
        a = {k: day_sums(E2f[k][:, h], m) for k in E2f}
        row["leads"][h + 1] = {"station_days": int(m.sum()), "rmse": {k: rmse(a[k]) for k in a},
                               "gain_vs_cams": gain(a["fc"], a["cams"], IDXF), "gain_vs_aurora": gain(a["fc"], a["aur"], IDXF),
                               "member_gain_vs_cams": gain(a["fc0"], a["cams"], IDXF), "member_gain_vs_aurora": gain(a["fc0"], a["aur"], IDXF)}
    out["D"]["groups"][name] = row
    P(f"\n  {name}: stations {row['stations']}")
    for h in range(4):
        d = row["leads"][h + 1]; r_ = d["rmse"]
        P(f"    {h + 1} d n={d['station_days']:>8d} RMSE model {r_['fc']:6.2f} (one member {r_['fc0']:6.2f})  CAMS {r_['cams']:6.2f}  Aurora {r_['aur']:6.2f} | gain vs CAMS "
          f"{fmt(d['gain_vs_cams'])}  vs Aurora {fmt(d['gain_vs_aurora'])} | one member: vs CAMS {fmt(d['member_gain_vs_cams'])} vs Aurora {fmt(d['member_gain_vs_aurora'])}")

OUT.mkdir(exist_ok=True)
tag = "" if R == 10000 else f"_R{R}"
json.dump(out, open(OUT / f"numbers{tag}.json", "w"), indent=1)
open(OUT / f"tables{tag}.txt", "w").write("\n".join(lines) + "\n")
print("saved", OUT / f"numbers{tag}.json", OUT / f"tables{tag}.txt")
