#!/usr/bin/env python3
"""Emission and transport gains at equal training, recomputed from the 2022 evaluation files (read only).

Four models, each evaluated with the same command (4 members, --steps 8 --refine-tail 3 = 12 Euler steps, every second
start day of 2022, leads 1-7; eval_global_green.py):
  A     base model: transport inputs, no anthropogenic emission channels        runs/..._whres_ghres/eval_final
  C     same network trained without the transport inputs                       runs/..._whres_notransport/eval_final
  e02   A fine-tuned with the six emission channels (x 10), epoch index 2 of 12 runs/..._ghres_ftemis12/eval_final_e02
  ctrl  the same fine-tuning with the emission channels x 0, epoch index 2      runs/..._ghres_ftemis12ctrl0/eval_final_e02
Comparisons (gain = 100 (1 - score_model / score_reference), positive = model better):
  emissions          e02  against ctrl   equal training: the only difference is the emission information
  transport          A    against C      equal training: the only difference is the transport inputs
  fine_tuning_alone  ctrl against A      what the three extra epochs do without any emission information
  submitted          e02  against A      the comparison of the submitted Extended Data Fig. 9 (mixes the two)
Two statistics, both against the 1 km estimates (GHAP) on identical pixels / cells and forecasts:
  km1   sums over the 1 km land pixels of each region with a valid target (array `recs` of the evaluation files): RMSE of
        the ensemble mean, fair CRPS of the four members, spread-skill ratio sqrt((M+1)/M mean variance) / RMSE. This is
        the statistic of Fig. 5j (figdata_transport_gain.py) and of control_report.py.
  c25   the 0.25 degree cell means of the ensemble mean and of GHAP stored for every forecast (`map25`, `map25_ghap`): RMSE
        of the ensemble mean over the land cells of each box, each cell once. The members are not stored at 0.25 degrees, so
        the CRPS exists only as km1. Boxes: the nine regions, "All regions" (their sum, as km1: overlaps counted twice),
        "All land" (every valid cell between 84N and 60S once), North China Plain, Indo-Gangetic Plain.
Intervals: 95 %, paired moving-block bootstrap over start days (the same resampled start days for the two models; blocks
of 7 consecutive start days, i.e. 14 calendar days since every second day is a start day; 10,000 resamples; generator
seed 0 as in figdata_transport_gain.py, which makes the stored files reproducible to the last digit). Sensitivity: two
more seeds and blocks of 4 and 14 start days.
Extra rows: leads pooled ("1-7", "1-3", "5-7": one RMSE / CRPS over all forecasts of these leads) and, for the emissions,
the change of the gain between leads 1-3 and leads 5-7 with its interval ("holds" when it includes zero).

    python recompute.py            -> out/gains.csv, scores.csv, sensitivity.csv, trend.csv, check_vs_author.txt, summary.json
    python recompute.py --km1-only    (records only, a few seconds; no 0.25 degree maps)
"""
import csv
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
RUNS = "/scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km/runs/green_flow_shift25_global_L7_noant_whres"
EV = {"A": "_ghres/eval_final", "C": "_notransport/eval_final", "e02": "_ghres_ftemis12/eval_final_e02",
      "ctrl": "_ghres_ftemis12ctrl0/eval_final_e02"}
MODELS = list(EV)
COMP = [("emissions", "ctrl", "e02"), ("transport", "C", "A"), ("fine_tuning_alone", "A", "ctrl"), ("submitted", "A", "e02")]
AUTHOR = Path("/scratch/project_462001804/ammar/eccv/cran-pm/results/emission_by_lead")
AUTHOR_LABEL = {"emissions": "emissions (e02 vs control)", "transport": "transport (A vs C)",
                "fine_tuning_alone": "fine-tuning alone (control vs A)"}
PLAINS = {"North China Plain": (41.0, 34.0, 112.0, 120.0), "Indo-Gangetic Plain": (31.0, 24.0, 74.0, 88.0)}   # N, S, W, E
REGIONS = {"Europe": (72, 35, -25, 45), "North America": (72, 15, -170, -50), "South America": (15, -56, -95, -30),
           "Africa": (37, -35, -20, 52), "Middle East": (42, 12, 35, 63), "South Asia": (37, 5, 60, 97),
           "East Asia": (54, 18, 97, 146), "Southeast Asia": (18, -11, 92, 142), "Oceania": (-10, -48, 110, 180)}   # eval_global_green.py
POOLS = {"1-7": range(7), "1-3": range(0, 3), "5-7": range(4, 7)}
NB, BLOCK, SEED = 10000, 7, 0


def block_index(nd, block=BLOCK, n=NB, seed=SEED):
    """resampled start-day indices (n, nd): moving blocks of `block` consecutive start days, joined and cut to nd
    (the generator and the arithmetic of figdata_transport_gain.py and control_report.py)"""
    rng = np.random.default_rng(seed)
    return (rng.integers(0, nd - block + 1, size=(n, -(-nd // block)))[:, :, None] + np.arange(block)).reshape(n, -1)[:, :nd]


def block_counts(nd, block=BLOCK, n=NB, seed=SEED):
    """the same resamples as a matrix of counts (n, nd): how many times each start day is drawn. Every score here is a
    ratio of sums over start days, so the resampled sums are counts @ daily sums (identical to indexing, and fast)."""
    idx = block_index(nd, block, n, seed)
    W = np.zeros((n, nd))
    np.add.at(W, (np.arange(n)[:, None], idx), 1.0)
    return W


def rmse(X):
    """X (..., days, [n, sse, crps, ...]) -> RMSE over the days"""
    return np.sqrt(X[..., 1].sum(-1) / X[..., 0].sum(-1))


def crps(X):
    return X[..., 2].sum(-1) / X[..., 0].sum(-1)


SCORE = {"rmse": rmse, "crps": crps}


def boot(X, metric, W):
    """the score of every resample: X (days, stats), W (resamples, days)"""
    S = W @ X[:, :3]
    return np.sqrt(S[:, 1] / S[:, 0]) if metric == "rmse" else S[:, 2] / S[:, 0]


def gain(R_, A_, metric, W):
    """gain of A_ over R_ (days, stats), with its bootstrap interval (the same resampled days for both: paired)"""
    f = SCORE[metric]
    pt = 100 * (1 - f(A_) / f(R_))
    bs = 100 * (1 - boot(A_, metric, W) / boot(R_, metric, W))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(pt), float(lo), float(hi), bs


# ------------------------------------------------------------------------------------------------ 1 km records
def load_recs(ev):
    recs, cols, regions, info = [], None, None, None
    for p in sorted(glob.glob(f"{RUNS}{ev}/part_*.npz")):
        z = np.load(p)
        recs.append(z["recs"]); cols = [str(c) for c in z["rec_cols"]]; regions = [str(r) for r in z["regions"]]
        info = [str(v) for v in z["info"]]
    R = np.concatenate(recs)
    ix = [cols.index(s) for s in ("n", "sse", "crps", "var", "sum_err", "sum_t")]
    return {(int(r[0]), int(r[1]), int(r[2])): r[ix] for r in R}, regions, info


def km1():
    data = {k: load_recs(v) for k, v in EV.items()}
    regions = data["A"][1]
    assert all(d[1] == regions for d in data.values()) and regions == list(REGIONS)
    names = regions + ["All regions"]
    for k, d in data.items():
        print(f"  {k:5s} {d[2][0]} | {d[2][1]} | members {d[2][5]} steps {d[2][6]} | {len(d[0])} records", flush=True)
    keys = sorted(set.intersection(*[set(d[0]) for d in data.values()]))
    days = sorted({k[0] for k in keys}); di = {d: i for i, d in enumerate(days)}
    T = np.zeros((len(MODELS), len(names), 7, len(days), 6))
    nmax = 0.0
    for (d, h, g) in keys:
        n0 = data["A"][0][(d, h, g)][0]
        for m, k in enumerate(MODELS):
            x = data[k][0][(d, h, g)]
            nmax = max(nmax, abs(x[0] - n0))                         # identical pixels in every model
            T[m, g, h - 1, di[d]] += x; T[m, -1, h - 1, di[d]] += x
    nfc = len({(d, h) for d, h, _ in keys})
    print(f"km1: {len(keys)} common records (start day, lead, region) = {nfc} forecasts, {len(days)} start days "
          f"(day of year {days[0]}..{days[-1]}); largest difference of pixel count between models: {nmax:g}", flush=True)
    members = {k: int(d[2][5]) for k, d in data.items()}
    return T, names, days, members, nfc


# ------------------------------------------------------------------------------------------------ 0.25 degree maps
def c25(days_ref):
    NR, NC = 576, 1440
    lat = 84.0 - 0.25 * np.arange(NR); lon = -180.0 + 0.25 * np.arange(NC)     # cell k is centred on fine row 25 k (cell_mean)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    boxes = dict(REGIONS); boxes.update(PLAINS)
    masks = {k: (LA <= b[0]) & (LA > b[1]) & (LO >= b[2]) & (LO < b[3]) for k, b in boxes.items()}
    names = list(REGIONS) + ["All regions", "All land"] + list(PLAINS)
    index = {}
    for k, ev in EV.items():
        index[k] = {}
        for p in sorted(glob.glob(f"{RUNS}{ev}/part_*.npz")):
            for n, (d, h, _) in enumerate(np.load(p)["keys"]):
                index[k][(int(d), int(h))] = (p, n)
    common = sorted(set.intersection(*[set(v) for v in index.values()]))
    days = sorted({d for d, _ in common}); di = {d: i for i, d in enumerate(days)}
    assert days == days_ref, "start days of the maps differ from those of the records"
    T = np.zeros((len(MODELS), len(names), 7, len(days), 6))             # n, sse, (unused), sum sd^2, sum err, sum target
    cache = {}

    def arrays(path, with_ghap):
        if path not in cache:
            if len(cache) > 8:
                cache.pop(next(iter(cache)))
            z = np.load(path)
            cache[path] = (z["map25"], z["map25_sd"], z["map25_ghap"] if with_ghap else None)
        return cache[path]

    by_part = {}
    for (d, h) in common:
        by_part.setdefault(index["A"][(d, h)][0], []).append((d, h))
    t0 = time.time(); gdiff = 0.0
    for ip, (pA, fcs) in enumerate(by_part.items()):
        for (d, h) in fcs:
            F, S = [], []
            y = None
            for k in MODELS:
                p, n = index[k][(d, h)]
                m25, sd25, g25 = arrays(p, True)
                F.append(m25[n].astype(np.float32)); S.append(sd25[n].astype(np.float32))
                g = g25[n].astype(np.float32)
                if y is None:
                    y = g
                else:
                    both = np.isfinite(y) & np.isfinite(g)
                    assert (np.isfinite(y) == np.isfinite(g)).all()
                    gdiff = max(gdiff, float(np.abs(y[both] - g[both]).max()))
            ok = np.isfinite(y) & (y > 0)
            for f in F:
                ok &= np.isfinite(f)
            for b, nm in enumerate(names):
                if nm == "All regions":
                    continue
                mk = ok if nm == "All land" else (ok & masks[nm])
                if not mk.any():
                    continue
                yy = y[mk]
                for m in range(len(MODELS)):
                    e = F[m][mk] - yy
                    row = [mk.sum(), float((e.astype(np.float64) ** 2).sum()), 0.0, float((S[m][mk].astype(np.float64) ** 2).sum()),
                           float(e.sum(dtype=np.float64)), float(yy.sum(dtype=np.float64))]
                    T[m, b, h - 1, di[d]] += row
                    if nm in REGIONS:
                        T[m, names.index("All regions"), h - 1, di[d]] += row
        if ip % 16 == 0:
            print(f"  c25: part {ip + 1}/{len(by_part)} {time.time() - t0:.0f}s", flush=True)
    print(f"c25: {len(common)} common forecasts, {len(days)} start days; largest difference of the GHAP cell means between the "
          f"evaluation files of two models: {gdiff:g}", flush=True)
    return T, names


# ------------------------------------------------------------------------------------------------ tables
def tables(stat, T, names, days, members, metrics, grows, srows, trows, sens):
    nd = len(days); idx = block_counts(nd)
    alt_idx = {("seed", s): block_counts(nd, seed=s) for s in (1, 2)}
    alt_idx.update({("block", b): block_counts(nd, block=b) for b in (4, 14)})
    for g, nm in enumerate(names):
        for h in list(range(7)) + list(POOLS):
            X = T[:, g, h] if isinstance(h, int) else T[:, g, list(POOLS[h])].sum(1)       # (models, days, stats)
            lead = h + 1 if isinstance(h, int) else h
            if X[0, :, 0].sum() == 0:
                continue
            for m, k in enumerate(MODELS):
                M = members[k]
                r = float(rmse(X[m]))
                ss = float(np.sqrt((M + 1) / M * X[m, :, 3].sum() / X[m, :, 0].sum()) / r) if stat == "km1" else float("nan")
                srows.append([stat, nm, lead, k, round(r, 4), round(float(crps(X[m])), 4) if stat == "km1" else "",
                              round(ss, 4) if stat == "km1" else "", round(float(X[m, :, 4].sum() / X[m, :, 0].sum()), 4),
                              round(float(X[m, :, 5].sum() / X[m, :, 0].sum()), 4), int(X[m, :, 0].sum()), int((X[m, :, 0] > 0).sum())])
            for label, ref, alt in COMP:
                R_, A_ = X[MODELS.index(ref)], X[MODELS.index(alt)]
                for metric in metrics:
                    pt, lo, hi, _ = gain(R_, A_, metric, idx)
                    grows.append([stat, nm, lead, label, ref, alt, metric, round(pt, 3), round(lo, 3), round(hi, 3),
                                  int(lo > 0) - int(hi < 0)])
                    if label in ("emissions", "transport"):
                        for (kind, v), ix in alt_idx.items():
                            p2, l2, h2, _ = gain(R_, A_, metric, ix)
                            sens.append([stat, nm, lead, label, metric, kind, v, round(l2, 3), round(h2, 3), int(l2 > 0) - int(h2 < 0)])
        # does the emission gain hold? gain over leads 5-7 minus gain over leads 1-3, paired bootstrap
        for label, ref, alt in COMP[:2]:
            for metric in metrics:
                f = SCORE[metric]
                e, l = [T[:, g, list(POOLS[k])].sum(1) for k in ("1-3", "5-7")]
                R1, A1, R2, A2 = e[MODELS.index(ref)], e[MODELS.index(alt)], l[MODELS.index(ref)], l[MODELS.index(alt)]
                if R1[:, 0].sum() == 0:
                    continue
                g1 = 100 * (1 - f(A1) / f(R1)); g2 = 100 * (1 - f(A2) / f(R2))
                b1 = 100 * (1 - boot(A1, metric, idx) / boot(R1, metric, idx)); b2 = 100 * (1 - boot(A2, metric, idx) / boot(R2, metric, idx))
                lo, hi = np.percentile(b2 - b1, [2.5, 97.5])
                trows.append([stat, nm, label, metric, round(float(g1), 3), round(float(g2), 3), round(float(g2 - g1), 3),
                              round(float(lo), 3), round(float(hi), 3), "holds" if lo <= 0 <= hi else ("grows" if lo > 0 else "fades")])


def check_author(grows, srows, f):
    G = {(r[0], r[1], r[2], r[3], r[6]): r[7:10] for r in grows}
    S = {(r[0], r[1], r[2], r[3]): r for r in srows}
    worst = 0.0; n = 0
    for r in csv.DictReader(open(AUTHOR / "control_gains.csv")):
        lab = {v: k for k, v in AUTHOR_LABEL.items()}[r["comparison"]]
        mine = G[("km1", r["region"], int(r["lead_days"]), lab, r["metric"])]
        d = max(abs(mine[0] - float(r["gain_pct"])), abs(mine[1] - float(r["ci95_lo"])), abs(mine[2] - float(r["ci95_hi"])))
        worst = max(worst, d); n += 1
    f.write(f"control_gains.csv (stored, 13 Oct): {n} rows (gain, lower, upper); largest absolute difference with this "
            f"recomputation: {worst:.4f} percentage points (the stored file is rounded to 0.01)\n")
    worst = [0.0, 0.0, 0.0]; n = 0
    for r in csv.DictReader(open(AUTHOR / "control_scores.csv")):
        mine = S[("km1", r["region"], int(r["lead_days"]), r["model"])]
        for j, (col, k) in enumerate((("rmse", 4), ("crps", 5), ("spread_skill", 6))):
            worst[j] = max(worst[j], abs(float(mine[k]) - float(r[col])))
        n += 1
    f.write(f"control_scores.csv: {n} rows; largest absolute difference: RMSE {worst[0]:.4f}, CRPS {worst[1]:.5f}, "
            f"spread-skill {worst[2]:.4f} (the stored file rounded to 0.001, 0.0001, 0.001)\n")
    if any(r[0] == "c25" for r in grows):
        worst = 0.0; n = 0
        lab = {("e02", "ctrl"): "emissions", ("ctrl", "A"): "fine_tuning_alone", ("e02", "A"): "submitted"}
        for r in csv.DictReader(open(AUTHOR / "pixel_regions_ctrl.csv")):
            if r["kind"] != "gain":
                continue
            mine = G[("c25", r["box"], int(r["lead_days"]), lab[(r["model"], r["reference"])], "rmse")]
            lo, hi = (float(v) for v in r["ci95"].split(";"))
            worst = max(worst, abs(mine[0] - float(r["rmse_gain_pct"])), abs(mine[1] - lo), abs(mine[2] - hi)); n += 1
        f.write(f"pixel_regions_ctrl.csv (two plains, 0.25 degree cells; the stored bootstrap resamples the forecasts common to A, "
                f"e02 and ctrl, this one those common to the four models: same start days): {n} rows; largest absolute "
                f"difference: {worst:.4f} percentage points\n")


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    print("evaluation files:", flush=True)
    T1, names1, days, members, nfc = km1()
    grows, srows, trows, sens = [], [], [], []
    tables("km1", T1, names1, days, members, ("rmse", "crps"), grows, srows, trows, sens)
    if "--km1-only" not in sys.argv[1:]:
        T2, names2 = c25(days)
        tables("c25", T2, names2, days, members, ("rmse",), grows, srows, trows, sens)
        np.savez_compressed(OUT / "sums_c25.npz", T=T2, names=np.array(names2), models=np.array(MODELS), days=np.array(days),
                            stats=np.array(["n", "sse", "unused", "sum_sd2", "sum_err", "sum_target"]))
    np.savez_compressed(OUT / "sums_km1.npz", T=T1, names=np.array(names1), models=np.array(MODELS), days=np.array(days),
                        stats=np.array(["n", "sse", "crps", "var", "sum_err", "sum_target"]))
    with open(OUT / "gains.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["statistic", "region", "lead_days", "comparison", "reference", "model", "metric", "gain_pct", "ci95_lo", "ci95_hi",
                    "sign_if_interval_excludes_zero"])
        w.writerows(grows)
    with open(OUT / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["statistic", "region", "lead_days", "model", "rmse", "crps", "spread_skill", "bias", "mean_target", "n_pixel_or_cell_days",
                    "n_start_days"])
        w.writerows(srows)
    with open(OUT / "sensitivity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["statistic", "region", "lead_days", "comparison", "metric", "changed", "value", "ci95_lo", "ci95_hi",
                    "sign_if_interval_excludes_zero"])
        w.writerows(sens)
    with open(OUT / "trend.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["statistic", "region", "comparison", "metric", "gain_leads_1-3", "gain_leads_5-7", "difference", "ci95_lo", "ci95_hi",
                    "verdict"])
        w.writerows(trows)
    with open(OUT / "check_vs_author.txt", "w") as f:
        check_author(grows, srows, f)
    print(open(OUT / "check_vs_author.txt").read(), flush=True)
    # how many conclusions (sign of an interval that excludes zero) change with the seed or the block length
    base = {(r[0], r[1], r[2], r[3], r[6]): r[10] for r in grows}
    flips = [s for s in sens if s[9] != base[(s[0], s[1], s[2], s[3], s[4])]]
    json.dump(dict(forecasts=nfc, start_days=len(days), first_day=days[0], last_day=days[-1], members=members,
                   bootstrap=dict(resamples=NB, block_start_days=BLOCK, seed=SEED),
                   sensitivity_rows=len(sens), sensitivity_sign_changes=len(flips),
                   sensitivity_changes=[dict(statistic=s[0], region=s[1], lead=s[2], comparison=s[3], metric=s[4], changed=s[5], value=s[6],
                                             lo=s[7], hi=s[8]) for s in flips]),
              open(OUT / "summary.json", "w"), indent=1)
    print(f"sensitivity: {len(flips)} of {len(sens)} interval verdicts change with another seed or block length", flush=True)
    print(f"done in {time.time() - t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
