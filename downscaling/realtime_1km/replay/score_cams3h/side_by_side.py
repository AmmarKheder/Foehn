#!/usr/bin/env python3
"""cams3h: the real-time scores against the CAMS forecast with its two daily means, side by side (evaluation only).

  old = three instants (1/4, 1/2, 1/4 at 00, 12, 24 UTC): the arm `cams` of score/ = `cams_old` here
  new = the paper's rule, the eight 3-hourly steps 00 .. 21 UTC of the valid day: the arm `cams` of score_cams3h/

RULE: every score that the scorer's tables contain is READ from the two runs of the unmodified scorer
(score/tables, score_cams3h/tables: same station-days, same bootstrap draws). What the tables lack is computed here
WITH THE SCORER'S OWN CODE (sc_lib.Data, sc_lib.Sample, score.contrast, imported read-only, no bytecode) on the station-days
where the observation and km, rt04, cams, clim, bcsd AND cams_old are finite, issue days 2026-09-06..2026-11-11:
  * the RMSE of the two CAMS readings with intervals and their ratio (pooled, mean over regions, by lead day and region);
  * the 0.4 degree input against each CAMS reading;
  * the share of 2-degree cells (>= 100 station-days), of monitors (>= 30) and of issue days where the 1 km forecast has the
    lower RMSE (definitions of paper/natcomms/figures_new/fig_rt_skill/rt_skill_data.py, copied);
  * sensitivity of the daily-mean rule (trapezoid 00 .. 24 UTC; steps 03 .. 24 UTC; float32 instead of float16 steps).
Each recomputation is checked against the tables (key "checks").
    python -I -B side_by_side.py [--B 2000]   -> side_by_side.json, side_by_side.md (this directory)"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
R = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay")
OLD, NEW = R / "score", R / "score_cams3h"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(R / "cams3h/code"))
sys.path.insert(0, str(OLD))
import numpy as np  # noqa: E402
import safe_io  # noqa: E402
import sc_lib as L  # noqa: E402   (the scorer's library, read-only)
import score as SC  # noqa: E402   (the scorer's tables code; main() is guarded)

CLEAN = ("2026-09-06", "2026-11-11")
SET = "with the CAMS forecast"
MIN_CELL, MIN_MON = 100, 30
FIG = Path("/scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/figures_new/fig_rt_skill/numbers.json")


def tab(run, name):
    return json.loads((run / "tables" / f"{name}.json").read_text())


def ec(d):
    if d is None:
        return None
    o = dict(est=float(d["est"]), ci=[float(d["ci"][0]), float(d["ci"][1])])
    if "n" in d:
        o["n"] = int(d["n"])
    return o


def f1(d, nd=1):
    return "n/a" if d is None else f"{d['est']:+.{nd}f} [{d['ci'][0]:+.{nd}f}, {d['ci'][1]:+.{nd}f}]"


def sdays(desc):
    """'1,819,647' from a sample description of the scorer"""
    import re
    m = re.search(r"([0-9,]+) station-days", desc)
    return m.group(1) if m else ""


def f2(d, nd=2):
    return "n/a" if d is None else f"{d['est']:.{nd}f} [{d['ci'][0]:.{nd}f}, {d['ci'][1]:.{nd}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    t0 = time.time()
    out, checks, md = {}, {}, []

    def check(name, mine, theirs, tol=1e-9):
        ok = bool(np.allclose(np.asarray(mine, float), np.asarray(theirs, float), rtol=0, atol=tol))
        checks[name] = dict(recomputed_here=np.asarray(mine, float).round(6).tolist(), reference=np.asarray(theirs, float).round(6).tolist(), identical=ok)
        print(f"   check {name}: {'identical' if ok else 'DIFFERENT'}", flush=True)
        return ok

    # ------------------------------------------------------------------ the two station-day tables hold the same rows
    zo = np.load(OLD / "work" / "pairs.npz")
    zn = np.load(NEW / "work" / "pairs.npz")
    same_rows = all(np.array_equal(zo[k], zn[k]) for k in ("issue", "lead", "mon", "vday", "src", "issue_days", "id"))
    same_obs = bool(np.array_equal(zo["obs"], zn["obs"], equal_nan=True))
    same_arm = {k: bool(np.array_equal(zo[f"m_{k}"], zn[f"m_{k}"], equal_nan=True)) for k in ("km", "rt04", "clim", "bcsd", "km_ring10", "km_cell_lin", "an")}
    old_cams_kept = bool(np.array_equal(zo["m_cams"], zn["m_cams_old"], equal_nan=True))
    recomputed_same = bool(np.array_equal(zn["m_cams_old"], zn["m_cams3i"], equal_nan=True))
    d3 = np.abs(zn["m_cams3i"].astype(np.float64) - zo["m_cams"].astype(np.float64))
    out["tables"] = dict(rows_old=int(zo["obs"].size), rows_new=int(zn["obs"].size), same_rows_issue_lead_monitor=bool(same_rows), same_observations=same_obs,
                         other_arms_identical=same_arm, existing_cams_arm_carried_as_cams_old_identical=old_cams_kept,
                         cams3i_identical_to_existing_cams=recomputed_same, cams3i_vs_existing_cams_max_abs_diff=float(np.nanmax(d3)),
                         cams3i_vs_existing_cams_rows_compared=int(np.isfinite(d3).sum()))
    print(f"[{time.time() - t0:5.1f} s] tables: {out['tables']}", flush=True)
    del zo, zn

    D = L.Data(str(NEW / "work" / "pairs.npz"))
    base = ["rt04", "cams", "clim", "bcsd"]
    B5 = ["km"] + base
    m_scorer = D.finite(B5) & D.issues(*CLEAN)
    m_both = D.finite(B5 + ["cams_old"]) & D.issues(*CLEAN)
    smp = L.Sample(D, m_both, B=a.B)
    T01o, T01n = tab(OLD, "T01_samples"), tab(NEW, "T01_samples")
    out["sample"] = dict(description=smp.describe(), station_days=int(smp.n), station_days_scorer_mask_new_run=int(m_scorer.sum()),
                         station_days_old_run=int(T01o["samples"][f"clean|{SET}"]["station_days"]), station_days_new_run=int(T01n["samples"][f"clean|{SET}"]["station_days"]),
                         lost_by_requiring_both_cams_arms=int(m_scorer.sum() - m_both.sum()), monitors=int(smp.n_mon), cells=int(smp.C), week_blocks=int(smp.W),
                         issue_days=int(smp.n_issue), time_blocks=smp.blocks,
                         all_53_issue_days=dict(station_days_scorer_mask=int((D.finite(B5) & D.issues("2026-09-01", "2026-11-11")).sum()),
                                                station_days_both_arms=int((D.finite(B5 + ["cams_old"]) & D.issues("2026-09-01", "2026-11-11")).sum())))
    print(f"[{time.time() - t0:5.1f} s] {smp.describe()}", flush=True)
    T03o, T03n = tab(OLD, "T03_contrasts")[SET]["contrasts"], tab(NEW, "T03_contrasts")[SET]["contrasts"]
    T02o, T02n = tab(OLD, "T02_scores")[SET], tab(NEW, "T02_scores")[SET]

    # ------------------------------------------------------------------ contrasts computed here on ONE sample (both arms)
    c_new = SC.contrast(smp, "km", "cams", "rmse")
    c_old = SC.contrast(smp, "km", "cams_old", "rmse")
    check("1 km vs CAMS (3-hourly mean), pooled est + interval, against score_cams3h T03", [c_new["all"]["all"]["est"]] + c_new["all"]["all"]["ci"],
          [T03n["km_vs_cams|rmse"]["all"]["all"]["est"]] + T03n["km_vs_cams|rmse"]["all"]["all"]["ci"])
    check("1 km vs CAMS (three instants), pooled est + interval, against score T03", [c_old["all"]["all"]["est"]] + c_old["all"]["all"]["ci"],
          [T03o["km_vs_cams|rmse"]["all"]["all"]["est"]] + T03o["km_vs_cams|rmse"]["all"]["all"]["ci"])
    check("1 km vs CAMS (3-hourly mean), mean over regions, against score_cams3h T03", [c_new["all"]["regmean"]["est"]] + c_new["all"]["regmean"]["ci"],
          [T03n["km_vs_cams|rmse"]["all"]["regmean"]["est"]] + T03n["km_vs_cams|rmse"]["all"]["regmean"]["ci"])
    check("1 km vs CAMS (three instants), mean over regions, against score T03", [c_old["all"]["regmean"]["est"]] + c_old["all"]["regmean"]["ci"],
          [T03o["km_vs_cams|rmse"]["all"]["regmean"]["est"]] + T03o["km_vs_cams|rmse"]["all"]["regmean"]["ci"])
    for b in ("rt04", "clim", "bcsd"):                                   # arms that do not involve CAMS: the two runs must agree to the last digit
        check(f"1 km vs {b}: the two runs of the scorer agree (pooled est + interval)", [T03n[f"km_vs_{b}|rmse"]["all"]["all"]["est"]] + T03n[f"km_vs_{b}|rmse"]["all"]["all"]["ci"],
              [T03o[f"km_vs_{b}|rmse"]["all"]["all"]["est"]] + T03o[f"km_vs_{b}|rmse"]["all"]["all"]["ci"])
    rows = smp.rows
    obs = D.obs[rows]
    se = {x: (D.m[x][rows] - obs) ** 2 for x in ["km", "rt04", "cams", "cams_old", "clim", "bcsd", "cams3i", "cams3h", "cams3h_f32", "cams3h_trap", "cams3h_0324"]}
    check("pooled RMSE km, CAMS three instants (score T02)", [np.sqrt(se["km"].mean()), np.sqrt(se["cams_old"].mean())], [T02o["km"]["all|all"]["rmse"], T02o["cams"]["all|all"]["rmse"]])
    check("pooled RMSE km, CAMS 3-hourly (score_cams3h T02)", [np.sqrt(se["km"].mean()), np.sqrt(se["cams"].mean())], [T02n["km"]["all|all"]["rmse"], T02n["cams"]["all|all"]["rmse"]])

    cnt = smp.q("n", lambda: np.ones(smp.n))
    vn, wn = L.views(cnt[0]), L.views(cnt[1])

    def rmse_ci(x):
        q = smp.q(f"rmse_{x}", lambda x=x: se[x])
        va, wa = L.views(q[0]), L.views(q[1])
        o = {}
        for key, nm in [(("all", "all"), "all")] + [((k, "all"), f"lead {k}") for k in range(1, 6)] + [(("all", g), L.REGIONS[g]) for g in range(8)]:
            o[nm] = dict(est=float(np.sqrt(va[key] / vn[key])), ci=L.pct(np.sqrt(wa[key] / wn[key])), n=int(vn[key]))
        return o

    def flat(c):
        """contrast dict -> {'all', 'regmean', regions..., 'lead 1'.. (pooled), 'lead 1 regmean'..}"""
        o = {g: ec(c["all"].get(g)) for g in ["all", "regmean"] + L.REGIONS}
        for k in range(1, 6):
            o[f"lead {k}"] = ec(c[str(k)]["all"])
            o[f"lead {k} regmean"] = ec(c[str(k)].get("regmean"))
        o["regions_lower"] = int(c["all"]["regmean"]["n_negative"])
        o["regions_interval_below_zero"] = int(c["all"]["regmean"]["n_ci_below_zero"])
        o["regions_interval_above_zero"] = int(c["all"]["regmean"]["n_ci_above_zero"])
        return o

    out["rmse"] = {x: rmse_ci(x) for x in ("km", "rt04", "cams_old", "cams", "clim", "bcsd")}
    out["bias_r_mae"] = dict(cams_old={k: T02o["cams"]["all|all"][k] for k in ("rmse", "mae", "bias", "r", "fc_mean", "obs_mean")},
                             cams={k: T02n["cams"]["all|all"][k] for k in ("rmse", "mae", "bias", "r", "fc_mean", "obs_mean")},
                             by_region={g: dict(cams_old={k: T02o["cams"][f"all|{g}"][k] for k in ("rmse", "bias", "r")},
                                                cams={k: T02n["cams"][f"all|{g}"][k] for k in ("rmse", "bias", "r")}) for g in L.REGIONS})
    out["km_vs_cams"] = dict(old=flat(c_old), new=flat(c_new))
    hand = SC.contrast(smp, "cams_old", "cams", "rmse")                   # 100 x (RMSE three instants / RMSE 3-hourly - 1)
    out["handicap_three_instants_vs_3hourly_pct"] = flat(hand)
    hand_mae = SC.contrast(smp, "cams_old", "cams", "mae")
    out["handicap_mae_pct"] = dict(all=ec(hand_mae["all"]["all"]), regmean=ec(hand_mae["all"]["regmean"]))
    out["rt04_vs_cams"] = dict(old=flat(SC.contrast(smp, "rt04", "cams_old", "rmse")), new=flat(SC.contrast(smp, "rt04", "cams", "rmse")))
    out["clim_vs_cams"] = dict(old=flat(SC.contrast(smp, "clim", "cams_old", "rmse")), new=flat(SC.contrast(smp, "clim", "cams", "rmse")))
    out["bcsd_vs_cams"] = dict(old=flat(SC.contrast(smp, "bcsd", "cams_old", "rmse")), new=flat(SC.contrast(smp, "bcsd", "cams", "rmse")))
    cr_new, cr_old = SC.contrast(smp, "km", "cams", "crps"), SC.contrast(smp, "km", "cams_old", "crps")
    out["km_fair_crps_vs_cams_abs_error"] = dict(old=flat(cr_old), new=flat(cr_new), fair_crps_km=float(D.crps("km")[rows].mean()),
                                                 mae_cams_old=float(np.abs(D.m["cams_old"][rows] - obs).mean()), mae_cams=float(np.abs(D.m["cams"][rows] - obs).mean()))
    check("fair CRPS of the 1 km forecast vs absolute error of CAMS (3-hourly), against score_cams3h T03", [cr_new["all"]["all"]["est"]] + cr_new["all"]["all"]["ci"],
          [T03n["km_vs_cams|crps"]["all"]["all"]["est"]] + T03n["km_vs_cams|crps"]["all"]["all"]["ci"])
    check("fair CRPS of the 1 km forecast vs absolute error of CAMS (three instants), against score T03", [cr_old["all"]["all"]["est"]] + cr_old["all"]["all"]["ci"],
          [T03o["km_vs_cams|crps"]["all"]["all"]["est"]] + T03o["km_vs_cams|crps"]["all"]["all"]["ci"])
    # sensitivity of the rule
    sens = {}
    for x, nm in (("cams_old", "three instants 00, 12, 24 UTC (existing arm)"), ("cams3i", "three instants, recomputed from the new files"),
                  ("cams", "paper's rule: eight steps 00 .. 21 UTC (float16 steps, as the chain stores them)"), ("cams3h_f32", "eight steps 00 .. 21 UTC from float32 values"),
                  ("cams3h_trap", "nine steps 00 .. 24 UTC, half weights on the ends (trapezoid)"), ("cams3h_0324", "eight steps 03 .. 24 UTC")):
        c = SC.contrast(smp, "km", x, "rmse")
        sens[x] = dict(rule=nm, rmse=float(np.sqrt(se[x].mean())), bias=float((D.m[x][rows] - obs).mean()), km_vs=dict(all=ec(c["all"]["all"]), regmean=ec(c["all"]["regmean"])))
    out["sensitivity_of_the_rule"] = sens
    print(f"[{time.time() - t0:5.1f} s] contrasts done", flush=True)

    # ------------------------------------------------------------------ shares of cells, monitors, issue days (rt_skill_data.py definitions)
    mon = D.mon[rows]
    ci, W, w_row = smp.ci, smp.W, smp.w

    def share(unit, n_units, cell_of_unit, x, y, min_n):
        k = unit.astype(np.int64) * W + w_row
        sx = np.bincount(k, weights=se[x], minlength=n_units * W).reshape(n_units, W)
        sy = np.bincount(k, weights=se[y], minlength=n_units * W).reshape(n_units, W)
        nn = np.bincount(k, minlength=n_units * W).reshape(n_units, W).astype(np.float64)
        ok = nn.sum(1) >= min_n
        low = sx.sum(1)[ok] < sy.sum(1)[ok]
        bx, by, bn = sx[ok] @ smp.Mw.T, sy[ok] @ smp.Mw.T, nn[ok] @ smp.Mw.T
        wt = smp.Mc[:, cell_of_unit[ok]].T * (bn > 0)
        bs = 100.0 * ((bx < by) * wt).sum(0) / wt.sum(0)
        chg = 100.0 * (np.sqrt(sx.sum(1)[ok] / sy.sum(1)[ok]) - 1.0)
        return dict(est=float(100.0 * low.mean()), ci=L.pct(bs), units=int(ok.sum()), units_lower=int(low.sum()), median_change_pct=float(np.median(chg)),
                    units_below_minus2=int((chg < -2).sum()), units_within_2=int((np.abs(chg) <= 2).sum()), units_above_plus2=int((chg > 2).sum()))

    ucell = np.unique(D.cell_of_mon[mon])
    um, mi = np.unique(mon, return_inverse=True)
    cell_of_um = np.searchsorted(ucell, D.cell_of_mon[um])
    sh = {}
    for tag, y in (("old", "cams_old"), ("new", "cams")):
        sh[tag] = dict(cells=share(ci, smp.C, np.arange(smp.C), "km", y, MIN_CELL), monitors=share(mi, um.size, cell_of_um, "km", y, MIN_MON))
    out["share_lower_rmse"] = sh
    idx_of = {d: i for i, d in enumerate(D.issue_days)}
    days = smp.issue_list
    per = dict(issue_day=days, station_days=[], old=[], new=[], old_regmean=[], new_regmean=[], rmse={x: [] for x in ("km", "cams_old", "cams")})
    for d in days:
        sd = L.Sample(D, m_both & (D.issue == idx_of[d]), B=a.B)
        per["station_days"].append(int(sd.n))
        for x in ("km", "cams_old", "cams"):
            per["rmse"][x].append(float(np.sqrt(((D.m[x][sd.rows] - D.obs[sd.rows]) ** 2).mean())))
        for tag, y in (("old", "cams_old"), ("new", "cams")):
            c = SC.contrast(sd, "km", y, "rmse", "c")["all"]
            per[tag].append(ec(c["all"]))
            per[f"{tag}_regmean"].append(ec(c["regmean"]))
    per["counts"] = {}
    for tag in ("old", "new"):
        e_ = np.array([v["est"] for v in per[tag]])
        hi = np.array([v["ci"][1] for v in per[tag]])
        er = np.array([v["est"] for v in per[f"{tag}_regmean"]])
        live = np.array([d >= "2026-11-03" for d in days])
        per["counts"][tag] = dict(issue_days=len(days), pooled_lower=int((e_ < 0).sum()), pooled_interval_below_zero_cells_only=int((hi < 0).sum()), regmean_lower=int((er < 0).sum()),
                                  pooled_min=float(e_.min()), pooled_max=float(e_.max()), pooled_median=float(np.median(e_)), live_lower=int((e_[live] < 0).sum()), live_days=int(live.sum()),
                                  least_favourable_day=days[int(e_.argmax())])
    out["by_issue_day"] = per
    if FIG.exists():                                                      # the figure's own numbers for the old arm (read-only)
        fg = json.loads(FIG.read_text())
        f_c = fg["a_map"]["share_lower_rmse_pct"]["cells_vs_cams"]
        check("share of cells with the lower RMSE, three instants, against fig_rt_skill/numbers.json", [sh["old"]["cells"]["est"]] + sh["old"]["cells"]["ci"] + [sh["old"]["cells"]["units"]],
              [f_c["est"]] + f_c["ci"] + [f_c["units"]])
        f_m = fg["a_map"]["share_lower_rmse_pct"]["monitors_vs_cams"]
        check("share of monitors with the lower RMSE, three instants, against fig_rt_skill/numbers.json", [sh["old"]["monitors"]["est"]] + sh["old"]["monitors"]["ci"], [f_m["est"]] + f_m["ci"])
        f_d = fg["e_days"]["counts"]["cams"]
        check("issue days with the lower RMSE, three instants, against fig_rt_skill/numbers.json",
              [per["counts"]["old"]["pooled_lower"], per["counts"]["old"]["pooled_min"], per["counts"]["old"]["pooled_max"], per["counts"]["old"]["pooled_median"], per["counts"]["old"]["regmean_lower"]],
              [f_d["pooled_lower"], f_d["pooled_min"], f_d["pooled_max"], f_d["pooled_median"], f_d["regmean_lower"]])
    print(f"[{time.time() - t0:5.1f} s] shares done", flush=True)

    # ------------------------------------------------------------------ read from the two runs of the scorer
    def both(fn):
        return dict(old=fn(OLD), new=fn(NEW))

    T11 = both(lambda r: tab(r, "T11_periods"))
    out["periods"] = {}
    for p, nm in (("sep", "issue days 6-20 September"), ("oct", "issue days 10 October - 2 November"), ("live", "issue days 3-11 November (live)"),
                  ("octnov", "issue days 10 October - 11 November"), ("clean", "48 issue days"), ("all53", "53 issue days (1-5 September included)")):
        k = f"{p}|with CAMS"
        out["periods"][p] = dict(name=nm, sample=T11["new"][k]["sample"], same_sample=T11["new"][k]["sample"] == T11["old"][k]["sample"],
                                 **{t: dict(pooled=ec(T11[t][k]["contrasts"]["km_vs_cams|rmse"]["all"]["all"]), regmean=ec(T11[t][k]["contrasts"]["km_vs_cams|rmse"]["all"]["regmean"]),
                                            pooled_cells_only_ci=T11[t][k]["contrasts"]["km_vs_cams|rmse"]["all"]["all"].get("ci_cells_only"),
                                            rmse_cams=float(T11[t][k]["scores"]["cams"]["all|all"]["rmse"]), rmse_km=float(T11[t][k]["scores"]["km"]["all|all"]["rmse"]),
                                            by_region={g: ec(T11[t][k]["contrasts"]["km_vs_cams|rmse"]["all"].get(g)) for g in L.REGIONS if g in T11[t][k]["contrasts"]["km_vs_cams|rmse"]["all"]})
                                    for t in ("old", "new")})
    k = "clean|final observations only"
    out["periods"]["final_obs_only"] = dict(name="48 issue days without the valid days whose observations are not final", sample=T11["new"][k]["sample"],
                                            **{t: dict(pooled=ec(T11[t][k]["contrasts"]["km_vs_cams|rmse"]["all"]["all"]), regmean=ec(T11[t][k]["contrasts"]["km_vs_cams|rmse"]["all"]["regmean"]))
                                               for t in ("old", "new")})
    T08 = both(lambda r: tab(r, "T08_rings")[SET])
    out["rings"] = dict(sample=T08["new"]["sample"], same_sample=T08["new"]["sample"] == T08["old"]["sample"],
                        **{t: {kk.split("_vs_")[0]: dict(pooled=ec(v["all"]["all"]), regmean=ec(v["all"]["regmean"])) for kk, v in T08[t]["contrasts"].items() if kk.endswith("_vs_cams|rmse")}
                           for t in ("old", "new")},
                        rmse_cams=dict(old=float(T08["old"]["scores"]["cams"]["all|all"]["rmse"]), new=float(T08["new"]["scores"]["cams"]["all|all"]["rmse"])))
    T13 = both(lambda r: tab(r, "T13_official_sites")[SET])
    gn = [g for g in T13["new"]["contrasts"]["km_vs_cams|rmse"]["all"] if g not in ("all", "regmean")]
    out["official_sites"] = dict(sample=T13["new"]["sample"], same_sample=T13["new"]["sample"] == T13["old"]["sample"],
                                 **{t: {g: dict(rmse=ec(T13[t]["contrasts"]["km_vs_cams|rmse"]["all"][g]), crps=ec(T13[t]["contrasts"]["km_vs_cams|crps"]["all"][g]),
                                                rmse_cams=float(T13[t]["scores"]["cams"][f"all|{g}"]["rmse"]), rmse_km=float(T13[t]["scores"]["km"][f"all|{g}"]["rmse"])) for g in gn}
                                    for t in ("old", "new")})
    T04 = both(lambda r: tab(r, "T04_temporal"))
    out["temporal"] = {}
    for key, nm in ((SET, "whole record"), (f"within sep|{SET}", "inside September"), (f"within octnov|{SET}", "inside October-November")):
        out["temporal"][nm] = dict(n_series=int(T04["new"][key]["n_series"]), same_series=T04["new"][key]["n_series"] == T04["old"][key]["n_series"],
                                   **{t: {g: dict(r_cams=ec(T04[t][key]["groups"][g]["cams"]["r"]), amp_cams=ec(T04[t][key]["groups"][g]["cams"]["amp"]),
                                                  r_km=ec(T04[t][key]["groups"][g]["km"]["r"]), amp_km=ec(T04[t][key]["groups"][g]["km"]["amp"]),
                                                  r_rt04=ec(T04[t][key]["groups"][g]["rt04"]["r"]),
                                                  km_minus_cams_r_median_of_diff=ec(T04[t][key]["groups"][g]["km-cams"]["r_median_of_diff"]),
                                                  share_series_km_r_higher=T04[t][key]["groups"][g]["km-cams"].get("share_series_r_higher"),
                                                  spatial_r_cams=T04[t][key]["groups"][g]["cams"].get("spatial_r"))
                                          for g in ["all", "regmean"] + [x for x in L.REGIONS if x in T04[t][key]["groups"]]} for t in ("old", "new")},
                                   by_lead={t: {Lk: dict(r=float(v["cams"]["r"]), amp=float(v["cams"]["amp"])) for Lk, v in T04[t][key]["leads"].items()} for t in ("old", "new")})
    T07 = both(lambda r: tab(r, "T07_sedi")[SET])
    out["sedi"] = {en: dict(n_events=int(T07["new"][en]["n_events"]), n_valid=int(T07["new"][en]["n_valid"]),
                            **{t: dict(cams=ec(T07[t][en]["arms"]["cams"]["all|all"]["sedi"]), km=ec(T07[t][en]["arms"]["km"]["all|all"]["sedi"]),
                                       clim=ec(T07[t][en]["arms"]["clim"]["all|all"]["sedi"]), km_minus_cams=ec(T07[t][en]["diff"]["km-cams"]["all|all"]),
                                       cams_hit_rate=float(T07[t][en]["arms"]["cams"]["all|all"]["hit_rate"]), cams_false_alarm_rate=float(T07[t][en]["arms"]["cams"]["all|all"]["false_alarm_rate"]),
                                       cams_frequency_bias=float(T07[t][en]["arms"]["cams"]["all|all"]["frequency_bias"])) for t in ("old", "new")}) for en in T07["new"]}
    T05 = both(lambda r: tab(r, "T05_ensemble"))
    out["crps_tables"] = {t: dict(km_fair_crps=ec(T05[t][SET]["km"]["all|all"]["crps"]), cams_abs_error=ec(T05[t][SET]["cams"]["all|all"]["crps"]),
                                  change=dict(pooled=ec(T05[t][f"{SET}|crps_contrasts"]["km_vs_cams|crps"]["all"]["all"]),
                                              regmean=ec(T05[t][f"{SET}|crps_contrasts"]["km_vs_cams|crps"]["all"]["regmean"])),
                                  sample_note="T05 requires the BCSD members as well (same station-days here)") for t in ("old", "new")}
    T12 = both(lambda r: tab(r, "T12_title_rule"))
    out["title_rule"] = {t: dict(verdict_strict=T12[t]["verdict_strict"], verdict_lenient=T12[t]["verdict_lenient"], regions_with_lower_rmse=T12[t]["cams"]["regions_with_lower_rmse"],
                                 regions_with_interval_below_zero=T12[t]["cams"]["regions_with_interval_below_zero"], regmean=ec(T12[t]["cams"]["regmean"])) for t in ("old", "new")}
    T10 = both(lambda r: tab(r, "T10_europe"))
    eu = {}
    for t in ("old", "new"):
        e_ = T10[t]["episode"]["contrasts"]["km_vs_cams"]["all"]
        eu[t] = dict(episode={g: ec(v) for g, v in e_.items() if g not in ("all", "regmean")},
                     subregions={g: ec(v) for g, v in T10[t][f"subregions|{SET}"]["contrasts"]["km_vs_cams|rmse"]["all"].items()},
                     periods={p: ec(T10[t][f"period|{p}|with CAMS"]["contrasts"]["km_vs_cams"]["two_way"]) for p in ("sep", "oct", "live", "clean") if f"period|{p}|with CAMS" in T10[t]})
    out["europe"] = eu
    T09 = both(lambda r: tab(r, "T09_analysis_mode"))
    if "with_cams" in T09["new"]:
        out["analysis_mode_days"] = {t: dict(sample=T09[t]["with_cams"]["sample"], **{kk.split("|")[0]: dict(pooled=ec(v["all"]["all"]), regmean=ec(v["all"]["regmean"]))
                                                                                      for kk, v in T09[t]["with_cams"]["contrasts"].items() if kk.endswith("_vs_cams|rmse")}) for t in ("old", "new")}
    out["checks"] = checks
    out["all_checks_identical"] = all(v["identical"] for v in checks.values())
    out["sources"] = dict(old_run=str(OLD), new_run=str(NEW), station_day_table=str(NEW / "work" / "pairs.npz"), scorer_code=[str(OLD / "sc_lib.py"), str(OLD / "score.py")],
                          bootstrap="two-way block bootstrap of the scorer (2-degree cells x ISO weeks of the valid day, 2,000 resamples, seed 20261119); per issue day: 2-degree cells only",
                          resamples=a.B)
    safe_io.write_json(HERE / "side_by_side.json", out)

    # ------------------------------------------------------------------ one-screen tables
    S = out
    md += ["# CAMS forecast: three instants against the 3-hourly daily mean, side by side\n",
           f"Sample: {S['sample']['description']}. Station-days lost by requiring both CAMS readings: {S['sample']['lost_by_requiring_both_cams_arms']}. "
           "RMSE change in %, negative = the first system is better; 95 % intervals of the scorer's two-way block bootstrap (2-degree cells x weeks, 2,000 resamples).\n",
           "old = CAMS daily mean from three instants (00, 12, 24 UTC; weights 1/4, 1/2, 1/4), as `score/`; new = mean of the eight 3-hourly steps 00 .. 21 UTC, the paper's rule.\n",
           "## RMSE of the CAMS forecast (ug m-3) and the handicap of the three-instant reading\n",
           L.md_table(["group", "station-days", "1 km", "CAMS, three instants", "CAMS, 3-hourly mean", "three instants vs 3-hourly, %"],
                      [[g, f"{S['rmse']['km'][g]['n']:,}", f2(S["rmse"]["km"][g]), f2(S["rmse"]["cams_old"][g]), f2(S["rmse"]["cams"][g]),
                        f1(S["handicap_three_instants_vs_3hourly_pct"]["all" if g == "all" else g])] for g in ["all"] + [f"lead {k}" for k in range(1, 6)] + L.REGIONS]),
           f"Mean over regions of the handicap: {f1(S['handicap_three_instants_vs_3hourly_pct']['regmean'])} %. Bias of CAMS: {S['bias_r_mae']['cams_old']['bias']:+.2f} (three instants), "
           f"{S['bias_r_mae']['cams']['bias']:+.2f} (3-hourly); Pearson r over station-days {S['bias_r_mae']['cams_old']['r']:.3f} / {S['bias_r_mae']['cams']['r']:.3f}; "
           f"MAE {S['bias_r_mae']['cams_old']['mae']:.2f} / {S['bias_r_mae']['cams']['mae']:.2f}.\n",
           "## 1 km forecast against the CAMS forecast\n",
           L.md_table(["group", "old (three instants)", "new (3-hourly mean)"],
                      [[nm, f1(S["km_vs_cams"]["old"][g]), f1(S["km_vs_cams"]["new"][g])] for g, nm in
                       [("all", "all monitors, pooled"), ("regmean", "mean over the 8 regions")] + [(f"lead {k}", f"lead day {k}, pooled") for k in range(1, 6)] +
                       [(f"lead {k} regmean", f"lead day {k}, mean over regions") for k in range(1, 6)] + [(g, g) for g in L.REGIONS]]),
           f"Regions with a lower RMSE: {S['km_vs_cams']['old']['regions_lower']} of 8 (old), {S['km_vs_cams']['new']['regions_lower']} of 8 (new); with the interval entirely below zero: "
           f"{S['km_vs_cams']['old']['regions_interval_below_zero']} (old), {S['km_vs_cams']['new']['regions_interval_below_zero']} (new).\n",
           "## By period (issue days; scorer table T11)\n",
           L.md_table(["period", "station-days", "old pooled", "new pooled", "old mean over regions", "new mean over regions", "RMSE CAMS old / new"],
                      [[v["name"], sdays(v["sample"]),
                        f1(v["old"]["pooled"]), f1(v["new"]["pooled"]), f1(v["old"]["regmean"]), f1(v["new"]["regmean"]),
                        (f"{v['old']['rmse_cams']:.2f} / {v['new']['rmse_cams']:.2f}" if "rmse_cams" in v["old"] else "")] for p, v in S["periods"].items()]),
           "## Away from the monitor's pixel (T08) and at the reference-site proxies (T13)\n",
           f"Rings sample: {S['rings']['sample']}.\n",
           L.md_table(["where the 1 km forecast is read", "old pooled", "new pooled", "old mean over regions", "new mean over regions"],
                      [[nm, f1(S["rings"]["old"][k]["pooled"]), f1(S["rings"]["new"][k]["pooled"]), f1(S["rings"]["old"][k]["regmean"]), f1(S["rings"]["new"][k]["regmean"])]
                       for k, nm in (("km", "monitor's pixel"), ("km_ring1", "ring 1 km"), ("km_ring3", "ring 3 km"), ("km_ring10", "ring 10 km"), ("km_cell_lin", "0.25 degree cell mean"))]),
           f"Official-sites sample: {S['official_sites']['sample']}.\n",
           L.md_table(["group", "old RMSE change", "new RMSE change", "old CRPS change", "new CRPS change", "RMSE CAMS old / new"],
                      [[g, f1(S["official_sites"]["old"][g]["rmse"]), f1(S["official_sites"]["new"][g]["rmse"]), f1(S["official_sites"]["old"][g]["crps"]), f1(S["official_sites"]["new"][g]["crps"]),
                        f"{S['official_sites']['old'][g]['rmse_cams']:.2f} / {S['official_sites']['new'][g]['rmse_cams']:.2f}"] for g in gn]),
           "## Shares with the lower RMSE (1 km forecast against CAMS)\n",
           L.md_table(["unit", "old", "new"],
                      [[f"2-degree cells with at least {MIN_CELL} station-days ({sh['new']['cells']['units']})", f"{f2(sh['old']['cells'], 1)} % ({sh['old']['cells']['units_lower']} cells)",
                        f"{f2(sh['new']['cells'], 1)} % ({sh['new']['cells']['units_lower']} cells)"],
                       [f"monitors with at least {MIN_MON} station-days ({sh['new']['monitors']['units']:,})", f"{f2(sh['old']['monitors'], 1)} % ({sh['old']['monitors']['units_lower']:,})",
                        f"{f2(sh['new']['monitors'], 1)} % ({sh['new']['monitors']['units_lower']:,})"],
                       ["issue days, pooled RMSE of the day", f"{per['counts']['old']['pooled_lower']} of {len(days)}", f"{per['counts']['new']['pooled_lower']} of {len(days)}"],
                       ["issue days, interval of the day (cells only) below zero", f"{per['counts']['old']['pooled_interval_below_zero_cells_only']} of {len(days)}",
                        f"{per['counts']['new']['pooled_interval_below_zero_cells_only']} of {len(days)}"],
                       ["issue days, mean over regions", f"{per['counts']['old']['regmean_lower']} of {len(days)}", f"{per['counts']['new']['regmean_lower']} of {len(days)}"],
                       ["range of the daily change, % (median)", f"{per['counts']['old']['pooled_min']:+.1f} .. {per['counts']['old']['pooled_max']:+.1f} ({per['counts']['old']['pooled_median']:+.1f})",
                        f"{per['counts']['new']['pooled_min']:+.1f} .. {per['counts']['new']['pooled_max']:+.1f} ({per['counts']['new']['pooled_median']:+.1f})"],
                       ["live issue days (3-11 November)", f"{per['counts']['old']['live_lower']} of {per['counts']['old']['live_days']}", f"{per['counts']['new']['live_lower']} of {per['counts']['new']['live_days']}"]]),
           "## 0.4 degree input against the CAMS forecast\n",
           L.md_table(["group", "old (three instants)", "new (3-hourly mean)"],
                      [[nm, f1(S["rt04_vs_cams"]["old"][g]), f1(S["rt04_vs_cams"]["new"][g])] for g, nm in
                       [("all", "all monitors, pooled"), ("regmean", "mean over the 8 regions")] + [(f"lead {k}", f"lead day {k}, pooled") for k in range(1, 6)] + [(g, g) for g in L.REGIONS]]),
           "## Fair CRPS of the eight 1 km members against the absolute error of the CAMS forecast\n",
           f"Fair CRPS of the 1 km forecast {S['km_fair_crps_vs_cams_abs_error']['fair_crps_km']:.3f} ug m-3; mean absolute error of CAMS {S['km_fair_crps_vs_cams_abs_error']['mae_cams_old']:.3f} "
           f"(three instants), {S['km_fair_crps_vs_cams_abs_error']['mae_cams']:.3f} (3-hourly).\n",
           L.md_table(["group", "old", "new"], [[nm, f1(S["km_fair_crps_vs_cams_abs_error"]["old"][g]), f1(S["km_fair_crps_vs_cams_abs_error"]["new"][g])]
                                                 for g, nm in [("all", "all monitors, pooled"), ("regmean", "mean over the 8 regions")] + [(g, g) for g in L.REGIONS]]),
           "## Day-to-day signal of the CAMS forecast at each monitor (T04)\n"]
    for nm, v in S["temporal"].items():
        md.append(L.md_table([nm + f" ({v['n_series']:,} series)", "temporal r, old", "temporal r, new", "amplitude ratio, old", "amplitude ratio, new", "1 km minus CAMS, r (median of differences): old", "new"],
                             [[g, f2(v["old"][g]["r_cams"]), f2(v["new"][g]["r_cams"]), f2(v["old"][g]["amp_cams"]), f2(v["new"][g]["amp_cams"]),
                               f1(v["old"][g]["km_minus_cams_r_median_of_diff"], 3), f1(v["new"][g]["km_minus_cams_r_median_of_diff"], 3)] for g in v["new"]]))
    md.append("## SEDI (T07): rows that involve the CAMS forecast\n")
    md.append(L.md_table(["event", "events / station-days", "CAMS old", "CAMS new", "1 km", "1 km minus CAMS, old", "1 km minus CAMS, new"],
                         [[en, f"{v['n_events']:,} / {v['n_valid']:,}", f2(v["old"]["cams"]), f2(v["new"]["cams"]), f2(v["new"]["km"]), f1(v["old"]["km_minus_cams"], 2), f1(v["new"]["km_minus_cams"], 2)]
                          for en, v in S["sedi"].items()]))
    md.append("## Sensitivity to the rule of the daily mean (pooled, same station-days)\n")
    md.append(L.md_table(["daily mean of the CAMS forecast", "RMSE of CAMS", "bias", "1 km vs CAMS, pooled", "mean over regions"],
                         [[v["rule"], f"{v['rmse']:.3f}", f"{v['bias']:+.2f}", f1(v["km_vs"]["all"]), f1(v["km_vs"]["regmean"])] for v in sens.values()]))
    md.append(f"## Title rule (T12)\n\nold: strict {S['title_rule']['old']['verdict_strict']}, lenient {S['title_rule']['old']['verdict_lenient']} (lower in {S['title_rule']['old']['regions_with_lower_rmse']} of 8 regions, "
              f"interval below zero in {S['title_rule']['old']['regions_with_interval_below_zero']}); new: strict {S['title_rule']['new']['verdict_strict']}, lenient {S['title_rule']['new']['verdict_lenient']} "
              f"(lower in {S['title_rule']['new']['regions_with_lower_rmse']} of 8, interval below zero in {S['title_rule']['new']['regions_with_interval_below_zero']}).\n")
    md.append("## Checks\n")
    md.append(L.md_table(["check", "result"], [[k_, "identical" if v_["identical"] else f"DIFFERENT: {v_['recomputed_here']} vs {v_['reference']}"] for k_, v_ in checks.items()]))
    md.append(f"Station-day tables of the two runs: same rows {S['tables']['same_rows_issue_lead_monitor']}, same observations {S['tables']['same_observations']}, other arms identical "
              f"{all(S['tables']['other_arms_identical'].values())}; the existing arm carried as `cams_old` is identical to the arm `cams` of score/: {S['tables']['existing_cams_arm_carried_as_cams_old_identical']}; "
              f"`cams3i` (three instants recomputed from the new files) against the existing arm: largest difference {S['tables']['cams3i_vs_existing_cams_max_abs_diff']:.3g} ug m-3 on "
              f"{S['tables']['cams3i_vs_existing_cams_rows_compared']:,} rows.\n")
    safe_io.write_text(HERE / "side_by_side.md", "\n".join(md) + "\n")
    print(f"[{time.time() - t0:5.1f} s] wrote side_by_side.json, side_by_side.md; all checks identical: {out['all_checks_identical']}", flush=True)


if __name__ == "__main__":
    main()
