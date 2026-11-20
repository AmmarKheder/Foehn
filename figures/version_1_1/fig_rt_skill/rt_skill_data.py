#!/usr/bin/env python3
"""Data of fig_rt_skill: the real-time 1 km forecasts verified at the monitors (48 issue days, lead days 1-5).

RULE: every score that the scorer's tables contain is READ from them (score/tables/T*.json); nothing is re-scored.
Four breakdowns that the tables lack are computed here WITH THE SCORER'S OWN CODE (sc_lib.Data, sc_lib.Sample,
score.contrast, score.ensemble, imported read-only) on the IDENTICAL station-days of its primary set (every arm finite,
issue days 2026-09-06..2026-11-11) and with its bootstrap draws:
  (1) per 2-degree cell (the space blocks of the bootstrap): station-days, monitors, RMSE of each arm   -> panel a
  (2) per issue day: pooled RMSE of each arm, RMSE change with the cells-only interval, mean over regions -> panel e
  (3) the eight members: RMSE of the ensemble mean in classes of ensemble spread                         -> panel f
  (4) the 0.4 degree input against the CAMS forecast on the station-days of the rings table            -> panel d (line)
plus the share of cells / monitors with the lower RMSE (two-way block bootstrap) and the RMSE by lead day with intervals.
Each computation is checked against a value of the tables (same sample, same draws): key "checks" of numbers.json.
The monitors are evaluation only: nothing is fitted, selected or corrected here.

    python -B rt_skill_data.py [--score DIR] [--arm km] [--arm2 NAME --name2 "label"] [--tag TAG] [--B 2000]
--score : the scorer's output directory (tables/, work/pairs.npz); default = the run of 2026-11-19
--arm   : the 1 km arm of the figure (first arm of the scorer's FOCUS)
--arm2  : optional second ensemble arm for the ensemble panels (e.g. a calibrated ensemble scored by the scorer with
          FOCUS "km,kmcal"); both ensembles are then scored on the station-days where both exist
--tag   : suffix of the outputs (data/rt_skill_<tag>.npz, numbers_<tag>.json)"""
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCORE = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay/score")
CLEAN = ("2026-09-06", "2026-11-11")            # the scorer's primary period (score.PERIODS["clean"])
LIVE0 = "2026-11-03"                            # first issue day of the live 1 km forecasts (score.PERIODS["live"])
MIN_CELL, MIN_MON = 100, 30                     # station-days needed to colour a cell / to count a monitor
NAMES = {"km": "Foehn in real time (1 km)", "rt04": "Foehn in real time (40 km)", "cams": "CAMS forecast",
         "clim": "Climatology (1 km)", "bcsd": "Statistical downscaling of the 40 km members"}
REG_FIG = {"Europe": "Europe", "East Asia": "East Asia", "South Asia": "South Asia", "North America": "North America",
           "Africa": "Africa", "South America": "South America", "North": "North of 60°N", "Other": "Elsewhere"}
# boxes of the monitor catalogue (realtime_1km/verif/verif.py, REGIONS; the first box that contains a monitor wins)
BOXES = [("Europe", 35.0, 60.0, -10.0, 35.0), ("East Asia", 20.0, 50.0, 100.0, 135.0), ("South Asia", 5.0, 35.0, 60.0, 95.0),
         ("North America", 15.0, 50.0, -130.0, -60.0), ("Africa", -35.0, 35.0, -18.0, 52.0),
         ("South America", -40.0, 12.0, -82.0, -34.0), ("North", 60.0, 90.0, -180.0, 180.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", default=str(SCORE))
    ap.add_argument("--code", default=str(SCORE))
    ap.add_argument("--arm", default="km")
    ap.add_argument("--arm2", default="")
    ap.add_argument("--name2", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    t0 = time.time()
    S = Path(a.score)
    sys.path.insert(0, a.code)
    import numpy as np
    import sc_lib as L                                                    # the scorer's library (read-only import)
    import score as SC                                                    # the scorer's tables code (main() is guarded)
    suf = f"_{a.tag}" if a.tag else ""
    A = a.arm
    src = {}                                                              # name -> where the number comes from

    def table(name):
        return json.loads((S / "tables" / f"{name}.json").read_text())

    def ec(d, n=None):
        o = dict(est=float(d["est"]), ci=[float(d["ci"][0]), float(d["ci"][1])])
        if "n" in d or n is not None:
            o["n"] = int(d.get("n", n))
        return o

    T01, T02, T03 = table("T01_samples"), table("T02_scores"), table("T03_contrasts")
    T05, T08, T11 = table("T05_ensemble"), table("T08_rings"), table("T11_periods")
    SET = "with the CAMS forecast"
    C3 = T03[SET]["contrasts"]
    focus = [k.split("_vs_")[0] for k in C3 if k.endswith("_vs_rt04|rmse")]           # the scorer's FOCUS, in its order
    assert A == focus[0], f"--arm {A} is not the first arm of the scorer's FOCUS {focus}"
    base = ["rt04", "cams", "clim", "bcsd"]
    B5 = focus + base

    # ------------------------------------------------------------------ the table of station-days and the primary sample
    D = L.Data(str(S / "work" / "pairs.npz"))
    mask = D.finite(B5) & D.issues(*CLEAN)
    smp = L.Sample(D, mask, B=a.B)
    print(f"[{time.time() - t0:5.1f} s] {smp.describe()}", flush=True)
    samp = T01["samples"][f"clean|{SET}"]
    checks = {}

    def check(name, mine, theirs, tol=1e-6):
        ok = bool(np.allclose(np.asarray(mine, float), np.asarray(theirs, float), rtol=0, atol=tol))
        checks[name] = dict(recomputed_here=np.round(np.asarray(mine, float), 6).tolist(), scorer_table=np.round(np.asarray(theirs, float), 6).tolist(), identical=ok)
        print(f"   check {name}: {'identical' if ok else 'DIFFERENT'}", flush=True)
        assert ok, name

    check("station_days of the primary set (T01)", smp.n, samp["station_days"])
    c0 = SC.contrast(smp, A, "cams", "rmse")
    check("RMSE change 1 km vs CAMS, pooled: estimate and two-way interval (T03)", [c0["all"]["all"]["est"]] + c0["all"]["all"]["ci"],
          [C3[f"{A}_vs_cams|rmse"]["all"]["all"]["est"]] + C3[f"{A}_vs_cams|rmse"]["all"]["all"]["ci"])
    check("RMSE change 1 km vs CAMS, mean over regions (T03)", [c0["all"]["regmean"]["est"]] + c0["all"]["regmean"]["ci"],
          [C3[f"{A}_vs_cams|rmse"]["all"]["regmean"]["est"]] + C3[f"{A}_vs_cams|rmse"]["all"]["regmean"]["ci"])
    rows = smp.rows
    obs = D.obs[rows]
    se = {x: (D.m[x][rows] - obs) ** 2 for x in B5}
    check("pooled RMSE of the five arms (T02)", [np.sqrt(se[x].mean()) for x in B5], [T02[SET][x]["all|all"]["rmse"] for x in B5])

    num = dict(figure="fig_rt_skill", made=dt.datetime.now().isoformat(timespec="seconds"), arm=A, arm2=a.arm2 or None,
               names={**NAMES, A: NAMES["km"]}, scorer_directory=str(S))
    reg_n = {g: dict(station_days=int(v[0]), monitors=int(v[1])) for g, v in samp["by_region"].items()}
    num["sample"] = dict(issue_days=len(samp["issue_days"]), first_issue_day=samp["issue_days"][0], last_issue_day=samp["issue_days"][-1],
                         issue_day_list=samp["issue_days"], station_days=int(samp["station_days"]), monitors=int(samp["monitors"]),
                         cells_2deg=int(smp.C), week_blocks=int(smp.W), time_blocks=samp["time_blocks"], lead_days=[1, 2, 3, 4, 5],
                         by_region=reg_n, description=T03[SET]["sample"],
                         live_issue_days=[d for d in samp["issue_days"] if d >= LIVE0],
                         reforecast_issue_days=[d for d in samp["issue_days"] if d < LIVE0],
                         mean_observation=float(T02[SET][A]["all|all"]["obs_mean"]))
    src["sample"] = "tables/T01_samples.json ['samples']['clean|with the CAMS forecast'] and T03_contrasts.json [SET]['sample'] (cells, week blocks)"

    # ------------------------------------------------------------------ a: the 2-degree cells (computed here, scorer's sample)
    mon = D.mon[rows]
    cell_row = D.cell_of_mon[mon]
    ucell = np.unique(cell_row)
    assert ucell.size == smp.C and np.array_equal(ucell[smp.ci], cell_row)             # same cells, same order as the bootstrap
    ci = smp.ci
    n_c = np.bincount(ci, minlength=smp.C)
    pair = np.unique(ci.astype(np.int64) * D.id.size + mon)
    nmon_c = np.bincount(pair // D.id.size, minlength=smp.C)
    rm_c = {x: np.sqrt(np.bincount(ci, weights=se[x], minlength=smp.C) / n_c) for x in B5}
    lat0 = 2.0 * (ucell // 1000 - 90)
    lon0 = 2.0 * (ucell % 1000 - 180)
    okc = n_c >= MIN_CELL
    reg_c = np.array([np.bincount(D.reg[rows][ci == k], minlength=8).argmax() for k in range(smp.C)])
    w_row = smp.w
    W = smp.W

    def share(unit, n_units, cell_of_unit, x, y, min_n):
        """share of the units (cells or monitors) with at least min_n station-days where arm x has the lower RMSE than arm y;
        two-way block bootstrap with the scorer's draws: weeks reweighted inside each unit (m_w), units weighted by the
        multiplicity m_c of their 2-degree cell"""
        k = unit.astype(np.int64) * W + w_row
        sx = np.bincount(k, weights=se[x], minlength=n_units * W).reshape(n_units, W)
        sy = np.bincount(k, weights=se[y], minlength=n_units * W).reshape(n_units, W)
        nn = np.bincount(k, minlength=n_units * W).reshape(n_units, W).astype(np.float64)
        ok = nn.sum(1) >= min_n
        low = sx.sum(1)[ok] < sy.sum(1)[ok]
        bx, by, bn = sx[ok] @ smp.Mw.T, sy[ok] @ smp.Mw.T, nn[ok] @ smp.Mw.T          # (units, B)
        wt = smp.Mc[:, cell_of_unit[ok]].T * (bn > 0)
        bs = 100.0 * ((bx < by) * wt).sum(0) / wt.sum(0)
        return dict(est=float(100.0 * low.mean()), ci=L.pct(bs), units=int(ok.sum()), units_lower=int(low.sum()))

    um, mi = np.unique(mon, return_inverse=True)
    cell_of_um = np.searchsorted(ucell, D.cell_of_mon[um])
    shares = {}
    for b in base:
        shares[f"cells_vs_{b}"] = share(ci, smp.C, np.arange(smp.C), A, b, MIN_CELL)
        shares[f"monitors_vs_{b}"] = share(mi, um.size, cell_of_um, A, b, MIN_MON)
    chg_c = {b: 100.0 * (rm_c[A] / rm_c[b] - 1.0) for b in base}
    SC_BOUNDS = [-30, -20, -15, -10, -5, -2, 2, 5, 10, 15, 20, 30]                    # gc_style.SC_BOUNDS (scorecards of the paper)
    hist = np.histogram(chg_c["cams"][okc], bins=[-np.inf] + SC_BOUNDS + [np.inf])[0]
    hist11 = np.concatenate([[hist[0] + hist[1]], hist[2:-2], [hist[-2] + hist[-1]]])   # the 11 colours drawn (open-ended beyond +-20 %)
    num["a_map"] = dict(
        quantity="100 x (RMSE of the 1 km ensemble mean / RMSE of the CAMS forecast - 1) per 2-degree cell, all lead days and issue days pooled",
        cells_with_monitors=int(smp.C), min_station_days_to_colour=MIN_CELL, cells_coloured=int(okc.sum()), cells_too_few=int((~okc).sum()),
        station_days_in_coloured_cells_pct=float(100.0 * n_c[okc].sum() / n_c.sum()),
        class_bounds=SC_BOUNDS, cells_per_class=[int(v) for v in hist], cells_per_class_pct=[float(100.0 * v / okc.sum()) for v in hist],
        colour_class_bounds=SC_BOUNDS[1:-1], cells_per_colour_class=[int(v) for v in hist11],
        cells_per_colour_class_pct=[float(100.0 * v / okc.sum()) for v in hist11],
        cells_below_minus2=int((chg_c["cams"][okc] < -2).sum()), cells_within_2=int((np.abs(chg_c["cams"][okc]) <= 2).sum()),
        cells_above_plus2=int((chg_c["cams"][okc] > 2).sum()), median_change_pct=float(np.median(chg_c["cams"][okc])),
        share_lower_rmse_pct=shares, min_station_days_per_monitor=MIN_MON, region_boxes=[dict(name=n, lat=[s, t], lon=[w, e]) for n, s, t, w, e in BOXES])
    src["a_map"] = ("computed here with sc_lib.Data / sc_lib.Sample on the scorer's primary set (rt_skill_data.py, section a); "
                    "cell = (floor(lat/2)+90)*1000 + floor(lon/2)+180 as in sc_lib; shares: two-way block bootstrap with the Sample's draws")
    print(f"[{time.time() - t0:5.1f} s] cells: {okc.sum()} coloured of {smp.C}; lower RMSE than CAMS in {shares['cells_vs_cams']}", flush=True)
    print(f"          monitors: {shares['monitors_vs_cams']}; vs climatology cells {shares['cells_vs_clim']['est']:.1f}, monitors {shares['monitors_vs_clim']['est']:.1f}", flush=True)

    # ------------------------------------------------------------------ b: by lead day (tables T02, T03; intervals of the RMSE computed here)
    cnt = smp.q("n", lambda: np.ones(smp.n))
    vn, wn = L.views(cnt[0]), L.views(cnt[1])
    lead = dict(rmse={}, rmse_interval_two_way={}, change_pooled={}, change_regmean={}, station_days=[int(T02[SET][A][f"{k}|all"]["n"]) for k in range(1, 6)])
    for x in B5:
        lead["rmse"][x] = [float(T02[SET][x][f"{k}|all"]["rmse"]) for k in range(1, 6)]
        q = smp.q(f"rmse_{x}", lambda x=x: se[x])
        va, wa = L.views(q[0]), L.views(q[1])
        mine = [float(np.sqrt(va[(k, "all")] / vn[(k, "all")])) for k in range(1, 6)]
        if x == A:
            check("RMSE of the 1 km forecast by lead day (T02)", mine, lead["rmse"][x])
        lead["rmse_interval_two_way"][x] = [L.pct(np.sqrt(wa[(k, "all")] / wn[(k, "all")])) for k in range(1, 6)]
    for b in base:
        r = C3[f"{A}_vs_{b}|rmse"]
        lead["change_pooled"][b] = [ec(r[str(k)]["all"]) for k in range(1, 6)]
        lead["change_regmean"][b] = [ec(r[str(k)]["regmean"]) for k in range(1, 6)]
    num["b_lead"] = lead
    src["b_lead"] = ("rmse: tables/T02_scores.json [SET][arm]['<lead>|all']['rmse']; change_pooled / change_regmean: tables/T03_contrasts.json "
                     "[SET]['contrasts']['<arm>_vs_<baseline>|rmse']['<lead>']['all' | 'regmean'] (two-way block bootstrap); "
                     "rmse_interval_two_way: computed here from the same Sample (not drawn)")

    # ------------------------------------------------------------------ c: by region (table T03)
    reg = dict(order_of_scorer=L.REGIONS, figure_names=REG_FIG, monitors={g: reg_n[g]["monitors"] for g in L.REGIONS},
               station_days={g: reg_n[g]["station_days"] for g in L.REGIONS}, change={}, rmse={}, bias={})
    for b in base:
        r = C3[f"{A}_vs_{b}|rmse"]["all"]
        reg["change"][b] = {g: ec(r[g]) for g in ["all", "regmean"] + L.REGIONS}
        reg["change"][b]["regions_lower"] = int(r["regmean"]["n_negative"])
        reg["change"][b]["regions_interval_below_zero"] = int(r["regmean"]["n_ci_below_zero"])
        reg["change"][b]["regions_interval_above_zero"] = int(r["regmean"]["n_ci_above_zero"])
    for x in B5:
        reg["rmse"][x] = {g: float(T02[SET][x][f"all|{g}"]["rmse"]) for g in ["all"] + L.REGIONS}
        reg["bias"][x] = {g: float(T02[SET][x][f"all|{g}"]["bias"]) for g in ["all"] + L.REGIONS}
    reg["share_of_squared_error_pct"] = {g: float(v[A]) for g, v in T02[SET]["share_of_squared_error_pct"].items()}
    reg["share_of_station_days_pct"] = {g: float(v["station_days_pct"]) for g, v in T02[SET]["share_of_squared_error_pct"].items()}
    num["c_regions"] = reg
    src["c_regions"] = ("change: tables/T03_contrasts.json [SET]['contrasts']['<arm>_vs_<baseline>|rmse']['all'][<region> | 'all' | 'regmean'] "
                        "(est, ci two-way; n_negative, n_ci_below_zero); rmse, bias, shares: tables/T02_scores.json; monitors: tables/T01_samples.json")

    # ------------------------------------------------------------------ d: away from the monitor's pixel (table T08)
    R8 = T08[SET]
    rings = [("km", "pixel", 0.0), ("km_ring1", "1 km", 1.0), ("km_ring3", "3 km", 3.0), ("km_ring10", "10 km", 10.0), ("km_cell_lin", "0.25° cell mean", None)]
    rings = [(A if k == "km" else k, lab_, d_) for k, lab_, d_ in rings]
    dr = dict(sample=R8["sample"], where=[lab_ for _, lab_, _ in rings], arms=[k for k, _, _ in rings], vs={}, vs_pixel={}, rmse={})
    for b in base:
        dr["vs"][b] = dict(pooled=[ec(R8["contrasts"][f"{k}_vs_{b}|rmse"]["all"]["all"]) for k, _, _ in rings],
                           regmean=[ec(R8["contrasts"][f"{k}_vs_{b}|rmse"]["all"]["regmean"]) for k, _, _ in rings])
    dr["vs_pixel"] = dict(pooled=[None] + [ec(R8["contrasts"][f"{k}_vs_{A}|rmse"]["all"]["all"]) for k, _, _ in rings[1:]],
                          regmean=[None] + [ec(R8["contrasts"][f"{k}_vs_{A}|rmse"]["all"]["regmean"]) for k, _, _ in rings[1:]])
    for k in [k for k, _, _ in rings] + base:
        dr["rmse"][k] = float(R8["scores"][k]["all|all"]["rmse"])
    arms_r = [A] + [k for k, _, _ in rings[1:]] + base
    smr = L.Sample(D, D.finite(arms_r) & D.issues(*CLEAN), B=a.B)
    cr = SC.contrast(smr, "km_ring10", "cams", "rmse")["all"]["all"]
    check("rings sample: 10 km ring vs CAMS, estimate and interval (T08)", [cr["est"]] + cr["ci"],
          [R8["contrasts"]["km_ring10_vs_cams|rmse"]["all"]["all"]["est"]] + R8["contrasts"]["km_ring10_vs_cams|rmse"]["all"]["all"]["ci"])
    ci_ = SC.contrast(smr, "rt04", "cams", "rmse")["all"]
    dr["input_vs_cams_same_station_days"] = dict(pooled=ec(ci_["all"]), regmean=ec(ci_["regmean"]), station_days=int(smr.n))
    num["d_rings"] = dr
    src["d_rings"] = ("vs, vs_pixel, rmse: tables/T08_rings.json [SET]['contrasts']['<ring arm>_vs_<baseline>|rmse']['all']['all' | 'regmean'] and ['scores']; "
                      "input_vs_cams_same_station_days: computed here with score.contrast on the station-days of T08 (sample checked against T08)")
    del smr

    # ------------------------------------------------------------------ e: by issue day (computed here, scorer's code, cells-only intervals)
    days = samp["issue_days"]
    idx_of = {d: i for i, d in enumerate(D.issue_days)}
    per = dict(issue_day=days, live=[d >= LIVE0 for d in days], station_days=[], mean_observation=[], rmse={x: [] for x in B5},
               change_pooled={b: [] for b in base}, change_regmean={b: [] for b in base})
    for d in days:
        sd = L.Sample(D, mask & (D.issue == idx_of[d]), B=a.B)
        per["station_days"].append(int(sd.n))
        per["mean_observation"].append(float(D.obs[sd.rows].mean()))
        for x in B5:
            per["rmse"][x].append(float(np.sqrt(((D.m[x][sd.rows] - D.obs[sd.rows]) ** 2).mean())))
        for b in base:
            c = SC.contrast(sd, A, b, "rmse", "c")["all"]
            per["change_pooled"][b].append(ec(c["all"]))
            per["change_regmean"][b].append(dict(**ec(c["regmean"]), n_groups=int(c["regmean"]["n_groups"])))
    assert sum(per["station_days"]) == smp.n
    per["counts"] = {}
    for b in base:
        e_ = np.array([v["est"] for v in per["change_pooled"][b]])
        hi = np.array([v["ci"][1] for v in per["change_pooled"][b]])
        lo = np.array([v["ci"][0] for v in per["change_pooled"][b]])
        er = np.array([v["est"] for v in per["change_regmean"][b]])
        per["counts"][b] = dict(issue_days=len(days), pooled_lower=int((e_ < 0).sum()), pooled_interval_below_zero=int((hi < 0).sum()),
                                pooled_interval_above_zero=int((lo > 0).sum()), regmean_lower=int((er < 0).sum()),
                                pooled_min=float(e_.min()), pooled_max=float(e_.max()), pooled_median=float(np.median(e_)),
                                live_lower=int((e_[np.array(per["live"])] < 0).sum()), live_days=int(np.sum(per["live"])))
    per["periods_two_way"] = {}
    for p, nm in (("sep", "issue days 6-20 September"), ("oct", "issue days 10 October - 2 November"), ("live", "issue days 3-11 November (live)")):
        P = T11[f"{p}|with CAMS"]
        per["periods_two_way"][p] = dict(name=nm, sample=P["sample"], rmse={x: float(P["scores"][x]["all|all"]["rmse"]) for x in B5},
                                         **{f"vs_{b}": dict(pooled=ec(P["contrasts"][f"{A}_vs_{b}|rmse"]["all"]["all"]),
                                                            regmean=ec(P["contrasts"][f"{A}_vs_{b}|rmse"]["all"]["regmean"])) for b in base})
    num["e_days"] = per
    src["e_days"] = ("per issue day: computed here, one sc_lib.Sample per issue day (all its lead days 1-5; the station-days of the primary set), "
                     "score.contrast(mode 'c') = RMSE change with the interval of the 2-degree-cell bootstrap (a single issue day has one or two "
                     "week blocks: no time resampling), 'regmean' = mean over the eight regions; periods_two_way: tables/T11_periods.json")
    print(f"[{time.time() - t0:5.1f} s] issue days: " + "; ".join(f"vs {b}: lower on {v['pooled_lower']}/{v['issue_days']}, interval below zero on {v['pooled_interval_below_zero']}"
                                                                  for b, v in per["counts"].items()), flush=True)

    # ------------------------------------------------------------------ f: the eight members at the monitors (table T05; spread classes here)
    ens_arms = [A] + ([a.arm2] if a.arm2 else [])
    m_e = mask & D.finite(ens_arms)
    same = int(m_e.sum()) == smp.n
    sme = smp if same else L.Sample(D, m_e, B=a.B)
    E = SC.ensemble(sme, ens_arms, [])
    t5 = T05[SET][A]["all|all"]
    if same:
        check("ensemble of the 1 km forecast: spread/skill, coverage, share outside (T05)",
              [E[A]["all|all"]["spread_skill"]["est"]] + E[A]["all|all"]["spread_skill"]["ci"] + [E[A]["all|all"]["cover"]["est"], E[A]["all|all"]["outside"]],
              [t5["spread_skill"]["est"]] + t5["spread_skill"]["ci"] + [t5["cover"]["est"], t5["outside"]])
        check("rank histogram of the 1 km forecast (T05)", E[A]["all|all"]["rank_hist"], t5["rank_hist"])
    ens = dict(station_days=int(sme.n), identical_to_primary_set=same, members=int(t5["members"]), arms=ens_arms,
               names={A: NAMES["km"], **({a.arm2: a.name2 or a.arm2} if a.arm2 else {})},
               expected=dict(rank_share_pct=100.0 / (t5["members"] + 1), outside_pct=200.0 / (t5["members"] + 1), coverage_10_90_pct=64.0,
                             note="a reliable 8-member ensemble: flat rank histogram (11.1 % per rank, 22.2 % outside the members); its 10th-90th "
                                  "percentile range (linear quantiles of 8 members) holds about 64 % (scorer's simulation, methods.md), not 80 %"))
    spread_bins = None
    for x in ens_arms:
        e_ = (t5 if (x == A and same) else E[x]["all|all"])
        st = D.ens_stats(x)
        M = st["M"]
        f_ = (M + 1) / M
        var = st["var"][sme.rows]
        sq = (D.m[x][sme.rows] - D.obs[sme.rows]) ** 2
        sp = np.sqrt(f_ * var)
        if spread_bins is None:                                                   # classes of spread: factor 2**0.5, from the first arm
            lo_, hi_ = np.percentile(sp, [1.0, 99.9])
            spread_bins = 2.0 ** (np.arange(np.floor(2 * np.log2(lo_)), np.ceil(2 * np.log2(hi_)) + 1) / 2)
        edges = np.concatenate([[0.0], spread_bins, [np.inf]])
        k = np.digitize(sp, edges) - 1
        nb = edges.size - 1
        nk = np.bincount(k, minlength=nb)
        keep = nk >= 5000
        bins = dict(spread=np.sqrt(f_ * np.bincount(k, weights=var, minlength=nb)[keep] / nk[keep]).tolist(),
                    rmse=np.sqrt(np.bincount(k, weights=sq, minlength=nb)[keep] / nk[keep]).tolist(),
                    station_days=nk[keep].tolist(), lower_edge=edges[:-1][keep].tolist(), upper_edge=[float(v) if np.isfinite(v) else None for v in edges[1:][keep]],
                    min_station_days_per_class=5000, classes_dropped=int((~keep & (nk > 0)).sum()), station_days_dropped=int(nk[~keep].sum()))
        ens[x] = dict(rank_hist_pct=[100.0 * float(v) for v in e_["rank_hist"]], outside_pct=float(e_["outside"]),
                      coverage_10_90_pct=ec(e_["cover"]), below_pct=float(e_["below"]), above_pct=float(e_["above"]),
                      spread_skill=ec(e_["spread_skill"]), fair_crps=ec(e_["crps"]), spread=float(e_["spread"]), rmse=float(e_["rmse"]),
                      spread_classes=bins, source="tables/T05_ensemble.json" if (x == A and same) else "computed here with score.ensemble")
    ens["by_lead"] = {x: dict(spread_skill=[ec(E[x][f"{k}|all"]["spread_skill"]) for k in range(1, 6)],
                              coverage_10_90_pct=[ec(E[x][f"{k}|all"]["cover"]) for k in range(1, 6)]) for x in ens_arms}
    for x, nm_ in (("rt04", "members of the 40 km forecast"), ("bcsd_m", "members of the statistical downscaling")):   # for the text, not drawn
        if x in T05[SET] and x not in ens_arms:
            v = T05[SET][x]["all|all"]
            ens.setdefault("other_ensembles_same_station_days", {})[x] = dict(
                name=nm_, rank_hist_pct=[100.0 * float(q) for q in v["rank_hist"]], outside_pct=float(v["outside"]), coverage_10_90_pct=ec(v["cover"]),
                spread_skill=ec(v["spread_skill"]), fair_crps=ec(v["crps"]), source="tables/T05_ensemble.json")
    c5 = T05.get(f"{SET}|crps_contrasts", {})
    ens["crps_change_pct"] = {k.split("|")[0]: dict(pooled=ec(v["all"]["all"]), regmean=ec(v["all"]["regmean"])) for k, v in c5.items() if k.startswith(f"{A}_vs_")}
    num["f_ensemble"] = ens
    src["f_ensemble"] = ("rank_hist_pct, outside_pct, coverage_10_90_pct, below/above, spread_skill, fair_crps: tables/T05_ensemble.json [SET][arm]['all|all'] "
                         "(for a second arm: score.ensemble on the common station-days); spread_classes: computed here on the same station-days from "
                         "sc_lib.Data.ens_stats (unbiased member variance): classes of sqrt((M+1)/M x variance) of width 2**0.5, per class "
                         "sqrt((M+1)/M x mean variance) and RMSE of the ensemble mean; crps_change_pct: tables/T05_ensemble.json ['...|crps_contrasts']")

    # ------------------------------------------------------------------ headline numbers of the legend (all from the tables)
    al = lambda b: ec(C3[f"{A}_vs_{b}|rmse"]["all"]["all"])
    rg = lambda b: ec(C3[f"{A}_vs_{b}|rmse"]["all"]["regmean"])
    num["legend_numbers"] = dict(
        rmse_pooled={x: float(T02[SET][x]["all|all"]["rmse"]) for x in B5},
        change_pooled={b: al(b) for b in base}, change_regmean={b: rg(b) for b in base},
        final_observations_only={b: dict(pooled=ec(T11["clean|final observations only"]["contrasts"][f"{A}_vs_{b}|rmse"]["all"]["all"]),
                                         regmean=ec(T11["clean|final observations only"]["contrasts"][f"{A}_vs_{b}|rmse"]["all"]["regmean"])) for b in base},
        with_1_to_5_september={b: dict(pooled=ec(T11["all53|with CAMS"]["contrasts"][f"{A}_vs_{b}|rmse"]["all"]["all"]),
                                       regmean=ec(T11["all53|with CAMS"]["contrasts"][f"{A}_vs_{b}|rmse"]["all"]["regmean"])) for b in base})
    src["legend_numbers"] = "tables/T02_scores.json, T03_contrasts.json, T11_periods.json (keys 'clean|final observations only', 'all53|with CAMS')"
    # ------------------------------------------------------------------ every number printed in legend.tex: value, printed form, source
    pc = lambda d: f"{d['est']:+.1f}% [{d['ci'][0]:+.1f}, {d['ci'][1]:+.1f}]"                   # change in %, as printed (ASCII minus)
    T3 = "tables/T03_contrasts.json ['with the CAMS forecast']['contrasts']"
    r10 = dr["vs"]["cams"]["pooled"][3]
    sh = shares["cells_vs_cams"]
    nN = reg["change"]["cams"]["North"]
    num["legend_text_values"] = {
        "monitors": dict(value=num["sample"]["monitors"], printed=f"{num['sample']['monitors']:,} monitors", source="tables/T01_samples.json ['samples']['clean|with the CAMS forecast']['monitors']"),
        "issue_days": dict(value=num["sample"]["issue_days"], printed=f"{num['sample']['issue_days']} issue days", source="tables/T01_samples.json (same key) ['issue_days']"),
        "station_days": dict(value=num["sample"]["station_days"], printed=f"{num['sample']['station_days']:,} station-days", source="tables/T01_samples.json (same key) ['station_days']"),
        "station_days_panel_d": dict(value=dr["input_vs_cams_same_station_days"]["station_days"], printed=f"{dr['input_vs_cams_same_station_days']['station_days']:,} in d",
                                     source="tables/T08_rings.json ['with the CAMS forecast']['sample'] (station-days with every ring; recomputed here, identical)"),
        "min_station_days_per_cell": dict(value=MIN_CELL, printed=f"fewer than {MIN_CELL} station-days", source="rt_skill_data.py, constant MIN_CELL (the scorer's minimum for a group, sc_lib.ratio_contrast min_n)"),
        "cells_coloured": dict(value=int(okc.sum()), printed=f"{int(okc.sum())} coloured cells", source="computed here: a_map.cells_coloured"),
        "share_of_cells_with_lower_rmse": dict(value=sh, printed=f"{sh['est']:.0f}% [{sh['ci'][0]:.0f}, {sh['ci'][1]:.0f}]", source="computed here: a_map.share_lower_rmse_pct.cells_vs_cams (two-way block bootstrap)"),
        "all_monitors_vs_cams": dict(value=al("cams"), printed=pc(al("cams")), source=f"{T3}['{A}_vs_cams|rmse']['all']['all']"),
        "all_monitors_vs_climatology": dict(value=al("clim"), printed=pc(al("clim")), source=f"{T3}['{A}_vs_clim|rmse']['all']['all']"),
        "mean_of_regions_vs_cams": dict(value=rg("cams"), printed=pc(rg("cams")), source=f"{T3}['{A}_vs_cams|rmse']['all']['regmean']"),
        "mean_of_regions_vs_climatology": dict(value=rg("clim"), printed=pc(rg("clim")), source=f"{T3}['{A}_vs_clim|rmse']['all']['regmean']"),
        "north_of_60N_interval_upper_bound": dict(value=nN["ci"][1], printed=f"{nN['ci'][1]:+.0f}%", source=f"{T3}['{A}_vs_cams|rmse']['all']['North']['ci'][1]"),
        "not_below_cams": dict(value={g: reg["change"]["cams"][g] for g in L.REGIONS if reg["change"]["cams"][g]["est"] > 0}, printed="In Europe and north of 60°N",
                               source=f"{T3}['{A}_vs_cams|rmse']['all'][region]: the regions with a positive estimate"),
        "ring_10km_vs_cams": dict(value=r10, printed=f"{-r10['est']:.1f}% [{-r10['ci'][1]:.1f}, {-r10['ci'][0]:.1f}]",
                                  source="tables/T08_rings.json ['with the CAMS forecast']['contrasts']['km_ring10_vs_cams|rmse']['all']['all'] (printed as a reduction)"),
        "issue_days_below_both": dict(value=dict(vs_cams=per["counts"]["cams"]["pooled_lower"], vs_climatology=per["counts"]["clim"]["pooled_lower"], of=len(days)),
                                      printed=f"{min(per['counts']['cams']['pooled_lower'], per['counts']['clim']['pooled_lower'])} of {len(days)} days",
                                      source="computed here: e_days.counts (pooled RMSE of the issue day, lead days 1-5)"),
        "live_issue_days": dict(value=num["sample"]["live_issue_days"], printed="3--11 November", source="score.py PERIODS['live'] (issue days 2026-11-03..2026-11-11)"),
        "outside_the_members": dict(value=ens[A]["outside_pct"], printed=f"{ens[A]['outside_pct']:.0f}%", source="tables/T05_ensemble.json ['with the CAMS forecast'][arm]['all|all']['outside']"),
        "outside_expected": dict(value=ens["expected"]["outside_pct"], printed=f"{ens['expected']['outside_pct']:.0f}% expected", source="2 / (M + 1) for M = 8 exchangeable members"),
    }
    num["checks"] = checks
    num["sources"] = dict(per_block=src, scorer_report=str(S / "REPORT.md"), scorer_numbers=str(S / "numbers.json"),
                          scorer_tables=str(S / "tables"), station_day_table=str(S / "work" / "pairs.npz"),
                          scorer_code=[str(Path(a.code) / "sc_lib.py"), str(Path(a.code) / "score.py")],
                          bootstrap="two-way block bootstrap of the scorer (2-degree cells x ISO weeks of the valid day, 2,000 resamples, seed 20261119); "
                                    "per issue day: 2-degree cells only",
                          this_script=str(HERE / "rt_skill_data.py"))

    # ------------------------------------------------------------------ files
    (HERE / "data").mkdir(exist_ok=True)
    np.savez_compressed(HERE / "data" / f"rt_skill{suf}.npz", cell=ucell, cell_lat0=lat0, cell_lon0=lon0, cell_n=n_c, cell_monitors=nmon_c,
                        cell_region=np.array(L.REGIONS)[reg_c], **{f"cell_rmse_{x}": rm_c[x] for x in B5},
                        mon_id=D.id[um], mon_lat=D.lat[um], mon_lon=D.lon[um], mon_n=np.bincount(mi),
                        **{f"mon_rmse_{x}": np.sqrt(np.bincount(mi, weights=se[x]) / np.bincount(mi)) for x in B5})
    (HERE / f"numbers{suf}.json").write_text(json.dumps(num, indent=1))
    with open(HERE / "data" / f"cells{suf}.csv", "w") as f:                               # source data of panel a
        f.write("cell,lat_south,lon_west,region,station_days,monitors," + ",".join(f"rmse_{x}" for x in B5) + ",change_vs_cams_pct,coloured\n")
        for k in range(smp.C):
            f.write(f"{ucell[k]},{lat0[k]:g},{lon0[k]:g},{L.REGIONS[reg_c[k]]},{n_c[k]},{nmon_c[k]}," + ",".join(f"{rm_c[x][k]:.3f}" for x in B5) +
                    f",{chg_c['cams'][k]:.2f},{int(okc[k])}\n")
    with open(HERE / "data" / f"issue_days{suf}.csv", "w") as f:                          # source data of panel e
        f.write("issue_day,live,station_days,mean_observation," + ",".join(f"rmse_{x}" for x in B5) +
                "," + ",".join(f"change_vs_{b}_pct,lo,hi,regmean_vs_{b}_pct" for b in base) + "\n")
        for i, d in enumerate(days):
            f.write(f"{d},{int(per['live'][i])},{per['station_days'][i]},{per['mean_observation'][i]:.2f}," + ",".join(f"{per['rmse'][x][i]:.3f}" for x in B5) + "," +
                    ",".join(f"{per['change_pooled'][b][i]['est']:.2f},{per['change_pooled'][b][i]['ci'][0]:.2f},{per['change_pooled'][b][i]['ci'][1]:.2f},"
                             f"{per['change_regmean'][b][i]['est']:.2f}" for b in base) + "\n")
    print(f"[{time.time() - t0:5.1f} s] wrote data/rt_skill{suf}.npz, numbers{suf}.json, data/cells{suf}.csv, data/issue_days{suf}.csv", flush=True)


if __name__ == "__main__":
    main()
