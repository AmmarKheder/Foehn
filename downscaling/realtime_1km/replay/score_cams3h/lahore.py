#!/usr/bin/env python3
"""cams3h: the numbers of the figure 'a real-time forecast at 1 km' that involve the CAMS forecast, with its two daily means.

paper/natcomms/figures_new/fig_rt_day/numbers.json (rt_day_data.py, read-only here) reads the CAMS forecast with the
three-instant daily mean. Here the same numbers are recomputed twice with the figure's own loaders (rt_loaders.py, imported
read-only: cities, city boxes, monitors, observations, 1 km snapshots and tiles, bilinear reading on the 84 N .. 60 S grid):
  old  = rt_loaders.cams_daily (three instants) -> must reproduce numbers.json (key "reproduces_numbers_json");
  new  = the paper's rule, the eight 3-hourly steps 00 .. 21 UTC of the valid day, from cams3h/raw (c3h_lib).
Cases as in numbers.json (variant "strict"): Lahore on 2026-11-11 (city monitors; the 221 monitors of the window
72.6-79.0 E, 27.4-32.6 N), Delhi as its neighbour, the series of 3-11 November, Karachi on 2026-11-10.
Evaluation only; nothing is written outside this directory.
    python -I -B lahore.py   -> lahore.json, lahore.md"""
import datetime as dt
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
R0 = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay")
FIG = Path("/scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/figures_new/fig_rt_day")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(R0 / "cams3h/code"))
sys.path.insert(0, str(FIG))
import numpy as np  # noqa: E402
import c3h_lib as C  # noqa: E402
import safe_io  # noqa: E402
import rt_loaders as R  # noqa: E402   (the figure's loaders; nothing in it writes)

NMIN = 10
OBS_DAYS = [dt.date(2026, 11, 3) + dt.timedelta(k) for k in range(13)]
REGION = (72.6, 79.0, 27.4, 32.6)
DAY1, CITY1, NEIGHBOUR = dt.date(2026, 11, 11), "Lahore", "Delhi"
DAY2, CITY2 = dt.date(2026, 11, 10), "Karachi"


def main():
    ref = json.loads((FIG / "numbers.json").read_text())
    assert ref["case1"]["city"] == CITY1 and ref["case1"]["day"] == str(DAY1) and ref["case2"]["city"] == CITY2 and ref["case2"]["day"] == str(DAY2)
    ids, lat, lon = R.catalogue()
    cities = R.cities()
    box = {n: R.city_box(la, lo) for n, la, lo in cities}
    inbox = {n: (lat >= b[0]) & (lat <= b[1]) & (lon >= b[2]) & (lon <= b[3]) for n, b in box.items()}
    obs = {d: R.observations(d, ids)[0] for d in OBS_DAYS}
    snap = {d: R.snapshot(d) for d in R.ISSUES}
    rt = {(d, L): R.daily04(d, L) for d in R.ISSUES for L in range(1, 6)}
    # ---- the two daily means of the CAMS forecast on the grid of the figure (361 x 900: 84 N .. 60 S, 180 W .. 179.6 E)
    cf = dict(old={(d, L): R.cams_daily(d, L) for d in R.ISSUES for L in range(1, 6)}, new={}, three_from_new_files={})
    idx = C.index_raw()
    cache = {}
    for d in R.ISSUES:
        f, i = idx[d]
        if f not in cache:
            cache.clear()
            cache[f] = C.read_file(f)
        r16 = C.roll180(C.to_f16(cache[f][1][i]))
        for L in range(1, 6):
            cf["new"][d, L] = C.daily_rule(r16, L, "paper")[15:15 + 361]
            cf["three_from_new_files"][d, L] = C.daily_three(r16, L)[15:15 + 361]
    cache.clear()
    same_fields = max(float(np.abs(cf["old"][k].astype(np.float64) - cf["three_from_new_files"][k].astype(np.float64)).max()) for k in cf["old"])

    def at_monitors(sel, v, which):
        o = {}
        for L in (1, 2, 3, 4, 5):
            d = v - dt.timedelta(L - 1)
            if d in snap:
                o[f"cams{L}"] = float(R.bilinear(cf[which][d, L], lat[sel], lon[sel]).mean())
                o[f"km{L}"] = float(snap[d][0][L - 1][sel].mean())
                o[f"rt{L}"] = float(R.bilinear(rt[d, L], lat[sel], lon[sel]).mean())
        return o

    sd = lambda f, o: dict(rmse=float(np.sqrt(np.mean((f - o) ** 2))), bias=float(np.mean(f - o)), r=float(np.corrcoef(o, f)[0, 1]))  # noqa: E731
    out = dict(sources=dict(figure_numbers=str(FIG / "numbers.json"), loaders=str(FIG / "rt_loaders.py"), raw=str(C.RAW),
                            rule_new="mean of the eight 3-hourly steps 00 .. 21 UTC of the valid day (float16 steps, floor at 0), read as the figure reads the three-instant mean"),
               three_instant_fields_from_new_files_max_abs_diff_vs_figure_loader=same_fields)
    W_, E_, S_, N_ = REGION
    mon1 = inbox[CITY1] & np.isfinite(obs[DAY1])
    monn = inbox[NEIGHBOUR] & np.isfinite(obs[DAY1])
    mreg = (lat >= S_) & (lat <= N_) & (lon >= W_) & (lon <= E_) & np.isfinite(obs[DAY1])
    o_r = obs[DAY1][mreg]
    si, sj, _ = R.cells04(REGION)
    incity = (inbox[CITY1] | inbox[NEIGHBOUR])[mreg]
    cell = np.round((84.0 - lat[mreg]) / 0.4).astype(np.int64) * 1000 + np.round((lon[mreg] + 180.0) / 0.4).astype(np.int64)
    m2 = inbox[CITY2] & np.isfinite(obs[DAY2])
    s2, n2, w2, e2 = box[CITY2]
    hx, hext = R.window(DAY2, 1, "mean", s2, n2, w2, e2)
    r0, r1, c0, c1 = R.grid_window(hext)
    pop = np.load(R.G / "pop1km_global.npy", mmap_mode="r")
    hok = np.isfinite(hx) & R.inside(hext, hx.shape, box[CITY2])
    hpop = np.where(hok, np.asarray(pop[r0:r1, c0:c1], np.float32), 0.0)
    habove = hok & (hx > R.THR)
    res = {}
    for which in ("old", "new"):
        c = cf[which]
        o = {}
        # ---- case 1
        o["lahore_city_monitors"] = dict(monitors=int(mon1.sum()), measured=float(obs[DAY1][mon1].mean()), **at_monitors(mon1, DAY1, which))
        o["delhi_city_monitors"] = dict(monitors=int(monn.sum()), measured=float(obs[DAY1][monn].mean()), **at_monitors(monn, DAY1, which))
        o["plain_monitors"] = dict(monitors=int(mreg.sum()), measured=float(o_r.mean()), **at_monitors(mreg, DAY1, which))
        sc = {}
        for L in (1, 2, 3, 4, 5):
            d = DAY1 - dt.timedelta(L - 1)
            sc[f"cams_lead{L}"] = sd(R.bilinear(c[d, L], lat[mreg], lon[mreg]), o_r)
            sc[f"km_lead{L}"] = sd(snap[d][0][L - 1][mreg], o_r)
            sc[f"rt04_lead{L}"] = sd(R.bilinear(rt[d, L], lat[mreg], lon[mreg]), o_r)
        o["plain_scores"] = sc
        F1 = dict(km=snap[DAY1][0][0][mreg].astype(np.float64), rt04=R.bilinear(rt[DAY1, 1], lat[mreg], lon[mreg]), cams=R.bilinear(c[DAY1, 1], lat[mreg], lon[mreg]))
        part = lambda q: dict(monitors=int(q.sum()), measured=float(o_r[q].mean()), cams=dict(mean=float(F1["cams"][q].mean()), bias=float((F1["cams"][q] - o_r[q]).mean()),  # noqa: E731
                                                                                             rmse=float(np.sqrt(np.mean((F1["cams"][q] - o_r[q]) ** 2)))))
        rng = np.random.default_rng(0)

        def boot(fa, fb, groups, B=10000):
            g, inv = np.unique(groups, return_inverse=True)
            sa = np.bincount(inv, (fa - o_r) ** 2, g.size)
            sb = np.bincount(inv, (fb - o_r) ** 2, g.size)
            k = rng.integers(0, g.size, (B, g.size))
            dd = 100 * (np.sqrt(sa[k].sum(1) / sb[k].sum(1)) - 1)
            return [round(float(100 * (np.sqrt(sa.sum() / sb.sum()) - 1)), 1), round(float(np.percentile(dd, 2.5)), 1), round(float(np.percentile(dd, 97.5)), 1)]
        b1 = boot(F1["km"], F1["cams"], np.arange(o_r.size))
        _ = boot(F1["km"], F1["rt04"], np.arange(o_r.size))                # keeps the order of the random draws of rt_day_data.py
        b2 = boot(F1["km"], F1["cams"], cell)
        o["plain_lead_day_1"] = dict(in_the_two_city_boxes=part(incity), outside_the_city_boxes=part(~incity),
                                     contrast_city_minus_outside=dict(measured=float(o_r[incity].mean() - o_r[~incity].mean()), cams=float(F1["cams"][incity].mean() - F1["cams"][~incity].mean()),
                                                                      km=float(F1["km"][incity].mean() - F1["km"][~incity].mean()), rt04=float(F1["rt04"][incity].mean() - F1["rt04"][~incity].mean())),
                                     r_between_monitors=dict(cams=float(np.corrcoef(o_r, F1["cams"])[0, 1]), km=float(np.corrcoef(o_r, F1["km"])[0, 1])),
                                     share_of_monitors_closer_than_cams={k: float(np.mean(np.abs(F1[k] - o_r) < np.abs(F1["cams"] - o_r))) for k in ("km", "rt04")},
                                     km_vs_cams_rmse_change_pct=dict(resampling_monitors=b1, resampling_0p4_degree_cells=b2, cells=int(np.unique(cell).size)),
                                     rt04_vs_cams_rmse_change_pct=float(100 * (np.sqrt(np.mean((F1["rt04"] - o_r) ** 2)) / np.sqrt(np.mean((F1["cams"] - o_r) ** 2)) - 1)))
        o["plain_cams_cell_max"] = float(c[DAY1, 1][si, sj].max())
        # ---- series at the monitors of the two cities
        ser = {}
        for n in (CITY1, NEIGHBOUR):
            S = {k: np.full(len(OBS_DAYS), np.nan) for k in ("n", "obs", "km1", "cams1", "cams3", "rt1")}
            for t, v in enumerate(OBS_DAYS):
                m = inbox[n] & np.isfinite(obs[v])
                S["n"][t] = m.sum()
                if m.sum() < NMIN:
                    continue
                S["obs"][t] = obs[v][m].mean()
                a = at_monitors(m, v, which)
                for k in S:
                    if k in a:
                        S[k][t] = a[k]
            a1 = np.isfinite(S["obs"]) & np.isfinite(S["km1"])
            a3 = np.isfinite(S["obs"]) & np.isfinite(S["cams3"])
            ser[n] = dict(cams1={str(v): (None if not np.isfinite(S["cams1"][t]) else round(float(S["cams1"][t]), 2)) for t, v in enumerate(OBS_DAYS)},
                          cams3={str(v): (None if not np.isfinite(S["cams3"][t]) else round(float(S["cams3"][t]), 2)) for t, v in enumerate(OBS_DAYS)},
                          n_days_lead1=int(a1.sum()), measured_mean=float(S["obs"][a1].mean()), km1_mean=float(S["km1"][a1].mean()), rt1_mean=float(S["rt1"][a1].mean()),
                          cams1_mean=float(S["cams1"][a1].mean()), rmse_of_daily_means_cams1=float(np.sqrt(np.mean((S["cams1"][a1] - S["obs"][a1]) ** 2))),
                          rmse_of_daily_means_km1=float(np.sqrt(np.mean((S["km1"][a1] - S["obs"][a1]) ** 2))),
                          n_days_lead3=int(a3.sum()), cams3_mean=float(S["cams3"][a3].mean()), measured_mean_lead3_days=float(S["obs"][a3].mean()))
        o["series"] = ser
        # ---- case 2
        hcf = R.nearest04(c[DAY2, 1], hext, hx.shape)
        days2 = {}
        for d in R.ISSUES:
            m = inbox[CITY2] & np.isfinite(obs[d])
            days2[str(d)] = dict(monitors=int(m.sum()), measured=float(obs[d][m].mean()), cams1=float(R.bilinear(c[d, 1], lat[m], lon[m]).mean()),
                                 km1=float(snap[d][0][0][m].mean()), rt1=float(R.bilinear(rt[d, 1], lat[m], lon[m]).mean()))
        o["karachi"] = dict(monitors=int(m2.sum()), measured=float(obs[DAY2][m2].mean()), **at_monitors(m2, DAY2, which),
                            cams_monitors_above_37p5=int((R.bilinear(c[DAY2, 1], lat[m2], lon[m2]) > R.THR).sum()),
                            cams_at_monitors_min_max=[float(R.bilinear(c[DAY2, 1], lat[m2], lon[m2]).min()), float(R.bilinear(c[DAY2, 1], lat[m2], lon[m2]).max())],
                            cells_max_cams=float(hcf[hok].max()), popw_cams=float((hpop * hcf).sum() / hpop.sum()),
                            hidden_if_cells_of_cams_forecast_million=float((hpop * (habove & (hcf <= R.THR))).sum()) / 1e6,
                            city_on_every_lead1_day=days2, cams1_mean_over_those_days=float(np.mean([v["cams1"] for v in days2.values()])))
        res[which] = o
    out.update(res)

    # ---- does "old" reproduce numbers.json?
    c1, c2 = ref["case1"], ref["case2"]
    cmp = []

    def chk(name, mine, theirs, nd):
        cmp.append(dict(number=name, recomputed=round(float(mine), nd), numbers_json=theirs, identical=bool(abs(round(float(mine), nd) - float(theirs)) < 0.5 * 10 ** (-nd) + 1e-12)))
    for L in (1, 2, 3, 4, 5):
        chk(f"Lahore, CAMS at the city monitors, lead day {L}", res["old"]["lahore_city_monitors"][f"cams{L}"], c1["at_city_monitors"][f"cams{L}"], 2)
        chk(f"Delhi, CAMS at the city monitors, lead day {L}", res["old"]["delhi_city_monitors"][f"cams{L}"], c1["neighbour"][f"cams{L}"], 2)
        chk(f"plain, CAMS at the 221 monitors, lead day {L}", res["old"]["plain_monitors"][f"cams{L}"], c1["region"]["at_monitors"][f"cams{L}"], 2)
        for k in ("rmse", "bias", "r"):
            chk(f"plain, CAMS {k}, lead day {L}", res["old"]["plain_scores"][f"cams_lead{L}"][k], c1["region"]["scores_at_monitors"][f"cams_lead{L}"][k], 4)
        chk(f"Karachi, CAMS at the monitors, lead day {L}", res["old"]["karachi"][f"cams{L}"], c2["at_monitors"][f"cams{L}"], 2)
    chk("Lahore, monitors", res["old"]["lahore_city_monitors"]["monitors"], c1["monitors"], 0)
    chk("Lahore, measured", res["old"]["lahore_city_monitors"]["measured"], c1["measured"], 2)
    chk("Lahore, 1 km at the city monitors, lead day 1", res["old"]["lahore_city_monitors"]["km1"], c1["at_city_monitors"]["km1"], 2)
    chk("plain, monitors", res["old"]["plain_monitors"]["monitors"], c1["region"]["monitors"], 0)
    chk("plain, 1 km RMSE lead day 1", res["old"]["plain_scores"]["km_lead1"]["rmse"], c1["region"]["scores_at_monitors"]["km_lead1"]["rmse"], 4)
    d1 = c1["region"]["lead_day_1_detail"]
    chk("plain, contrast city minus outside, CAMS", res["old"]["plain_lead_day_1"]["contrast_city_minus_outside"]["cams"], d1["contrast_city_minus_outside"]["cams"], 2)
    chk("plain, share of monitors where the 1 km forecast is closer than CAMS", res["old"]["plain_lead_day_1"]["share_of_monitors_closer_than_cams"]["km"], d1["share_of_monitors_closer_than_cams"]["km"], 3)
    for j, nm in enumerate(("estimate", "lower bound", "upper bound")):
        chk(f"plain, 1 km vs CAMS RMSE change, resampling monitors, {nm}", res["old"]["plain_lead_day_1"]["km_vs_cams_rmse_change_pct"]["resampling_monitors"][j], d1["rmse_change_percent_95ci"]["resampling_monitors"]["km_vs_cams"][j], 1)
        chk(f"plain, 1 km vs CAMS RMSE change, resampling 0.4 degree cells, {nm}", res["old"]["plain_lead_day_1"]["km_vs_cams_rmse_change_pct"]["resampling_0p4_degree_cells"][j], d1["rmse_change_percent_95ci"]["resampling_0p4_degree_cells"]["km_vs_cams"][j], 1)
    chk("plain, CAMS cell maximum", res["old"]["plain_cams_cell_max"], c1["region"]["cams_cell_max"], 1)
    for n in (CITY1, NEIGHBOUR):
        chk(f"{n}, series mean of CAMS lead day 1", res["old"]["series"][n]["cams1_mean"], ref["series_summary"][n]["cams1_mean"], 2)
        chk(f"{n}, RMSE of the daily means, CAMS lead day 1", res["old"]["series"][n]["rmse_of_daily_means_cams1"], ref["series_summary"][n]["rmse_of_daily_means"]["cams1"], 2)
    chk("Karachi, monitors", res["old"]["karachi"]["monitors"], c2["monitors"], 0)
    chk("Karachi, CAMS monitors above 37.5", res["old"]["karachi"]["cams_monitors_above_37p5"], c2["cams_monitors_above"], 0)
    chk("Karachi, CAMS cell maximum", res["old"]["karachi"]["cells_max_cams"], c2["cells_max"]["cams"], 1)
    chk("Karachi, population-weighted CAMS", res["old"]["karachi"]["popw_cams"], c2["popw_cams"], 2)
    chk("Karachi, hidden if cells of the CAMS forecast (million)", res["old"]["karachi"]["hidden_if_cells_of_cams_forecast_million"], c2["hidden_if_cells_of_cams_forecast_million"], 3)
    chk("Karachi, mean of CAMS lead day 1 over 3-11 November", res["old"]["karachi"]["cams1_mean_over_those_days"], c2["city_mean_over_those_days"]["cams1"], 2)
    out["reproduces_numbers_json"] = dict(numbers_compared=len(cmp), identical=sum(x["identical"] for x in cmp), different=[x for x in cmp if not x["identical"]], all=cmp)
    safe_io.write_json(HERE / "lahore.json", out)

    o, n = res["old"], res["new"]
    g = lambda a, *ks: __import__("functools").reduce(lambda q, k: q[k], ks, a)  # noqa: E731
    rowsmd = []

    def row(name, *ks, nd=1, sign=False):
        f = (lambda v: f"{v:+.{nd}f}") if sign else (lambda v: f"{v:.{nd}f}")
        rowsmd.append(f"| {name} | {f(g(o, *ks))} | {f(g(n, *ks))} |")
    md = ["# The Lahore and Karachi cases: numbers that involve the CAMS forecast\n",
          f"old = three instants (reproduces `fig_rt_day/numbers.json`: {out['reproduces_numbers_json']['identical']} of {len(cmp)} numbers identical); "
          "new = the eight 3-hourly steps 00 .. 21 UTC of the valid day. Monitors, observations, the 1 km and 0.4 degree forecasts are unchanged.\n",
          "| number (ug m-3 unless stated) | old | new |", "|---|---|---|"]
    row(f"Lahore, 11 Nov, CAMS at the {o['lahore_city_monitors']['monitors']} city monitors, lead day 1 (measured {o['lahore_city_monitors']['measured']:.1f}; 1 km {o['lahore_city_monitors']['km1']:.1f}; 0.4 deg {o['lahore_city_monitors']['rt1']:.1f})", "lahore_city_monitors", "cams1")
    for L in (2, 3, 4, 5):
        row(f"Lahore, 11 Nov, CAMS, lead day {L}", "lahore_city_monitors", f"cams{L}")
    row(f"Delhi, 11 Nov, CAMS at the {o['delhi_city_monitors']['monitors']} city monitors, lead day 1 (measured {o['delhi_city_monitors']['measured']:.1f})", "delhi_city_monitors", "cams1")
    row(f"plain, {o['plain_monitors']['monitors']} monitors, CAMS mean, lead day 1 (measured {o['plain_monitors']['measured']:.1f})", "plain_monitors", "cams1")
    row("plain, CAMS RMSE, lead day 1", "plain_scores", "cams_lead1", "rmse")
    row("plain, CAMS bias, lead day 1", "plain_scores", "cams_lead1", "bias", sign=True)
    row("plain, CAMS correlation between monitors, lead day 1", "plain_scores", "cams_lead1", "r", nd=2)
    for L in (2, 3, 4, 5):
        rowsmd.append(f"| plain, CAMS RMSE / bias / r, lead day {L} | {o['plain_scores'][f'cams_lead{L}']['rmse']:.1f} / {o['plain_scores'][f'cams_lead{L}']['bias']:+.1f} / {o['plain_scores'][f'cams_lead{L}']['r']:.2f} | "
                      f"{n['plain_scores'][f'cams_lead{L}']['rmse']:.1f} / {n['plain_scores'][f'cams_lead{L}']['bias']:+.1f} / {n['plain_scores'][f'cams_lead{L}']['r']:.2f} |")
    b = lambda x: f"{x[0]:+.1f} [{x[1]:+.1f}, {x[2]:+.1f}]"  # noqa: E731
    rowsmd.append(f"| plain, 1 km vs CAMS, RMSE change %, resampling monitors | {b(o['plain_lead_day_1']['km_vs_cams_rmse_change_pct']['resampling_monitors'])} | {b(n['plain_lead_day_1']['km_vs_cams_rmse_change_pct']['resampling_monitors'])} |")
    rowsmd.append(f"| plain, 1 km vs CAMS, RMSE change %, resampling 0.4 degree cells | {b(o['plain_lead_day_1']['km_vs_cams_rmse_change_pct']['resampling_0p4_degree_cells'])} | {b(n['plain_lead_day_1']['km_vs_cams_rmse_change_pct']['resampling_0p4_degree_cells'])} |")
    row("plain, 0.4 deg input vs CAMS, RMSE change %", "plain_lead_day_1", "rt04_vs_cams_rmse_change_pct", sign=True)
    row("plain, CAMS in the two city boxes: mean", "plain_lead_day_1", "in_the_two_city_boxes", "cams", "mean")
    row("plain, CAMS in the two city boxes: RMSE", "plain_lead_day_1", "in_the_two_city_boxes", "cams", "rmse")
    row("plain, CAMS outside the city boxes: mean", "plain_lead_day_1", "outside_the_city_boxes", "cams", "mean")
    row("plain, contrast city minus outside, CAMS (measured %.1f; 1 km %.1f; 0.4 deg %.1f)" % (o["plain_lead_day_1"]["contrast_city_minus_outside"]["measured"], o["plain_lead_day_1"]["contrast_city_minus_outside"]["km"],
                                                                                             o["plain_lead_day_1"]["contrast_city_minus_outside"]["rt04"]), "plain_lead_day_1", "contrast_city_minus_outside", "cams")
    rowsmd.append(f"| plain, share of monitors where the 1 km forecast is closer than CAMS, % | {100 * o['plain_lead_day_1']['share_of_monitors_closer_than_cams']['km']:.1f} | {100 * n['plain_lead_day_1']['share_of_monitors_closer_than_cams']['km']:.1f} |")
    row("plain, largest CAMS cell of the window", "plain_cams_cell_max")
    for nme in (CITY1, NEIGHBOUR):
        row(f"{nme}, 3-11 Nov, mean of CAMS lead day 1 over {o['series'][nme]['n_days_lead1']} days (measured {o['series'][nme]['measured_mean']:.1f}; 1 km {o['series'][nme]['km1_mean']:.1f}; 0.4 deg {o['series'][nme]['rt1_mean']:.1f})", "series", nme, "cams1_mean")
        row(f"{nme}, RMSE of the daily city means, CAMS lead day 1 (1 km {o['series'][nme]['rmse_of_daily_means_km1']:.1f})", "series", nme, "rmse_of_daily_means_cams1")
    row(f"Karachi, 10 Nov, CAMS at the {o['karachi']['monitors']} monitors, lead day 1 (measured {o['karachi']['measured']:.1f}; 1 km {o['karachi']['km1']:.1f}; 0.4 deg {o['karachi']['rt1']:.1f})", "karachi", "cams1")
    for L in (2, 3, 4, 5):
        row(f"Karachi, 10 Nov, CAMS, lead day {L}", "karachi", f"cams{L}")
    row("Karachi, largest CAMS cell over the city box", "karachi", "cells_max_cams")
    row("Karachi, population-weighted CAMS", "karachi", "popw_cams", nd=2)
    rowsmd.append(f"| Karachi, monitors with CAMS above 37.5 | {o['karachi']['cams_monitors_above_37p5']} | {n['karachi']['cams_monitors_above_37p5']} |")
    row("Karachi, residents above 37.5 at 1 km in CAMS cells at or below it (million)", "karachi", "hidden_if_cells_of_cams_forecast_million", nd=3)
    row("Karachi, mean of CAMS lead day 1 over 3-11 November", "karachi", "cams1_mean_over_those_days", nd=2)
    md += rowsmd
    md.append("\nSeries of the city means, CAMS lead day 1 (old / new), by valid day:\n")
    md.append("| city | " + " | ".join(str(v)[5:] for v in OBS_DAYS[:9]) + " |")
    md.append("|---|" + "---|" * 9)
    for nme in (CITY1, NEIGHBOUR):
        md.append(f"| {nme} | " + " | ".join(("n/a" if o["series"][nme]["cams1"][str(v)] is None else f"{o['series'][nme]['cams1'][str(v)]:.1f} / {n['series'][nme]['cams1'][str(v)]:.1f}") for v in OBS_DAYS[:9]) + " |")
    md.append(f"| Karachi | " + " | ".join(f"{o['karachi']['city_on_every_lead1_day'][str(v)]['cams1']:.1f} / {n['karachi']['city_on_every_lead1_day'][str(v)]['cams1']:.1f}" for v in OBS_DAYS[:9]) + " |")
    if out["reproduces_numbers_json"]["different"]:
        md.append("\nNumbers of `numbers.json` NOT reproduced by the old reading:\n")
        for x in out["reproduces_numbers_json"]["different"]:
            md.append(f"- {x['number']}: recomputed {x['recomputed']}, numbers.json {x['numbers_json']}")
    md.append(f"\nThree-instant fields recomputed from the new files against the figure's loader: largest difference {same_fields:.3g} ug m-3 over the 45 fields.\n")
    safe_io.write_text(HERE / "lahore.md", "\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
