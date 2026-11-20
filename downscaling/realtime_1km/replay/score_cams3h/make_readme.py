#!/usr/bin/env python3
"""cams3h: README.md and reading.md of this directory, with every number taken from the JSON files written by the checks
(side_by_side.json, lahore.json, check_manual.json, ../cams3h/build_report.json, ../cams3h/raw/*.json). No score is
computed here.    python -I -B make_readme.py"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent
sys.path.insert(0, str(R / "cams3h/code"))
import safe_io  # noqa: E402

REG = ["Europe", "East Asia", "South Asia", "North America", "Africa", "South America", "North", "Other"]
REGNAME = {"East Asia": "East Asia (Korea, Japan west of 135 E, Taiwan)", "North": "North of 60 N", "Other": "Other"}


def f1(d, nd=1):
    return "n/a" if d is None else f"{d['est']:+.{nd}f} [{d['ci'][0]:+.{nd}f}, {d['ci'][1]:+.{nd}f}]"


def f2(d, nd=2):
    return "n/a" if d is None else f"{d['est']:.{nd}f} [{d['ci'][0]:.{nd}f}, {d['ci'][1]:.{nd}f}]"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def main():
    S = json.loads((HERE / "side_by_side.json").read_text())
    LH = json.loads((HERE / "lahore.json").read_text())
    CM = json.loads((HERE / "check_manual.json").read_text())
    BR = json.loads((R / "cams3h" / "build_report.json").read_text())["summary"]
    FS = json.loads((HERE / "figure_script_check.json").read_text())
    raw = [json.loads(p.read_text()) for p in sorted((R / "cams3h" / "raw").glob("2026-*.json"))]
    kc, ho = S["km_vs_cams"], S["handicap_three_instants_vs_3hourly_pct"]
    o, n = kc["old"], kc["new"]
    per, rg, of, sh, dy = S["periods"], S["rings"], S["official_sites"], S["share_lower_rmse"], S["by_issue_day"]["counts"]
    r4, cr, tw, sd = S["rt04_vs_cams"], S["km_fair_crps_vs_cams_abs_error"], S["temporal"]["whole record"], S["sedi"]
    lo, ln = LH["old"], LH["new"]
    us, eu = "United States, at an official PM2.5 site", "Europe, at an official PM2.5 site"
    sens = S["sensitivity_of_the_rule"]
    nchk = len(S["checks"])
    rows = [
        ["RMSE of the CAMS forecast, pooled (ug m-3)", f2(S["rmse"]["cams_old"]["all"]), f2(S["rmse"]["cams"]["all"])],
        ["three instants against 3-hourly mean, RMSE of CAMS", "", f"ratio {S['rmse']['cams_old']['all']['est'] / S['rmse']['cams']['all']['est']:.4f}: {f1(ho['all'])} % pooled, {f1(ho['regmean'])} % mean over regions"],
        ["**1 km vs CAMS, all monitors pooled**", f"**{f1(o['all'])}**", f"**{f1(n['all'])}**"],
        ["**1 km vs CAMS, mean over the 8 regions**", f"**{f1(o['regmean'])}**", f"**{f1(n['regmean'])}**"]]
    rows += [[f"lead day {k}, pooled", f1(o[f"lead {k}"]), f1(n[f"lead {k}"])] for k in range(1, 6)]
    rows += [[REGNAME.get(g, g), f1(o[g]), f1(n[g])] for g in REG]
    rows += [[f"{per[p]['name']}", f1(per[p]["old"]["pooled"]), f1(per[p]["new"]["pooled"])] for p in ("sep", "oct", "live")]
    rows += [["53 issue days (1-5 September included)", f1(per["all53"]["old"]["pooled"]), f1(per["all53"]["new"]["pooled"])],
             ["without the valid days whose observations are not final", f1(per["final_obs_only"]["old"]["pooled"]), f1(per["final_obs_only"]["new"]["pooled"])],
             ["read 10 km from the monitor (rings sample, 1,658,975 station-days)", f1(rg["old"]["km_ring10"]["pooled"]), f1(rg["new"]["km_ring10"]["pooled"])],
             ["same sample, at the monitor's pixel", f1(rg["old"]["km"]["pooled"]), f1(rg["new"]["km"]["pooled"])],
             ["United States, monitors at official sites (proxy)", f1(of["old"][us]["rmse"]), f1(of["new"][us]["rmse"])],
             ["Europe, monitors at official sites (proxy)", f1(of["old"][eu]["rmse"]), f1(of["new"][eu]["rmse"])],
             ["issue days with the lower pooled RMSE", f"{dy['old']['pooled_lower']} of 48 (range {dy['old']['pooled_min']:+.1f} .. {dy['old']['pooled_max']:+.1f}, median {dy['old']['pooled_median']:+.1f})",
              f"{dy['new']['pooled_lower']} of 48 (range {dy['new']['pooled_min']:+.1f} .. {dy['new']['pooled_max']:+.1f}, median {dy['new']['pooled_median']:+.1f})"],
             ["issue days whose own interval (2-degree cells) excludes zero", f"{dy['old']['pooled_interval_below_zero_cells_only']} of 48", f"{dy['new']['pooled_interval_below_zero_cells_only']} of 48"],
             [f"2-degree cells with the lower RMSE (of {sh['new']['cells']['units']} with at least 100 station-days), %", f"{f2(sh['old']['cells'], 1)} ({sh['old']['cells']['units_lower']} cells)",
              f"{f2(sh['new']['cells'], 1)} ({sh['new']['cells']['units_lower']} cells)"],
             [f"monitors with the lower RMSE (of {sh['new']['monitors']['units']:,} with at least 30 station-days), %", f2(sh["old"]["monitors"], 1), f2(sh["new"]["monitors"], 1)],
             ["0.4 degree input vs CAMS, pooled", f1(r4["old"]["all"]), f1(r4["new"]["all"])],
             ["0.4 degree input vs CAMS, mean over regions", f1(r4["old"]["regmean"]), f1(r4["new"]["regmean"])],
             [f"fair CRPS of the 1 km members ({cr['fair_crps_km']:.2f}) vs absolute error of CAMS ({cr['mae_cams_old']:.2f} / {cr['mae_cams']:.2f}), pooled", f1(cr["old"]["all"]), f1(cr["new"]["all"])],
             ["the same, mean over regions", f1(cr["old"]["regmean"]), f1(cr["new"]["regmean"])],
             ["per-monitor temporal correlation of CAMS (median over series)", f2(tw["old"]["all"]["r_cams"]), f2(tw["new"]["all"]["r_cams"])],
             ["1 km minus CAMS, temporal correlation (median of paired differences)", f1(tw["old"]["all"]["km_minus_cams_r_median_of_diff"], 3), f1(tw["new"]["all"]["km_minus_cams_r_median_of_diff"], 3)],
             ["amplitude ratio of CAMS, sd(forecast) / sd(observation)", f2(tw["old"]["all"]["amp_cams"]), f2(tw["new"]["all"]["amp_cams"])]]
    for en, short in (("above 37.5 ug m-3", "SEDI of CAMS, above 37.5 ug m-3"), ("above 75 ug m-3", "SEDI of CAMS, above 75 ug m-3"),
                      ("above the monitor's own 95th percentile, same threshold for the forecast", "SEDI of CAMS, monitor's own 95th percentile, same threshold"),
                      ("above the monitor's own 95th percentile, forecast above its own 95th percentile", "SEDI of CAMS, own 95th percentile of each series"),
                      ("above the monitor's own 99th percentile, same threshold for the forecast", "SEDI of CAMS, monitor's own 99th percentile, same threshold"),
                      ("above the monitor's own 99th percentile, forecast above its own 99th percentile", "SEDI of CAMS, own 99th percentile of each series")):
        v = sd[en]
        rows.append([f"{short} (1 km: {v['new']['km']['est']:.2f})", f"{f2(v['old']['cams'])}; 1 km minus CAMS {f1(v['old']['km_minus_cams'], 2)}",
                     f"{f2(v['new']['cams'])}; 1 km minus CAMS {f1(v['new']['km_minus_cams'], 2)}"])
    pl = "plain_lead_day_1"
    b = lambda x: f"{x[0]:+.1f} [{x[1]:+.1f}, {x[2]:+.1f}]"  # noqa: E731
    rows += [[f"Lahore, 11 November, CAMS at the {lo['lahore_city_monitors']['monitors']} city monitors (measured {lo['lahore_city_monitors']['measured']:.1f}, 1 km {lo['lahore_city_monitors']['km1']:.1f})",
              f"{lo['lahore_city_monitors']['cams1']:.1f}", f"{ln['lahore_city_monitors']['cams1']:.1f}"],
             [f"the {lo['plain_monitors']['monitors']} monitors of the plain, CAMS RMSE / bias / r between monitors",
              f"{lo['plain_scores']['cams_lead1']['rmse']:.1f} / {lo['plain_scores']['cams_lead1']['bias']:+.1f} / {lo['plain_scores']['cams_lead1']['r']:.2f}",
              f"{ln['plain_scores']['cams_lead1']['rmse']:.1f} / {ln['plain_scores']['cams_lead1']['bias']:+.1f} / {ln['plain_scores']['cams_lead1']['r']:.2f}"],
             ["the plain, 1 km vs CAMS, RMSE change (monitors resampled; 0.4 degree cells resampled)",
              f"{b(lo[pl]['km_vs_cams_rmse_change_pct']['resampling_monitors'])}; {b(lo[pl]['km_vs_cams_rmse_change_pct']['resampling_0p4_degree_cells'])}",
              f"{b(ln[pl]['km_vs_cams_rmse_change_pct']['resampling_monitors'])}; {b(ln[pl]['km_vs_cams_rmse_change_pct']['resampling_0p4_degree_cells'])}"],
             ["the plain, city-minus-outside contrast of CAMS (measured %.1f, 1 km %.1f)" % (lo[pl]["contrast_city_minus_outside"]["measured"], lo[pl]["contrast_city_minus_outside"]["km"]),
              f"{lo[pl]['contrast_city_minus_outside']['cams']:.1f}", f"{ln[pl]['contrast_city_minus_outside']['cams']:.1f}"],
             [f"Lahore / Delhi, mean of CAMS over the lead-day-1 days of 3-11 November", f"{lo['series']['Lahore']['cams1_mean']:.1f} / {lo['series']['Delhi']['cams1_mean']:.1f}",
              f"{ln['series']['Lahore']['cams1_mean']:.1f} / {ln['series']['Delhi']['cams1_mean']:.1f}"],
             [f"Karachi, 10 November, CAMS at the {lo['karachi']['monitors']} monitors (measured {lo['karachi']['measured']:.1f}, 1 km {lo['karachi']['km1']:.1f})",
              f"{lo['karachi']['cams1']:.1f}", f"{ln['karachi']['cams1']:.1f}"],
             ["Karachi, largest CAMS cell over the city box; residents above 37.5 at 1 km in CAMS cells at or below it (million)",
              f"{lo['karachi']['cells_max_cams']:.1f}; {lo['karachi']['hidden_if_cells_of_cams_forecast_million']:.3f}", f"{ln['karachi']['cells_max_cams']:.1f}; {ln['karachi']['hidden_if_cells_of_cams_forecast_million']:.3f}"]]
    one_screen = table(["quantity (RMSE change in %, 95 % interval, unless stated)", "old: three instants (`score/`)", "new: 3-hourly mean (this directory)"], rows)

    wa = CM["A_largest_abs_differences"]
    cc = CM["C_pooled_rmse_plain_numpy"]
    ep = S["europe"]
    # facts stated in the text, derived from the data (never typed by hand)
    ho_leads = [ho[f"lead {k}"]["est"] for k in range(1, 6)]
    sig_new = [g for g in REG if n[g]["ci"][1] < 0]
    sig_old = [g for g in REG if o[g]["ci"][1] < 0]
    assert sig_new == sig_old, (sig_old, sig_new)
    lead_pooled_excl = sum(n[f"lead {k}"]["ci"][1] < 0 for k in range(1, 6))
    lead_regmean_excl = sum(n[f"lead {k} regmean"]["ci"][1] < 0 for k in range(1, 6))
    per_pooled_excl = [p for p in ("sep", "oct", "live") if per[p]["new"]["pooled"]["ci"][1] < 0]
    per_regmean_incl = [per[p]["name"] for p in ("sep", "oct", "live") if per[p]["new"]["regmean"]["ci"][1] >= 0]
    na = {e["name"]: e for e in json.loads((R / "score" / "numbers.json").read_text())}
    nb = {e["name"]: e for e in json.loads((HERE / "numbers.json").read_text())}
    assert set(na) == set(nb)
    changed = [k for k in na if (na[k]["value"], na[k]["interval"]) != (nb[k]["value"], nb[k]["interval"])]
    assert all("cams" in k.lower() for k in changed)
    n_sd_diff = sum(na[k]["station_days"] != nb[k]["station_days"] for k in na)
    km_rt = json.loads((HERE / "tables" / "T03_contrasts.json").read_text())["with the CAMS forecast"]["contrasts"]["km_vs_rt04|rmse"]["all"]["all"]
    rule_rmse = {k: sens[k]["rmse"] for k in ("cams", "cams3h_trap", "cams3h_0324")}
    assert min(rule_rmse, key=rule_rmse.get) == "cams", rule_rmse
    readme = f"""# Real-time scores against the CAMS forecast with the daily mean of the paper's rule (2026-11-19)

One job: the arm `cams` of the real-time scorer read the CAMS forecast at three instants of each day; the paper's rule is the
mean of its 3-hourly steps. The steps were fetched for the 53 cycles, the arm was rebuilt at the monitors, and the UNMODIFIED
scorer (`../score/run_all.sh`, same station-days, same two-way block bootstrap, same seed) was rerun into this directory.
**In this directory the arm `cams` IS the 3-hourly daily mean**, so the tables, `numbers.json` and `REPORT.md` have the layout
of `../score/` and can replace it for any script that takes a scorer directory. Monitors are evaluation only: nothing was
fitted, selected or tuned. Nothing outside `../cams3h/` and this directory was written; no existing file was changed.

## 1. The two definitions (written before anything was changed)

Day: lead day L = 1..5 of issue day d (00 UTC cycle) is the UTC day d + L - 1, forecast hours 24(L-1) .. 24L.

**Existing arm `cams`** (`../code/assemble.py` l.86-146 and `../code/cams_points.py`, lines of `realtime_1km/rc/coupling_live.py`):
files `realtime/data/<d>/cams_fc.npz` (cycles from 2026-10-11) or `../camsfc/<d>/cams_fc.npz` (1-20 September, 10 October):
the ten leads 12, 24, ..., 120 h, float16, 451 x 900 at 0.4 degree (90 N .. 90 S, 0 .. 359.6 E, rolled to 180 W); the 0 h
instant is the analysis `realtime/data/<d>/cams.npy[1, 1]` (84 N .. 60 S). Daily mean of lead day L =
max(1/4 x(24(L-1)) + 1/2 x(24L-12) + 1/4 x(24L), 0) in float32: **three instants, 00, 12 and 24 UTC**. Reading:
bilinear in linear space on the 0.4 degree grid, `coupling_live.bilinear(F, 90, 0.4, -180, 0.4, lat, lon)`.

**The paper's rule for 2022** (Methods: "The daily mean of the CAMS forecast is the mean of its 3-hourly steps over the target
day"). Code: `eccv/world/data_pipeline/dl_cams_forecast.py`, `to_daily_leads`: steps 0, 3, ..., 117 h of the 00 UTC run, values
float32 x 1e9, lead day = mean of the steps with 24 L <= hour < 24 (L + 1): **the eight steps 00, 03, ..., 21 UTC of the day,
equal weights; the step of 24 UTC is not in the day** (neither 03 .. 24 nor 00 .. 24 with half weights). Stored in
`eccv/data/world/cams_forecast_daily/2022.zarr` and read by `experiments/figures/compare_fair.py` (l.71, 131-132:
`cams[d + 1, 0, h - 1]`, bilinear on the grid 90 N, 0 E, 0.4 degree, linear space).

**New arms** (`../cams3h/points/<d>.npz`, (5 lead days, 12,136 monitors), catalogue order; code `../cams3h/code/`):
- `cams3h` = the eight steps 24(L-1) + 0, 3, ..., 21 h, equal weights; steps in the float16 values that the chain stores,
  floor at 0, the SAME bilinear function as the existing arm. Only the time sampling differs from `cams`.
- `cams3i` = the three instants recomputed from the new files with the lines of the existing arm.
- extras: `cams3h_f32` (float32 steps), `cams3h_trap` (nine steps 00 .. 24 UTC, half weights on the ends), `cams3h_0324`
  (eight steps 03 .. 24 UTC).
The 0.4 degree input of the 1 km forecast (`rt04`) keeps its three instants: its members exist every 12 h only.

## 2. Data: the download succeeded for all 53 cycles

Dataset `cams-global-atmospheric-composition-forecasts` of the Atmosphere Data Store, type forecast, 00 UTC, variable
`particulate_matter_2.5um`, lead hours 0, 3, ..., 120 (the chain's request, `realtime/fetch_camsfc.py`, with 41 lead hours and
date ranges), through the chain's own API client (`realtime/fetch.py`, `ads()`: the only code that reads the credentials;
nothing here prints, copies or logs them). Five requests (job 23870417, sent 20:36-20:37 UTC), each answered within 2 to
3.5 minutes; files in `../cams3h/raw/` (2.16 GB, SHA-256 in the `.json` beside each, re-read and identical):

{table(["file", "cycles", "fields", "MB", "request finished (UTC)"], [[x["file"], x["request"]["date"][0], 41 * int(x["content"][0].split("'forecast_reference_time': ")[1].split(",")[0]), f"{x['bytes'] / 1e6:.1f}", x["finished"][:19]] for x in raw])}
The first job was cancelled after 3 minutes because of an error of mine in `fetch3h.py` (it expected a zip; the service
answers a one-variable request with a bare netCDF file); job 23870445 re-attached to the five requests already sent and
downloaded them. No request was sent twice and none is left in the queue.

Already on disk and used only as an independent check: `cams_forecast_daily/2026.zarr`, daily means written by the paper's own
builder from its own download, cycles to 2026-11-03 (45 of the 53; it holds no steps, and not the cycles of 4-11 November).

## 3. Result in one screen

Primary set unchanged: **{S['sample']['station_days']:,} station-days**, 48 issue days (6-20 September, 10 October - 11 November 2026), lead days 1-5,
{S['sample']['monitors']:,} monitors, {S['sample']['cells']} 2-degree cells, {S['sample']['week_blocks']} week blocks; station-days lost: {S['sample']['lost_by_requiring_both_cams_arms']} (the new arm is finite wherever the old one is; all 53
issue days: {S['sample']['all_53_issue_days']['station_days_both_arms']:,}). RMSE of the 1 km forecast {S['rmse']['km']['all']['est']:.2f} ug m-3 in both columns.

{one_screen}
Regions with a lower RMSE than CAMS: {o['regions_lower']} of 8 (old), {n['regions_lower']} of 8 (new); interval entirely below zero in {o['regions_interval_below_zero']} and {n['regions_interval_below_zero']}
(the same regions: {', '.join(sig_new)}). Title rule (T12): unchanged ({S['title_rule']['new']['verdict_strict']} in the strict
reading, {S['title_rule']['new']['verdict_lenient']} in the lenient one). Full tables: `side_by_side.md` (both readings), `tables/` and `REPORT.md` (new
reading, every table of the scorer), `lahore.md`.

**What the change is.** Read at three instants, the CAMS forecast had an RMSE {ho['all']['est']:.1f} % [{ho['all']['ci'][0]:.1f}, {ho['all']['ci'][1]:.1f}] higher than with its eight steps
(the review notes give about 4 % at the 2022 stations). The handicap is largest in East Asia ({f1(ho['East Asia'])} %) and "Other"
({f1(ho['Other'])} %); it is {f1(ho['Europe'])} % in Europe, {f1(ho['South Asia'])} % in South Asia, {f1(ho['North America'])} % in North America, and there is none in
Africa ({f1(ho['Africa'])} %), South America ({f1(ho['South America'])} %) and north of 60 N ({f1(ho['North'])} %). By lead day it is {min(ho_leads):.1f} to {max(ho_leads):.1f} %.
The headline goes from 17.0 % to **{-n['all']['est']:.1f} % [{-n['all']['ci'][1]:.1f}, {-n['all']['ci'][0]:.1f}]** pooled and from 14.3 % to **{-n['regmean']['est']:.1f} % [{-n['regmean']['ci'][1]:.1f}, {-n['regmean']['ci'][0]:.1f}]** as the mean over
regions. The pooled interval excludes zero at {lead_pooled_excl} of 5 lead days and in {len(per_pooled_excl)} of the 3 periods; the interval of the mean over regions at
{lead_regmean_excl} of 5 lead days (by period it includes zero for: {', '.join(per_regmean_incl) or 'none'}, as with three instants). The pooled RMSE of the day is
lower on {dy['new']['pooled_lower']} of 48 issue days ({dy['new']['pooled_interval_below_zero_cells_only']} with an interval of the day that excludes zero).

**What it does not change.** Every number that does not involve the CAMS forecast is identical to `../score/` to the last
digit ({len(na) - len(changed):,} of the {len(na):,} entries of `numbers.json`; the {len(changed)} that changed all carry `cams` in their name; entries with a
different number of station-days: {n_sd_diff}).

**Statements of `../score/reading.md` that need the new value** (old -> new):
- Europe against CAMS: +0.6 [-5.5, +6.3] -> {f1(n['Europe'])}; episode of 6-9 November {f1(ep['old']['episode']['valid days 6-9 November'])} -> {f1(ep['new']['episode']['valid days 6-9 November'])}
  (cells-only intervals); the other live days {f1(ep['old']['episode']['other valid days of 3-15 November'])} -> {f1(ep['new']['episode']['other valid days of 3-15 November'])}; by period {ep['old']['periods']['sep']['est']:+.1f} / {ep['old']['periods']['oct']['est']:+.1f} / {ep['old']['periods']['live']['est']:+.1f} ->
  {ep['new']['periods']['sep']['est']:+.1f} / {ep['new']['periods']['oct']['est']:+.1f} / {ep['new']['periods']['live']['est']:+.1f} % (September, October, live).
- The 0.4 degree input against CAMS: {f1(r4['old']['all'])} -> {f1(r4['new']['all'])} pooled; as the mean over regions the interval now includes
  zero ({f1(r4['new']['regmean'])}); lead day 1 {f1(r4['old']['lead 1'])} -> {f1(r4['new']['lead 1'])}.
- SEDI above 37.5 and 75 ug m-3: CAMS {sd['above 37.5 ug m-3']['old']['cams']['est']:.2f} -> {sd['above 37.5 ug m-3']['new']['cams']['est']:.2f} and {sd['above 75 ug m-3']['old']['cams']['est']:.2f} -> {sd['above 75 ug m-3']['new']['cams']['est']:.2f}; the difference 1 km minus CAMS no longer
  excludes zero ({f1(sd['above 37.5 ug m-3']['new']['km_minus_cams'], 2)} and {f1(sd['above 75 ug m-3']['new']['km_minus_cams'], 2)}).
- Temporal correlation: CAMS {tw['old']['all']['r_cams']['est']:.2f} -> {tw['new']['all']['r_cams']['est']:.2f} (1 km: {tw['new']['all']['r_km']['est']:.2f}); inside October-November the paired difference 1 km minus CAMS is
  {f1(S['temporal']['inside October-November']['old']['all']['km_minus_cams_r_median_of_diff'], 3)} -> {f1(S['temporal']['inside October-November']['new']['all']['km_minus_cams_r_median_of_diff'], 3)}.
- Days with the analysis-mode arm (27 issue days): 1 km against CAMS {S['analysis_mode_days']['old']['km_vs_cams']['pooled']['est']:+.1f} -> {f1(S['analysis_mode_days']['new']['km_vs_cams']['pooled'])}.
- Live period, East Asia: {f1(per['live']['old']['by_region']['East Asia'])} -> {f1(per['live']['new']['by_region']['East Asia'])}.

**Sensitivity to the rule** (same station-days; `side_by_side.md`): 1 km against CAMS {f1(sens['cams']['km_vs']['all'])} with the paper's eight
steps 00 .. 21 UTC, {f1(sens['cams3h_trap']['km_vs']['all'])} with the trapezoid 00 .. 24 UTC, {f1(sens['cams3h_0324']['km_vs']['all'])} with the steps 03 .. 24 UTC; RMSE of CAMS
{sens['cams']['rmse']:.2f}, {sens['cams3h_trap']['rmse']:.2f}, {sens['cams3h_0324']['rmse']:.2f}: the paper's rule gives CAMS its lowest RMSE of the three. Float32 instead of float16 steps:
RMSE {sens['cams3h_f32']['rmse']:.3f} against {sens['cams']['rmse']:.3f}, same contrast.

## 4. Checks

1. **The new files are the product of the existing arm**: for {BR['a_ten_leads_bit_identical']} of {BR['a_ten_leads_checked']} cycles the leads 12 .. 120 h of the download are
   bit-identical to the ten-lead file that the existing arm read, and for {BR['b_step0_identical_to_analysis']} of 53 the step 0 h equals the analysis used as
   its 0 h instant (`../cams3h/build_report.json`).
2. **`cams3i` reproduces the existing arm `cams`**: largest difference {BR['c_cams3i_max_abs_diff']:g} ug m-3, {100 * BR['c_cams3i_min_share_identical']:.0f} % of the values identical, same finite
   monitors, on every one of the 53 cycles ({S['tables']['cams3i_vs_existing_cams_rows_compared']:,} rows of the station-day table: largest difference {S['tables']['cams3i_vs_existing_cams_max_abs_diff']:g}).
3. **The rule is the paper's**: for the {BR['d_cycles_in_paper_builder']} cycles that the paper's own builder holds (to 3 November), its daily means and
   mine agree to {BR['d_grid_max_abs_diff']:.1g} ug m-3 on the grid (relative {BR['d_grid_max_rel_diff']:.1g}) and to {BR['d_monitors_max_abs_diff_f32']:.1g} at the monitors (float32 steps);
   the float16 storage of the steps moves a monitor value by at most {BR['e_f16_vs_f32_monitors_max_abs_diff']:.3f} ug m-3.
4. **By hand from the raw files** (`check_manual.md`): issue days 2026-09-12 (lead days 2, 5), 2026-10-24 (1, 4) and 2026-11-08
   (3, 5), ten monitors drawn once: scalar arithmetic on the four grid nodes, each step interpolated then averaged. Largest
   difference with the stored `cams3h` {wa['paper_f16_vs_cams3h']:.1g}, with `cams3i` {wa['three_f16_vs_cams3i']:.1g}, with the existing arm `cams` {wa['three_f16_vs_existing_cams']:.1g} ug m-3.
   Every row of `work/pairs.npz` equals the points files ({CM['B_table_against_points']['cams']['identical']:,} of {CM['B_table_against_points']['cams']['rows']:,} for each arm).
5. **Pooled RMSE without scorer code** (plain numpy on `work/pairs.npz`): {cc['station_days']:,} station-days; 1 km {cc['rmse']['km']:.4f}, CAMS three
   instants {cc['rmse']['cams_old']:.4f}, CAMS 3-hourly {cc['rmse']['cams']:.4f}, ratio {cc['ratio_three_instants_over_3hourly']:.4f}; changes {cc['km_vs_cams_three_instants_pct']:+.3f} % and {cc['km_vs_cams3h_pct']:+.3f} %: the values of
   the scorer's tables T02 / T03 of `../score/` and of this directory.
6. **Same station-days, same draws**: the two station-day tables have the same rows and observations; the contrasts of the
   1 km forecast against its input, the climatology and BCSD are identical in the two runs; `side_by_side.py` recomputes both
   headline contrasts on one sample and finds the values of both runs ({nchk} checks, all identical, among them the shares of
   cells, monitors and issue days of `fig_rt_skill/numbers.json` for the old reading).
7. **Lahore and Karachi** (`lahore.md`): the old reading reproduces {LH['reproduces_numbers_json']['identical']} of {LH['reproduces_numbers_json']['numbers_compared']} numbers of `fig_rt_day/numbers.json`.
8. Expected and not an error: table T14 of this directory compares the arm `cams` with the first live comparison, which used
   three instants, so its `cams` row differs; the agreement of the old arm is in `../score/tables/T14_check_live.md`.
9. **This directory works as a scorer directory for the figure of the real-time skill** (`figure_script_check.json`): a copy of
   `fig_rt_skill/rt_skill_data.py`, run in a temporary directory with `--score` set to this directory (nothing written under
   the paper folder), passes its {len(FS['script_checks'])} own checks against these tables ({'all identical' if FS['script_checks_all_identical'] else 'NOT all identical'}) and gives the shares of cells, monitors and
   issue days of the table above (identical: {all(FS['identical_to_side_by_side'].values())}). On the rings sample it gives the 0.4 degree input against CAMS at
   {f1(FS['input_vs_cams_on_the_rings_sample']['pooled'])} % (the figure's status file quotes -7.2 [-10.1, -2.6] with three instants).

Limits, unchanged: eight week blocks, so the intervals are indicative; South Asia holds 9 % of the station-days and most of
the pooled squared error.

## 5. Files

`../cams3h/` (new): `raw/2026-09a.nc`, `2026-09b.nc`, `2026-10a.nc`, `2026-10b.nc`, `2026-11.nc` (+ one `.json` each,
`requests.json`); `points/<d>.npz` (53 files: `cams3h`, `cams3i`, `cams3h_f32`, `cams3h_trap`, `cams3h_0324`, `id`, `issue`,
`valid`); `build_report.json`; `code/` (`fetch3h.py`, `fetch3h_job.sh`, `c3h_lib.py`, `build_arms.py`, `safe_io.py`); `logs/`.

This directory (new): `README.md`; `work/pairs.npz` (+ `pairs_meta.json`, `refgrade.*`): arms of the scorer with
`cams` = `cams3h`, plus `cams_old` (the existing arm), `cams3i`, `cams3h`, `cams3h_f32`, `cams3h_trap`, `cams3h_0324`;
`tables/T01..T14.{{md,json}}`, `numbers.json`, `REPORT.md`, `methods.md`, `reading.md` (scorer output and its two text parts);
`side_by_side.{{py,json,md}}`; `check_manual.{{py,json,md}}`; `lahore.{{py,json,md}}`; `check_figure_script.py`,
`figure_script_check.json`; `make_readme.py`; `logs/` (job 23870539 = the scorer; `arms.txt` = its ARMS argument).

## 6. Re-run

```bash
R=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay; PY="/projappl/project_462001804/envs/world/bin/python -I -B"
sbatch $R/cams3h/code/fetch3h_job.sh --deadline <UTC time>      # only if raw/ is empty; re-attaches to its own requests
$PY $R/cams3h/code/build_arms.py                               # raw -> points + build_report.json (2 min, login node)
sbatch --output=$R/score_cams3h/logs/score_%j.out $R/score/run_all_job.sh $R/points $R/score_cams3h "$(cat $R/score_cams3h/logs/arms.txt)" km
$PY $R/score_cams3h/check_manual.py; $PY $R/score_cams3h/side_by_side.py; $PY $R/score_cams3h/lahore.py
$PY $R/score_cams3h/check_figure_script.py; $PY $R/score_cams3h/make_readme.py
$PY $R/score/make_report.py --out $R/score_cams3h              # REPORT.md again, with reading.md of this directory
```
"""
    reading = f"""# Reading (one page)

**This run.** The scores of `../score/` recomputed with ONE change: the CAMS forecast (`cams`) has the daily mean of the paper's
rule, the mean of its eight 3-hourly steps 00 .. 21 UTC of the valid day, instead of three instants (00, 12, 24 UTC). Same
{S['sample']['station_days']:,} station-days (48 issue days, {S['sample']['monitors']:,} monitors), same two-way block bootstrap and draws. Every number that does not
involve the CAMS forecast is identical to `../score/REPORT.md`, whose reading stays valid for them. Details, checks and the
two readings side by side: `README.md`, `side_by_side.md`.

**Against the operational forecast.** RMSE {S['rmse']['km']['all']['est']:.2f} against {S['rmse']['cams']['all']['est']:.2f} ug m-3 for the CAMS forecast of the same cycle ({S['rmse']['cams_old']['all']['est']:.2f} when it is
read at three instants: {f1(ho['all'])} %): {f1(n['all'])} % pooled and {f1(n['regmean'])} % as the mean over regions (three instants: -17.0
[-23.1, -10.0] and -14.3 [-19.0, -5.8]), at every lead day ({n['lead 1']['est']:+.1f} % on lead day 1, {n['lead 5']['est']:+.1f} % on lead day 5) and in each period (September
{per['sep']['new']['pooled']['est']:+.1f} %, October {per['oct']['new']['pooled']['est']:+.1f} %, live {per['live']['new']['pooled']['est']:+.1f} %; T03, T11). The RMSE is lower in {n['regions_lower']} of 8 regions, with intervals below zero in
{', '.join(f"{g} ({n[g]['est']:+.1f} %)" for g in sig_new)}; on {dy['new']['pooled_lower']} of 48 issue days; in
{sh['new']['cells']['est']:.1f} % [{sh['new']['cells']['ci'][0]:.1f}, {sh['new']['cells']['ci'][1]:.1f}] of the 2-degree cells. Read 10 km from the monitor it is still {-rg['new']['km_ring10']['pooled']['est']:.1f} % [{-rg['new']['km_ring10']['pooled']['ci'][1]:.1f}, {-rg['new']['km_ring10']['pooled']['ci'][0]:.1f}] below CAMS (T08); at
official sites (proxy) {f1(of['new'][us]['rmse'])} % in the United States and {f1(of['new'][eu]['rmse'])} % in Europe (T13). Without the valid days whose
observations are provisional: {per['final_obs_only']['new']['pooled']['est']:+.1f} % pooled, {per['final_obs_only']['new']['regmean']['est']:+.1f} % over regions (T11).

**Limits that involve the CAMS forecast.**
- *Europe is not below CAMS:* {f1(n['Europe'])} % over the record; the episode of 6-9 November {f1(ep['new']['episode']['valid days 6-9 November'])} %, the other live days
  {f1(ep['new']['episode']['other valid days of 3-15 November'])} % (T10, cells-only intervals).
- *The 0.4-degree input against CAMS:* {f1(r4['new']['all'])} % pooled, {f1(r4['new']['regmean'])} % over regions (`side_by_side.md`; three instants:
  {f1(r4['old']['all'])} and {f1(r4['old']['regmean'])}); the 1 km forecast is {f1(km_rt)} % against that input (unchanged).
- *Day-to-day signal:* per-monitor temporal correlation {tw['new']['all']['r_km']['est']:.2f} for the 1 km forecast against {tw['new']['all']['r_cams']['est']:.2f} for CAMS (paired difference
  {f1(tw['new']['all']['km_minus_cams_r_median_of_diff'], 3)}); amplitude ratio of CAMS {tw['new']['all']['amp_cams']['est']:.2f} (T04).
- *Extremes:* SEDI above 37.5 ug m-3 {sd['above 37.5 ug m-3']['new']['km']['est']:.2f} against {sd['above 37.5 ug m-3']['new']['cams']['est']:.2f} for CAMS, a difference whose interval now includes zero
  ({f1(sd['above 37.5 ug m-3']['new']['km_minus_cams'], 2)}); for a monitor's own 95th percentile {sd["above the monitor's own 95th percentile, same threshold for the forecast"]['new']['km']['est']:.2f} against {sd["above the monitor's own 95th percentile, same threshold for the forecast"]['new']['cams']['est']:.2f} (T07).
- *A CRPS against an absolute error* ({f1(cr['new']['all'])} % against CAMS) does not measure the same thing.

**Title rule (T12):** unchanged, not met as written (mean over regions {f1(n['regmean'])} %; intervals excluding zero in {n['regions_interval_below_zero']} of 8 regions).
"""
    safe_io.write_text(HERE / "README.md", readme)
    safe_io.write_text(HERE / "reading.md", reading)
    print(readme)
    print("=" * 30, "reading.md")
    print(reading)


if __name__ == "__main__":
    main()
