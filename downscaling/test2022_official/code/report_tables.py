#!/usr/bin/env python3
"""A7: markdown tables from results/scores_official_2022.json (written by score_official.py) -> results/tables_official_2022.md.
No data is read here other than that result file.
    python -I report_tables.py [--selftest]"""
import argparse
import json
import time
from pathlib import Path

O = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/test2022_official")
ALL = "All six networks"
LAB = {"ds_sameday": "downscaler, same-day inputs", "cams0_bil": "CAMS first-day field, bilinear", "bcsd": "BCSD of the CAMS first-day field",
       "clim_ens": "climatology ensemble (8 draws)", "clim_med": "climatology median of the downscaler", "clim_paper": "climatology (paper's definition)",
       "ghap": "GHAP 2022 itself (reference, not a model)", "chain": "chain (0.4° forecast -> downscaler)",
       "chain_t6fc": "chain, transported PM2.5 from the forecast", "chain_rtseed": "chain, real-time adapter's noise seeds",
       "coarse": "0.4° forecast (chain input), bilinear", "camsfc": "CAMS forecast (paper, equal issuance)", "aurora": "Aurora (paper, equal issuance)",
       "camsfc_ds": "CAMS forecast -> downscaler", "bcsd_fc": "BCSD of the 0.4° forecast (8 members)", "persist_ds": "downscaled field of the day before the issue, persisted",
       "foehn": "Foehn 1 km (paper; retrospective inputs)", "obs_persist": "station value of the day before the issue"}
WN = {"pixel": "station pixel", "ring1km": "ring 1 km", "ring3km": "ring 3 km", "ring10km": "ring 10 km"}


def pct(r, key="rel_rmse"):
    if r is None or r.get(key) is None:
        return "–"
    s = f"{100 * r[key]:+.1f} %"
    ci = r.get(key + "_ci")
    if ci:
        star = "" if ci[0] <= 0 <= ci[1] else "*"
        s += f" [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]{star}"
    return s


def num(v, nd=2, sign=False):
    if v is None:
        return "–"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def arm_rows(T, arms, group, leads, key, nd=2, sign=False):
    L = ["| system | " + " | ".join(f"lead day {x}" if x.isdigit() else x for x in leads) + " |", "|---|" + "---|" * len(leads)]
    for a in arms:
        if a not in T["arms"]:
            continue
        g = T["arms"][a][group]
        L.append(f"| {LAB.get(a, a)} (`{a}`) | " + " | ".join(num((g.get(x) or {}).get(key), nd, sign) for x in leads) + " |")
    return L


def contrast_rows(T, pairs, group, leads, key="rel_rmse", extra=None):
    L = ["| A − B | " + " | ".join(f"lead day {x}" if x.isdigit() else x for x in leads) + " |", "|---|" + "---|" * len(leads)]
    for A, B in pairs:
        c = T["contrasts"].get(f"{A} - {B}")
        if c is None and extra:
            c = (extra.get(f"{A} - {B}") or {}).get("groups")
        if c is None:
            continue
        L.append(f"| `{A}` − `{B}` | " + " | ".join(pct(c[group].get(x), key) for x in leads) + " |")
    return L


def group_rows(T, A, B, groups, leads, key="rel_rmse"):
    c = T["contrasts"].get(f"{A} - {B}")
    if c is None:
        return []
    L = [f"`{A}` − `{B}`", "", "| network | station-days (all leads) | " + " | ".join(f"lead day {x}" if x.isdigit() else x for x in leads) + " |",
         "|---|---|" + "---|" * len(leads)]
    for g in groups:
        last = c[g].get(leads[-1]) or {}
        L.append(f"| {g} | {last.get('n', '–'):,} | " + " | ".join(pct(c[g].get(x), key) for x in leads) + " |" if last else f"| {g} | – |" + " – |" * len(leads))
    return L + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    src = O / "results" / ("selftest_scores.json" if a.selftest else "scores_official_2022.json")
    R = json.loads(src.read_text())
    groups = R["groups"]
    P = R["tables"]["pixel"]
    S, C, F = P["S"], P["C"], P.get("F")
    L5 = ["1", "2", "3", "4", "5", "all"]
    out = [f"# Amendment A7: tables ({'SELFTEST, SYNTHETIC OBSERVATIONS, NOT RESULTS' if R['selftest_FAKE_observations'] else 'official stations of the paper, 2022'})", "",
           f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `code/report_tables.py` from `{src}` (written {R['written']}). "
           f"System: `{R['system']['system']}`, {R['system']['label']}. Units µg m⁻³. Relative changes are A / B − 1 on identical "
           f"station-days; brackets = 95 % interval of the {R['bootstrap']}; * = interval excludes zero. "
           "CRPS of a single field = its absolute error.", "",
           f"**Caution for every forecast-mode table:** {R['caution']}.", ""]
    st = R["stations"]
    out += ["## 0. Stations", "",
            f"Official stations with ≥ 30 valid days: {st['official_ge30_days_total']} ({st['official_ge30_days']}); outside the "
            f"downscaler's mask: {st['dropped_not_in_mask_total']} ({st['dropped_not_in_mask']}); scored: {st['n_stations']} stations on "
            f"{st['n_pixels']} pixels ({st['by_network']}). Rings kept (half-ring rule): "
            + ", ".join(f"{k} {v['station_pixels_with_half_ring_in_mask']} pixels" for k, v in st["rings"].items()) + ".", ""]
    if R.get("checks"):
        out += ["## 1. Checks of the reading (A7(g))", ""]
        for k, c in R["checks"].items():
            out += [f"- `{k}`: pass = {c['pass']}, largest RMSE difference {c['max_abs_rmse_diff']:.2e} µg m⁻³ ({c['file']})."]
            for r in c["rows"][:4 if k != "frozen_test_pixel_station_rmse" else 15]:
                out += ["  - " + ", ".join(f"{kk} {vv:.3f}" if isinstance(vv, float) else f"{kk} {vv}" for kk, vv in r.items())]
        out += [""]
    # ---------------------------------------------------------------- table S
    out += ["## 2. Same-day mode (table S): the frozen downscaler with the frozen test's inputs, all 365 days of 2022", "",
            S["note"] + f". {S['n_station_days']:,} station-days, {S['n_stations']} stations.", "",
            "### 2.1 Station pixel, by network: RMSE", "",
            "| network | station-days | " + " | ".join(f"`{x}`" for x in ("ds_sameday", "cams0_bil", "bcsd", "clim_paper", "clim_med", "clim_ens", "ghap")) + " |",
            "|---|---|---|---|---|---|---|---|---|"]
    for g in groups:
        row = [num((S["arms"][x][g]["sameday"] or {}).get("rmse")) for x in ("ds_sameday", "cams0_bil", "bcsd", "clim_paper", "clim_med", "clim_ens", "ghap")]
        n = (S["arms"]["ds_sameday"][g]["sameday"] or {}).get("n", 0)
        out.append(f"| {g} | {n:,} | " + " | ".join(row) + " |")
    out += ["", "### 2.2 Station pixel, by network: relative RMSE change of the downscaler", "",
            "| network | − CAMS first-day field | − BCSD | − climatology (paper) | − climatology median | share of stations better than CAMS | median station r: downscaler / CAMS / BCSD | amplitude ratio: downscaler / CAMS |",
            "|---|---|---|---|---|---|---|---|"]
    for g in groups:
        cc = lambda B: pct(S["contrasts"][f"ds_sameday - {B}"][g]["sameday"])            # noqa: E731
        d, c0, b0 = (S["arms"][x][g]["sameday"] or {} for x in ("ds_sameday", "cams0_bil", "bcsd"))
        out.append(f"| {g} | {cc('cams0_bil')} | {cc('bcsd')} | {cc('clim_paper')} | {cc('clim_med')} | "
                   f"{num(d.get('share_stations_rmse_below_cams0_bil'))} of {d.get('n_stations_ge_min_days', '–')} | "
                   f"{num(d.get('median_station_r'), 3)} / {num(c0.get('median_station_r'), 3)} / {num(b0.get('median_station_r'), 3)} | "
                   f"{num(d.get('median_station_amp'))} / {num(c0.get('median_station_amp'))} |")
    out += ["", "### 2.3 Station pixel, by network: bias, CRPS, spread/skill", "",
            "| network | bias: downscaler / CAMS / BCSD / climatology (paper) | CRPS: downscaler / climatology ensemble / CAMS (abs. error) | CRPS change vs climatology ensemble | spread/skill: downscaler / climatology ensemble | observations inside the member range: downscaler / climatology ensemble (expected 0.78) |",
            "|---|---|---|---|---|---|"]
    for g in groups:
        A = {x: (S["arms"][x][g]["sameday"] or {}) for x in S["arms"]}
        out.append(f"| {g} | {num(A['ds_sameday'].get('bias'), 2, True)} / {num(A['cams0_bil'].get('bias'), 2, True)} / {num(A['bcsd'].get('bias'), 2, True)} / "
                   f"{num(A['clim_paper'].get('bias'), 2, True)} | {num(A['ds_sameday'].get('crps'))} / {num(A['clim_ens'].get('crps'))} / {num(A['cams0_bil'].get('crps'))} | "
                   f"{pct(S['contrasts']['ds_sameday - clim_ens'][g]['sameday'], 'rel_crps')} | {num(A['ds_sameday'].get('spread_skill'))} / {num(A['clim_ens'].get('spread_skill'))} | "
                   f"{num(A['ds_sameday'].get('coverage_member_range'))} / {num(A['clim_ens'].get('coverage_member_range'))} |")
    out += ["", "### 2.4 Station pixel and rings: relative RMSE change of the downscaler, and GHAP's own RMSE", ""]
    for B in ("cams0_bil", "bcsd", "clim_med"):
        out += [f"`ds_sameday` − `{B}`", "", "| network | " + " | ".join(WN[w] for w in R["where"]) + " |", "|---|---|---|---|---|"]
        for g in groups:
            out.append(f"| {g} | " + " | ".join(pct(R["tables"][w]["S"]["contrasts"][f"ds_sameday - {B}"][g]["sameday"]) for w in R["where"]) + " |")
        out.append("")
    out += ["RMSE of `ds_sameday` / `ghap` (and stations in the table)", "", "| network | " + " | ".join(WN[w] for w in R["where"]) + " |", "|---|---|---|---|---|"]
    for g in groups:
        cells = []
        for w in R["where"]:
            T = R["tables"][w]["S"]["arms"]
            d, gh = T["ds_sameday"][g]["sameday"] or {}, T["ghap"][g]["sameday"] or {}
            cells.append(f"{num(d.get('rmse'))} / {num(gh.get('rmse'))} ({d.get('n_stations_ge_min_days', '–')})")
        out.append(f"| {g} | " + " | ".join(cells) + " |")
    # ---------------------------------------------------------------- table C
    armsC = ["chain", "chain_t6fc", "chain_rtseed", "coarse", "camsfc", "aurora", "camsfc_ds", "bcsd_fc", "persist_ds", "ds_sameday",
             "cams0_bil", "bcsd", "clim_paper", "clim_med", "clim_ens", "ghap"]
    out += ["", "## 3. Forecast mode (table C): 214 issue days, 1 May – 30 November 2022, lead days 1–5", "",
            C["note"], "", f"{C['n_station_days']:,} station-days, {C['n_stations']} stations, {C['n_start_dates']} issue days. Lead day 5 of "
            "Aurora is computed here with the paper's rule (the paper reports lead days 1–4).", ""]
    for key, title, nd, sign in (("rmse", "RMSE", 2, False), ("crps", "fair CRPS (absolute error for single fields)", 2, False), ("bias", "bias", 2, True),
                                 ("spread_skill", "spread/skill", 2, False), ("median_station_r", "median per-station temporal correlation", 3, False),
                                 ("median_station_amp", "median per-station amplitude ratio (s.d. of forecast / s.d. of observation)", 2, False),
                                 ("share_stations_rmse_below_camsfc", "share of stations with a lower RMSE than the CAMS forecast", 2, False)):
        leads = L5 if key in ("rmse", "crps", "bias", "spread_skill") else L5[:-1]
        out += [f"### 3.{['rmse', 'crps', 'bias', 'spread_skill', 'median_station_r', 'median_station_amp', 'share_stations_rmse_below_camsfc'].index(key) + 1} "
                f"All six networks, station pixel: {title}", ""] + arm_rows(C, armsC, ALL, leads, key, nd, sign) + [""]
    out += ["### 3.7b All six networks, station pixel: share of observations inside the member range (below lowest / above highest member); "
            "expected 0.78 for 8 exchangeable members", "",
            "| system | " + " | ".join(f"lead day {x}" if x.isdigit() else x for x in L5) + " |", "|---|" + "---|" * len(L5)]
    for a_ in ("chain", "chain_t6fc", "chain_rtseed", "coarse", "camsfc_ds", "bcsd_fc", "persist_ds", "ds_sameday", "clim_ens"):
        g = C["arms"][a_][ALL]
        out.append(f"| {LAB.get(a_, a_)} (`{a_}`) | " + " | ".join(
            "–" if not g.get(x) or g[x].get("coverage_member_range") is None else
            f"{g[x]['coverage_member_range']:.2f} ({g[x]['share_obs_below_lowest_member']:.2f} / {g[x]['share_obs_above_highest_member']:.2f})" for x in L5) + " |")
    out += [""]
    n30 = (C["arms"]["chain"][ALL]["1"] or {}).get("n_stations_ge_min_days")
    out += [f"Per-station statistics: stations with at least {C['min_common_days_for_station_statistics']} common days at that lead ({n30} at lead day 1).", ""]
    pairs1 = [("chain", b) for b in ("coarse", "camsfc", "aurora", "clim_paper", "clim_med", "bcsd_fc", "camsfc_ds", "persist_ds", "ds_sameday", "ghap")]
    pairs2 = [("chain_t6fc", b) for b in ("chain", "coarse", "camsfc", "aurora", "clim_paper", "bcsd_fc")] + [("chain_rtseed", "chain")]
    pairs3 = [("coarse", "camsfc"), ("coarse", "aurora"), ("aurora", "camsfc"), ("camsfc_ds", "camsfc"), ("camsfc_ds", "coarse"), ("bcsd_fc", "coarse"),
              ("bcsd_fc", "camsfc"), ("persist_ds", "camsfc"), ("ds_sameday", "camsfc"), ("ds_sameday", "cams0_bil"), ("ds_sameday", "bcsd"),
              ("ds_sameday", "clim_paper"), ("camsfc", "clim_paper"), ("coarse", "clim_paper"), ("chain", "obs_persist"), ("camsfc", "obs_persist")]
    out += ["### 3.8 All six networks, station pixel: relative RMSE change", ""] + contrast_rows(C, pairs1, ALL, L5) + [""] + \
           contrast_rows(C, pairs2, ALL, L5) + [""] + contrast_rows(C, pairs3, ALL, L5, extra=C.get("contrasts_own_days")) + \
           ["", "(`obs_persist` rows: own common station-days, those where the station also has a value on the day before the issue.)", "",
            "### 3.9 All six networks, station pixel: relative CRPS change (ensemble against ensemble unless B is a single field)", ""] + \
           contrast_rows(C, [("chain", "coarse"), ("chain", "clim_ens"), ("chain", "bcsd_fc"), ("chain", "camsfc_ds"), ("chain", "persist_ds"),
                             ("chain", "ds_sameday"), ("chain_t6fc", "chain"), ("chain_t6fc", "clim_ens"), ("chain_t6fc", "coarse"),
                             ("chain", "camsfc"), ("chain", "aurora"), ("ds_sameday", "clim_ens")], ALL, L5, "rel_crps") + [""]
    out += ["### 3.10 By network, station pixel: relative RMSE change", ""]
    for A, B in (("chain", "camsfc"), ("chain", "coarse"), ("chain", "aurora"), ("chain", "clim_paper"), ("chain", "bcsd_fc"), ("chain", "ds_sameday"),
                 ("chain", "camsfc_ds"), ("chain", "persist_ds"), ("chain_t6fc", "chain"), ("chain_t6fc", "camsfc"), ("chain_t6fc", "coarse"),
                 ("chain_t6fc", "clim_paper"), ("chain_rtseed", "chain"), ("coarse", "camsfc"), ("camsfc_ds", "camsfc"), ("ds_sameday", "cams0_bil"),
                 ("ds_sameday", "camsfc"), ("camsfc", "clim_paper")):
        out += group_rows(C, A, B, groups, L5)
    out += ["### 3.11 By network, station pixel: RMSE, all lead days pooled, and per-station statistics at lead days 1 and 3", "",
            "| network | station-days | " + " | ".join(f"`{x}`" for x in ("chain", "chain_t6fc", "coarse", "camsfc", "aurora", "camsfc_ds", "bcsd_fc", "ds_sameday", "clim_paper", "ghap")) + " |",
            "|---|---|" + "---|" * 10]
    for g in groups:
        out.append(f"| {g} | {(C['arms']['chain'][g]['all'] or {}).get('n', 0):,} | " + " | ".join(
            num((C["arms"][x][g]["all"] or {}).get("rmse")) for x in ("chain", "chain_t6fc", "coarse", "camsfc", "aurora", "camsfc_ds", "bcsd_fc", "ds_sameday", "clim_paper", "ghap")) + " |")
    for ld in ("1", "3"):
        out += ["", f"Lead day {ld}: share of stations better than the CAMS forecast (chain / chain_t6fc / coarse / aurora), median station r "
                f"(chain / coarse / camsfc), amplitude ratio (chain / coarse / camsfc), bias (chain / coarse / camsfc)", "",
                "| network | stations | share better than CAMS forecast | median station r | amplitude ratio | bias |", "|---|---|---|---|---|---|"]
        for g in groups:
            A = {x: (C["arms"][x][g][ld] or {}) for x in ("chain", "chain_t6fc", "coarse", "aurora", "camsfc")}
            k = "share_stations_rmse_below_camsfc"
            out.append(f"| {g} | {A['chain'].get('n_stations_ge_min_days', '–')} | {num(A['chain'].get(k))} / {num(A['chain_t6fc'].get(k))} / {num(A['coarse'].get(k))} / {num(A['aurora'].get(k))} | "
                       f"{num(A['chain'].get('median_station_r'), 3)} / {num(A['coarse'].get('median_station_r'), 3)} / {num(A['camsfc'].get('median_station_r'), 3)} | "
                       f"{num(A['chain'].get('median_station_amp'))} / {num(A['coarse'].get('median_station_amp'))} / {num(A['camsfc'].get('median_station_amp'))} | "
                       f"{num(A['chain'].get('bias'), 2, True)} / {num(A['coarse'].get('bias'), 2, True)} / {num(A['camsfc'].get('bias'), 2, True)} |")
    # ---------------------------------------------------------------- table F
    if F:
        armsF = ["foehn", "chain", "chain_t6fc", "coarse", "camsfc", "aurora", "camsfc_ds", "bcsd_fc", "ds_sameday", "clim_paper", "ghap"]
        out += ["", "## 4. With the paper's 1 km forecast (table F): the issue days that have a Foehn forecast", "", F["note"], "",
                f"{F['n_station_days']:,} station-days, {F['n_stations']} stations, {F['n_start_dates']} issue days.", "",
                "### 4.1 All six networks, station pixel: RMSE", ""] + arm_rows(F, armsF, ALL, L5, "rmse") + \
               ["", "### 4.2 All six networks, station pixel: fair CRPS", ""] + arm_rows(F, armsF, ALL, L5, "crps") + \
               ["", "### 4.3 All six networks, station pixel: spread/skill, bias", ""] + arm_rows(F, ["foehn", "chain", "chain_t6fc", "coarse"], ALL, L5, "spread_skill") + [""] + \
               arm_rows(F, ["foehn", "chain", "chain_t6fc", "coarse", "camsfc", "aurora"], ALL, L5, "bias", 2, True) + \
               ["", "### 4.4 All six networks, station pixel: relative RMSE change", ""] + \
               contrast_rows(F, [("chain", "foehn"), ("chain_t6fc", "foehn"), ("foehn", "camsfc"), ("foehn", "aurora"), ("foehn", "coarse"), ("foehn", "ds_sameday"),
                                 ("foehn", "clim_paper"), ("chain", "camsfc"), ("chain_t6fc", "camsfc"), ("chain", "coarse"), ("coarse", "camsfc")], ALL, L5) + \
               ["", "Relative CRPS change", ""] + contrast_rows(F, [("chain", "foehn"), ("chain_t6fc", "foehn"), ("chain", "coarse")], ALL, L5, "rel_crps") + \
               ["", "### 4.5 By network, station pixel: relative RMSE change", ""]
        for A, B in (("chain", "foehn"), ("chain_t6fc", "foehn"), ("foehn", "camsfc"), ("chain", "camsfc")):
            out += group_rows(F, A, B, groups, L5)
        out += ["| network | station-days | " + " | ".join(f"`{x}` RMSE (all leads)" for x in ("foehn", "chain", "chain_t6fc", "coarse", "camsfc", "aurora")) + " |", "|---|---|---|---|---|---|---|---|"]
        for g in groups:
            out.append(f"| {g} | {(F['arms']['foehn'][g]['all'] or {}).get('n', 0):,} | " + " | ".join(
                num((F["arms"][x][g]["all"] or {}).get("rmse")) for x in ("foehn", "chain", "chain_t6fc", "coarse", "camsfc", "aurora")) + " |")
    # ---------------------------------------------------------------- rings, forecast mode
    out += ["", "## 5. Forecast mode on the rings (table C per 'where'; all lead days pooled)", "",
            "Each ring has its own station set (half-ring rule). The coarse systems keep their value at the station coordinates.", ""]
    for A, B in (("chain", "camsfc"), ("chain", "coarse"), ("chain", "bcsd_fc"), ("chain", "clim_med"), ("chain", "ds_sameday"), ("chain_t6fc", "camsfc")):
        out += [f"`{A}` − `{B}`, relative RMSE change", "", "| network | " + " | ".join(WN[w] for w in R["where"]) + " |", "|---|---|---|---|---|"]
        for g in groups:
            out.append(f"| {g} | " + " | ".join(pct((R["tables"][w]["C"]["contrasts"].get(f"{A} - {B}") or {}).get(g, {}).get("all")) for w in R["where"]) + " |")
        out.append("")
    out += ["RMSE of `chain` / `ghap` / `camsfc`, all lead days pooled (stations at lead day 1)", "", "| network | " + " | ".join(WN[w] for w in R["where"]) + " |", "|---|---|---|---|---|"]
    for g in groups:
        cells = []
        for w in R["where"]:
            T = R["tables"][w]["C"]["arms"]
            cells.append(f"{num((T['chain'][g]['all'] or {}).get('rmse'))} / {num((T['ghap'][g]['all'] or {}).get('rmse'))} / {num((T['camsfc'][g]['all'] or {}).get('rmse'))} "
                         f"({(T['chain'][g]['1'] or {}).get('n_stations_ge_min_days', '–')})")
        out.append(f"| {g} | " + " | ".join(cells) + " |")
    # ---------------------------------------------------------------- GHAP vs stations
    if "ghap_vs_stations_2022" in R:
        out += ["", "## 6. GHAP 2022 against these stations (description of the target, not a model score)", "",
                "Share of station-days with |GHAP − station| ≤ 0.5 µg m⁻³ / share of stations with daily R > 0.999 / median station R / RMSE", "",
                "| network | " + " | ".join(WN[w] for w in R["where"]) + " |", "|---|---|---|---|---|"]
        for g in groups:
            cells = []
            for w in R["where"]:
                p = R["ghap_vs_stations_2022"][g][w]
                cells.append("–" if p["share_station_days_within_0p5"] is None else
                             f"{100 * p['share_station_days_within_0p5']:.1f} % / {100 * p['share_stations_R_gt_0p999']:.1f} % / {p['median_station_R']:.3f} / {p['rmse']:.2f}")
            out.append(f"| {g} | " + " | ".join(cells) + " |")
    # ---------------------------------------------------------------- pixel against ring
    rings = [w for w in R["where"] if w != "pixel" and "C_pixel_vs_ring" in R["tables"][w]]
    if rings:
        out += ["", "## 7. The station pixel against the ring mean of the same arm (paired; stations that have the ring)", "",
                "Relative RMSE change of the value at the station pixel against the ring mean (negative = the pixel is closer to the "
                "station than the mean of the ring around it). `ghap` shows how much of a pixel advantage is the station fit of the target.", "",
                "### 7.1 Forecast mode (table C), all lead days pooled", ""]
        for n in ("chain", "chain_t6fc", "camsfc_ds", "bcsd_fc", "ds_sameday", "clim_med", "ghap"):
            out += [f"`{n}`: pixel − ring", "", "| network | " + " | ".join(WN[w] for w in rings) + " |", "|---|" + "---|" * len(rings)]
            for g in groups:
                out.append(f"| {g} | " + " | ".join(pct(R["tables"][w]["C_pixel_vs_ring"]["contrasts"][f"{n}@pixel - {n}@ring"][g].get("all")) for w in rings) + " |")
            out.append("")
        out += ["`chain`: pixel − ring by lead day, all six networks", "", "| ring | " + " | ".join(f"lead day {x}" if x.isdigit() else x for x in L5) + " |", "|---|" + "---|" * len(L5)]
        for w in rings:
            c = R["tables"][w]["C_pixel_vs_ring"]["contrasts"]["chain@pixel - chain@ring"][ALL]
            out.append(f"| {WN[w]} | " + " | ".join(pct(c.get(x)) for x in L5) + " |")
        out += ["", "`chain`: relative CRPS change pixel − ring, all lead days pooled", "", "| network | " + " | ".join(WN[w] for w in rings) + " |", "|---|" + "---|" * len(rings)]
        for g in groups:
            out.append(f"| {g} | " + " | ".join(pct(R["tables"][w]["C_pixel_vs_ring"]["contrasts"]["chain@pixel - chain@ring"][g].get("all"), "rel_crps") for w in rings) + " |")
        out += ["", "### 7.2 Same-day mode (table S), 365 days", ""]
        for n in ("ds_sameday", "bcsd", "clim_med", "ghap"):
            out += [f"`{n}`: pixel − ring", "", "| network | " + " | ".join(WN[w] for w in rings) + " |", "|---|" + "---|" * len(rings)]
            for g in groups:
                out.append(f"| {g} | " + " | ".join(pct(R["tables"][w]["S_pixel_vs_ring"]["contrasts"][f"{n}@pixel - {n}@ring"][g].get("sameday")) for w in rings) + " |")
            out.append("")
    dst = O / "results" / ("selftest_tables.md" if a.selftest else "tables_official_2022.md")
    dst.write_text("\n".join(out) + "\n")
    print(f"wrote {dst} ({len(out)} lines)", flush=True)


if __name__ == "__main__":
    main()
