#!/usr/bin/env python3
"""WP1b step 2: every score and table of the real-time-mode reforecasts at the monitors.
Evaluation only: nothing is trained, selected, tuned or corrected with these scores.

  python score.py --pairs S/work/pairs.npz --out S [--focus km[,kmcal]] [--B 2000] [--Bt 400] [--only T03,T05]

--focus: the 1 km arms (the first is the reference 1 km forecast; a second one, e.g. a calibrated ensemble read by
build_pairs.py from another points directory, is scored on the same station-days and contrasted with every baseline and
with the first). Every table states its arms, issue days and station-days; all arms of a table share the station-days.
Bootstrap: sc_lib.Sample (two-way block bootstrap over 2-degree cells and calendar weeks)."""
import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sc_lib as L  # noqa: E402
from sc_lib import REGIONS, Sample, basic_scores, fci, fmt, md_table, pct, ratio_contrast, rel_mean, rel_rmse, views

PERIODS = {
    "clean": ("2026-09-06", "2026-11-11", "issue days 6-20 September and 10 October - 11 November 2026"),
    "sep": ("2026-09-06", "2026-09-20", "September reforecast, issue days 6-20 September"),
    "oct": ("2026-10-10", "2026-11-02", "reforecast, issue days 10 October - 2 November"),
    "live": ("2026-11-03", "2026-11-11", "live and published, issue days 3-11 November"),
    "octnov": ("2026-10-10", "2026-11-11", "issue days 10 October - 11 November"),
    "all53": ("2026-09-01", "2026-11-11", "all 53 issue days (1-5 September included)"),
}
THR = [15.0, 37.5, 75.0]
MIN_DAYS, MIN_DAYS_PCT = 15, 30
LABEL = {"km": "1 km", "rt04": "0.4 deg input", "cams": "CAMS forecast", "clim": "1 km climatology", "bcsd": "BCSD",
         "an": "analysis mode", "bcsd_m": "BCSD members", "km_ring1": "ring 1 km", "km_ring3": "ring 3 km",
         "km_ring10": "ring 10 km", "km_cell_lin": "0.25 deg cell mean"}
EU_SUB = ["British Isles, France, Benelux", "Iberia and south-west", "Central", "Italy and Po Valley",
          "Denmark and southern Scandinavia", "East and south-east"]


def lab(a):
    return LABEL.get(a, a)


def sets(same, items):
    """the comparison sets of a table: when the CAMS forecast is on disk for every issue day the two sets are the same
    station-days, and the set 'without the CAMS forecast' is dropped"""
    return [x for x in items if not (same and str(x[0]).startswith("without"))]


def eu_sub(lat, lon):
    """sub-regions of the catalogue's Europe box (35-60 N, 10 W-35 E), boxes fixed before any score was read"""
    s = np.full(lat.shape, -1, np.int64)
    s[(lon < 6) & (lat >= 44)] = 0
    s[(lon < 6) & (lat < 44)] = 1
    s[(lon >= 6) & (lon < 19) & (lat >= 46.5) & (lat < 55)] = 2
    s[(lon >= 6) & (lon < 19) & (lat < 46.5)] = 3
    s[(lon >= 6) & (lon < 19) & (lat >= 55)] = 4
    s[lon >= 19] = 5
    return s


class Ctx:
    def __init__(self, D, out, B, Bt, focus):
        self.D, self.out, self.B, self.Bt, self.focus = D, Path(out), B, Bt, focus
        self.num, self.md, self.samples = [], {}, {}
        (self.out / "tables").mkdir(parents=True, exist_ok=True)

    def sample(self, arms, period, extra=None, key=None, group=None, gnames=None):
        k = key or ("|".join(arms), period)
        if k not in self.samples:
            m = self.D.finite(arms) & self.D.issues(*PERIODS[period][:2])
            if extra is not None:
                m &= extra
            self.samples[k] = Sample(self.D, m, group, gnames, B=self.B) if m.sum() >= 1000 else None
        return self.samples[k]

    def add(self, name, value, ci, smp, table, unit="%", n=None, note=None):
        ok = value is not None and np.isfinite(value)
        e = dict(name=name, value=round(float(value), 4) if ok else None,
                 interval=[round(float(ci[0]), 4), round(float(ci[1]), 4)] if ci is not None and np.all(np.isfinite(ci)) else None,
                 unit=unit, issue_days=smp.n_issue, issue_day_range=f"{smp.issue_list[0]}..{smp.issue_list[-1]}",
                 station_days=int(n if n is not None else smp.n), table=f"tables/{table}.md")
        if note:
            e["note"] = note
        self.num.append(e)

    def write(self, name, md, js):
        (self.out / "tables" / f"{name}.md").write_text(md)
        (self.out / "tables" / f"{name}.json").write_text(json.dumps(js, indent=1, default=float))
        self.md[name] = md
        print(f"[{time.strftime('%H:%M:%S')}] wrote tables/{name}.md ({len(md):,} chars)", flush=True)


def contrast(smp, a, b, metric="rmse", mode="cw"):
    D = smp.D
    fn = {"rmse": lambda x: (D.m[x][smp.rows] - D.obs[smp.rows]) ** 2, "crps": lambda x: D.crps(x)[smp.rows],
          "mae": lambda x: np.abs(D.m[x][smp.rows] - D.obs[smp.rows])}[metric]
    qa = smp.q(f"{metric}_{a}", lambda: fn(a), mode)
    qb = smp.q(f"{metric}_{b}", lambda: fn(b), mode)
    cnt = smp.q("n", lambda: np.ones(smp.n), mode)
    return ratio_contrast(smp, qa, qb, cnt, rel_rmse if metric == "rmse" else rel_mean)


def contrasts(smp, pairs, metrics=("rmse", "crps"), cells_only=True):
    out = {}
    for a, b in pairs:
        for m in metrics:
            out[f"{a}_vs_{b}|{m}"] = contrast(smp, a, b, m)
        if cells_only:                                           # the cells-only interval of the first live comparison, for reference
            c = contrast(smp, a, b, "rmse", "c")
            for Lk, o in out[f"{a}_vs_{b}|rmse"].items():
                for gk, e in o.items():
                    e["ci_cells_only"] = c[Lk][gk]["ci"]
    return out


def md_contrasts(smp, res, pairs, metric, by_lead=True, grid=True, groups=None):
    names = [f"{a}_vs_{b}|{metric}" for a, b in pairs]
    gl = [("all", "all monitors, pooled"), ("regmean", "mean over regions")] + [(g, g) for g in smp.gnames]
    if groups is not None:
        gl = groups
    hdr = ["group", "station-days"] + [f"{lab(a)} vs {lab(b)}" for a, b in pairs]
    rows = [[t, f"{res[names[0]]['all'][g]['n']:,}"] + [fci(res[nm]["all"].get(g)) for nm in names] for g, t in gl if g in res[names[0]]["all"]]
    md = md_table(hdr, rows) + "\n"
    if by_lead:
        hdr = ["lead day", "station-days"]
        for a, b in pairs:
            hdr += [f"{lab(a)} vs {lab(b)}, pooled", "mean over regions"]
        rows = []
        for Lk in ["1", "2", "3", "4", "5"]:
            r = [Lk, f"{res[names[0]][Lk]['all']['n']:,}"]
            for nm in names:
                r += [fci(res[nm][Lk]["all"]), fci(res[nm][Lk].get("regmean"))]
            rows.append(r)
        md += "By lead day:\n\n" + md_table(hdr, rows) + "\n"
    if grid:
        for (a, b), nm in zip(pairs, names):
            rows = [[g] + [fci(res[nm][Lk].get(g)) for Lk in ["1", "2", "3", "4", "5"]] for g in smp.gnames if g in res[nm]["all"]]
            md += f"{lab(a)} vs {lab(b)}, by region and lead day:\n\n" + md_table(["region", "lead 1", "lead 2", "lead 3", "lead 4", "lead 5"], rows) + "\n"
    return md


def group_counts(smp):
    mon = smp.D.mon[smp.rows]
    out = {"all": (smp.n, int(np.unique(mon).size))}
    for g, nm in enumerate(smp.gnames):
        m = smp.g == g
        out[nm] = (int(m.sum()), int(np.unique(mon[m]).size))
    return out


def scores(smp, arms):
    D = smp.D
    sc = {a: basic_scores(D, smp.rows, a, smp.g, smp.G) for a in arms}
    js = {a: {f"{k[0]}|{k[1] if k[1] == 'all' else smp.gnames[k[1]]}": v for k, v in sc[a].items()} for a in arms}
    return sc, js


def md_scores(smp, sc, arms, metrics=(("rmse", 2, False), ("mae", 2, False), ("bias", 2, True), ("r", 3, False)), leads=True):
    gc = group_counts(smp)
    md = ""
    for metric, nd, sign in metrics:
        rows = []
        for gk, gname in [("all", "all monitors")] + [(g, n) for g, n in enumerate(smp.gnames)]:
            if ("all", gk) not in sc[arms[0]]:
                continue
            b = sc[arms[0]][("all", gk)]
            rows.append([gname, f"{b['n']:,}", f"{gc['all' if gk == 'all' else gname][1]:,}", f"{b['obs_mean']:.1f}"] +
                        [fmt(sc[a][("all", gk)][metric], nd, sign) for a in arms])
        if leads:
            for Ld in range(1, 6):
                if (Ld, "all") in sc[arms[0]]:
                    b = sc[arms[0]][(Ld, "all")]
                    rows.append([f"lead day {Ld}", f"{b['n']:,}", "", f"{b['obs_mean']:.1f}"] + [fmt(sc[a][(Ld, "all")][metric], nd, sign) for a in arms])
        name = {"rmse": "RMSE (ug m-3)", "mae": "MAE (ug m-3)", "bias": "bias, forecast minus observation (ug m-3)", "r": "Pearson r over station-days"}[metric]
        md += f"{name}:\n\n" + md_table(["group", "station-days", "monitors", "mean obs"] + [lab(a) for a in arms], rows) + "\n"
    return md


# ---- per-monitor temporal statistics -------------------------------------------------------------------------------------
def temporal(smp, arms, focus, Bt=400, min_days=MIN_DAYS, boot=True, pool_leads=False):
    """per (monitor, lead day) series with >= min_days days: temporal correlation and amplitude ratio sd(forecast)/sd(obs);
    medians over series by region; intervals: two-way block bootstrap (weeks reweighted inside every series through the
    sufficient statistics, cells through a weighted median over series). Spatial r of monitor means (>= min_days station-days)."""
    D, rows = smp.D, smp.rows
    us, s = np.unique(D.mon[rows] * 5 + (0 if pool_leads else (D.lead[rows] - 1)), return_inverse=True)
    ns, slead = us.size, us % 5
    sg, sc_ = np.zeros(ns, np.int64), np.zeros(ns, np.int64)
    sg[s], sc_[s] = smp.g, smp.ci
    W = smp.W
    key = s * W + smp.w

    def sw(q):
        return np.bincount(key, weights=q, minlength=ns * W).reshape(ns, W)
    y = D.obs[rows]
    n_sw, y_sw, yy_sw = sw(np.ones(y.size)), sw(y), sw(y * y)
    X = {}
    for a in arms:
        x = D.m[a][rows]
        X[a] = (sw(x), sw(x * x), sw(x * y))

    def stats(n, sy, syy, sx, sxx, sxy, nmin):
        with np.errstate(divide="ignore", invalid="ignore"):
            mx, my = sx / n, sy / n
            vx, vy, cov = sxx / n - mx * mx, syy / n - my * my, sxy / n - mx * my
            ok = (n >= nmin) & (vx > 1e-9) & (vy > 1e-9)
            return np.where(ok, cov / np.sqrt(vx * vy), np.nan), np.where(ok, np.sqrt(vx / vy), np.nan)
    n = n_sw.sum(1)
    R, A = {}, {}
    for a in arms:
        R[a], A[a] = stats(n, y_sw.sum(1), yy_sw.sum(1), X[a][0].sum(1), X[a][1].sum(1), X[a][2].sum(1), min_days)
    ok = np.all([np.isfinite(R[a]) for a in arms], axis=0)
    pairs = [(f, b) for f in focus for b in arms if b != f]
    groups = [("all", ok)] + [(smp.gnames[g], ok & (sg == g)) for g in range(smp.G)]
    groups = [(gn, m) for gn, m in groups if m.sum() >= 20]
    out = dict(n_series=int(ok.sum()), n_monitors=int(np.unique(us[ok] // 5).size), min_days=min_days, groups={}, leads={})
    for gn, m in groups:
        o = dict(n_series=int(m.sum()), n_monitors=int(np.unique(us[m] // 5).size))
        for a in arms:
            o[a] = dict(r=dict(est=float(np.median(R[a][m]))), amp=dict(est=float(np.median(A[a][m]))), r_mean=float(R[a][m].mean()))
        for f, b in pairs:
            d = R[f][m] - R[b][m]
            o[f"{f}-{b}"] = dict(r_median_of_diff=dict(est=float(np.median(d))), r_diff_of_medians=dict(est=float(np.median(R[f][m]) - np.median(R[b][m]))),
                                 amp_diff_of_medians=dict(est=float(np.median(A[f][m]) - np.median(A[b][m]))), share_series_r_higher=float((d > 0).mean()))
        out["groups"][gn] = o
    for Ld in range(5):
        m = ok & (slead == Ld)
        if m.sum() >= 20:
            out["leads"][str(Ld + 1)] = dict(n_series=int(m.sum()), **{a: dict(r=float(np.median(R[a][m])), amp=float(np.median(A[a][m]))) for a in arms})
    rg = [gn for gn, _ in groups if gn != "all"]
    out["groups"]["regmean"] = dict(n_groups=len(rg), **{a: dict(r=dict(est=float(np.mean([out["groups"][g][a]["r"]["est"] for g in rg]))),
                                                              amp=dict(est=float(np.mean([out["groups"][g][a]["amp"]["est"] for g in rg])))) for a in arms},
                                    **{f"{f}-{b}": dict(r_median_of_diff=dict(est=float(np.mean([out["groups"][g][f"{f}-{b}"]["r_median_of_diff"]["est"] for g in rg]))),
                                                        r_diff_of_medians=dict(est=float(np.mean([out["groups"][g][f"{f}-{b}"]["r_diff_of_medians"]["est"] for g in rg]))),
                                                        amp_diff_of_medians=dict(est=float(np.mean([out["groups"][g][f"{f}-{b}"]["amp_diff_of_medians"]["est"] for g in rg]))))
                                       for f, b in pairs})
    # spatial correlation of monitor means
    um, sm = np.unique(D.mon[rows], return_inverse=True)
    nm_ = np.bincount(sm)
    my = np.bincount(sm, y) / nm_
    mg = np.zeros(um.size, np.int64)
    mg[sm] = smp.g
    keep = nm_ >= min_days
    for a in arms:
        mx = np.bincount(sm, D.m[a][rows]) / nm_
        for gn, _ in groups:
            m = keep if gn == "all" else keep & (mg == smp.gnames.index(gn))
            if m.sum() > 5:
                out["groups"][gn][a]["spatial_r"] = float(np.corrcoef(mx[m], my[m])[0, 1])
                out["groups"][gn]["n_monitors_spatial"] = int(m.sum())
    if not boot:
        return out
    Bt = min(Bt, smp.B)
    idx = {gn: np.flatnonzero(m) for gn, m in groups}
    BS = {gn: {} for gn in idx}
    for gn in idx:
        for a in arms:
            BS[gn][("r", a)], BS[gn][("amp", a)] = np.full(Bt, np.nan), np.full(Bt, np.nan)
        for p in pairs:
            BS[gn][("d", p)] = np.full(Bt, np.nan)
    for b0 in range(0, Bt, 50):
        b1 = min(b0 + 50, Bt)
        mw, mc = smp.Mw[b0:b1], smp.Mc[b0:b1][:, sc_]
        nb, sy, syy = mw @ n_sw.T, mw @ y_sw.T, mw @ yy_sw.T
        Rb, Ab = {}, {}
        for a in arms:
            Rb[a], Ab[a] = stats(nb, sy, syy, mw @ X[a][0].T, mw @ X[a][1].T, mw @ X[a][2].T, 5)
        for gn, ii in idx.items():
            wt = mc[:, ii]
            for a in arms:
                BS[gn][("r", a)][b0:b1] = L.wmedian(Rb[a][:, ii], wt)
                BS[gn][("amp", a)][b0:b1] = L.wmedian(Ab[a][:, ii], wt)
            for f, b in pairs:
                BS[gn][("d", (f, b))][b0:b1] = L.wmedian(Rb[f][:, ii] - Rb[b][:, ii], wt)
    for gn in list(idx) + ["regmean"]:
        o = out["groups"][gn]
        g_ = rg if gn == "regmean" else [gn]
        for a in arms:
            o[a]["r"]["ci"] = pct(np.nanmean([BS[g][("r", a)] for g in g_], axis=0))
            o[a]["amp"]["ci"] = pct(np.nanmean([BS[g][("amp", a)] for g in g_], axis=0))
        for f, b in pairs:
            o[f"{f}-{b}"]["r_median_of_diff"]["ci"] = pct(np.nanmean([BS[g][("d", (f, b))] for g in g_], axis=0))
            o[f"{f}-{b}"]["r_diff_of_medians"]["ci"] = pct(np.nanmean([BS[g][("r", f)] - BS[g][("r", b)] for g in g_], axis=0))
            o[f"{f}-{b}"]["amp_diff_of_medians"]["ci"] = pct(np.nanmean([BS[g][("amp", f)] - BS[g][("amp", b)] for g in g_], axis=0))
    out["bootstrap_resamples"] = Bt
    return out


def md_temporal(smp, t, arms, focus):
    def c(d, nd=2, sign=False):
        if d is None:
            return "n/a"
        s = fmt(d["est"], nd, sign)
        if "ci" in d:
            s += f" [{fmt(d['ci'][0], nd, sign)}, {fmt(d['ci'][1], nd, sign)}]"
        return s
    gl = [("all", "all monitors"), ("regmean", "mean over regions")] + [(g, g) for g in smp.gnames]
    gl = [(g, n) for g, n in gl if g in t["groups"]]
    md = (f"Series = (monitor, lead day) with at least {t['min_days']} days: {t['n_series']:,} series at {t['n_monitors']:,} monitors. "
          f"Medians over series; intervals from {t.get('bootstrap_resamples', 0)} two-way resamples.\n\n")
    rows = [[n, f"{t['groups'][g].get('n_series', ''):,}" if g != "regmean" else ""] + [c(t["groups"][g][a]["r"]) for a in arms] for g, n in gl]
    md += "Median per-monitor temporal correlation:\n\n" + md_table(["group", "series"] + [lab(a) for a in arms], rows) + "\n"
    rows = [[n] + [c(t["groups"][g][a]["amp"]) for a in arms] for g, n in gl]
    md += "Median amplitude ratio sd(forecast) / sd(observation):\n\n" + md_table(["group"] + [lab(a) for a in arms], rows) + "\n"
    for f in focus:
        others = [b for b in arms if b != f]
        rows = [[n] + [c(t["groups"][g][f"{f}-{b}"]["r_median_of_diff"], 3, True) for b in others] for g, n in gl]
        md += f"Median over series of the difference of temporal correlation, {lab(f)} minus the other arm:\n\n" + md_table(["group"] + [lab(b) for b in others], rows) + "\n"
        rows = [[n] + [fmt(100 * t["groups"][g][f"{f}-{b}"]["share_series_r_higher"], 1) for b in others] for g, n in gl if g != "regmean"]
        md += f"Share of series (%) where {lab(f)} has the higher temporal correlation:\n\n" + md_table(["group"] + [lab(b) for b in others], rows) + "\n"
    rows = [[n] + [fmt(t["groups"][g][a].get("spatial_r", np.nan), 3) for a in arms] for g, n in gl if g != "regmean"]
    md += f"Spatial correlation of monitor means (monitors with at least {t['min_days']} station-days):\n\n" + md_table(["group"] + [lab(a) for a in arms], rows) + "\n"
    rows = [[k, f"{v['n_series']:,}"] + [f"{v[a]['r']:.2f} / {v[a]['amp']:.2f}" for a in arms] for k, v in t["leads"].items()]
    md += "By lead day, median temporal correlation / amplitude ratio (all monitors):\n\n" + md_table(["lead day", "series"] + [lab(a) for a in arms], rows) + "\n"
    return md


# ---- ensemble diagnostics ------------------------------------------------------------------------------------------------
def ensemble(smp, ens_arms, det_arms):
    D, rows = smp.D, smp.rows
    cnt = smp.q("n", lambda: np.ones(smp.n))
    vn, wn = views(cnt[0]), views(cnt[1])
    keys = [("all", "all")] + [("all", g) for g in range(smp.G)] + [(Ld, "all") for Ld in range(1, 6)]
    kname = lambda k: f"{k[0]}|{k[1] if k[1] == 'all' else smp.gnames[k[1]]}"
    out = {}
    for a in ens_arms:
        st = D.ens_stats(a)
        M = st["M"]
        Q = {k: smp.q(f"{k}_{a}", lambda k=k: st[k][rows]) for k in ("crps", "var", "inside", "below", "above", "width")}
        Q["se"] = smp.q(f"rmse_{a}", lambda: (D.m[a][rows] - D.obs[rows]) ** 2)
        V = {k: (views(v[0]), views(v[1])) for k, v in Q.items()}
        rk = np.bincount(smp.g * 9 + st["rank"][rows], minlength=smp.G * 9).reshape(smp.G, 9)
        rkl = np.bincount(smp.lead * 9 + st["rank"][rows], minlength=45).reshape(5, 9)
        o = {}
        for k in keys:
            if vn[k] < 100:
                continue
            f = (M + 1) / M
            e = dict(n=int(vn[k]), members=M,
                     crps=dict(est=float(V["crps"][0][k] / vn[k]), ci=pct(V["crps"][1][k] / wn[k])),
                     spread=float(np.sqrt(V["var"][0][k] / vn[k])), rmse=float(np.sqrt(V["se"][0][k] / vn[k])),
                     spread_skill=dict(est=float(np.sqrt(f * V["var"][0][k] / V["se"][0][k])), ci=pct(np.sqrt(f * V["var"][1][k] / V["se"][1][k]))),
                     cover=dict(est=float(100 * V["inside"][0][k] / vn[k]), ci=pct(100 * V["inside"][1][k] / wn[k])),
                     below=float(100 * V["below"][0][k] / vn[k]), above=float(100 * V["above"][0][k] / vn[k]),
                     mean_width=float(V["width"][0][k] / vn[k]))
            h = rk.sum(0) if k == ("all", "all") else rk[k[1]] if k[0] == "all" else rkl[k[0] - 1]
            e["rank_hist"] = [float(x) for x in h / h.sum()]
            e["outside"] = float(100 * (h[0] + h[-1]) / h.sum())
            o[kname(k)] = e
        out[a] = o
    for a in det_arms:
        Q = smp.q(f"crps_{a}", lambda: D.crps(a)[rows])
        V0, V1 = views(Q[0]), views(Q[1])
        out[a] = {kname(k): dict(n=int(vn[k]), crps=dict(est=float(V0[k] / vn[k]), ci=pct(V1[k] / wn[k]))) for k in keys if vn[k] >= 100}
    return out


def md_ensemble(smp, e, ens_arms, det_arms):
    gl = [("all|all", "all monitors")] + [(f"all|{g}", g) for g in smp.gnames] + [(f"{Ld}|all", f"lead day {Ld}") for Ld in range(1, 6)]
    gl = [(k, n) for k, n in gl if k in e[ens_arms[0]]]

    def c(d, nd=2):
        return f"{d['est']:.{nd}f} [{d['ci'][0]:.{nd}f}, {d['ci'][1]:.{nd}f}]"
    md = "Fair CRPS of the ensembles and absolute error (= CRPS) of the deterministic arms, ug m-3:\n\n"
    md += md_table(["group", "station-days"] + [lab(a) for a in ens_arms + det_arms],
                   [[n, f"{e[ens_arms[0]][k]['n']:,}"] + [c(e[a][k]["crps"]) for a in ens_arms + det_arms] for k, n in gl]) + "\n"
    md += ("Spread / skill = sqrt((M+1)/M x mean member variance) / RMSE of the ensemble mean (1 = spread matches the error; the "
           "measurement and representativeness error of the monitors is not in the spread):\n\n")
    md += md_table(["group"] + [lab(a) for a in ens_arms], [[n] + [c(e[a][k]["spread_skill"]) for a in ens_arms] for k, n in gl]) + "\n"
    md += ("Share of observations (%) inside the 10th-90th percentile range of the members (linear quantiles of 8 members, as the "
           "published products; a reliable 8-member ensemble gives about 62-64 %), and below / above it:\n\n")
    md += md_table(["group"] + [f"{lab(a)}: inside [95 %]; below / above" for a in ens_arms],
                   [[n] + [f"{c(e[a][k]['cover'], 1)}; {e[a][k]['below']:.1f} / {e[a][k]['above']:.1f}" for a in ens_arms] for k, n in gl]) + "\n"
    for a in ens_arms:
        md += f"Rank histogram of the observation among the members of {lab(a)} (% per rank 0..M; flat = {100 / (e[a]['all|all']['members'] + 1):.1f} %):\n\n"
        md += md_table(["group"] + [str(i) for i in range(len(e[a]["all|all"]["rank_hist"]))] + ["outside the ensemble"],
                       [[n] + [f"{100 * x:.1f}" for x in e[a][k]["rank_hist"]] + [f"{e[a][k]['outside']:.1f}"] for k, n in gl]) + "\n"
    return md


def reliability(smp, ens_arms, det_arms):
    D, rows = smp.D, smp.rows
    y = D.obs[rows]
    out = {}
    cnt = smp.q("n", lambda: np.ones(smp.n))
    vn = views(cnt[0])
    kname = lambda k: f"{k[0]}|{k[1] if k[1] == 'all' else smp.gnames[k[1]]}"
    keys = [("all", "all")] + [("all", g) for g in range(smp.G)] + [(Ld, "all") for Ld in range(1, 6)]
    for t in THR:
        o = (y > t).astype(np.float64)
        qo = smp.q(f"o_{t}", lambda: o)
        vo = views(qo[0])
        res = {"base_rate": {kname(k): float(vo[k] / vn[k]) for k in keys if vn[k] >= 100}, "arms": {}}
        Bq = {}
        for a in ens_arms + det_arms:
            p = D.pexc(a, t)[rows]
            Bq[a] = smp.q(f"brier_{a}_{t}", lambda: (p - o) ** 2)
            vb, wb = views(Bq[a][0]), views(Bq[a][1])
            wn = views(cnt[1])
            ar = {"brier": {kname(k): dict(est=float(vb[k] / vn[k]), ci=pct(wb[k] / wn[k])) for k in keys if vn[k] >= 100}}
            ar["bss_vs_base_rate"] = {kname(k): float(1 - (vb[k] / vn[k]) / max(res["base_rate"][kname(k)] * (1 - res["base_rate"][kname(k)]), 1e-12))
                                      for k in keys if vn[k] >= 100}
            if a in ens_arms:
                M = D.e[a].shape[1]
                fair = (p - o) ** 2 - p * (1 - p) / (M - 1)
                ar["brier_fair"] = {"all|all": float(fair.mean())}
                kk = np.rint(p * M).astype(np.int64)
                nk = np.bincount(kk, minlength=M + 1)
                ok_ = np.bincount(kk, weights=o, minlength=M + 1)
                ar["reliability"] = dict(prob=[i / M for i in range(M + 1)], n=[int(x) for x in nk],
                                         obs_freq=[float(a_ / b_) if b_ > 0 else None for a_, b_ in zip(ok_, nk)])
                byg = {}
                for g in range(smp.G):
                    m = smp.g == g
                    if m.sum() < 1000:
                        continue
                    nk_ = np.bincount(kk[m], minlength=M + 1)
                    ok2 = np.bincount(kk[m], weights=o[m], minlength=M + 1)
                    byg[smp.gnames[g]] = dict(n=[int(x) for x in nk_], obs_freq=[float(a_ / b_) if b_ > 0 else None for a_, b_ in zip(ok2, nk_)])
                ar["reliability_by_region"] = byg
            res["arms"][a] = ar
        res["contrast_brier"] = {}
        for f in ens_arms[:1]:
            for b in ens_arms[1:] + det_arms:
                res["contrast_brier"][f"{f}_vs_{b}"] = ratio_contrast(smp, Bq[f], Bq[b], cnt, rel_mean)["all"]
        out[str(t)] = res
    return out


def md_reliability(smp, r, ens_arms, det_arms):
    md = ""
    for t in THR:
        x = r[str(t)]
        md += f"**P(PM2.5 > {t:g} ug m-3)**: observed frequency {100 * x['base_rate']['all|all']:.1f} % of {smp.n:,} station-days.\n\n"
        rows = []
        for a in ens_arms + det_arms:
            ar = x["arms"][a]
            b = ar["brier"]["all|all"]
            rows.append([lab(a), f"{b['est']:.4f} [{b['ci'][0]:.4f}, {b['ci'][1]:.4f}]", f"{ar['brier_fair']['all|all']:.4f}" if "brier_fair" in ar else "",
                         f"{ar['bss_vs_base_rate']['all|all']:+.3f}"] + [f"{ar['brier'][f'all|{g}']['est']:.4f}" if f"all|{g}" in ar["brier"] else "n/a" for g in smp.gnames])
        md += ("Brier score (ensembles: probability = share of members above the threshold; deterministic arms: 0 or 1), fair Brier "
               "score, skill against the constant sample frequency, and Brier score by region:\n\n")
        md += md_table(["arm", "Brier [95 %]", "fair Brier", "skill vs sample frequency"] + smp.gnames, rows) + "\n"
        f = ens_arms[0]
        md += f"Relative change of the Brier score, {lab(f)} against each arm (%, negative = {lab(f)} better):\n\n"
        cb = x["contrast_brier"]
        md += md_table(["group"] + [lab(k.split('_vs_')[1]) for k in cb],
                       [[n] + [fci(cb[k].get(g)) for k in cb] for g, n in [("all", "all monitors, pooled"), ("regmean", "mean over regions")] + [(g, g) for g in smp.gnames]
                        if g in cb[next(iter(cb))]]) + "\n"
        for a in ens_arms:
            rel = x["arms"][a]["reliability"]
            md += f"Reliability of {lab(a)} (forecast probability = k/8 members above; observed frequency in %, station-days):\n\n"
            rows = [["all monitors"] + [f"{100 * o:.1f} ({n:,})" if o is not None else "n/a (0)" for o, n in zip(rel["obs_freq"], rel["n"])]]
            for g, v in x["arms"][a]["reliability_by_region"].items():
                rows.append([g] + [f"{100 * o:.1f} ({n:,})" if o is not None else "n/a (0)" for o, n in zip(v["obs_freq"], v["n"])])
            md += md_table(["group"] + [f"{100 * p:.1f} %" for p in rel["prob"]], rows) + "\n"
    return md


def sedi_table(smp, arms, focus):
    D, rows = smp.D, smp.rows
    y = D.obs[rows]
    us, s = np.unique(D.mon[rows] * 5 + (D.lead[rows] - 1), return_inverse=True)
    ns = us.size
    ev = {}
    for t in (37.5, 75.0):
        ev[f"above {t:g} ug m-3"] = (y > t, {a: D.m[a][rows] > t for a in arms}, np.ones(y.size, bool))
    for p in (95, 99):
        qo, n = L.group_percentile(y, s, ns, p)
        valid = (n >= MIN_DAYS_PCT)[s]
        eo = y > qo[s]
        ev[f"above the monitor's own {p}th percentile, same threshold for the forecast"] = (eo, {a: D.m[a][rows] > qo[s] for a in arms}, valid)
        ev[f"above the monitor's own {p}th percentile, forecast above its own {p}th percentile"] = (
            eo, {a: D.m[a][rows] > L.group_percentile(D.m[a][rows], s, ns, p)[0][s] for a in arms}, valid)
    out = {}
    kname = lambda k: f"{k[0]}|{k[1] if k[1] == 'all' else smp.gnames[k[1]]}"
    keys = [("all", "all")] + [("all", g) for g in range(smp.G)] + [(Ld, "all") for Ld in range(1, 6)]
    for en, (eo, ef, valid) in ev.items():
        res = {"n_valid": int(valid.sum()), "n_events": int((eo & valid).sum()), "arms": {}}
        bs = {}
        for a in arms:
            f = ef[a]
            tabs = [smp.boot(smp.sums(((x) & valid).astype(np.float64))) for x in (eo & f, ~eo & f, eo & ~f, ~eo & ~f)]
            P = [views(t_[0]) for t_ in tabs]
            Rr = [views(t_[1]) for t_ in tabs]
            ar = {}
            for k in keys:
                a_, b_, c_, d_ = (float(P[i][k]) for i in range(4))
                if a_ + c_ < 20:
                    continue
                sv = L.sedi(Rr[0][k], Rr[1][k], Rr[2][k], Rr[3][k])
                bs[(a, k)] = sv
                ar[kname(k)] = dict(sedi=dict(est=float(L.sedi(a_, b_, c_, d_)), ci=pct(sv)), hit_rate=a_ / (a_ + c_), false_alarm_rate=b_ / (b_ + d_),
                                    frequency_bias=(a_ + b_) / (a_ + c_), events=int(a_ + c_), n=int(a_ + b_ + c_ + d_))
            res["arms"][a] = ar
        res["diff"] = {}
        for f in focus:
            for b in arms:
                if b == f:
                    continue
                dd = {}
                for k in keys:
                    if (f, k) in bs and (b, k) in bs:
                        dd[kname(k)] = dict(est=res["arms"][f][kname(k)]["sedi"]["est"] - res["arms"][b][kname(k)]["sedi"]["est"], ci=pct(bs[(f, k)] - bs[(b, k)]))
                res["diff"][f"{f}-{b}"] = dd
        out[en] = res
    return out


def md_sedi(smp, r, arms, focus):
    def c(d, sign=False):
        if d is None or not np.isfinite(d["est"]):
            return "n/a"
        return f"{fmt(d['est'], 2, sign)} [{fmt(d['ci'][0], 2, sign)}, {fmt(d['ci'][1], 2, sign)}]"
    md = "SEDI of the ensemble mean (1 = perfect, 0 = no skill), all lead days and monitors pooled:\n\n"
    rows = []
    for en, x in r.items():
        rows.append([en, f"{x['n_events']:,} / {x['n_valid']:,}"] + [c(x["arms"][a].get("all|all", {}).get("sedi")) for a in arms])
    md += md_table(["event", "observed events / station-days"] + [lab(a) for a in arms], rows) + "\n"
    md += "Hit rate / false-alarm rate / frequency bias (forecast events / observed events), pooled:\n\n"
    rows = []
    for en, x in r.items():
        rows.append([en] + [f"{v['hit_rate']:.2f} / {v['false_alarm_rate']:.3f} / {v['frequency_bias']:.2f}" if (v := x["arms"][a].get("all|all")) else "n/a" for a in arms])
    md += md_table(["event"] + [lab(a) for a in arms], rows) + "\n"
    for f in focus:
        md += f"Difference of SEDI, {lab(f)} minus the other arm, pooled:\n\n"
        others = [b for b in arms if b != f]
        md += md_table(["event"] + [lab(b) for b in others], [[en] + [c(x["diff"][f"{f}-{b}"].get("all|all"), True) for b in others] for en, x in r.items()]) + "\n"
    for en, x in r.items():
        rows = []
        for k, n in [(f"all|{g}", g) for g in smp.gnames] + [(f"{Ld}|all", f"lead day {Ld}") for Ld in range(1, 6)]:
            if k in x["arms"][arms[0]]:
                rows.append([n, f"{x['arms'][arms[0]][k]['events']:,}"] + [c(x["arms"][a].get(k, {}).get("sedi")) for a in arms])
        md += f"SEDI by region and lead day, event: {en}:\n\n" + md_table(["group", "observed events"] + [lab(a) for a in arms], rows) + "\n"
    return md


# =========================================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--focus", default="km")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--Bt", type=int, default=400)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    t0 = time.time()
    D = L.Data(a.pairs)
    focus = [x for x in a.focus.split(",") if x]
    X = Ctx(D, a.out, a.B, a.Bt, focus)
    only = set(x for x in a.only.split(",") if x)
    want = lambda t: not only or t in only
    meta = json.loads((Path(a.pairs).parent / "pairs_meta.json").read_text())
    base = [b for b in ["rt04", "cams", "clim", "bcsd"] if b in D.arms]
    B5 = focus + base
    A4 = focus + [b for b in base if b != "cams"]
    f0 = focus[0]
    pairsB = [(f, b) for f in focus for b in base] + [(f, f0) for f in focus[1:]]
    pairsA = [(f, b) for f in focus for b in base if b != "cams"] + [(f, f0) for f in focus[1:]]
    print(f"loaded {D.N:,} station-days, arms {D.arms}, focus {focus} ({time.time() - t0:.0f} s)", flush=True)
    sB, sA = X.sample(B5, "clean"), X.sample(A4, "clean")
    SAME = sA is not None and sB is not None and sA.n == sB.n
    if SAME:
        print("the CAMS forecast is finite on every station-day of the clean period: one comparison set", flush=True)
    intro = lambda s: f"Sample: {s.describe()}. Time blocks: {'; '.join(s.blocks)}.\n\n"

    # ---- T01 samples ---------------------------------------------------------------------------------------------------
    if want("T01"):
        md = "## T01. Samples and observations\n\n"
        rows = []
        js = {}
        for p, (first, last, desc) in PERIODS.items():
            for nm, arms in (("with the CAMS forecast", B5), ("without the CAMS forecast", A4)):
                s = X.sample(arms, p)
                if s is None:
                    rows.append([desc, nm, "0", "0", "0", ""])
                    continue
                if nm.startswith("without") and (sw0 := X.sample(B5, p)) is not None and sw0.n == s.n:
                    continue
                gc = group_counts(s)
                rows.append([desc, nm, s.n_issue, f"{s.n:,}", f"{s.n_mon:,}", "; ".join(f"{g} {gc[g][0]:,}" for g in s.gnames)])
                js[f"{p}|{nm}"] = dict(issue_days=s.issue_list, station_days=s.n, monitors=s.n_mon, by_region={g: gc[g] for g in s.gnames}, time_blocks=s.blocks)
        md += ("Station-days with a finite value of every arm of the set (" + ", ".join(lab(x) for x in B5) + "), by period of ISSUE day. Lead day L of issue day d is "
               "valid on the UTC day d + L - 1.\n\n")
        md += md_table(["period (issue days)", "set", "issue days", "station-days", "monitors", "station-days by region"], rows) + "\n"
        ov = meta["obs_overlap"]
        tot = sum(v["both"] for k, v in ov.items() if k <= "2026-10-29")
        idn = sum(v["identical"] for k, v in ov.items() if k <= "2026-10-29")
        md += (f"Observations: {meta['obs_rule']}. On the {sum(1 for k in ov if k <= '2026-10-29')} days that both sources hold completely "
               f"({min(ov)}..2026-10-29) they contain the same monitors and {idn:,} of {tot:,} values are identical "
               f"(largest difference {max(v['max_abs_diff'] for k, v in ov.items() if k <= '2026-10-29'):.2f} ug m-3).\n\n")
        nf = [k for k, v in meta["obs_per_valid_day"].items() if v["final"] is False]
        md += f"Valid days whose archive files were read less than 7 days after the day (late measurements may still arrive): {', '.join(nf)}.\n\n"
        days = meta["obs_per_valid_day"]
        md += md_table(["valid day", "monitors with an observation", "source"], [[k, f"{v['n']:,}", v["source"]] for k, v in days.items()]) + "\n"
        X.write("T01_samples", md, dict(samples=js, obs_overlap=ov, obs_per_valid_day=days))

    # ---- T02 scores, T03 contrasts -------------------------------------------------------------------------------------
    resB = resA = None
    if want("T02") or want("T03") or want("T12"):
        md2 = "## T02. Scores of the ensemble mean on identical station-days\n\n"
        md3 = "## T03. Relative change of RMSE and CRPS, with 95 % intervals\n\nNegative = the first arm is better. Intervals: two-way block bootstrap (2-degree cells x calendar weeks).\n\n"
        js2, js3 = {}, {}
        for tag, s, arms, prs in sets(SAME, (("with the CAMS forecast", sB, B5, pairsB), ("without the CAMS forecast (more issue days)", sA, A4, pairsA))):
            if s is None:
                continue
            sc, js = scores(s, arms)
            md2 += f"### Set {tag}\n\n" + intro(s) + md_scores(s, sc, arms)
            tot = {a_: sc[a_][("all", "all")]["n"] * sc[a_][("all", "all")]["rmse"] ** 2 for a_ in arms}
            shr = {g: dict(station_days_pct=100 * sc[arms[0]][("all", gi)]["n"] / s.n,
                           **{a_: 100 * sc[a_][("all", gi)]["n"] * sc[a_][("all", gi)]["rmse"] ** 2 / tot[a_] for a_ in arms})
                   for gi, g in enumerate(s.gnames) if ("all", gi) in sc[arms[0]]}
            md2 += ("Share of the station-days and of each arm's summed squared error by region (%):\n\n" +
                    md_table(["region", "station-days"] + [lab(a_) for a_ in arms], [[g, f"{v['station_days_pct']:.1f}"] + [f"{v[a_]:.1f}" for a_ in arms] for g, v in shr.items()]) + "\n")
            js["share_of_squared_error_pct"] = shr
            for g, v in shr.items():
                X.add(f"share_of_squared_error_pct.{f0}.{'withCAMS' if tag.startswith('with ') else 'noCAMS'}.{g}", v[f0], None, s, "T02_scores", n=sc[f0][("all", s.gnames.index(g))]["n"])
            js2[tag] = js
            res = contrasts(s, prs)
            js3[tag] = dict(sample=s.describe(), blocks=s.blocks, contrasts=res)
            md3 += f"### Set {tag}\n\n" + intro(s) + "RMSE of the ensemble mean, relative change in %:\n\n" + md_contrasts(s, res, prs, "rmse")
            md3 += "CRPS (fair CRPS of the 8 members; absolute error for the CAMS forecast, the climatology and BCSD), relative change in %:\n\n" + md_contrasts(s, res, prs, "crps", grid=False)
            if tag.startswith("with "):
                resB = res
            else:
                resA = res
            setk = "withCAMS" if tag.startswith("with ") else "noCAMS"
            for (f, b) in prs:
                for metric in ("rmse", "crps"):
                    r = res[f"{f}_vs_{b}|{metric}"]
                    for Lk in r:
                        for g in (["all", "regmean"] + (s.gnames if Lk == "all" else [])):
                            if g in r[Lk]:
                                X.add(f"{metric}_change_pct.{f}_vs_{b}.{setk}.lead_{Lk}.{g}", r[Lk][g]["est"], r[Lk][g]["ci"], s, "T03_contrasts", n=r[Lk][g]["n"])
            for arm in arms:
                for g in ["all"] + s.gnames:
                    k = ("all", "all") if g == "all" else ("all", s.gnames.index(g))
                    if k in sc[arm]:
                        for metric in ("rmse", "bias", "r", "mae"):
                            X.add(f"{metric}.{arm}.{setk}.lead_all.{g}", sc[arm][k][metric], None, s, "T02_scores", unit="ug m-3" if metric != "r" else "", n=sc[arm][k]["n"])
        X.write("T02_scores", md2, js2)
        X.write("T03_contrasts", md3, js3)

    # ---- T04 temporal --------------------------------------------------------------------------------------------------
    tB = tA = None
    if want("T04") or want("T12"):
        md = ("## T04. Day-to-day signal at each monitor\n\nPer-monitor temporal correlation and amplitude ratio are computed on each (monitor, lead day) "
              "series; the series of a table are identical across its arms.\n\n")
        js = {}
        for tag, s, arms in sets(SAME, (("with the CAMS forecast", sB, B5), ("without the CAMS forecast (more issue days)", sA, A4))):
            if s is None:
                continue
            t = temporal(s, arms, focus, Bt=X.Bt)
            js[tag] = t
            md += f"### Set {tag}, whole record\n\n" + intro(s) + md_temporal(s, t, arms, focus)
            if tag.startswith("with "):
                tB = t
            else:
                tA = t
            setk = "withCAMS" if tag.startswith("with ") else "noCAMS"
            for g in ["all", "regmean"] + s.gnames:
                if g not in t["groups"]:
                    continue
                for arm in arms:
                    X.add(f"temporal_r_median.{arm}.{setk}.{g}", t["groups"][g][arm]["r"]["est"], t["groups"][g][arm]["r"].get("ci"), s, "T04_temporal", unit="")
                    X.add(f"amplitude_ratio_median.{arm}.{setk}.{g}", t["groups"][g][arm]["amp"]["est"], t["groups"][g][arm]["amp"].get("ci"), s, "T04_temporal", unit="")
                for f in focus:
                    for b in arms:
                        if b != f:
                            d = t["groups"][g][f"{f}-{b}"]["r_median_of_diff"]
                            X.add(f"temporal_r_median_of_diff.{f}-{b}.{setk}.{g}", d["est"], d.get("ci"), s, "T04_temporal", unit="")
        # inside one period, so that the step between September and October-November does not enter the correlation
        for p in ("sep", "octnov"):
            for tag, arms in (("with the CAMS forecast", B5), ("without the CAMS forecast", A4)):
                s = X.sample(arms, p)
                if s is None or not want("T04") or (tag.startswith("without") and (sw0 := X.sample(B5, p)) is not None and sw0.n == s.n):
                    continue
                t = temporal(s, arms, focus, Bt=X.Bt)
                setk = "withCAMS" if tag.startswith("with ") else "noCAMS"
                js[f"within {p}|{tag}"] = t
                md += f"### Inside one period only: {PERIODS[p][2]} (set {tag})\n\n" + intro(s) + md_temporal(s, t, arms, focus)
                for g in ["all", "regmean"] + s.gnames:
                    if g in t["groups"]:
                        for arm in arms:
                            X.add(f"temporal_r_median.{arm}.{setk}.{p}.{g}", t["groups"][g][arm]["r"]["est"], t["groups"][g][arm]["r"].get("ci"), s, "T04_temporal", unit="")
                            X.add(f"amplitude_ratio_median.{arm}.{setk}.{p}.{g}", t["groups"][g][arm]["amp"]["est"], t["groups"][g][arm]["amp"].get("ci"), s, "T04_temporal", unit="")
                        for f in focus:
                            for b in arms:
                                if b != f:
                                    d = t["groups"][g][f"{f}-{b}"]["r_median_of_diff"]
                                    X.add(f"temporal_r_median_of_diff.{f}-{b}.{setk}.{p}.{g}", d["est"], d.get("ci"), s, "T04_temporal", unit="")
        X.write("T04_temporal", md, js)

    # ---- T05 ensemble, T06 reliability ---------------------------------------------------------------------------------
    ens_extra = [x for x in ["rt04", "bcsd_m"] if x in D.arms]
    if want("T05"):
        md = "## T05. The eight-member ensembles at the monitors\n\n"
        js = {}
        for tag, s, det in sets(SAME, (("without the CAMS forecast (more issue days)", X.sample(A4 + ["bcsd_m"], "clean"), ["clim", "bcsd"]),
                                       ("with the CAMS forecast", X.sample(B5 + ["bcsd_m"], "clean"), ["cams", "clim", "bcsd"]))):
            if s is None:
                continue
            e = ensemble(s, focus + ens_extra, det)
            js[tag] = e
            md += f"### Set {tag}\n\n" + intro(s) + md_ensemble(s, e, focus + ens_extra, det)
            prs_e = [(f, b) for f in focus for b in ens_extra + det] + [(f, f0) for f in focus[1:]]
            res_e = contrasts(s, prs_e, metrics=("crps",), cells_only=False)
            js[tag + "|crps_contrasts"] = res_e
            md += ("Relative change of the CRPS in % (ensemble against ensemble for the 0.4 deg input and the BCSD members; ensemble CRPS against absolute error for the "
                   "deterministic arms):\n\n" + md_contrasts(s, res_e, prs_e, "crps", grid=False))
            for (a_, b_) in prs_e:
                for g in ["all", "regmean"] + s.gnames:
                    v = res_e[f"{a_}_vs_{b_}|crps"]["all"].get(g)
                    if v and b_ == "bcsd_m":
                        X.add(f"crps_change_pct.{a_}_vs_{b_}.{'withCAMS' if tag.startswith('with ') else 'noCAMS'}.lead_all.{g}", v["est"], v["ci"], s, "T05_ensemble", n=v["n"])
            setk = "withCAMS" if tag.startswith("with ") else "noCAMS"
            for arm in focus + ens_extra:
                for k, v in e[arm].items():
                    Lk, g = k.split("|")
                    X.add(f"coverage_10_90_pct.{arm}.{setk}.lead_{Lk}.{g}", v["cover"]["est"], v["cover"]["ci"], s, "T05_ensemble", n=v["n"])
                    X.add(f"spread_skill.{arm}.{setk}.lead_{Lk}.{g}", v["spread_skill"]["est"], v["spread_skill"]["ci"], s, "T05_ensemble", unit="", n=v["n"])
                    X.add(f"fair_crps.{arm}.{setk}.lead_{Lk}.{g}", v["crps"]["est"], v["crps"]["ci"], s, "T05_ensemble", unit="ug m-3", n=v["n"])
                    X.add(f"outside_ensemble_pct.{arm}.{setk}.lead_{Lk}.{g}", v["outside"], None, s, "T05_ensemble", n=v["n"])
            for arm in det:
                v = e[arm]["all|all"]
                X.add(f"mae_as_crps.{arm}.{setk}.lead_all.all", v["crps"]["est"], v["crps"]["ci"], s, "T05_ensemble", unit="ug m-3", n=v["n"])
        for p in ("sep", "oct", "live"):
            s = X.sample(A4 + ["bcsd_m"], p)
            if s is None:
                continue
            e = ensemble(s, focus + ens_extra, ["clim", "bcsd"])
            js[f"period {p}"] = e
            rows = []
            for arm in focus + ens_extra:
                v = e[arm]["all|all"]
                rows.append([lab(arm), f"{v['crps']['est']:.2f}", f"{v['spread_skill']['est']:.2f} [{v['spread_skill']['ci'][0]:.2f}, {v['spread_skill']['ci'][1]:.2f}]",
                             f"{v['cover']['est']:.1f} [{v['cover']['ci'][0]:.1f}, {v['cover']['ci'][1]:.1f}]", f"{v['below']:.1f} / {v['above']:.1f}", f"{v['outside']:.1f}"])
                X.add(f"coverage_10_90_pct.{arm}.noCAMS.{p}.all", v["cover"]["est"], v["cover"]["ci"], s, "T05_ensemble", n=v["n"])
                X.add(f"spread_skill.{arm}.noCAMS.{p}.all", v["spread_skill"]["est"], v["spread_skill"]["ci"], s, "T05_ensemble", unit="", n=v["n"])
            md += f"### {PERIODS[p][2]}\n\n" + intro(s) + md_table(["ensemble", "fair CRPS", "spread / skill", "inside 10-90 % (%)", "below / above (%)", "outside the ensemble (%)"], rows) + "\n"
        X.write("T05_ensemble", md, js)
    if want("T06"):
        md = "## T06. Probabilities of exceedance: reliability and Brier scores\n\n"
        js = {}
        for tag, s, det in sets(SAME, (("without the CAMS forecast (more issue days)", X.sample(A4 + ["bcsd_m"], "clean"), ["clim", "bcsd"]),
                                       ("with the CAMS forecast", X.sample(B5 + ["bcsd_m"], "clean"), ["cams", "clim", "bcsd"]))):
            if s is None:
                continue
            r = reliability(s, focus + ens_extra, det)
            js[tag] = r
            md += f"### Set {tag}\n\n" + intro(s) + md_reliability(s, r, focus + ens_extra, det)
            setk = "withCAMS" if tag.startswith("with ") else "noCAMS"
            for t in THR:
                for arm in focus + ens_extra + det:
                    b = r[str(t)]["arms"][arm]["brier"]["all|all"]
                    X.add(f"brier.P_gt_{t:g}.{arm}.{setk}.all", b["est"], b["ci"], s, "T06_reliability", unit="")
        X.write("T06_reliability", md, js)

    # ---- T07 SEDI ------------------------------------------------------------------------------------------------------
    if want("T07"):
        md = ("## T07. Extremes: SEDI\n\nEvent of the observation: the daily mean is above the threshold. Percentile thresholds are each monitor's own, computed on its "
              f"(monitor, lead day) series of the scored period (series with at least {MIN_DAYS_PCT} days; with 30-60 days the 95th percentile leaves the two or three "
              "highest days and the 99th percentile the single highest day). Two forecast events are given for them: the forecast above the SAME threshold as the observation "
              "(penalises a bias), and the forecast above its OWN percentile at that monitor (ranks the days only).\n\n")
        js = {}
        for tag, s, arms in sets(SAME, (("with the CAMS forecast", sB, B5), ("without the CAMS forecast (more issue days)", sA, A4))):
            if s is None:
                continue
            r = sedi_table(s, arms, focus)
            js[tag] = r
            md += f"### Set {tag}\n\n" + intro(s) + md_sedi(s, r, arms, focus)
            setk = "withCAMS" if tag.startswith("with ") else "noCAMS"
            for en, x in r.items():
                for arm in arms:
                    v = x["arms"][arm].get("all|all")
                    if v:
                        X.add(f"sedi.{en}.{arm}.{setk}.all", v["sedi"]["est"], v["sedi"]["ci"], s, "T07_sedi", unit="", n=v["n"])
        X.write("T07_sedi", md, js)

    # ---- T08 rings -----------------------------------------------------------------------------------------------------
    if want("T08"):
        rings = [x for x in ["km_ring1", "km_ring3", "km_ring10", "km_cell_lin"] if x in D.arms]
        md = ("## T08. Away from the monitor's pixel\n\nThe forecast is read on the square rings 1, 3 and 10 pixels (about 1, 3, 10 km) around the monitor's pixel "
              "(the pixel itself excluded; a ring is kept with at least half of its pixels inside the 1 km mask) and as the mean of the monitor's 0.25-degree cell "
              "(5 x 5 sample, at least 13 points inside the mask), and scored against the same monitor. All columns share the station-days. "
              "The baselines stay where they are in the other tables (climatology and BCSD at the monitor's pixel, the two 0.4-degree forecasts read bilinearly at the "
              "monitor): the columns 'vs' therefore ask whether a forecast read away from the pixel still beats a baseline read at the monitor.\n\n")
        js = {}
        for tag, arms0 in sets(SAME, (("with the CAMS forecast", [f0] + rings + base), ("without the CAMS forecast (more issue days)", [f0] + rings + [b for b in base if b != "cams"]))):
            s = X.sample(arms0, "clean")
            if s is None:
                continue
            bl = [b for b in arms0 if b in base]
            sc, jsc = scores(s, arms0)
            prs = [(r_, b) for r_ in [f0] + rings for b in bl] + [(r_, f0) for r_ in rings]
            res = contrasts(s, prs, metrics=("rmse",), cells_only=False)
            js[tag] = dict(sample=s.describe(), scores=jsc, contrasts=res)
            md += f"### Set {tag}\n\n" + intro(s)
            for gk, gt in (("all", "all monitors, pooled"), ("regmean", "mean over regions")):
                rows = []
                for r_ in [f0] + rings:
                    b0 = sc[r_][("all", "all")]
                    rows.append([lab(r_) if r_ != f0 else "monitor's pixel", f"{b0['rmse']:.2f}", f"{b0['bias']:+.2f}", f"{b0['r']:.3f}",
                                 fci(res[f"{r_}_vs_{f0}|rmse"]["all"].get(gk)) if r_ != f0 else "-"] + [fci(res[f"{r_}_vs_{b}|rmse"]["all"].get(gk)) for b in bl])
                md += f"RMSE change in %, {gt} (RMSE, bias and r are pooled):\n\n" + md_table(["where the 1 km forecast is read", "RMSE", "bias", "r", "vs the pixel"] + [f"vs {lab(b)}" for b in bl], rows) + "\n"
            for b in bl:
                rows = [[g] + [fci(res[f"{r_}_vs_{b}|rmse"]["all"].get(g)) for r_ in [f0] + rings] for g in s.gnames if g in res[f"{f0}_vs_{b}|rmse"]["all"]]
                md += f"RMSE change in % against {lab(b)}, by region:\n\n" + md_table(["region", "pixel"] + [lab(r_) for r_ in rings], rows) + "\n"
            rows = [[g] + [fci(res[f"{r_}_vs_{f0}|rmse"]["all"].get(g)) for r_ in rings] for g in s.gnames if g in res[f"{rings[0]}_vs_{f0}|rmse"]["all"]]
            md += "RMSE change in % of each ring against the monitor's pixel, by region:\n\n" + md_table(["region"] + [lab(r_) for r_ in rings], rows) + "\n"
            setk = "withCAMS" if tag.startswith("with ") else "noCAMS"
            for r_ in [f0] + rings:
                for b in bl + ([f0] if r_ != f0 else []):
                    for g in ("all", "regmean"):
                        v = res[f"{r_}_vs_{b}|rmse"]["all"].get(g)
                        if v:
                            X.add(f"rings.rmse_change_pct.{r_}_vs_{b}.{setk}.{g}", v["est"], v["ci"], s, "T08_rings", n=v["n"])
        X.write("T08_rings", md, js)

    # ---- T09 analysis mode ---------------------------------------------------------------------------------------------
    if want("T09") and "an" in D.arms:
        md = ("## T09. Forecast inputs against lead time: the analysis-mode downscaling of the same valid day\n\n'Analysis mode' = the same downscaler fed with the inputs of "
              "its retrospective tests for the valid day (first-day mean of the CAMS run, reanalysis weather, fires of the day, transport on reanalysis winds); it does not "
              "depend on the lead day. Available for valid days up to 2026-10-21. Lead day 1 isolates the effect of the real-time inputs; the later lead days add the lead time.\n\n")
        js = {}
        armsN = [f0, "an"] + [b for b in base if b != "cams"]
        s = X.sample(armsN, "clean")
        if s is not None:
            sc, jsc = scores(s, armsN)
            prs = [(f0, "an"), ("an", "clim"), ("an", "bcsd"), ("an", "rt04"), (f0, "clim"), (f0, "bcsd"), (f0, "rt04")]
            res = contrasts(s, prs, cells_only=False)
            t = temporal(s, armsN, [f0], Bt=X.Bt)
            e = ensemble(s, [f0, "an"], ["clim", "bcsd"])
            js["main"] = dict(sample=s.describe(), scores=jsc, contrasts=res, temporal=t, ensemble=e)
            md += "### All clean issue days with the analysis-mode arm\n\n" + intro(s) + md_scores(s, sc, armsN, metrics=(("rmse", 2, False), ("bias", 2, True), ("r", 3, False)))
            md += "RMSE change in %:\n\n" + md_contrasts(s, res, prs, "rmse", grid=False)
            rows = [[g] + [fci(res[f"{f0}_vs_an|rmse"][Lk].get(g)) for Lk in ["1", "2", "3", "4", "5"]] for g in ["all", "regmean"] + s.gnames if g in res[f"{f0}_vs_an|rmse"]["all"]]
            md += f"{lab(f0)} (forecast mode) vs analysis mode, RMSE change in % by region and lead day:\n\n" + md_table(["group", "lead 1", "lead 2", "lead 3", "lead 4", "lead 5"], rows) + "\n"
            md += "CRPS change in %:\n\n" + md_contrasts(s, res, [(f0, "an"), ("an", "clim"), (f0, "clim")], "crps", grid=False)
            md += "Day-to-day signal:\n\n" + md_temporal(s, t, armsN, [f0])
            rows = []
            for arm in (f0, "an"):
                for k, n in [("all|all", "all")] + [(f"{Ld}|all", f"lead day {Ld}") for Ld in range(1, 6)]:
                    v = e[arm][k]
                    rows.append([lab(arm), n, f"{v['crps']['est']:.2f}", f"{v['spread_skill']['est']:.2f}", f"{v['cover']['est']:.1f}", f"{v['outside']:.1f}"])
            md += "Ensembles:\n\n" + md_table(["arm", "group", "fair CRPS", "spread / skill", "inside 10-90 % (%)", "outside the ensemble (%)"], rows) + "\n"
            for Lk in ["all", "1", "2", "3", "4", "5"]:
                for g in ["all", "regmean"] + (s.gnames if Lk in ("all", "1") else []):
                    for (a_, b_) in prs:
                        v = res[f"{a_}_vs_{b_}|rmse"][Lk].get(g)
                        if v:
                            X.add(f"an.rmse_change_pct.{a_}_vs_{b_}.lead_{Lk}.{g}", v["est"], v["ci"], s, "T09_analysis_mode", n=v["n"])
            for arm in armsN:
                for g in ("all", "regmean"):
                    X.add(f"an.temporal_r_median.{arm}.{g}", t["groups"][g][arm]["r"]["est"], t["groups"][g][arm]["r"].get("ci"), s, "T09_analysis_mode", unit="")
                    X.add(f"an.amplitude_ratio_median.{arm}.{g}", t["groups"][g][arm]["amp"]["est"], t["groups"][g][arm]["amp"].get("ci"), s, "T09_analysis_mode", unit="")
        if "cams" in D.arms:
            armsC = [f0, "an", "cams"] + [b for b in base if b != "cams"]
            s = X.sample(armsC, "clean")
            if s is not None:
                sc, jsc = scores(s, armsC)
                prs = [(f0, "cams"), ("an", "cams"), (f0, "an")]
                res = contrasts(s, prs, cells_only=False)
                js["with_cams"] = dict(sample=s.describe(), scores=jsc, contrasts=res)
                md += "### The issue days that also have the CAMS forecast\n\n" + intro(s) + md_scores(s, sc, armsC, metrics=(("rmse", 2, False),), leads=True)
                md += "RMSE change in %:\n\n" + md_contrasts(s, res, prs, "rmse", grid=False)
                for (a_, b_) in prs:
                    for g in ("all", "regmean"):
                        v = res[f"{a_}_vs_{b_}|rmse"]["all"].get(g)
                        if v:
                            X.add(f"an_withCAMS.rmse_change_pct.{a_}_vs_{b_}.lead_all.{g}", v["est"], v["ci"], s, "T09_analysis_mode", n=v["n"])
        X.write("T09_analysis_mode", md, js)

    # ---- T10 Europe ----------------------------------------------------------------------------------------------------
    if want("T10"):
        md = "## T10. Europe (catalogue box 35-60 N, 10 W-35 E)\n\n"
        js = {}
        eur = D.reg == REGIONS.index("Europe")
        sub = eu_sub(D.lat, D.lon)[D.mon]
        for tag, arms, prs in sets(SAME, (("with the CAMS forecast", B5, pairsB), ("without the CAMS forecast (more issue days)", A4, pairsA))):
            s = X.sample(arms, "clean", extra=eur, key=("eu_sub", tag), group=sub, gnames=EU_SUB)
            if s is None:
                continue
            sc, jsc = scores(s, arms)
            res = contrasts(s, prs)
            js[f"subregions|{tag}"] = dict(sample=s.describe(), scores=jsc, contrasts=res)
            md += f"### Sub-regions, set {tag}\n\n" + intro(s)
            md += ("Sub-regions are boxes fixed before reading any score: west of 6 E and north of 44 N; west of 6 E and south of 44 N; 6-19 E and 46.5-55 N; 6-19 E south of "
                   "46.5 N; 6-19 E north of 55 N; east of 19 E. 'Mean over regions' is here the mean over these six sub-regions.\n\n")
            md += md_scores(s, sc, arms, metrics=(("rmse", 2, False), ("bias", 2, True), ("r", 3, False)))
            md += "RMSE change in %:\n\n" + md_contrasts(s, res, prs, "rmse", grid=False)
            md += "CRPS change in %:\n\n" + md_contrasts(s, res, prs, "crps", by_lead=False, grid=False)
            setk = "withCAMS" if tag.startswith("with ") else "noCAMS"
            for (a_, b_) in prs:
                for Lk in res[f"{a_}_vs_{b_}|rmse"]:
                    for g in ["all", "regmean"] + (EU_SUB if Lk == "all" else []):
                        v = res[f"{a_}_vs_{b_}|rmse"][Lk].get(g)
                        if v:
                            X.add(f"europe.rmse_change_pct.{a_}_vs_{b_}.{setk}.lead_{Lk}.{g}", v["est"], v["ci"], s, "T10_europe", n=v["n"])
            # by week of the valid day (groups = time blocks; cells-only bootstrap inside a week)
            sw_ = X.sample(arms, "clean", extra=eur, key=("eu_all", tag))
            wk = np.full(D.N, 0, np.int64)
            wk[sw_.rows] = sw_.w
            s2 = Sample(D, np.isin(np.arange(D.N), sw_.rows), group=wk, gnames=sw_.blocks, B=X.B)
            sc2, jsc2 = scores(s2, arms)
            rows = []
            jw = {}
            for gi, bl_ in enumerate(s2.gnames):
                r = [bl_, f"{sc2[arms[0]][('all', gi)]['n']:,}", f"{sc2[arms[0]][('all', gi)]['obs_mean']:.1f}"] + [f"{sc2[a_][('all', gi)]['rmse']:.2f}" for a_ in arms]
                jw[bl_] = {}
                for (a_, b_) in prs:
                    c = contrast(s2, a_, b_, "rmse", "c")["all"].get(bl_)
                    r.append(fci(c))
                    jw[bl_][f"{a_}_vs_{b_}"] = c
                rows.append(r)
            js[f"weeks|{tag}"] = dict(scores=jsc2, contrasts=jw)
            md += ("By week of the valid day (all lead days; intervals: 2-degree cells only, because a single week is the time block):\n\n" +
                   md_table(["valid days", "station-days", "mean obs"] + [f"RMSE {lab(a_)}" for a_ in arms] + [f"{lab(a_)} vs {lab(b_)}" for a_, b_ in prs], rows) + "\n")
        # the episode of 6-9 November and the two halves of the live period
        if sB is not None:
            ep = (D.vday >= L.doy0("2026-11-06")) & (D.vday <= L.doy0("2026-11-09"))
            livev = (D.vday >= L.doy0("2026-11-03")) & (D.vday <= L.doy0("2026-11-15"))
            grp = np.where(ep, 0, np.where(livev, 1, 2)).astype(np.int64)
            gn = ["valid days 6-9 November", "other valid days of 3-15 November", "all other valid days"]
            s = X.sample(B5, "clean", extra=eur, key=("eu_episode",), group=grp, gnames=gn)
            sc, jsc = scores(s, B5)
            res = {f"{a_}_vs_{b_}": contrast(s, a_, b_, "rmse", "c") for a_, b_ in pairsB}
            js["episode"] = dict(sample=s.describe(), scores=jsc, contrasts=res)
            rows = []
            for gi, g in enumerate(gn):
                b0 = sc[B5[0]][("all", gi)]
                rows.append([g, f"{b0['n']:,}", f"{b0['obs_mean']:.1f}"] + [f"{sc[a_][('all', gi)]['rmse']:.2f} / {sc[a_][('all', gi)]['bias']:+.1f} / {sc[a_][('all', gi)]['r']:.2f}" for a_ in B5])
            md += "### The episode of 6-9 November 2026 (set with the CAMS forecast)\n\n" + intro(s) + "RMSE / bias / r by arm:\n\n"
            md += md_table(["valid days", "station-days", "mean obs"] + [lab(a_) for a_ in B5], rows) + "\n"
            rows = [[g] + [fci(res[f"{a_}_vs_{b_}"]["all"].get(g)) for a_, b_ in pairsB] for g in gn]
            md += "RMSE change in % (intervals: 2-degree cells only):\n\n" + md_table(["valid days"] + [f"{lab(a_)} vs {lab(b_)}" for a_, b_ in pairsB], rows) + "\n"
            rows = []
            for Lk in ["1", "2", "3", "4", "5"]:
                rows.append([Lk] + [fci(res[f"{a_}_vs_{b_}"][Lk].get(gn[0])) for a_, b_ in pairsB])
            md += "Valid days 6-9 November by lead day, RMSE change in %:\n\n" + md_table(["lead day"] + [f"{lab(a_)} vs {lab(b_)}" for a_, b_ in pairsB], rows) + "\n"
            for g in gn:
                for a_, b_ in pairsB:
                    v = res[f"{a_}_vs_{b_}"]["all"].get(g)
                    if v:
                        X.add(f"europe.episode.rmse_change_pct.{a_}_vs_{b_}.{g}", v["est"], v["ci"], s, "T10_europe", n=v["n"], note="interval: cells only")
            # periods in Europe
            rows = []
            for p in ("sep", "oct", "live", "clean"):
                for tag, arms, prs in (("with CAMS", B5, pairsB), ("without CAMS", A4, pairsA)):
                    sp = X.sample(arms, p, extra=eur, key=("eu_period", p, tag))
                    if sp is None or (tag == "without CAMS" and (sw0 := X.sample(B5, p, extra=eur, key=("eu_period", p, "with CAMS"))) is not None and sw0.n == sp.n):
                        continue
                    scp, _ = scores(sp, arms)
                    r = [PERIODS[p][2], tag, sp.n_issue, f"{sp.n:,}", f"{scp[arms[0]][('all', 'all')]['obs_mean']:.1f}"]
                    jp = {}
                    for (a_, b_) in pairsB:
                        if b_ in arms:
                            c = contrast(sp, a_, b_, "rmse")["all"]["all"]
                            c2 = contrast(sp, a_, b_, "rmse", "c")["all"]["all"]
                            r.append(f"{fci(c)}; cells only [{c2['ci'][0]:+.1f}, {c2['ci'][1]:+.1f}]")
                            jp[f"{a_}_vs_{b_}"] = dict(two_way=c, cells_only=c2)
                            X.add(f"europe.period_{p}.{tag.replace(' ', '_')}.rmse_change_pct.{a_}_vs_{b_}", c["est"], c["ci"], sp, "T10_europe")
                        else:
                            r.append("n/a")
                    js[f"period|{p}|{tag}"] = dict(sample=sp.describe(), blocks=sp.blocks, contrasts=jp)
                    rows.append(r)
            md += ("### Europe by period\n\nRMSE change in %, all European monitors pooled; two-way interval, then the cells-only interval (a period of two or three weeks "
                   "has too few time blocks for the two-way interval to mean much):\n\n")
            md += md_table(["period", "set", "issue days", "station-days", "mean obs"] + [f"{lab(a_)} vs {lab(b_)}" for a_, b_ in pairsB], rows) + "\n"
        # were the live weeks typical? weekly mean of the European network through 2026
        md_t, js_t = europe_typical(D, meta)
        md += md_t
        js["typical"] = js_t
        X.write("T10_europe", md, js)

    # ---- T11 periods ---------------------------------------------------------------------------------------------------
    if want("T11"):
        md = ("## T11. The periods separately\n\nRMSE change in %: all monitors pooled, then the mean over regions. Two-way intervals; a period with fewer than four week "
              "blocks has too few time blocks for the time part of the interval to be trusted (cells-only interval given in the JSON file).\n\n")
        js = {}
        rows, rows2 = [], []
        for p in ("sep", "oct", "live", "octnov", "clean", "all53"):
            for tag, arms, prs in (("with CAMS", B5, pairsB), ("without CAMS", A4, pairsA)):
                s = X.sample(arms, p)
                if s is None or (tag == "without CAMS" and (sw0 := X.sample(B5, p)) is not None and sw0.n == s.n):
                    continue
                sc, jsc = scores(s, arms)
                res = contrasts(s, prs)
                js[f"{p}|{tag}"] = dict(sample=s.describe(), blocks=s.blocks, scores=jsc, contrasts=res)
                b0 = sc[arms[0]][("all", "all")]
                r = [PERIODS[p][2], tag, s.n_issue, f"{s.n:,}", s.W, f"{b0['obs_mean']:.1f}"]
                r2 = [PERIODS[p][2], tag] + [f"{sc[a_][('all', 'all')]['rmse']:.2f}" if a_ in arms else "n/a" for a_ in B5]
                for (a_, b_) in pairsB:
                    if b_ in arms:
                        v = res[f"{a_}_vs_{b_}|rmse"]["all"]
                        r.append(f"{fci(v['all'])}; {fci(v.get('regmean'))}")
                        for g in ["all", "regmean"] + s.gnames:
                            if g in v:
                                X.add(f"period_{p}.{tag.replace(' ', '_')}.rmse_change_pct.{a_}_vs_{b_}.{g}", v[g]["est"], v[g]["ci"], s, "T11_periods", n=v[g]["n"])
                    else:
                        r.append("n/a")
                rows.append(r)
                rows2.append(r2)
        nf = [L.doy0(k) for k, v in meta["obs_per_valid_day"].items() if v["final"] is False]
        s = X.sample(B5, "clean", extra=~np.isin(D.vday, nf), key=("clean_final_obs",))
        if s is not None and nf:
            res = contrasts(s, pairsB)
            sc, jsc = scores(s, B5)
            js["clean|final observations only"] = dict(sample=s.describe(), blocks=s.blocks, scores=jsc, contrasts=res, excluded_valid_days=[L.day_of(k) for k in nf])
            rows.append([PERIODS["clean"][2] + ", WITHOUT the valid days whose observations are not final (" + ", ".join(L.day_of(k)[5:] for k in nf) + ")", "with CAMS", s.n_issue,
                         f"{s.n:,}", s.W, f"{sc[B5[0]][('all', 'all')]['obs_mean']:.1f}"] +
                        [f"{fci(res[f'{a_}_vs_{b_}|rmse']['all']['all'])}; {fci(res[f'{a_}_vs_{b_}|rmse']['all'].get('regmean'))}" for a_, b_ in pairsB])
            rows2.append([PERIODS["clean"][2] + ", final observations only", "with CAMS"] + [f"{sc[a_][('all', 'all')]['rmse']:.2f}" for a_ in B5])
            for a_, b_ in pairsB:
                for g in ("all", "regmean"):
                    v = res[f"{a_}_vs_{b_}|rmse"]["all"].get(g)
                    X.add(f"final_obs_only.rmse_change_pct.{a_}_vs_{b_}.{g}", v["est"], v["ci"], s, "T11_periods", n=v["n"])
        md += md_table(["period (issue days)", "set", "issue days", "station-days", "week blocks", "mean obs"] + [f"{lab(a_)} vs {lab(b_)}: pooled; mean over regions" for a_, b_ in pairsB], rows) + "\n"
        md += "RMSE of each arm (ug m-3), all monitors pooled:\n\n" + md_table(["period (issue days)", "set"] + [lab(a_) for a_ in B5], rows2) + "\n"
        for p in ("sep", "oct", "live"):
            for tag, arms, prs in (("with CAMS", B5, pairsB), ("without CAMS", A4, pairsA)):
                k = f"{p}|{tag}"
                if k not in js or (tag == "without CAMS" and f"{p}|with CAMS" in js):
                    continue
                res = js[k]["contrasts"]
                s = X.sample(arms, p)
                rows = [[g] + [fci(res[f"{a_}_vs_{b_}|rmse"]["all"].get(g)) for a_, b_ in prs] for g in ["all", "regmean"] + s.gnames if g in res[f"{prs[0][0]}_vs_{prs[0][1]}|rmse"]["all"]]
                md += f"{PERIODS[p][2]}, set {tag}, by region (RMSE change in %):\n\n" + md_table(["group"] + [f"{lab(a_)} vs {lab(b_)}" for a_, b_ in prs], rows) + "\n"
        X.write("T11_periods", md, js)

    # ---- T12 title rule ------------------------------------------------------------------------------------------------
    if want("T12") and resB is not None and tB is not None:
        md = ("## T12. The title rule, evaluated\n\nRule: 'Real-time' enters the title only if the 1 km forecast is below BOTH the CAMS forecast and the 1 km climatology in RMSE "
              "with intervals excluding zero, as the mean over regions and in most regions, AND its per-monitor temporal correlation is not below its input's.\n\n" + intro(sB))
        js = {}
        strict, lenient = [], []
        rows = []
        for b in ("cams", "clim"):
            v = resB[f"{f0}_vs_{b}|rmse"]["all"]
            rm = v["regmean"]
            c1 = rm["ci"][1] < 0
            c2 = rm["n_negative"] > rm["n_groups"] / 2
            c3 = rm["n_ci_below_zero"] > rm["n_groups"] / 2
            js[b] = dict(pooled=v["all"], regmean=rm, regions={g: v[g] for g in sB.gnames if g in v}, mean_over_regions_interval_below_zero=bool(c1),
                         regions_with_lower_rmse=rm["n_negative"], regions_with_interval_below_zero=rm["n_ci_below_zero"], n_regions=rm["n_groups"])
            rows.append([f"RMSE below the {lab(b)}, mean over regions, interval excluding zero", fci(rm), "yes" if c1 else "NO"])
            rows.append([f"RMSE below the {lab(b)} in most regions",
                         f"lower in {rm['n_negative']} of {rm['n_groups']} regions; interval entirely below zero in {rm['n_ci_below_zero']}, entirely above zero in {rm['n_ci_above_zero']}",
                         "yes" if c3 else ("point estimates only (intervals exclude zero in fewer than half of the regions)" if c2 else "NO")])
            rows.append([f"for reference: pooled over station-days, against the {lab(b)}", fci(v["all"]), "(not part of the rule)"])
            strict += [c1, c3]
            lenient += [c1, c2]
        d = tB["groups"]["all"][f"{f0}-rt04"]
        dr = tB["groups"]["regmean"][f"{f0}-rt04"]
        c4s = d["r_median_of_diff"]["est"] >= 0 and dr["r_median_of_diff"]["est"] >= 0
        c4l = d["r_median_of_diff"]["ci"][1] >= 0
        js["temporal"] = dict(all=d, regmean=dr, km=tB["groups"]["all"][f0]["r"], rt04=tB["groups"]["all"]["rt04"]["r"],
                              by_region={g: tB["groups"][g][f"{f0}-rt04"] for g in sB.gnames if g in tB["groups"]})
        nreg = [g for g in sB.gnames if g in tB["groups"]]
        nlow = sum(tB["groups"][g][f"{f0}-rt04"]["r_median_of_diff"]["est"] < 0 for g in nreg)
        nlow_sig = sum(tB["groups"][g][f"{f0}-rt04"]["r_median_of_diff"]["ci"][1] < 0 for g in nreg)
        nhigh_sig = sum(tB["groups"][g][f"{f0}-rt04"]["r_median_of_diff"]["ci"][0] > 0 for g in nreg)
        rows.append(["per-monitor temporal correlation not below the 0.4 deg input's",
                     f"median over series {tB['groups']['all'][f0]['r']['est']:.3f} against {tB['groups']['all']['rt04']['r']['est']:.3f}; median of the paired difference "
                     f"{d['r_median_of_diff']['est']:+.3f} [{d['r_median_of_diff']['ci'][0]:+.3f}, {d['r_median_of_diff']['ci'][1]:+.3f}]; mean over regions of that median "
                     f"{dr['r_median_of_diff']['est']:+.3f} [{dr['r_median_of_diff']['ci'][0]:+.3f}, {dr['r_median_of_diff']['ci'][1]:+.3f}]; higher at "
                     f"{100 * d['share_series_r_higher']:.0f} % of the series; lower in {nlow} of {len(nreg)} regions (interval entirely below zero in {nlow_sig}, entirely above in {nhigh_sig})",
                     "yes" if c4s else ("lower in the pooled median, the interval includes zero" if c4l else "NO")])
        strict.append(c4s)
        lenient.append(c4l)
        js["verdict_strict"] = "met" if all(strict) else "not met"
        js["verdict_lenient"] = "met" if all(lenient) else "not met"
        md += md_table(["condition", "value", "met"], rows) + "\n"
        md += (f"**Verdict on {sB.n_issue} issue days ({sB.issue_list[0]}..{sB.issue_list[-1]}): the rule as written is {'MET' if all(strict) else 'NOT MET'}.** "
               f"Strict reading = every interval excludes zero in more than half of the regions and the temporal correlation is not lower than the input's in the point "
               f"estimates: {js['verdict_strict']}. Lenient reading = mean over regions with an interval excluding zero, lower RMSE in more than half of the regions by the point "
               f"estimates, and a temporal correlation not significantly below the input's: {js['verdict_lenient']}.\n\n")
        if resA is not None and not SAME:
            v = resA[f"{f0}_vs_clim|rmse"]["all"]
            md += (f"On the larger set without the CAMS forecast ({sA.n_issue} issue days, {sA.n:,} station-days) the change against the 1 km climatology is "
                   f"{fci(v['regmean'])} as the mean over regions (lower in {v['regmean']['n_negative']} of {v['regmean']['n_groups']} regions, interval entirely below zero in "
                   f"{v['regmean']['n_ci_below_zero']}, entirely above zero in {v['regmean']['n_ci_above_zero']}) and {fci(v['all'])} pooled.")
            js["clim_larger_set"] = v
            if tA is not None:
                d2, dr2 = tA["groups"]["all"][f"{f0}-rt04"], tA["groups"]["regmean"][f"{f0}-rt04"]
                md += (f" On that set the median temporal correlation is {tA['groups']['all'][f0]['r']['est']:.3f} for the 1 km forecast against "
                       f"{tA['groups']['all']['rt04']['r']['est']:.3f} for its input; median of the paired difference {d2['r_median_of_diff']['est']:+.3f} "
                       f"[{d2['r_median_of_diff']['ci'][0]:+.3f}, {d2['r_median_of_diff']['ci'][1]:+.3f}], mean over regions {dr2['r_median_of_diff']['est']:+.3f} "
                       f"[{dr2['r_median_of_diff']['ci'][0]:+.3f}, {dr2['r_median_of_diff']['ci'][1]:+.3f}].")
                js["temporal_larger_set"] = dict(all=d2, regmean=dr2, km=tA["groups"]["all"][f0]["r"], rt04=tA["groups"]["all"]["rt04"]["r"])
            md += "\n\n"
        X.write("T12_title_rule", md, js)

    # ---- T13 monitors at official sites (proxy for reference-grade instruments) -----------------------------------------
    rg_file = Path(a.pairs).parent / "refgrade.npz"
    if want("T13") and rg_file.exists():
        rg = np.load(rg_file)
        tol = float(rg["tol"])
        eu_m = D.region_name == "Europe"
        off_eu = eu_m & (rg["d_eea"] <= tol)
        us_m = rg["usa_box"]
        off_us = us_m & (np.minimum(rg["d_epa88101"], rg["d_epa88502"]) <= tol)
        gm = np.full(D.id.size, -1, np.int64)
        gm[eu_m & ~off_eu], gm[off_eu], gm[us_m & ~off_us], gm[off_us] = 1, 0, 3, 2
        gn = ["Europe, at an official PM2.5 site", "Europe, other monitors", "United States, at an official PM2.5 site", "United States, other monitors"]
        grp = gm[D.mon]
        md = ("## T13. Monitors at the sites of official reference networks (a proxy)\n\nNo file on disk gives the instrument type of the OpenAQ locations. Proxy: a monitor is 'at an official "
              f"PM2.5 site' when it lies within {tol:.0f} m of a PM2.5 sampling point of the European network registry (`data/stations/eea/PanEuropean_metadata.csv`, pollutant 6001) or of a "
              "United States site of the 2024 files of parameters 88101 / 88502 (`data/stations/epa_aqs`). The other monitors of the two regions are a mixture of low-cost sensors and of "
              "official sites missing from these registries. United States = catalogue monitors of the box 24.5-49.5 N, 125-66.5 W. 'Mean over regions' is not used here.\n\n")
        js = {}
        for tag, arms, prs in sets(SAME, (("with the CAMS forecast", B5, pairsB), ("without the CAMS forecast (more issue days)", A4, pairsA))):
            s = X.sample(arms, "clean", extra=grp >= 0, key=("refgrade", tag), group=np.maximum(grp, 0), gnames=gn)
            if s is None:
                continue
            sc, jsc = scores(s, arms)
            res = contrasts(s, prs)
            t = temporal(s, arms, focus, boot=False)
            e = ensemble(s, focus, [])
            js[tag] = dict(sample=s.describe(), scores=jsc, contrasts=res, temporal=t, ensemble=e)
            gc = group_counts(s)
            md += f"### Set {tag}\n\n" + intro(s)
            rows = []
            for gi, g in enumerate(gn):
                b0 = sc[arms[0]][("all", gi)]
                rows.append([g, f"{b0['n']:,}", f"{gc[g][1]:,}", f"{b0['obs_mean']:.1f}"] + [f"{sc[a_][('all', gi)]['rmse']:.2f} / {sc[a_][('all', gi)]['bias']:+.1f} / {sc[a_][('all', gi)]['r']:.2f}" for a_ in arms])
            md += "RMSE / bias / r by arm:\n\n" + md_table(["group", "station-days", "monitors", "mean obs"] + [lab(a_) for a_ in arms], rows) + "\n"
            rows = [[g] + [fci(res[f"{a_}_vs_{b_}|rmse"]["all"].get(g)) for a_, b_ in prs] for g in gn]
            md += "RMSE change in %:\n\n" + md_table(["group"] + [f"{lab(a_)} vs {lab(b_)}" for a_, b_ in prs], rows) + "\n"
            rows = [[g] + [fci(res[f"{a_}_vs_{b_}|crps"]["all"].get(g)) for a_, b_ in prs] for g in gn]
            md += "CRPS change in %:\n\n" + md_table(["group"] + [f"{lab(a_)} vs {lab(b_)}" for a_, b_ in prs], rows) + "\n"
            rows = [[g] + [f"{t['groups'][g][a_]['r']['est']:.2f} / {t['groups'][g][a_]['amp']['est']:.2f}" for a_ in arms] for g in gn if g in t["groups"]]
            md += "Median per-monitor temporal correlation / amplitude ratio:\n\n" + md_table(["group"] + [lab(a_) for a_ in arms], rows) + "\n"
            rows = [[g, f"{e[f0][f'all|{g}']['spread_skill']['est']:.2f}", f"{e[f0][f'all|{g}']['cover']['est']:.1f}", f"{e[f0][f'all|{g}']['below']:.1f} / {e[f0][f'all|{g}']['above']:.1f}",
                     f"{e[f0][f'all|{g}']['outside']:.1f}"] for g in gn if f"all|{g}" in e[f0]]
            md += f"Ensemble of {lab(f0)}:\n\n" + md_table(["group", "spread / skill", "inside 10-90 % (%)", "below / above (%)", "outside the ensemble (%)"], rows) + "\n"
            setk = "withCAMS" if tag.startswith("with ") else "noCAMS"
            for g in gn:
                for a_, b_ in prs:
                    v = res[f"{a_}_vs_{b_}|rmse"]["all"].get(g)
                    if v:
                        X.add(f"official_sites.rmse_change_pct.{a_}_vs_{b_}.{setk}.{g}", v["est"], v["ci"], s, "T13_official_sites", n=v["n"])
        X.write("T13_official_sites", md, js)

    # ---- T14 check against the first live comparison -------------------------------------------------------------------
    if want("T14"):
        md, js = check_live(D, X)
        X.write("T14_check_live", md, js)

    if not only:
        import re
        HEAD = [r"^rmse_change_pct\.km_vs_(cams|clim|bcsd|rt04)\.withCAMS\.lead_(all|[1-5])\.(all|regmean)$",
                r"^rmse_change_pct\.km_vs_(cams|clim|bcsd|rt04)\.withCAMS\.lead_all\.", r"^crps_change_pct\.km_vs_(cams|clim|bcsd|rt04|bcsd_m)\.withCAMS\.lead_all\.(all|regmean)$",
                r"^(rmse|bias|r)\.(km|rt04|cams|clim|bcsd)\.withCAMS\.lead_all\.all$", r"^temporal_r_median\.(km|rt04|cams|clim|bcsd)\.withCAMS\.(all|regmean)$",
                r"^amplitude_ratio_median\.(km|rt04|cams|clim|bcsd)\.withCAMS\.(all|regmean)$", r"^temporal_r_median_of_diff\.km-(rt04|cams)\.withCAMS\.(all|regmean)$",
                r"^(coverage_10_90_pct|spread_skill|fair_crps|outside_ensemble_pct)\.(km|rt04|bcsd_m)\.withCAMS\.lead_all\.all$", r"^mae_as_crps\.", r"^brier\.P_gt_.*\.(km|cams|clim)\.withCAMS\.all$",
                r"^sedi\..*\.(km|cams|clim)\.withCAMS\.all$", r"^rings\.rmse_change_pct\.km_(ring1|ring3|ring10|cell_lin)_vs_(km|cams)\.withCAMS\.(all|regmean)$",
                r"^an\.rmse_change_pct\.km_vs_an\.lead_(all|1|5)\.(all|regmean)$", r"^an\.rmse_change_pct\.an_vs_clim\.lead_all\.(all|regmean)$",
                r"^europe\.rmse_change_pct\.km_vs_(cams|clim|bcsd|rt04)\.withCAMS\.lead_all\.all$", r"^europe\.episode\.rmse_change_pct\.km_vs_cams\.",
                r"^period_(sep|oct|live)\.with_CAMS\.rmse_change_pct\.km_vs_(cams|clim|rt04|bcsd)\.(all|regmean)$", r"^official_sites\.rmse_change_pct\.km_vs_(cams|clim|bcsd)\.withCAMS\.",
                r"^share_of_squared_error_pct\.km\.withCAMS\.South Asia$", r"^final_obs_only\."]
        for e in X.num:
            e["headline"] = any(re.search(h, e["name"]) for h in HEAD)
        X.num.sort(key=lambda e: not e["headline"])
        print(f"numbers.json: {sum(e['headline'] for e in X.num)} headline entries first")
        (X.out / "numbers.json").write_text(json.dumps(X.num, indent=0, default=float))
        print(f"numbers.json: {len(X.num)} entries")
    print(f"done in {time.time() - t0:.0f} s")


def europe_typical(D, meta):
    """weekly mean of the European monitors' daily means through 2026 (truth file to 2026-10-21, archive after)"""
    W = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
    tr = np.load(W / "hindcast/2026/st_truth_2026.npz")
    tix = {s: i for i, s in enumerate(tr["id"].astype(str).tolist())}
    eu = np.flatnonzero(D.region_name == "Europe")
    daily = tr["daily"][np.array([tix[s] for s in D.id[eu]])]                     # (n_eu, 365)
    cix = {s: i for i, s in enumerate(D.id.tolist())}
    series = {}
    for k in range(daily.shape[1]):
        d = L.D0 + dt.timedelta(k)
        if d <= dt.date(2026, 10, 21):
            v = daily[:, k]
        else:
            f = W / "realtime_1km/verif/obs" / f"{d}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            v = np.full(D.id.size, np.nan)
            kk = np.array([cix.get(s, -1) for s in z["id"].astype(str).tolist()], np.int64)
            v[kk[kk >= 0]] = z["value"][kk >= 0]
            v = v[eu]
        if np.isfinite(v).sum() >= 300:
            series[d] = (float(np.nanmean(v)), float(np.nanmedian(v)), float(np.mean(v[np.isfinite(v)] > 25.0)), int(np.isfinite(v).sum()))
    weeks = {}
    for d, x in series.items():
        weeks.setdefault(tuple(d.isocalendar())[:2], []).append((d, x))
    rows = {}
    for w, lst in sorted(weeks.items()):
        if len(lst) >= 4:
            rows[f"{w[0]}-W{w[1]:02d}"] = dict(first=str(min(d for d, _ in lst)), last=str(max(d for d, _ in lst)), days=len(lst), mean=float(np.mean([x[0] for _, x in lst])),
                                               median=float(np.mean([x[1] for _, x in lst])), share_above_25=float(np.mean([x[2] for _, x in lst])),
                                               monitors=float(np.mean([x[3] for _, x in lst])))
    means = np.array([v["mean"] for v in rows.values()])
    md = ("### Were the first live weeks typical for Europe?\n\nWeekly mean over the days of the week of the mean of the European monitors' daily means (every ISO week of 2026 "
          "with at least four days of observations at 300 or more monitors):\n\n")
    tab = []
    for k, v in rows.items():
        if v["first"] >= "2026-08-31":
            rank = int((means > v["mean"]).sum()) + 1
            v["rank_from_highest"] = rank
            tab.append([k, f"{v['first']}..{v['last']}", v["days"], f"{v['monitors']:.0f}", f"{v['mean']:.1f}", f"{v['median']:.1f}", f"{100 * v['share_above_25']:.0f}", f"{rank} of {means.size}"])
    md += md_table(["week", "days", "days with data", "monitors (mean)", "mean (ug m-3)", "median monitor (ug m-3)", "monitors above 25 ug m-3 (%)", "rank of the mean among the weeks of 2026"], tab) + "\n"
    q = np.percentile(means, [10, 25, 50, 75, 90])
    md += (f"Over the {means.size} weeks of 2026 with data the weekly mean has percentiles 10/25/50/75/90 = {q[0]:.1f} / {q[1]:.1f} / {q[2]:.1f} / {q[3]:.1f} / {q[4]:.1f} ug m-3 "
           f"(the first weeks of the year are winter weeks).\n\n")
    return md, dict(weeks=rows, percentiles_of_weekly_mean=dict(zip(["p10", "p25", "p50", "p75", "p90"], [float(x) for x in q])))


def check_live(D, X):
    """row-level comparison with the pairs of the first live comparison (rc/coupling_pairs.npz) and its scores"""
    W = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/rc")
    z = np.load(W / "coupling_pairs.npz")
    cols = [str(c) for c in z["cols"]]
    A = z["pairs"].astype(np.float64)
    key_old = (A[:, cols.index("issue_doy0")].astype(np.int64) * 10 + A[:, cols.index("lead")].astype(np.int64)) * 100000 + A[:, cols.index("monitor")].astype(np.int64)
    key_new = (D.iday * 10 + D.lead) * 100000 + D.mon
    o_new = np.argsort(key_new)
    pos = np.searchsorted(key_new[o_new], key_old)
    pos = np.minimum(pos, key_new.size - 1)
    hit = key_new[o_new][pos] == key_old
    rn = o_new[pos[hit]]
    js = dict(rows_first_comparison=int(key_old.size), found_here=int(hit.sum()))
    md = ("## T14. Agreement with the first live comparison (issue days 3-11 November)\n\n" +
          f"`rc/coupling_pairs.npz` holds {key_old.size:,} station-days (issue days 3-11 November; every one of its 16 arms finite). {int(hit.sum()):,} of them are in the present table "
          "(same issue day, lead day, monitor).\n\n")
    rows = []
    for a_new, a_old in (("obs", "obs"), ("km", "km"), ("rt04", "rt04"), ("cams", "cams"), ("clim", "clim"), ("bcsd", "bcsd")):
        xo = A[hit, cols.index(a_old)]
        xn = D.obs[rn] if a_new == "obs" else D.m[a_new][rn]
        ok = np.isfinite(xn)
        d = np.abs(xn[ok] - xo[ok])
        js[a_new] = dict(n=int(ok.sum()), max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()), share_within_0p01=float((d <= 0.01).mean()))
        rows.append([a_new, f"{int(ok.sum()):,}", f"{d.max():.4f}", f"{d.mean():.5f}", f"{100 * (d <= 0.01).mean():.2f}"])
    md += "Values on the common station-days (the first comparison read the 1 km mean from the published float16 tiles and took the 0.4-degree member mean before the floor at 0):\n\n"
    md += md_table(["column", "station-days", "largest |difference| (ug m-3)", "mean |difference|", "within 0.01 ug m-3 (%)"], rows) + "\n"
    # scores on the two windows of the first comparison, here with every station-day of the present table
    out_rows = []
    for win, first, last, fjs in (("dev", "2026-11-03", "2026-11-07", "coupling_score_dev.json"), ("test", "2026-11-08", "2026-11-11", "coupling_score_test.json")):
        ref = json.loads((W / fjs).read_text())
        m = D.finite(["km", "rt04", "cams", "clim", "bcsd"]) & D.issues(first, last)
        s = Sample(D, m, B=X.B)
        t = temporal(s, ["km", "rt04", "cams", "clim", "bcsd"], ["km"], min_days=5, boot=False, pool_leads=True)
        js[win] = dict(sample=s.describe())
        for g in ["all", "Europe", "East Asia", "South Asia", "North America"]:
            gk = "all" if g == "all" else g
            new, old = {}, {}
            for b in ("cams", "rt04", "clim"):
                c = contrast(s, "km", b, "rmse", "c")["all"].get(gk)
                new[b] = c
            rg = ref.get(g, {})
            if rg:
                old["cams"] = rg["km"]["vs_cams"][0]
                old["rt04"] = 100 * (rg["km"]["rmse"] / rg["rt04"]["rmse"] - 1)
                old["clim"] = 100 * (rg["km"]["rmse"] / rg["clim"]["rmse"] - 1)
            tg = t["groups"].get(g)
            out_rows.append([win, g, f"{new['cams']['n']:,}" if new["cams"] else "", *[f"{fci(new[b])} | {old.get(b, float('nan')):+.1f}" for b in ("cams", "rt04", "clim")],
                             f"{tg['km']['r']['est']:.2f} / {tg['rt04']['r']['est']:.2f} / {tg['cams']['r']['est']:.2f} | {rg['km']['temporal_r']:.2f} / {rg['rt04']['temporal_r']:.2f} / {rg['cams']['temporal_r']:.2f}" if tg and rg else "",
                             f"{tg['km']['amp']['est']:.2f} / {tg['rt04']['amp']['est']:.2f} / {tg['cams']['amp']['est']:.2f} | {rg['km']['amplitude']:.2f} / {rg['rt04']['amplitude']:.2f} / {rg['cams']['amplitude']:.2f}" if tg and rg else ""])
            js[win][g] = dict(here=new, first_comparison=old)
    md += ("Scores on the two windows of the first comparison (issue days 3-7 and 8-11 November). 'here | first' = this scoring (cells-only interval, to compare like with like) | "
           "the value of `rc/coupling_score_{dev,test}.json`. For this check only, the temporal statistics pool the lead days of a monitor (at least 5 station-days), as the first "
           "comparison did (which, in addition, required a non-zero spread of every one of its 16 arms).\n\n")
    md += md_table(["window", "group", "station-days here", "1 km vs CAMS: here | first", "1 km vs 0.4 deg input: here | first", "1 km vs climatology: here | first",
                    "temporal r km / input / CAMS: here | first", "amplitude km / input / CAMS: here | first"], out_rows) + "\n"
    return md, js


if __name__ == "__main__":
    main()
