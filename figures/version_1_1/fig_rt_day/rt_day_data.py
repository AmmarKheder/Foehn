#!/usr/bin/env python3
"""Data of the figure 'a real-time forecast at 1 km': the two case rules, the fields of every panel, the numbers a legend
may quote. Read-only on the project; writes data/rt_day.npz, data/rule_measured.csv, data/rule_hidden.csv, data/series.csv
and numbers.json beside this script (suffix _<variant> for a variant other than the default).

Rules (nothing is chosen by looking at a forecast score):
  candidates  the 60 cities of the public list of the real-time chain (city box: +-0.5 degree of latitude and the same
              distance in longitude) on the days when at least 10 monitors of the box have a daily mean;
  days        valid days with the 1 km real-time forecast on disk at lead days 1, 3 and 5 (7-11 November 2026) and,
              in the strict variant (default), a lead-day-1 forecast computed on its issue day (8-11 November: the
              cycle of 7 November was computed on 8 November);
  case 1      the city-day with the highest mean measured PM2.5;
  case 2      the city-day with the most residents above 37.5 ug m-3 (WHO interim target 3) in the 1 km forecast of lead
              day 1 whose 0.4 degree cell is forecast at or below it by the real-time 0.4 degree forecast.
The CAMS forecast enters neither rule: case 1 uses the measurements, case 2 the two real-time forecasts.

Daily means (lead day L of the 00 UTC cycle of day d = UTC day d + L - 1):
  CAMS forecast     the mean of its EIGHT 3-hourly steps 00, 03, ..., 21 UTC of the valid day, floor 0 (the rule of the
                    paper's Methods and of the real-time scores, replay/score_cams3h), from the raw files replay/cams3h/raw:
                    rt_loaders.cams_daily_3h. It feeds the CAMS map (cams), the scores written on it, the CAMS lines of the
                    two series and every key of numbers.json and column of the csv files with "cams" in its name.
                    Until 2026-11-19 it was read at three instants (00, 12, 24 UTC, weights 1/4, 1/2, 1/4): the outputs of
                    that reading are kept beside the new ones as <name>_3instants.<ext>.
  0.4 degree        UNCHANGED, three instants (00, 12, 24 UTC, weights 1/4, 1/2, 1/4; the 0 h instant is the CAMS analysis):
                    the members of the real-time 0.4 degree forecast exist every 12 h only (rt_loaders.daily04).
  1 km              UNCHANGED: the daily fields of the downscaler (mean, p10, p90 of the 8 members).
    python rt_day_data.py            strict variant (default)
    python rt_day_data.py literal    days 7-11 November for both cases (the rule as first written)
    python rt_day_data.py ontime     days whose three cycles (lead days 1, 3, 5) were all computed on their day of issue
                                     (8 and 10 November)
Every output is written through rt_safe_io (temporary file, fsync, read back, os.replace)."""
import csv
import datetime as dt
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True                                                   # nothing is written beside the modules that are read
import rt_loaders as R  # noqa: E402
import rt_safe_io  # noqa: E402

HERE = Path(__file__).resolve().parent
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "strict"
OUT = HERE / "data"; OUT.mkdir(exist_ok=True)
SUF = "" if VARIANT == "strict" else f"_{VARIANT}"
NMIN = 10
OBS_DAYS = [dt.date(2026, 11, 3) + dt.timedelta(k) for k in range(13)]           # 3-15 November: the live period with monitors
REGION = (72.6, 79.0, 27.4, 32.6)                                                # W, E, S, N on 0.4 degree cell edges
PAIR = {"Lahore": "Delhi", "Delhi": "Lahore"}                                    # the other large city of the same haze
# cycles whose 1 km forecast was computed the day after the issue day (file times read on 2026-11-19 and the journals of
# the 0.4 degree chain: 3 November, the GHAP-trained system was run on 4 November 08:23 UTC; 7 November, catch-up run
# of the whole chain on 8 November 12:52-13:45 UTC)
LATE_CYCLES = {dt.date(2026, 11, 3), dt.date(2026, 11, 7)}


def utc_day(path):
    return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)


def predicted_utc(day):
    """when the 0.4 degree members of a cycle were computed, from the journal of the chain ('... predict <day>: ...')"""
    for line in (R.RT / "data" / str(day) / "journal.txt").read_text().splitlines():
        if f"predict {day}" in line:
            return dt.datetime.strptime(line[:20], "%Y-%m-%d %H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    return None


def at_monitors(sel, v, snap, rt, cf, lat, lon):
    """means over the monitors `sel` of valid day v of each system and lead day (NaN where there is no forecast)"""
    out = {}
    for L in (1, 2, 3, 4, 5):
        d = v - dt.timedelta(L - 1)
        if d in snap:
            mean, p10, p90 = snap[d]
            out[f"km{L}"] = float(mean[L - 1][sel].mean()); out[f"km{L}_p10"] = float(p10[L - 1][sel].mean())
            out[f"km{L}_p90"] = float(p90[L - 1][sel].mean())
            out[f"rt{L}"] = float(R.bilinear(rt[d, L], lat[sel], lon[sel]).mean())
            out[f"cams{L}"] = float(R.bilinear(cf[d, L], lat[sel], lon[sel]).mean())
    return out


def main():
    ids, lat, lon = R.catalogue()
    C = R.cities(); assert len(C) == 60
    pop = np.load(R.G / "pop1km_global.npy", mmap_mode="r")
    land = np.load(R.G / "land1km_global.npy", mmap_mode="r")                    # 1 km land mask of the paper
    obs, final = {}, {}
    for d in OBS_DAYS:
        obs[d], final[d] = R.observations(d, ids)
    snap = {d: R.snapshot(d) for d in R.ISSUES}; assert all(s is not None for s in snap.values())
    rt = {(d, L): R.daily04(d, L) for d in R.ISSUES for L in range(1, 6)}
    cf, cams_info = R.cams_daily_3h(R.ISSUES)                                    # the paper's rule: eight 3-hourly steps of the valid day
    assert sorted(cf) == sorted(rt) and all(cf[k].shape == rt[k].shape for k in cf)
    print("CAMS forecast:", cams_info["rule"], "|", cams_info["fields"], "fields from", [Path(f).name for f in cams_info["raw_files"]],
          "| three instants from the same files against the former loader:", cams_info["three_instants_from_the_same_raw_files_vs_former_loader"])
    made = {d: utc_day(R.KM / str(d) / R.TAG / "lead1_part0.npz") for d in R.ISSUES}      # when the 1 km files were written
    made04 = {d: predicted_utc(d) for d in R.ISSUES}
    on_day = {d: d not in LATE_CYCLES for d in R.ISSUES}
    for d in R.ISSUES:                                                           # the file times still tell the same story
        if (made[d].date() - d).days in (0, 1):
            assert (made[d].date() == d) == on_day[d], (d, made[d])
        else:
            print(f"warning: the 1 km files of {d} were rewritten on {made[d]:%Y-%m-%d}; LATE_CYCLES is used")
    has135 = [v for v in OBS_DAYS if all((v - dt.timedelta(L - 1)) in snap for L in (1, 3, 5))]
    if VARIANT == "strict":                                                      # lead day 1 computed on its day of issue
        days = [v for v in has135 if on_day[v]]
    elif VARIANT == "ontime":                                                    # the three cycles shown computed on their day of issue
        days = [v for v in has135 if all(on_day[v - dt.timedelta(L - 1)] for L in (1, 3, 5))]
    else:                                                                        # "literal": every day with lead days 1, 3 and 5
        days = has135
    print("variant", VARIANT, "| days with lead days 1, 3, 5:", [str(v) for v in has135], "| rule days:", [str(v) for v in days])
    print("1 km files written (UTC):", {str(d): made[d].strftime("%m-%d %H:%M") for d in R.ISSUES})
    box = {n: R.city_box(la, lo) for n, la, lo in C}
    inbox = {n: (lat >= b[0]) & (lat <= b[1]) & (lon >= b[2]) & (lon <= b[3]) for n, b in box.items()}

    # ---- rule 1: highest mean measured PM2.5
    rows1 = []
    for n, _, _ in C:
        for v in OBS_DAYS:
            m = inbox[n] & np.isfinite(obs[v])
            if m.sum() >= NMIN:
                rows1.append(dict(city=n, day=str(v), monitors=int(m.sum()), measured=float(obs[v][m].mean()), rule_day=v in days,
                                  leads135=v in has135, **at_monitors(m, v, snap, rt, cf, lat, lon)))
    rows1.sort(key=lambda r: -r["measured"])
    cand1 = [r for r in rows1 if r["rule_day"]]
    case1 = cand1[0]; city1, day1 = case1["city"], dt.date.fromisoformat(case1["day"])
    NEIGHBOUR = PAIR[city1]                                                      # the window REGION holds both (checked below)
    print(f"rule 1: {len(cand1)} city-days, {len({r['city'] for r in cand1})} cities ->", [(r["city"], r["day"], round(r["measured"], 1), r["monitors"]) for r in cand1[:5]])

    # ---- rule 2: residents above 37.5 at 1 km (lead day 1) in 0.4 degree cells forecast at or below it
    rows2 = []
    for d in R.ISSUES:
        for n, la, lo in C:
            s, nn, w, e = box[n]
            x, ext = R.window(d, 1, "mean", s, nn, w, e)
            r0, r1, c0, c1 = R.grid_window(ext)
            ok = np.isfinite(x) & R.inside(ext, x.shape, box[n]); p = np.where(ok, np.asarray(pop[r0:r1, c0:c1], np.float64), 0.0)
            c04, ccf = R.nearest04(rt[d, 1], ext, x.shape), R.nearest04(cf[d, 1], ext, x.shape)
            above = ok & (x > R.THR); m = inbox[n] & np.isfinite(obs[d])
            rows2.append(dict(city=n, day=str(d), monitors=int(m.sum()), measured=float(obs[d][m].mean()) if m.any() else float("nan"),
                              residents=float(p.sum()), above=float((p * above).sum()), hidden=float((p * (above & (c04 <= R.THR))).sum()),
                              hidden_cams=float((p * (above & (ccf <= R.THR))).sum()),
                              popw_km=float((p * np.nan_to_num(x)).sum() / max(p.sum(), 1)), popw_rt04=float((p * c04).sum() / max(p.sum(), 1)),
                              popw_cams=float((p * ccf).sum() / max(p.sum(), 1)), rule_day=d in days, computed_on_issue_day=bool(on_day[d]),
                              candidate=bool(d in days and m.sum() >= NMIN)))
    rows2.sort(key=lambda r: -r["hidden"])
    cand2 = [r for r in rows2 if r["candidate"]]
    case2 = cand2[0]; city2, day2 = case2["city"], dt.date.fromisoformat(case2["day"])
    print("rule 2:", [(r["city"], r["day"], round(r["hidden"] / 1e6, 2)) for r in cand2[:5]])
    best = {}
    for r in cand2:
        best.setdefault(r["city"], r)
    print("rule 2, best day of each city:", [(r["city"], r["day"], round(r["hidden"] / 1e6, 2)) for r in list(best.values())[:6]])
    for name, rows in (("rule_measured", rows1), ("rule_hidden", rows2)):
        with io.StringIO(newline="") as f:
            keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("city", "day"), k))
            w_ = csv.DictWriter(f, keys); w_.writeheader(); w_.writerows(rows)
            rt_safe_io.write_text(OUT / f"{name}{SUF}.csv", f.getvalue())

    # ---- a: the world at 1 km, forecast of the case day for that day
    wld = R.world(day1, 1, "mean", k=5)[120:]                                    # 84 N .. 60 S, 0.05 degree
    # ---- b-e: the region of the case
    W_, E_, S_, N_ = REGION
    km = {L: R.window(day1 - dt.timedelta(L - 1), L, "mean", S_, N_, W_, E_) for L in (1, 3, 5)}
    ext_r = km[1][1]; assert np.allclose(ext_r, REGION, atol=1e-6), ext_r
    for n in (city1, NEIGHBOUR):
        assert S_ <= box[n][0] and box[n][1] <= N_ and W_ <= box[n][2] and box[n][3] <= E_, (n, box[n])
    si, sj, ext04 = R.cells04(REGION); assert np.allclose(ext04, REGION, atol=1e-6), ext04
    mreg = (lat >= S_) & (lat <= N_) & (lon >= W_) & (lon <= E_) & np.isfinite(obs[day1])
    # ---- f-g: series at the monitors of the two cities
    series = {}
    for n in (city1, NEIGHBOUR):
        S = {k: np.full(len(OBS_DAYS), np.nan) for k in ("n", "obs", "km1", "km1_p10", "km1_p90", "km3", "km5", "rt1", "rt3", "cams1", "cams3", "in1090")}
        for t, v in enumerate(OBS_DAYS):
            m = inbox[n] & np.isfinite(obs[v]); S["n"][t] = m.sum()
            if m.sum() < NMIN:
                continue
            S["obs"][t] = obs[v][m].mean()
            a = at_monitors(m, v, snap, rt, cf, lat, lon)
            for k in S:
                if k in a:
                    S[k][t] = a[k]
            if v in snap:
                S["in1090"][t] = np.mean((obs[v][m] >= snap[v][1][0][m]) & (obs[v][m] <= snap[v][2][0][m]))
        series[n] = S
    # ---- h-j: hidden in real time
    s2, n2, w2, e2 = box[city2]
    hx, hext = R.window(day2, 1, "mean", s2, n2, w2, e2)
    r0, r1, c0, c1 = R.grid_window(hext)
    hok = np.isfinite(hx) & R.inside(hext, hx.shape, box[city2]); hpop = np.where(hok, np.asarray(pop[r0:r1, c0:c1], np.float32), 0.0)
    h04 = R.nearest04(rt[day2, 1], hext, hx.shape); hcf = R.nearest04(cf[day2, 1], hext, hx.shape)
    _, _, cext = R.cells04(hext)                                                  # the 1 km forecast averaged over whole 0.4 degree cells
    xb, extb = R.window(day2, 1, "mean", cext[2], cext[3], cext[0], cext[1])
    nr, nc = xb.shape[0] // 40, xb.shape[1] // 40; okb = np.isfinite(xb)
    with np.errstate(all="ignore"):
        cm = np.where(okb, xb, 0).reshape(nr, 40, nc, 40).sum((1, 3)) / okb.reshape(nr, 40, nc, 40).sum((1, 3))
    oi, oj = int(round((extb[3] - hext[3]) * 100)), int(round((hext[0] - extb[0]) * 100))
    hown = np.repeat(np.repeat(cm, 40, 0), 40, 1)[oi:oi + hx.shape[0], oj:oj + hx.shape[1]]
    m2 = inbox[city2] & np.isfinite(obs[day2])
    habove = hok & (hx > R.THR)
    hid = {}
    for L in (1, 2, 3):                                                           # the same count from the forecasts issued earlier
        d = day2 - dt.timedelta(L - 1)
        if d in snap:
            xl, _ = R.window(d, L, "mean", s2, n2, w2, e2); cl = R.nearest04(rt[d, L], hext, hx.shape)
            ab = hok & np.isfinite(xl) & (xl > R.THR)
            hid[L] = dict(issue=str(d), above=float((hpop * ab).sum()), hidden=float((hpop * (ab & (cl <= R.THR))).sum()),
                          popw_km=float((hpop * np.nan_to_num(xl)).sum() / hpop.sum()), popw_rt04=float((hpop * cl).sum() / hpop.sum()))
    a2 = at_monitors(m2, day2, snap, rt, cf, lat, lon)
    rt_safe_io.write_npz(
        OUT / f"rt_day{SUF}.npz", compressed=True, world=wld.astype(np.float16), day1=str(day1), city1=city1, city2=city2, day2=str(day2), neighbour=NEIGHBOUR,
        region=np.array(REGION), km1=km[1][0].astype(np.float16), km3=km[3][0].astype(np.float16), km5=km[5][0].astype(np.float16),
        rt04=rt[day1, 1][si, sj], cams=cf[day1, 1][si, sj].astype(np.float32), reg_lon=lon[mreg], reg_lat=lat[mreg], reg_obs=obs[day1][mreg],
        days=np.array([str(v) for v in OBS_DAYS]), **{f"s_{n}_{k}": a for n, S in series.items() for k, a in S.items()},
        box1=np.array(box[city1]), boxn=np.array(box[NEIGHBOUR]), box2=np.array(box[city2]),
        h_km=hx.astype(np.float16), h_ext=np.array(hext), h_rt04=h04.astype(np.float32), h_cams=hcf.astype(np.float32), h_pop=hpop,
        h_lon=lon[m2], h_lat=lat[m2], h_obs=obs[day2][m2], h_land=np.asarray(land[r0:r1, c0:c1]),
        reg_land=np.asarray(land[slice(*R.grid_window(REGION)[:2]), slice(*R.grid_window(REGION)[2:])]), centre1=np.array([c[1:] for c in C if c[0] == city1][0]),
        centren=np.array([c[1:] for c in C if c[0] == NEIGHBOUR][0]), centre2=np.array([c[1:] for c in C if c[0] == city2][0]))

    # ---- numbers a legend may quote, each with its source
    src = dict(
        monitors=f"{R.VERIF}/obs/<UTC day>.npz (UTC daily means, at least 18 valid hours; OpenAQ public archive) and {R.VERIF}/catalogue.npz",
        km_at_monitors=f"{R.VERIF}/fc/<issue day>.npz (mean, p10, p90 of the 8 members at the monitor pixel, tag {R.TAG})",
        km_fields=f"{R.KM}/<issue day>/{R.TAG}/lead<L>_part<r>.npz (product 'mean')",
        rt04=f"{R.RT}/out/<issue day>/pred.npy (8 members) with {R.RT}/data/<issue day>/cams.npy [1, 1] as the 0 h instant",
        cams=f"{R.C3H}/raw/*.nc (operational forecast of the same 00 UTC cycle, 3-hourly steps 0 .. 120 h): daily mean of the eight "
             "steps 00, 03, ..., 21 UTC of the valid day (key cams_daily_mean)",
        daily_means="CAMS forecast: mean of its eight 3-hourly steps 00 .. 21 UTC of the valid day, floor 0 (the rule of the paper's Methods; "
                    "three instants until 2026-11-19, see numbers_3instants.json). Real-time 0.4 degree forecast: unchanged, 1/4, 1/2, 1/4 of "
                    "the instants at 00, 12 and 24 UTC, because its members exist every 12 h only. 1 km forecast: unchanged, daily fields.",
        residents=f"{R.G}/pop1km_global.npy (GHS-POP 2020 on the 1 km grid of the paper)",
        cities=f"{R.RT}/products.py, CITIES (60 cities)", rules=f"{OUT}/rule_measured{SUF}.csv, {OUT}/rule_hidden{SUF}.csv")
    mon1 = inbox[city1] & np.isfinite(obs[day1]); monn = inbox[NEIGHBOUR] & np.isfinite(obs[day1])
    reg_at = at_monitors(mreg, day1, snap, rt, cf, lat, lon)
    sd = lambda f, o: dict(rmse=float(np.sqrt(np.mean((f - o) ** 2))), bias=float(np.mean(f - o)), r=float(np.corrcoef(o, f)[0, 1]))
    o_r = obs[day1][mreg]
    reg_scores = {}
    for L in (1, 2, 3, 4, 5):                                                    # every system at the monitors of the window, by lead day
        d = day1 - dt.timedelta(L - 1)
        reg_scores[f"km_lead{L}"] = sd(snap[d][0][L - 1][mreg], o_r)
        reg_scores[f"rt04_lead{L}"] = sd(R.bilinear(rt[d, L], lat[mreg], lon[mreg]), o_r)
        reg_scores[f"cams_lead{L}"] = sd(R.bilinear(cf[d, L], lat[mreg], lon[mreg]), o_r)
    # what the monitors of the window show on the case day: lead day 1, the monitors of the two city boxes against the others,
    # the correlation between monitors, and how far the one-day RMSE differences hold when the monitors are resampled
    # (singly, and by 0.4 degree cell: monitors of one cell are not independent)
    F1 = dict(km=snap[day1][0][0][mreg].astype(np.float64), rt04=R.bilinear(rt[day1, 1], lat[mreg], lon[mreg]),
              cams=R.bilinear(cf[day1, 1], lat[mreg], lon[mreg]))
    incity = (inbox[city1] | inbox[NEIGHBOUR])[mreg]
    part = lambda q: dict(monitors=int(q.sum()), measured=round(float(o_r[q].mean()), 2),
                          **{k: dict(mean=round(float(f[q].mean()), 2), bias=round(float((f[q] - o_r[q]).mean()), 2),
                                     rmse=round(float(np.sqrt(np.mean((f[q] - o_r[q]) ** 2))), 2)) for k, f in F1.items()})
    cell = np.round((84.0 - lat[mreg]) / 0.4).astype(np.int64) * 1000 + np.round((lon[mreg] + 180.0) / 0.4).astype(np.int64)
    rng = np.random.default_rng(0)

    def boot(fa, fb, groups, B=10000):
        g, inv = np.unique(groups, return_inverse=True)
        sa = np.bincount(inv, (fa - o_r) ** 2, g.size); sb = np.bincount(inv, (fb - o_r) ** 2, g.size)
        k = rng.integers(0, g.size, (B, g.size)); dd = 100 * (np.sqrt(sa[k].sum(1) / sb[k].sum(1)) - 1)
        return [round(float(100 * (np.sqrt(sa.sum() / sb.sum()) - 1)), 1), round(float(np.percentile(dd, 2.5)), 1), round(float(np.percentile(dd, 97.5)), 1)]

    reg_detail = dict(
        in_the_two_city_boxes=part(incity), outside_the_city_boxes=part(~incity),
        contrast_city_minus_outside=dict(measured=round(float(o_r[incity].mean() - o_r[~incity].mean()), 2),
                                         **{k: round(float(f[incity].mean() - f[~incity].mean()), 2) for k, f in F1.items()}),
        r_between_monitors={k: round(float(np.corrcoef(o_r, f)[0, 1]), 3) for k, f in F1.items()},
        share_of_monitors_closer_than_cams={k: round(float(np.mean(np.abs(F1[k] - o_r) < np.abs(F1["cams"] - o_r))), 3) for k in ("km", "rt04")},
        rmse_change_percent_95ci=dict(
            resampling_monitors=dict(km_vs_cams=boot(F1["km"], F1["cams"], np.arange(o_r.size)), km_vs_rt04=boot(F1["km"], F1["rt04"], np.arange(o_r.size))),
            resampling_0p4_degree_cells=dict(cells=int(np.unique(cell).size), km_vs_cams=boot(F1["km"], F1["cams"], cell),
                                             km_vs_rt04=boot(F1["km"], F1["rt04"], cell))))
    ser_tab ={n: {str(v): {k: (None if not np.isfinite(S[k][t]) else round(float(S[k][t]), 2)) for k in S} for t, v in enumerate(OBS_DAYS)}
               for n, S in series.items()}
    with io.StringIO(newline="") as f:                                           # source data of the two series panels
        keys = list(next(iter(series.values())))
        w_ = csv.writer(f); w_.writerow(["city", "valid_day"] + keys)
        for n, S in series.items():
            for t, v in enumerate(OBS_DAYS):
                w_.writerow([n, str(v)] + ["" if not np.isfinite(S[k][t]) else f"{S[k][t]:.3f}" for k in keys])
        rt_safe_io.write_text(OUT / f"series{SUF}.csv", f.getvalue())
    ser_sum = {}
    for n, S in series.items():                                                  # period means on the days that have both values
        a1 = np.isfinite(S["obs"]) & np.isfinite(S["km1"]); a3 = np.isfinite(S["obs"]) & np.isfinite(S["km3"])
        ins, tot = 0, 0
        for v in OBS_DAYS:
            m = inbox[n] & np.isfinite(obs[v])
            if v in snap and m.sum() >= NMIN:
                ins += int(((obs[v][m] >= snap[v][1][0][m]) & (obs[v][m] <= snap[v][2][0][m])).sum()); tot += int(m.sum())
        ser_sum[n] = dict(days_lead1=[str(OBS_DAYS[t]) for t in np.where(a1)[0]][::max(1, a1.sum() - 1)], n_days_lead1=int(a1.sum()),
                          measured_mean=round(float(S["obs"][a1].mean()), 2), km1_mean=round(float(S["km1"][a1].mean()), 2),
                          rt1_mean=round(float(S["rt1"][a1].mean()), 2), cams1_mean=round(float(S["cams1"][a1].mean()), 2),
                          rmse_of_daily_means=dict(km1=round(float(np.sqrt(np.mean((S["km1"][a1] - S["obs"][a1]) ** 2))), 2),
                                                   rt1=round(float(np.sqrt(np.mean((S["rt1"][a1] - S["obs"][a1]) ** 2))), 2),
                                                   cams1=round(float(np.sqrt(np.mean((S["cams1"][a1] - S["obs"][a1]) ** 2))), 2)),
                          days_below_measured_km1=int((S["km1"][a1] < S["obs"][a1]).sum()),
                          n_days_lead3=int(a3.sum()), measured_mean_lead3_days=round(float(S["obs"][a3].mean()), 2), km3_mean=round(float(S["km3"][a3].mean()), 2),
                          rmse_of_daily_means_km3=round(float(np.sqrt(np.mean((S["km3"][a3] - S["obs"][a3]) ** 2))), 2),
                          monitor_days_lead1=tot, share_inside_p10_p90_lead1=round(ins / max(tot, 1), 3),
                          monitors_per_day=[int(S["n"].min()), int(S["n"].max())],
                          std_of_daily_means=dict(measured=round(float(S["obs"][a1].std()), 1), km1=round(float(S["km1"][a1].std()), 1)))
    o2 = obs[day2][m2]; k2 = snap[day2][0][0][m2]
    c2days = {}                                                                  # the city of case 2 on every lead-day-1 day: how the day shown compares
    for r in sorted((r for r in rows2 if r["city"] == city2), key=lambda r: r["day"]):
        d = dt.date.fromisoformat(r["day"]); m = inbox[city2] & np.isfinite(obs[d]); a = at_monitors(m, d, snap, rt, cf, lat, lon)
        c2days[r["day"]] = dict(monitors=int(m.sum()), measured=round(float(obs[d][m].mean()), 2), monitors_above=int((obs[d][m] > R.THR).sum()),
                                km1=round(a["km1"], 2), rt1=round(a["rt1"], 2), cams1=round(a["cams1"], 2), hidden_million=round(r["hidden"] / 1e6, 3),
                                computed_on_issue_day=bool(on_day[d]))
    c2mean = {k: round(float(np.mean([v[k] for v in c2days.values()])), 2) for k in ("measured", "km1", "rt1", "cams1")}
    num = dict(
        variant=VARIANT, written=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), sources=src, cams_daily_mean=cams_info,
        rule=dict(cities=60, min_monitors=NMIN, days_with_lead_1_3_5=[str(v) for v in has135], rule_days=[str(v) for v in days],
                  km_files_written_utc={str(d): made[d].isoformat(timespec="minutes") for d in R.ISSUES},
                  rt04_members_computed_utc={str(d): made04[d].isoformat(timespec="minutes") for d in R.ISSUES},
                  rt04_computed_on_issue_day={str(d): bool(made04[d].date() == d) for d in R.ISSUES},
                  lead1_computed_on_issue_day={str(d): bool(on_day[d]) for d in R.ISSUES},
                  candidates_measured=len(cand1), candidate_cities=len({r["city"] for r in cand1}),
                  obs_files_final={str(v): bool(final[v]) for v in OBS_DAYS},
                  obs_files_mtime_utc={str(v): utc_day(R.VERIF / "obs" / f"{v}.npz").isoformat(timespec="minutes") for v in OBS_DAYS}),
        case1=dict(city=city1, day=str(day1), top5=[{k: (round(r[k], 2) if isinstance(r[k], float) else r[k]) for k in ("city", "day", "monitors", "measured")} for r in cand1[:5]],
                   at_city_monitors={k: round(v, 2) for k, v in at_monitors(mon1, day1, snap, rt, cf, lat, lon).items()},
                   measured=round(float(obs[day1][mon1].mean()), 2), monitors=int(mon1.sum()),
                   measured_min_max=[round(float(obs[day1][mon1].min()), 1), round(float(obs[day1][mon1].max()), 1)],
                   share_measured_inside_p10_p90=round(float(np.mean((obs[day1][mon1] >= snap[day1][1][0][mon1]) & (obs[day1][mon1] <= snap[day1][2][0][mon1]))), 3),
                   neighbour=dict(city=NEIGHBOUR, monitors=int(monn.sum()), measured=round(float(obs[day1][monn].mean()), 2),
                                  **{k: round(v, 2) for k, v in at_monitors(monn, day1, snap, rt, cf, lat, lon).items()}),
                   region=dict(extent_W_E_S_N=list(REGION), monitors=int(mreg.sum()), measured=round(float(o_r.mean()), 2),
                               at_monitors={k: round(v, 2) for k, v in reg_at.items()},
                               scores_at_monitors={k: {a: round(b, 4) for a, b in v.items()} for k, v in reg_scores.items()},
                               lead_day_1_detail=reg_detail,
                               km_field_max=round(float(np.nanmax(km[1][0])), 1), rt04_cell_max=round(float(rt[day1, 1][si, sj].max()), 1),
                               cams_cell_max=round(float(cf[day1, 1][si, sj].max()), 1)),
                   world=dict(issue=str(day1), lead_day=1, land_mean_every_5th_pixel=round(float(np.nanmean(wld)), 2), max=round(float(np.nanmax(wld)), 1),
                              pixels_1km=147289344)),
        series=ser_tab, series_summary=ser_sum,
        case2=dict(city=city2, day=str(day2), box_S_N_W_E=[round(float(b), 4) for b in box[city2]],
                   measured_min_max=[round(float(o2.min()), 1), round(float(o2.max()), 1)],
                   km_at_monitors_min_max=[round(float(k2.min()), 1), round(float(k2.max()), 1)],
                   r_between_monitors_km=round(float(np.corrcoef(o2, k2)[0, 1]), 2),
                   rt04_monitors_above=int((R.bilinear(rt[day2, 1], lat[m2], lon[m2]) > R.THR).sum()),
                   cams_monitors_above=int((R.bilinear(cf[day2, 1], lat[m2], lon[m2]) > R.THR).sum()),
                   km_files_written_utc=made[day2].isoformat(timespec="minutes"),
                   city_on_every_lead1_day=c2days, city_mean_over_those_days=c2mean,
                   days_with_measured_mean_above=int(sum(v["measured"] > R.THR for v in c2days.values())),
                   top5=[{k: (round(r[k] / 1e6, 3) if k in ("hidden", "above", "residents") else round(r[k], 2) if isinstance(r[k], float) else r[k])
                          for k in ("city", "day", "monitors", "measured", "residents", "above", "hidden")} for r in cand2[:5]],
                   best_day_of_each_city=[dict(city=r["city"], day=r["day"], hidden_million=round(r["hidden"] / 1e6, 3), above_million=round(r["above"] / 1e6, 3),
                                               monitors=r["monitors"], measured=round(r["measured"], 1)) for r in list(best.values())[:8]],
                   residents_million=round(float(hpop.sum()) / 1e6, 3), above_million=round(float((hpop * habove).sum()) / 1e6, 3),
                   hidden_million=round(float((hpop * (habove & (h04 <= R.THR))).sum()) / 1e6, 3),
                   seen_million=round(float((hpop * (habove & (h04 > R.THR))).sum()) / 1e6, 3),
                   hidden_if_cells_of_cams_forecast_million=round(float((hpop * (habove & (hcf <= R.THR))).sum()) / 1e6, 3),
                   hidden_if_cell_means_of_the_1km_forecast_million=round(float((hpop * (habove & (hown <= R.THR))).sum()) / 1e6, 3),
                   cells_max=dict(rt04=round(float(h04[hok].max()), 1), cams=round(float(hcf[hok].max()), 1), own_cell_mean=round(float(np.nanmax(hown[hok])), 1)),
                   km_field=dict(max=round(float(np.nanmax(hx)), 1), popw=round(float((hpop * np.nan_to_num(hx)).sum() / hpop.sum()), 2)),
                   popw_rt04=round(float((hpop * h04).sum() / hpop.sum()), 2), popw_cams=round(float((hpop * hcf).sum() / hpop.sum()), 2),
                   monitors=int(m2.sum()), measured=round(float(obs[day2][m2].mean()), 2), monitors_above=int((obs[day2][m2] > R.THR).sum()),
                   km_monitors_above=int((snap[day2][0][0][m2] > R.THR).sum()), at_monitors={k: round(v, 2) for k, v in a2.items()},
                   earlier_issues={str(L): {k: (round(v / 1e6, 3) if k in ("above", "hidden") else round(v, 2) if isinstance(v, float) else v) for k, v in h.items()}
                                   for L, h in hid.items()}))
    rt_safe_io.write_text(HERE / f"numbers{SUF}.json", json.dumps(num, indent=1))
    print(json.dumps({k: num[k] for k in ("case1", "case2")}, indent=1)[:6000])
    print("->", OUT / f"rt_day{SUF}.npz", HERE / f"numbers{SUF}.json")


if __name__ == "__main__":
    main()
