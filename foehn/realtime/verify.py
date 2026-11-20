#!/usr/bin/env python3
"""RT daily verification (public): the forecasts issued in the last five days are scored against the CAMS analyses
that have arrived since, taken from the day directories (data/<d>/cams.npy holds the analyses of 00 UTC of d and
12 UTC of d-1: no extra download). For each (issue day, lead) with a known valid time: RMSE of the ensemble mean over
all cells and over land, land bias, the same for the CAMS operational forecast (data/<issue>/cams_fc.npz, when its lead
was fetched) and for persistence (the 00 UTC analysis of the issue day), and the absolute error of the 60 published
cities. Rows accumulate in verif/scores_live.json (dedup by issue day and lead, 90 days kept) with a 30-day summary;
    python verify.py 2026-10-11            (today: the analyses up to 00 UTC of today are available)
    python verify.py 2026-09-21 --since 2026-09-02   (replay: every 'today' from --since to the date)"""
import argparse
import datetime as dt
import json

import numpy as np

import common as C
from products import CITIES, cell

VERIF = C.RT / "verif"; OUT = VERIF / "scores_live.json"
LAT = 84.0 - 0.4 * np.arange(361); W = np.cos(np.deg2rad(LAT))[:, None] * np.ones((1, 900)); W /= W.sum()


def truth(valid):
    """CAMS analysis PM2.5 at a valid time (00 or 12 UTC) from the day directories, or None"""
    if valid.hour == 0:
        f = C.DATA / valid.strftime("%Y-%m-%d") / "cams.npy"; i = 1
    else:
        f = C.DATA / (valid + dt.timedelta(days=1)).strftime("%Y-%m-%d") / "cams.npy"; i = 0
    return np.load(f, mmap_mode="r")[i, 1].astype(np.float32) if f.exists() else None


def scores(field, T, land):
    e = field - T
    return {"rmse_all": round(float(np.sqrt((W * e ** 2).sum())), 3), "rmse_land": round(float(np.sqrt((W * e ** 2)[land].sum() / W[land].sum())), 3),
            "bias_land": round(float((W * e)[land].sum() / W[land].sum()), 3)}


def city_vals(field):
    out = {}
    for name, la, lo in CITIES:
        i, j = cell(la, lo); out[name] = float(field[max(i - 1, 0):min(i + 2, 361), j - 1:j + 2].mean())
    return out


def one_day(today, rows, land):
    added = 0
    for back in range(1, 6):
        issue = today - dt.timedelta(days=back); O = C.OUT / issue.strftime("%Y-%m-%d"); D = C.DATA / issue.strftime("%Y-%m-%d")
        if not (O / "pred.npy").exists():
            continue
        pred = None; cf = None
        for k in range(1, 11):
            key = f"{issue}|{12 * k}"
            if key in rows:
                continue
            valid = dt.datetime(issue.year, issue.month, issue.day) + dt.timedelta(hours=12 * k)
            if valid > dt.datetime(today.year, today.month, today.day):
                break
            T = truth(valid)
            if T is None:
                continue
            if pred is None:
                pred = np.load(O / "pred.npy").astype(np.float32); mean = pred.mean(0)
                p0 = np.load(D / "cams.npy", mmap_mode="r")[1, 1].astype(np.float32)
                if (D / "cams_fc.npz").exists():
                    z = np.load(D / "cams_fc.npz"); cf = (list(z["leads"]), z["pm25"])
            ck = json.loads((O / "meta.json").read_text()).get("ckpt", "") if (O / "meta.json").exists() else ""
            r = {"issue": str(issue), "lead_h": 12 * k, "valid": valid.strftime("%Y-%m-%dT%H:00Z"), "hindcast": issue < dt.date(2026, 10, 10),
                 "model": "1.2" if "rt12" in ck else "1.1",
                 "rt": scores(mean[k - 1], T, land), "persistence": scores(p0, T, land), "cams": None, "members": int(pred.shape[0])}
            tc = city_vals(T); oc = city_vals(mean[k - 1]); r["cities"] = {n: {"truth": round(tc[n], 1), "rt": round(oc[n], 1)} for n in tc}
            if cf and 12 * k in cf[0]:
                a = np.roll(cf[1][cf[0].index(12 * k)].astype(np.float32)[15:376], 450, axis=-1)
                r["cams"] = scores(a, T, land); cc = city_vals(a)
                for n in tc:
                    r["cities"][n]["cams"] = round(cc[n], 1)
            rows[key] = r; added += 1
    return added


def summary(rows, today, days=30):
    first = today - dt.timedelta(days=days)
    out = {"window_days": days, "from": str(first), "to": str(today), "leads": {}}
    for L in range(12, 121, 12):
        sel = [r for r in rows.values() if r["lead_h"] == L and C.parse_date(r["issue"]) >= first]
        if not sel:
            continue
        g = lambda key, sub: round(float(np.mean([r[key][sub] for r in sel if r.get(key)])), 2) if any(r.get(key) for r in sel) else None
        cm = [abs(r["cities"][n]["rt"] - r["cities"][n]["truth"]) for r in sel for n in r["cities"]]
        cc = [abs(r["cities"][n]["cams"] - r["cities"][n]["truth"]) for r in sel for n in r["cities"] if "cams" in r["cities"][n]]
        out["leads"][str(L)] = {"n": len(sel), "rt": {s: g("rt", s) for s in ("rmse_all", "rmse_land", "bias_land")},
                                "persistence": {s: g("persistence", s) for s in ("rmse_all", "rmse_land", "bias_land")},
                                "cams": {s: g("cams", s) for s in ("rmse_all", "rmse_land", "bias_land")} if any(r.get("cams") for r in sel) else None,
                                "n_cams": sum(1 for r in sel if r.get("cams")),
                                "city_mae_rt": round(float(np.mean(cm)), 2) if cm else None, "city_mae_cams": round(float(np.mean(cc)), 2) if cc else None}
    # per city over the window at 24 h: typical absolute error of RT (shown in the place card)
    sel = [r for r in rows.values() if r["lead_h"] == 24 and C.parse_date(r["issue"]) >= first]
    out["cities_24h"] = {n: {"mae_rt": round(float(np.mean([abs(r["cities"][n]["rt"] - r["cities"][n]["truth"]) for r in sel])), 1),
                             "mean_truth": round(float(np.mean([r["cities"][n]["truth"] for r in sel])), 1), "n": len(sel)} for n in (sel[0]["cities"] if sel else {})}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("date"); ap.add_argument("--since", default=None); ap.add_argument("--keep", type=int, default=90)
    a = ap.parse_args(); today = C.parse_date(a.date)
    land = np.load(C.G / "cams_static.npz")["static"][0] > 0.5
    VERIF.mkdir(exist_ok=True)
    db = json.loads(OUT.read_text()) if OUT.exists() else {"rows": {}}
    rows = db["rows"]
    days = [today] if not a.since else [C.parse_date(a.since) + dt.timedelta(days=i) for i in range((today - C.parse_date(a.since)).days + 1)]
    added = sum(one_day(d, rows, land) for d in days)
    cut = today - dt.timedelta(days=a.keep)
    rows = {k: r for k, r in rows.items() if C.parse_date(r["issue"]) >= cut}
    db = {"updated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"), "truth": "CAMS analysis at the valid time", "unit": "ug m-3",
          "statistic": "RMSE of the ensemble mean (cosine-latitude weighted), bias over land; cities: 3 x 3 cells around the place",
          "summary": summary(rows, today), "rows": rows}
    OUT.write_text(json.dumps(db, separators=(",", ":")))
    s = db["summary"]["leads"]
    print(f"verification {today}: {added} new (issue, lead) rows, {len(rows)} kept; 30-day summary:", flush=True)
    for L, v in s.items():
        print(f"  +{L:>3} h n={v['n']:2d}  RT land {v['rt']['rmse_land']:6.2f}  CAMS {v['cams']['rmse_land'] if v['cams'] else '   -  '}  persistence {v['persistence']['rmse_land']:6.2f}  | city MAE RT {v['city_mae_rt']} CAMS {v['city_mae_cams']}")


if __name__ == "__main__":
    main()
