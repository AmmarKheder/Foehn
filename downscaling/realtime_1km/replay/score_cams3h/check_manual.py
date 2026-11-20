#!/usr/bin/env python3
"""cams3h, self-check before reporting (evaluation only). Independent of cams3h/code/build_arms.py and of the scorer code:

A. BY HAND FROM THE RAW FILES. For three issue days of different periods and two lead days each, at ten monitors: the
   daily mean of the CAMS forecast by both rules, recomputed with scalar arithmetic from the raw netCDF values (kg m-3)
   of the four grid nodes around each monitor, on the file's own grid (longitudes 0 .. 359.6, no roll):
       row = (90 - lat) / 0.4, column = (lon mod 360) / 0.4, bilinear weights from the fractional parts;
       paper's rule = mean of the eight steps 24(L-1) + 0, 3, ..., 21 h;  three instants = 1/4, 1/2, 1/4 at 24(L-1),
       24L-12, 24L h.  Each step is interpolated first, then the steps are averaged (the builder averages first).
   Compared with the stored arms (cams3h/points: cams3h, cams3i, cams3h_f32), with the existing arm (replay/points:
   cams) and with the station-day table (score_cams3h/work/pairs.npz) when the monitor has an observation that day.
   Two versions of the hand value: "exact" (float64 of the raw value x 1e9) and "f16" (each node value rounded to
   float16 first, as the chain stores it and as the arms cams3h and cams3i use it).
B. THE WHOLE TABLE: every row of pairs.npz against the points files (arms cams = cams3h, cams_old, cams3i), by indexing.
C. POOLED RMSE WITHOUT SCORER CODE: plain numpy on pairs.npz (primary set = issue days from 2026-09-06, observation and
   km, rt04, cams, clim, bcsd, cams_old finite): RMSE of the 1 km forecast, of the CAMS forecast with three instants
   and with the 3-hourly mean, their ratio and the two changes; compared with the scorer's tables of score/ and of
   score_cams3h/.
    python -I -B check_manual.py   -> check_manual.json, check_manual.md (this directory)"""
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

R = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay")
P = R.parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(R / "cams3h/code"))
import safe_io  # noqa: E402

CASES = [("2026-09-12", (2, 5)), ("2026-10-24", (1, 4)), ("2026-11-08", (3, 5))]      # September reforecast, October reforecast, live
N_MON = 10
SEED = 20261119


def raw_file_of(day):
    import netCDF4
    for f in sorted((R / "cams3h/raw").glob("*.nc")):
        ds = netCDF4.Dataset(str(f))
        ds.set_auto_mask(False)
        secs = [int(s) for s in ds.variables["forecast_reference_time"][:]]
        dates = [str(dt.date(1970, 1, 1) + dt.timedelta(days=s // 86400)) for s in secs]
        if day in dates:
            return ds, dates.index(day), f.name
        ds.close()
    raise FileNotFoundError(day)


def by_hand(ds, i, L, la, lo):
    """scalar recomputation at one monitor: (paper exact, paper f16, three exact, three f16), ug m-3"""
    v = ds.variables["pm2p5"]
    hours = [float(h) for h in ds.variables["forecast_period"][:]]
    r = (90.0 - la) / 0.4
    c = (lo % 360.0) / 0.4
    i0, j0 = int(math.floor(r)), int(math.floor(c)) % 900
    j1 = (j0 + 1) % 900
    fr, fc = r - math.floor(r), c - math.floor(c)

    def node(h, ii, jj, f16):
        k = hours.index(float(h))
        x = np.float32(v[k, i, ii, jj]) * np.float32(1e9)                    # the chain's arithmetic, float32
        return float(np.float16(x)) if f16 else float(np.float32(v[k, i, ii, jj])) * 1e9

    def interp(h, f16):
        return ((1 - fr) * (1 - fc) * node(h, i0, j0, f16) + (1 - fr) * fc * node(h, i0, j1, f16)
                + fr * (1 - fc) * node(h, i0 + 1, j0, f16) + fr * fc * node(h, i0 + 1, j1, f16))
    h0 = 24 * (L - 1)
    out = []
    for f16 in (False, True):
        paper = sum(interp(h0 + 3 * k, f16) for k in range(8)) / 8.0
        three = 0.25 * interp(h0, f16) + 0.5 * interp(h0 + 12, f16) + 0.25 * interp(h0 + 24, f16)
        out += [max(paper, 0.0), max(three, 0.0)]
    return out[0], out[2], out[1], out[3]                                    # paper exact, paper f16, three exact, three f16


def main():
    cat = np.load(P / "realtime_1km/verif/catalogue.npz")
    ids, lat, lon = cat["id"].astype(str), cat["lat"].astype(np.float64), cat["lon"].astype(np.float64)
    rng = np.random.default_rng(SEED)
    mons = np.sort(rng.choice(ids.size, N_MON, replace=False))
    z = np.load(HERE / "work" / "pairs.npz")
    issue_days = [str(x) for x in z["issue_days"]]
    issue, lead, mon, obs = z["issue"].astype(np.int64), z["lead"].astype(np.int64), z["mon"].astype(np.int64), z["obs"].astype(np.float64)
    arms = [str(x) for x in z["arms"]]
    assert np.array_equal(z["id"].astype(str), ids)
    out = dict(cases=[], monitors=[dict(index=int(k), id=str(ids[k]), lat=round(float(lat[k]), 4), lon=round(float(lon[k]), 4), region=str(cat["region"][k])) for k in mons])
    md = ["# cams3h: checks by hand\n", "## A. Daily means recomputed by hand from the raw files (ug m-3)\n",
          f"Ten monitors drawn once (seed {SEED}); three issue days of different periods, two lead days each. hand = scalar arithmetic on the raw netCDF values "
          "of the four grid nodes; 'f16' = each node value rounded to float16 first (what the chain stores and the arms cams3h / cams3i use).\n"]
    worst = dict(paper_f16_vs_cams3h=0.0, paper_exact_vs_cams3h_f32=0.0, paper_exact_vs_cams3h=0.0, three_f16_vs_cams3i=0.0, three_f16_vs_existing_cams=0.0,
                 pairs_cams_vs_points_cams3h=0.0, pairs_cams_old_vs_existing=0.0)
    n_pairs_rows = 0
    for day, leads in CASES:
        ds, i, fname = raw_file_of(day)
        new = np.load(R / "cams3h/points" / f"{day}.npz")
        old = np.load(R / "points" / f"{day}.npz")
        assert np.array_equal(new["id"].astype(str), ids) and np.array_equal(old["id"].astype(str), ids)
        ii = issue_days.index(day)
        for L in leads:
            valid = str(dt.date.fromisoformat(day) + dt.timedelta(L - 1))
            rows = []
            for k in mons:
                pe, pf, te, tf = by_hand(ds, i, L, float(lat[k]), float(lon[k]))
                s3h, s3i, s32 = float(new["cams3h"][L - 1, k]), float(new["cams3i"][L - 1, k]), float(new["cams3h_f32"][L - 1, k])
                ex = float(old["cams"][L - 1, k])
                q = np.flatnonzero((issue == ii) & (lead == L) & (mon == k))
                pc = float(z["m_cams"][q[0]]) if q.size else None
                po = float(z["m_cams_old"][q[0]]) if q.size else None
                n_pairs_rows += int(q.size)
                worst["paper_f16_vs_cams3h"] = max(worst["paper_f16_vs_cams3h"], abs(pf - s3h))
                worst["paper_exact_vs_cams3h_f32"] = max(worst["paper_exact_vs_cams3h_f32"], abs(pe - s32))
                worst["paper_exact_vs_cams3h"] = max(worst["paper_exact_vs_cams3h"], abs(pe - s3h))
                worst["three_f16_vs_cams3i"] = max(worst["three_f16_vs_cams3i"], abs(tf - s3i))
                worst["three_f16_vs_existing_cams"] = max(worst["three_f16_vs_existing_cams"], abs(tf - ex))
                if pc is not None:
                    worst["pairs_cams_vs_points_cams3h"] = max(worst["pairs_cams_vs_points_cams3h"], abs(pc - s3h))
                    worst["pairs_cams_old_vs_existing"] = max(worst["pairs_cams_old_vs_existing"], abs(po - ex))
                rows.append(dict(monitor=str(ids[k]), hand_paper_exact=pe, hand_paper_f16=pf, stored_cams3h=s3h, stored_cams3h_f32=s32, hand_three_exact=te,
                                 hand_three_f16=tf, stored_cams3i=s3i, existing_cams=ex, pairs_cams=pc, pairs_cams_old=po,
                                 observation=float(obs[q[0]]) if q.size else None))
            out["cases"].append(dict(issue_day=day, lead_day=L, valid_day=valid, raw_file=fname, rows=rows))
            md.append(f"\nIssue day {day}, lead day {L} (valid {valid}), file `raw/{fname}`:\n")
            md.append("| monitor | hand, 8 steps (exact) | hand, 8 steps (f16) | stored cams3h | hand, 3 instants (f16) | stored cams3i | existing arm cams | in pairs.npz: cams / cams_old | observation |")
            md.append("|---|---|---|---|---|---|---|---|---|")
            for r_ in rows:
                pp = f"{r_['pairs_cams']:.4f} / {r_['pairs_cams_old']:.4f}" if r_["pairs_cams"] is not None else "no observation"
                ob = f"{r_['observation']:.2f}" if r_["observation"] is not None else ""
                md.append(f"| {r_['monitor']} | {r_['hand_paper_exact']:.4f} | {r_['hand_paper_f16']:.4f} | {r_['stored_cams3h']:.4f} | {r_['hand_three_f16']:.4f} | "
                          f"{r_['stored_cams3i']:.4f} | {r_['existing_cams']:.4f} | {pp} | {ob} |")
        ds.close()
    out["A_largest_abs_differences"] = worst
    out["A_rows_found_in_pairs"] = n_pairs_rows
    md.append("\nLargest absolute differences over the 60 values (ug m-3):\n")
    md.append("| comparison | largest difference |\n|---|---|")
    names = dict(paper_f16_vs_cams3h="hand (8 steps, f16) against the stored arm cams3h", paper_exact_vs_cams3h_f32="hand (8 steps, exact) against the stored cams3h_f32",
                 paper_exact_vs_cams3h="hand (8 steps, exact) against the stored cams3h (float16 storage of the steps)",
                 three_f16_vs_cams3i="hand (3 instants, f16) against the stored cams3i", three_f16_vs_existing_cams="hand (3 instants, f16) against the EXISTING arm cams",
                 pairs_cams_vs_points_cams3h="pairs.npz arm cams against the stored cams3h", pairs_cams_old_vs_existing="pairs.npz arm cams_old against the existing arm cams")
    for k_, v_ in worst.items():
        md.append(f"| {names[k_]} | {v_:.3g} |")

    # ---- B. every row of the table against the points files
    B = dict(rows=int(obs.size))
    for arm, (pdir, key) in dict(cams=(R / "cams3h/points", "cams3h"), cams3h=(R / "cams3h/points", "cams3h"), cams3i=(R / "cams3h/points", "cams3i"),
                                 cams_old=(R / "points", "cams")).items():
        if arm not in arms:
            continue
        x = z[f"m_{arm}"]
        mx, same, n = 0.0, 0, 0
        for ii, day in enumerate(issue_days):
            q = np.flatnonzero(issue == ii)
            if q.size == 0:
                continue
            a = np.maximum(np.load(pdir / f"{day}.npz")[key], 0.0)[lead[q] - 1, mon[q]]
            mx = max(mx, float(np.abs(a.astype(np.float64) - x[q].astype(np.float64)).max()))
            same += int((a == x[q]).sum())
            n += int(q.size)
        B[arm] = dict(source=f"{pdir}/<d>.npz[{key}]", rows=n, identical=same, max_abs_diff=mx)
    B["cams_old_identical_to_cams3i"] = bool(np.array_equal(z["m_cams_old"], z["m_cams3i"]))
    B["cams_identical_to_cams3h"] = bool(np.array_equal(z["m_cams"], z["m_cams3h"]))
    out["B_table_against_points"] = B
    md.append("\n## B. Every row of the station-day table against the points files\n")
    md.append("| arm of pairs.npz | source | rows | identical | largest difference |\n|---|---|---|---|---|")
    for arm in ("cams", "cams3h", "cams3i", "cams_old"):
        if arm in B:
            md.append(f"| {arm} | `{B[arm]['source']}` | {B[arm]['rows']:,} | {B[arm]['identical']:,} | {B[arm]['max_abs_diff']:.3g} |")
    md.append(f"\n`cams_old` identical to `cams3i` on every row: {B['cams_old_identical_to_cams3i']}; `cams` identical to `cams3h`: {B['cams_identical_to_cams3h']}.\n")

    # ---- C. pooled RMSE with plain numpy
    d0 = issue_days.index("2026-09-06")
    need = ["km", "rt04", "cams", "clim", "bcsd", "cams_old"]
    ok = np.isfinite(obs) & (issue >= d0)
    for a in need:
        ok &= np.isfinite(z[f"m_{a}"])
    n = int(ok.sum())
    rm = {a: float(np.sqrt(np.mean((z[f"m_{a}"][ok].astype(np.float64) - obs[ok]) ** 2))) for a in ["km", "rt04", "cams", "cams_old", "clim", "bcsd"]}
    Cc = dict(station_days=n, issue_days=int(np.unique(issue[ok]).size), monitors=int(np.unique(mon[ok]).size), rmse=rm,
              km_vs_cams3h_pct=100 * (rm["km"] / rm["cams"] - 1), km_vs_cams_three_instants_pct=100 * (rm["km"] / rm["cams_old"] - 1),
              ratio_three_instants_over_3hourly=rm["cams_old"] / rm["cams"], rt04_vs_cams3h_pct=100 * (rm["rt04"] / rm["cams"] - 1),
              rt04_vs_cams_three_instants_pct=100 * (rm["rt04"] / rm["cams_old"] - 1))
    # the same set without requiring cams_old (the scorer's own mask)
    ok2 = np.isfinite(obs) & (issue >= d0)
    for a in ["km", "rt04", "cams", "clim", "bcsd"]:
        ok2 &= np.isfinite(z[f"m_{a}"])
    Cc["station_days_scorer_mask"] = int(ok2.sum())
    Cc["lost_by_requiring_both_cams_arms"] = int(ok2.sum() - ok.sum())
    tabs = {}
    for name, sdir in (("score (three instants)", R / "score"), ("score_cams3h (3-hourly mean)", HERE)):
        f = sdir / "tables" / "T02_scores.json"
        if f.exists():
            t = json.loads(f.read_text())["with the CAMS forecast"]
            t3 = json.loads((sdir / "tables" / "T03_contrasts.json").read_text())["with the CAMS forecast"]["contrasts"]["km_vs_cams|rmse"]["all"]["all"]
            tabs[name] = dict(rmse_km=t["km"]["all|all"]["rmse"], rmse_cams=t["cams"]["all|all"]["rmse"], n=t["km"]["all|all"]["n"], km_vs_cams_pct=t3["est"], ci=t3["ci"])
    Cc["scorer_tables"] = tabs
    out["C_pooled_rmse_plain_numpy"] = Cc
    md.append("## C. Pooled RMSE recomputed with plain numpy on pairs.npz (no scorer code)\n")
    md.append(f"Primary set: {n:,} station-days, {Cc['issue_days']} issue days, {Cc['monitors']:,} monitors (the scorer's own mask gives {Cc['station_days_scorer_mask']:,}; "
              f"lost by requiring both CAMS arms: {Cc['lost_by_requiring_both_cams_arms']}).\n")
    md.append("| quantity | plain numpy | scorer table |\n|---|---|---|")
    s_old, s_new = tabs.get("score (three instants)", {}), tabs.get("score_cams3h (3-hourly mean)", {})
    md.append(f"| RMSE 1 km forecast | {rm['km']:.4f} | {s_new.get('rmse_km', float('nan')):.4f} (score_cams3h), {s_old.get('rmse_km', float('nan')):.4f} (score) |")
    md.append(f"| RMSE CAMS, three instants | {rm['cams_old']:.4f} | {s_old.get('rmse_cams', float('nan')):.4f} (score) |")
    md.append(f"| RMSE CAMS, 3-hourly mean | {rm['cams']:.4f} | {s_new.get('rmse_cams', float('nan')):.4f} (score_cams3h) |")
    md.append(f"| ratio three instants / 3-hourly | {Cc['ratio_three_instants_over_3hourly']:.4f} | |")
    md.append(f"| 1 km vs CAMS three instants, % | {Cc['km_vs_cams_three_instants_pct']:+.3f} | {s_old.get('km_vs_cams_pct', float('nan')):+.3f} (score) |")
    md.append(f"| 1 km vs CAMS 3-hourly mean, % | {Cc['km_vs_cams3h_pct']:+.3f} | {s_new.get('km_vs_cams_pct', float('nan')):+.3f} (score_cams3h) |")
    md.append(f"| 0.4 deg input vs CAMS three instants / 3-hourly, % | {Cc['rt04_vs_cams_three_instants_pct']:+.3f} / {Cc['rt04_vs_cams3h_pct']:+.3f} | |")
    safe_io.write_json(HERE / "check_manual.json", out)
    safe_io.write_text(HERE / "check_manual.md", "\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
