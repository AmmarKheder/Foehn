#!/usr/bin/env python3
"""WP1a, instance B: the baseline 'cams' at the catalogue monitors, exactly as realtime_1km/rc/coupling_live.py (and
live_baselines.py) define it: for issue day d and lead day L = 1..5 (valid day d + L - 1), the daily mean of the CAMS
operational forecast of the 00 UTC cycle of d = max(1/4 x(24(L-1)) + 1/2 x(24L-12) + 1/4 x(24L), 0), where x(0) is the
CAMS analysis of t0 (realtime/data/<d>/cams.npy, rows 84 N .. 60 S placed at rows 15..375 of the 90 N .. 90 S grid, NaN
elsewhere) and x(h > 0) the forecast at + h hours; longitudes rolled to -180 .. 179.6; read bilinearly (linear space) at
the monitor (coupling_live.bilinear, imported). No observation is read here; nothing is scored.

The ten-lead forecast file of day d is realtime/data/<d>/cams_fc.npz when it has ten leads, otherwise
<replay>/camsfc/<d>/cams_fc.npz (decision A of the coordinator, fetched by camsfc_fetch.py); no file with ten leads ->
the day is skipped (its 'cams' stays NaN; no lead is interpolated).

  points      -> <replay>/camsfc/points/<d>.npz: cams (5, n) float32 in catalogue order, id, source (which file)
  check-live  compare with the 'cams' column of realtime_1km/rc/live_pairs.npz (issue days 2026-11-03..11-11; only the
              forecast columns and the indices of that file are read) -> <replay>/camsfc/check_live_pairs.json
  fill        fill 'cams' in <replay>/points/<d>.npz for the lead days where it is all-NaN and a value can be computed
              (atomic rewrite, every other array untouched) and report; compare where both exist
    python -I cams_points.py points | check-live | fill [--dry-run] [days ...]"""
import argparse
import datetime as dt
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

P = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
RT = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km/realtime")
R = P / "realtime_1km/replay"
CAT = P / "realtime_1km/verif/catalogue.npz"
sys.path.insert(0, str(P / "realtime_1km/rc"))
from coupling_live import bilinear  # noqa: E402  (the definition used for the live comparison; main() is guarded)

TARGET = [dt.date(2026, 9, 1) + dt.timedelta(k) for k in range(20)] + [dt.date(2026, 10, 10) + dt.timedelta(k) for k in range(33)]
LEADS = [12 * k for k in range(1, 11)]


def fc_file(d):
    """the ten-lead CAMS forecast file of issue day d, or None"""
    for p in (RT / "data" / str(d) / "cams_fc.npz", R / "camsfc" / str(d) / "cams_fc.npz"):
        if p.exists():
            try:
                if np.load(p)["leads"].tolist() == LEADS:
                    return p
            except Exception:  # noqa: BLE001
                pass
    return None


def cams_at(d, lat, lon):
    """(5, n) float64 and the file used; lines of coupling_live.main()"""
    cf = fc_file(d)
    if cf is None:
        return None, None
    an = np.load(RT / "data" / str(d) / "cams.npy")[:, 1].astype(np.float32)[1]
    c = np.load(cf)
    cl, co, cpm = c["lat"], c["lon"], c["pm25"].astype(np.float32)
    assert cpm.shape == (10, 451, 900) and abs(cl[0] - 90.0) < 1e-6 and abs((cl[0] - cl[1]) - 0.4) < 1e-6, (cpm.shape, cl[:3])
    if co[0] >= 0:
        k0 = int(np.argmin(np.abs(co - 180.0)))
        cpm = np.roll(cpm, -k0, axis=-1)
        co = np.remainder(np.roll(co, -k0) + 180.0, 360.0) - 180.0
    assert abs(co[0] + 180.0) < 1e-6, co[:3]
    an_full = np.full(cpm.shape[1:], np.nan, np.float32)
    an_full[15:15 + 361] = an
    out = np.full((5, lat.size), np.nan)
    for L in range(1, 6):
        a_, b_, c_ = 24 * (L - 1), 24 * L - 12, 24 * L
        F_cf = np.maximum(0.25 * (an_full if a_ == 0 else cpm[a_ // 12 - 1]) + 0.5 * cpm[b_ // 12 - 1] + 0.25 * cpm[c_ // 12 - 1], 0.0)
        out[L - 1] = bilinear(F_cf, 90.0, 0.4, -180.0, 0.4, lat, lon)
    return out, cf


def two_lead_check(old, new):
    """a fetched ten-lead file against the stored two-lead file of the same cycle (leads 24 and 120 h, float32, written by
    the September hindcast on 2026-10-10): the two shared leads must be the same data. None when not applicable."""
    old, new = Path(old), Path(new)
    if not (old.exists() and new.exists()):
        return None
    o, n_ = np.load(old), np.load(new)
    if o["leads"].tolist() != [24, 120] or n_["leads"].tolist() != LEADS:
        return None
    b = n_["pm25"][[LEADS.index(24), LEADS.index(120)]]
    a = o["pm25"].astype(b.dtype)
    df = np.abs(o["pm25"].astype(np.float64) - b.astype(np.float64))
    return dict(stored=str(old), fetched=str(new), stored_dtype=str(o["pm25"].dtype), fetched_dtype=str(b.dtype),
                identical_after_cast=bool(np.array_equal(a, b)), share_identical_after_cast=float((a == b).mean()),
                max_abs_diff=float(df.max()), mean_abs_diff=float(df.mean()),
                lat_lon_identical=bool(np.array_equal(o["lat"], n_["lat"]) and np.array_equal(o["lon"], n_["lon"])))


def catalogue():
    cat = np.load(CAT)
    return cat["id"], cat["lat"].astype(np.float64), cat["lon"].astype(np.float64)


def cmd_points(days):
    ids, lat, lon = catalogue()
    out = R / "camsfc" / "points"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in days:
        x, cf = cams_at(d, lat, lon)
        if x is None:
            rows.append(dict(day=str(d), file=None))
            print(f"{d}: no ten-lead file yet", flush=True)
            continue
        tmp = out / f"{d}.tmp.npz"
        np.savez(tmp, cams=x.astype(np.float32), id=ids, day=str(d), source=str(cf))
        tmp.rename(out / f"{d}.npz")
        rows.append(dict(day=str(d), file=str(cf), nan=int(np.isnan(x).sum()), mean_by_lead=[round(float(np.nanmean(v)), 3) for v in x]))
        if not str(cf).startswith(str(RT)):                # a fetched file: consistency with the stored two-lead file
            rows[-1]["two_lead_check"] = two_lead_check(RT / "data" / str(d) / "cams_fc.npz", cf)
            t = rows[-1]["two_lead_check"]
            if t is not None:
                print(f"   {d}: leads 24 and 120 h against the stored two-lead file: identical {t['identical_after_cast']}, "
                      f"max |diff| {t['max_abs_diff']:.3g} ug m-3", flush=True)
        print(f"{d}: cams {x.shape}, NaN {rows[-1]['nan']}, mean by lead day {rows[-1]['mean_by_lead']} ({'stored' if str(cf).startswith(str(RT)) else 'fetched'})", flush=True)
    (out / "index.json").write_text(json.dumps(rows, indent=1))
    chk = {r["day"]: r["two_lead_check"] for r in rows if r.get("two_lead_check")}
    if chk:
        (R / "camsfc" / "check_two_leads.json").write_text(json.dumps(dict(
            n_days=len(chk), all_identical_after_cast=all(v["identical_after_cast"] for v in chk.values()),
            max_abs_diff=max(v["max_abs_diff"] for v in chk.values()), days=chk), indent=1))
        print(f"two-lead check: {len(chk)} fetched days, all identical {all(v['identical_after_cast'] for v in chk.values())} "
              f"-> camsfc/check_two_leads.json", flush=True)
    have = [r["day"] for r in rows if r.get("file")]
    print(f"{len(have)} of {len(rows)} days have the ten leads; without: {[r['day'] for r in rows if not r.get('file')]}", flush=True)


def cmd_check_live():
    z = np.load(P / "realtime_1km/rc/live_pairs.npz")
    cols = z["cols"].tolist()
    A = z["pairs"]
    ids, lat, lon = catalogue()
    assert np.array_equal(z["id"], ids), "live_pairs.npz is not in catalogue order"
    L_, k_, cm, dd = (A[:, cols.index(c)] for c in ("lead", "monitor", "cams", "issue_doy0"))
    L_, k_, dd = L_.astype(int), k_.astype(int), dd.astype(int)
    res = dict(file=str(P / "realtime_1km/rc/live_pairs.npz"), columns_read=["lead", "monitor", "cams", "issue_doy0"], days={})
    worst = 0.0
    for doy0 in np.unique(dd):
        d = dt.date(2026, 1, 1) + dt.timedelta(int(doy0))
        x, cf = cams_at(d, lat, lon)
        if x is None:
            continue
        m = dd == doy0
        mine = x[L_[m] - 1, k_[m]].astype(np.float32)                     # the pairs are stored in float32
        diff = np.abs(mine.astype(np.float64) - cm[m].astype(np.float64))
        res["days"][str(d)] = dict(n=int(m.sum()), max_abs_diff=float(diff.max()), share_identical=float((mine == cm[m]).mean()))
        worst = max(worst, float(diff.max()))
        print(f"{d}: {int(m.sum())} pairs, max |diff| {diff.max():.3g}, identical {100 * (mine == cm[m]).mean():.2f} %", flush=True)
    res["max_abs_diff"] = worst
    (R / "camsfc" / "check_live_pairs.json").write_text(json.dumps(res, indent=1))
    print(f"max |diff| over {len(res['days'])} issue days: {worst:.3g} ug m-3 -> camsfc/check_live_pairs.json")


def cmd_fill(days, dry, pdir=None, report=None):
    ids, lat, lon = catalogue()
    pdir = Path(pdir) if pdir else R / "points"
    rep = []
    for d in days:
        f = pdir / f"{d}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=False)
        if "cams" not in z.files:
            rep.append(dict(day=str(d), action="no 'cams' key"))
            continue
        idk = "id" if "id" in z.files else "ids" if "ids" in z.files else None
        if idk is None or not np.array_equal(z[idk].astype(str), ids.astype(str)):
            rep.append(dict(day=str(d), action="ids differ from the catalogue: not touched"))
            continue
        old = z["cams"]
        x, cf = cams_at(d, lat, lon)
        if x is None:
            rep.append(dict(day=str(d), action="no ten-lead file", cams_all_nan=bool(np.isnan(old).all())))
            continue
        both = np.isfinite(old) & np.isfinite(x)
        row = dict(day=str(d), file=str(cf), old_nan=int(np.isnan(old).sum()), new_nan=int(np.isnan(x).sum()),
                   max_abs_diff_where_both=float(np.abs(old[both] - x[both].astype(old.dtype)).max()) if both.any() else None)
        need = [L for L in range(5) if np.isnan(old[L]).all() and np.isfinite(x[L]).any()]
        row["leads_filled"] = [L + 1 for L in need]
        if need and not dry:
            arrs = {k: z[k] for k in z.files}
            new = old.copy()
            for L in need:
                new[L] = x[L].astype(old.dtype)
            arrs["cams"] = new
            if "provenance" in arrs:                       # assemble.py: json string with a 'baselines' block
                try:
                    prov = json.loads(str(arrs["provenance"]))
                    b = prov.setdefault("baselines", {})
                    b.update(cams_forecast=str(cf), cams_missing_reason=None, cams_filled_by="code/cams_points.py fill",
                             cams_filled_leads=[L + 1 for L in need],
                             cams_filled_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
                    arrs["provenance"] = np.array(json.dumps(prov))
                except Exception as e:  # noqa: BLE001  (unknown layout: the array is filled, the json left as it is)
                    row["provenance_not_updated"] = repr(e)[:200]
            with zipfile.ZipFile(f) as zf:
                comp = any(i.compress_type != zipfile.ZIP_STORED for i in zf.infolist())
            tmp = f.with_name(f.stem + ".fill.tmp.npz")
            (np.savez_compressed if comp else np.savez)(tmp, **arrs)
            os.replace(tmp, f)
            row["action"] = "filled"
        else:
            row["action"] = "nothing to fill" if not need else "dry run"
        rep.append(row)
        print(row, flush=True)
    out = Path(report) if report else R / "camsfc" / ("fill_report_dry.json" if dry else "fill_report.json")
    out.write_text(json.dumps(rep, indent=1))
    print(f"{sum(r.get('action') == 'filled' for r in rep)} files filled, {len(rep)} examined -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["points", "check-live", "fill"])
    ap.add_argument("days", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--points-dir", default=None, help="fill: another directory of points files (tests)")
    ap.add_argument("--report", default=None, help="fill: another report file (tests)")
    a = ap.parse_args()
    days = [dt.date.fromisoformat(x) for x in a.days] or TARGET
    if a.cmd == "points":
        cmd_points(days)
    elif a.cmd == "check-live":
        cmd_check_live()
    else:
        cmd_fill(days, a.dry_run, a.points_dir, a.report)


if __name__ == "__main__":
    main()
