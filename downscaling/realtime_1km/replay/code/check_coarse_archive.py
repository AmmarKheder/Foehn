#!/usr/bin/env python3
"""WP1a check of step A on the days whose production forecast file was deleted (CPU; no monitor observation is read).

(1) October (2026-10-10..10-31): the forecast PUBLISHED on the day survives as RT/out/<d>/pub/rt_<d>.nc (ensemble mean,
    10th / 50th / 90th percentiles and P(> 25 ug m-3) of the 8 members at the ten leads, written by realtime/products.py
    with least_significant_digit = 1, i.e. rounded to 1/16 ug m-3). The same statistics of the replayed
    coarse/<d>/pred.npy must agree within half a rounding step (1/32) and the probability exactly.
(2) September (2026-09-01..09-20): the hindcast of the production weights ("a") was scored on 2026-10-11 against the CAMS
    analyses and its table is in RT/logs/validate_a.out (20 start days) and validate_ab_clean.out (2026-09-06..09-20):
    RMSE of the ensemble mean over all cells / land / fire-affected cells and land bias, cosine-latitude weighted, mean
    over the start days, by lead. The same numbers are recomputed from the replayed forecasts with the statements of
    realtime/hindcast.py score() (copied below) and compared at the printed precision (0.01).
    python -I check_coarse_archive.py -> replay/check/coarse_archive_check.json"""
import datetime as dt
import json
import re
from pathlib import Path

import numpy as np

RT = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km/realtime")
G = Path("/scratch/project_462001804/ammar/eccv/data/world/global1km")
R = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay")
THR = 25.0


def october():
    import netCDF4
    res = {}
    for k in range(22):
        d = dt.date(2026, 10, 10) + dt.timedelta(k)
        pp, nc = R / "coarse" / str(d) / "pred.npy", RT / "out" / str(d) / "pub" / f"rt_{d}.nc"
        if not pp.exists() or pp.is_symlink() or not nc.exists():
            continue
        pred = np.load(pp).astype(np.float32)
        mean = pred.mean(0)
        q10, q50, q90 = np.percentile(pred, [10, 50, 90], axis=0)
        prob = (pred > THR).mean(0)
        ds = netCDF4.Dataset(nc)
        r = {"published_title": ds.getncattr("title")}
        for n, x in (("pm25_mean", mean), ("pm25_p10", q10), ("pm25_p50", q50), ("pm25_p90", q90), ("prob_above_25", prob)):
            v = np.asarray(ds.variables[n][:], np.float64)
            dd = np.abs(v - x)
            r[n] = dict(max_abs=float(dd.max()), mean_abs=float(dd.mean()), share_within_half_step=float((dd <= 1 / 32 + 1e-6).mean()),
                        corr=float(np.corrcoef(v.ravel(), np.asarray(x, np.float64).ravel())[0, 1]))
        ds.close()
        r["reproduced"] = bool(all(r[n]["max_abs"] <= 1 / 32 + 1e-6 for n in ("pm25_mean", "pm25_p10", "pm25_p50", "pm25_p90")) and r["prob_above_25"]["max_abs"] == 0.0)
        res[str(d)] = r
        print(f"{d}: published NetCDF vs replay: max |diff| mean {r['pm25_mean']['max_abs']:.5f} p10 {r['pm25_p10']['max_abs']:.5f} p50 {r['pm25_p50']['max_abs']:.5f} "
              f"p90 {r['pm25_p90']['max_abs']:.5f} prob {r['prob_above_25']['max_abs']:.3f} -> {'REPRODUCED' if r['reproduced'] else 'DIFFERENT'} ({r['published_title'][:8]})", flush=True)
    return res


def crop_roll(a):
    return np.roll(a[..., 15:376, :], 450, axis=-1)


def september(first, last):
    """hindcast.py score(): ensemble-mean RMSE / bias against the CAMS analysis of the valid time, by lead"""
    tidx, parts = {}, []
    for vf in sorted((RT / "verif").glob("cams_pm25_*.npz")):
        V = np.load(vf)
        ts = V["times"].astype("datetime64[h]")
        for i, t in enumerate(ts):
            if str(t) not in tidx:
                tidx[str(t)] = len(parts)
                parts.append(V["pm25"][i])
    truth = np.stack(parts)
    lat = 84.0 - 0.4 * np.arange(361)
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, 900))
    land = np.load(G / "cams_static.npz")["static"][0] > 0.5
    days = [first + dt.timedelta(k) for k in range((last - first).days + 1)]
    fm = np.zeros((361, 900), bool)
    for d in days:
        x = np.load(RT / "data" / str(d) / "fires.npy").astype(np.float32)[1]
        ii = np.clip(np.round((84.0 - lat) / 0.25).astype(int), 0, 575)
        jj = np.clip(np.round((-180 + 0.4 * np.arange(900) + 180) / 0.25).astype(int), 0, 1439)
        fm |= x[ii[:, None], jj[None, :]] > 0.0
    rm = lambda e, m: float(np.sqrt((w * e ** 2)[m].sum() / w[m].sum()))
    bi = lambda e, m: float((w * e)[m].sum() / w[m].sum())
    allm = np.ones_like(land)
    res = {}
    for d in days:
        pp = R / "coarse" / str(d) / "pred.npy"
        if not pp.exists():
            return None
        t0 = dt.datetime(d.year, d.month, d.day)
        P = np.load(pp).astype(np.float32).mean(0)
        for k in range(10):
            vt = np.datetime64(t0 + dt.timedelta(hours=12 * (k + 1)), "h")
            if str(vt) not in tidx:
                continue
            e = P[k] - crop_roll(truth[tidx[str(vt)]].astype(np.float32))
            res.setdefault(k, []).append((rm(e, allm), rm(e, land), rm(e, fm), bi(e, land)))
    return {12 * (k + 1): [round(float(x), 2) for x in np.array(v).mean(0)] + [len(v)] for k, v in sorted(res.items())}, int(fm.sum())


def archived(logfile, tag="a"):
    """rows '  12 h | 1.1 ... | a  6.92  12.09  20.18  +0.02 | land ... | n 20' of the first '=== a vs' block"""
    out, on = {}, False
    for line in Path(logfile).read_text().splitlines():
        if line.startswith("==="):
            on = line.startswith(f"=== {tag} vs")
            continue
        m = re.match(r"\s*(\d+) h \| 1\.1 .*\| " + tag + r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+) \| land .* n (\d+)", line)
        if on and m:
            out[int(m.group(1))] = [float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)), int(m.group(6))]
    return out


def main():
    out = {"october_published_netcdf": october(), "september_hindcast_scores": {}}
    for name, first, last, log in (("2026-09-01..09-20", dt.date(2026, 9, 1), dt.date(2026, 9, 20), RT / "logs/validate_a.out"),
                                   ("2026-09-06..09-20", dt.date(2026, 9, 6), dt.date(2026, 9, 20), RT / "logs/validate_ab_clean.out")):
        r = september(first, last)
        if r is None:
            print(f"September {name}: replayed forecasts not all there yet")
            continue
        mine, nfire = r
        ref = archived(log)
        dmax = max(abs(mine[L][i] - ref[L][i]) for L in ref for i in range(4))
        same_n = all(mine[L][4] == ref[L][4] for L in ref)
        out["september_hindcast_scores"][name] = dict(archived_log=str(log), columns=["rmse_all", "rmse_land", "rmse_fire_cells", "bias_land", "start_days"],
                                                      replay=mine, archived=ref, fire_cells=nfire, max_abs_difference=dmax, same_number_of_days=same_n)
        print(f"September {name} ({log.name}): max |replay - archived| over 10 leads x 4 statistics = {dmax:.3f} (printed precision 0.01), "
              f"fire cells {nfire}, same day counts {same_n}")
        for L in sorted(ref):
            print(f"   {L:4d} h  replay {mine[L]}  archived {ref[L]}")
    oc = out["october_published_netcdf"]
    print(f"October: {sum(v['reproduced'] for v in oc.values())} of {len(oc)} replayed days reproduce the published NetCDF within the rounding step")
    (R / "check" / "coarse_archive_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
