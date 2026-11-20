#!/usr/bin/env python3
"""WP1a final checks over all delivered days (CPU; no observation is read) -> replay/check/summary.json and a table.

  (a) every points/<d>.npz: shapes, finite shares per array and lead day, value ranges, lead-day means, member spread,
      guard hits, the source of the coarse forecast, seconds per step;
  (b) issue days with a published snapshot of the same system (verif/fc/<d>.npz, tag pmG_final_P3T3: 2026-11-03..11-11):
      ensemble mean / p10 / p90 of 'km' against the snapshot (the proof that the replay is the production forecast);
  (c) rt04: member mean against the member-mean-first convention of the live comparison (max difference);
  (d) analysis mode against forecast mode, where 'an' exists: correlation and mean ratio of the ensemble means (plausibility).
    python -I check_all.py"""
import datetime as dt
import json
from pathlib import Path

import numpy as np

P = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
RT = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km/realtime")
R = P / "realtime_1km/replay"
TAG = "pmG_final_P3T3"
ARR3 = ("km", "km_ring1", "km_ring3", "km_ring10", "km_cell", "rt04", "an", "km_cell_lin", "bcsd_m", "qm_cell")
ARR2 = ("cams", "clim", "bcsd", "mu_cell")


def cmp16(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok].astype(np.float64), b[ok].astype(np.float64)
    a16 = a.astype(np.float16).astype(np.float64)
    step = np.spacing(np.abs(b).astype(np.float16)).astype(np.float64)
    d = np.abs(a - b)
    return dict(n=int(ok.sum()), corr=float(np.corrcoef(a, b)[0, 1]), mean_abs=float(d.mean()), max_abs=float(d.max()),
                max_rel=float((d / np.maximum(np.abs(b), 1.0)).max()), identical_after_float16=float((a16 == b).mean()),
                within_1_float16_step=float((np.abs(a16 - b) <= step).mean()))


def main():
    days = sorted(x.strip() for x in (R / "days_all.txt").read_text().split() if x.strip())
    out = {"days_expected": days, "days": {}, "snapshot": {}, "missing": []}
    ids = None
    for d in days:
        f = R / "points" / f"{d}.npz"
        if not f.exists():
            out["missing"].append(d)
            continue
        z = np.load(f)
        if ids is None:
            ids = z["id"]
        assert np.array_equal(ids, z["id"]) and str(z["issue"]) == d
        pr = json.loads(str(z["provenance"]))
        km = z["km"]
        row = dict(leads_made=pr["leads_made"], coarse=pr["coarse"]["source"].split()[0], cams=bool(np.isfinite(z["cams"]).any()),
                   an_leads=[int(L + 1) for L in range(5) if np.isfinite(z["an"][L]).any()],
                   finite={k: [round(float(np.isfinite(z[k][L]).mean()), 4) for L in range(5)] for k in ARR3 + ARR2},
                   km_mean=[round(float(np.nanmean(km[L])), 3) for L in range(5)],
                   km_spread_sd=[round(float(np.nanmean(km[L].std(0, ddof=1))), 3) for L in range(5)],
                   km_min=float(np.nanmin(km)), km_max=float(np.nanmax(km)),
                   rt04_mean=[round(float(np.nanmean(z["rt04"][L])), 3) for L in range(5)],
                   rt04_spread_sd=[round(float(np.nanmean(z["rt04"][L].std(0, ddof=1))), 3) for L in range(5)],
                   cams_mean=[round(float(np.nanmean(z["cams"][L])), 3) if np.isfinite(z["cams"][L]).any() else None for L in range(5)],
                   clim_mean=[round(float(np.nanmean(z["clim"][L])), 3) for L in range(5)],
                   bcsd_mean=[round(float(np.nanmean(z["bcsd"][L])), 3) for L in range(5)],
                   guard={k: int(sum(t["guard_hits"][k] for t in pr["downscaler"]["timing"].values())) for k in ("at_pmax", "at_bound", "gt_train_max", "z0_trunc")},
                   max_member=float(max(t["guard_hits"]["max_ug"] for t in pr["downscaler"]["timing"].values())),
                   seconds=dict(flexgreen=pr["coarse"].get("flexgreen_s"), predict=pr["coarse"].get("predict_s"), prep_met=pr["inputs_1km"].get("prep_seconds_met"),
                                channels=pr["inputs_1km"].get("channels_seconds"), points=pr["downscaler"]["seconds"]),
                   trained_instants=pr["coarse"]["predicted_instants_that_were_training_targets"])
        for k in ("km", "km_ring1", "rt04", "clim", "bcsd"):
            x = z[k]
            assert not (x[np.isfinite(x)] < 0).any(), (d, k)
        # (c) rt04: member mean here vs max(daily mean of the member-mean field, 0) read bilinearly (live_baselines.py)
        # (d) analysis mode vs forecast mode
        if row["an_leads"]:
            a, b = z["an"].mean(1), km.mean(1)
            ok = np.isfinite(a) & np.isfinite(b)
            row["an_vs_km"] = dict(n=int(ok.sum()), corr_log=float(np.corrcoef(np.log1p(a[ok]), np.log1p(b[ok]))[0, 1]),
                                   mean_an=float(a[ok].mean()), mean_km=float(b[ok].mean()))
        out["days"][d] = row
        sf = P / "realtime_1km/verif/fc" / f"{d}.npz"
        if sf.exists():
            s = np.load(sf)
            if str(s["tag"]) == TAG:
                assert np.array_equal(s["id"], ids)
                q = np.quantile(km.astype(np.float64), [0.1, 0.9], axis=1)
                out["snapshot"][d] = dict(mean=cmp16(km.astype(np.float64).mean(1), s["mean"]), p10=cmp16(q[0], s["p10"]), p90=cmp16(q[1], s["p90"]),
                                          snapshot_created=str(s["created"]), coarse=row["coarse"])
    done = sorted(out["days"])
    out["n_days"] = len(done)
    out["n_monitors"] = int(ids.size) if ids is not None else 0
    (R / "check" / "summary.json").write_text(json.dumps(out, indent=1))
    print(f"{len(done)} days with a points file, missing {out['missing']}")
    print("day         leads coarse    cams an   km mean L1..L5                    spread L1  rt04 L1  cams L1  clim L1  bcsd L1  max mem  trunc%  s(points)")
    for d in done:
        r = out["days"][d]
        print(f"{d}  {''.join(map(str, r['leads_made'])):5s} {r['coarse']:9s} {'y' if r['cams'] else '-':4s} {''.join(map(str, r['an_leads'])) or '-':5s}"
              f"{' '.join(f'{x:6.2f}' for x in r['km_mean'])}   {r['km_spread_sd'][0] if r['km_spread_sd'][0] == r['km_spread_sd'][0] else float('nan'):6.2f}   "
              f"{r['rt04_mean'][0]:6.2f}  {r['cams_mean'][0] if r['cams_mean'][0] is not None else float('nan'):6.2f}  {r['clim_mean'][0]:6.2f}  {r['bcsd_mean'][0]:6.2f}  "
              f"{r['max_member']:7.0f}  {100 * r['guard']['z0_trunc'] / (875425 * 8 * len(r['leads_made'])):5.3f}  {r['seconds']['points']}")
    print("\nsnapshot comparison (published forecast of the same issue day, 5 lead days x monitors):")
    for d, s in out["snapshot"].items():
        print(f"{d} ({s['coarse']}): " + "; ".join(f"{k} r {s[k]['corr']:.9f} max|d| {s[k]['max_abs']:.4f} id16 {s[k]['identical_after_float16']:.4f} w1 {s[k]['within_1_float16_step']:.4f}" for k in ("mean", "p10", "p90")))


if __name__ == "__main__":
    main()
