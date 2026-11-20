#!/usr/bin/env python3
"""cams3h step 2: the CAMS forecast at the catalogue monitors with the daily mean of the paper, and the checks.

For every issue day d of the 53 cycles -> <replay>/cams3h/points/<d>.npz, arrays (5 lead days, 12,136 monitors) float32
in catalogue order (the format read by score/build_pairs.py as NAME=<this directory>:KEY):
  cams3h       daily mean by the paper's rule for 2022: the eight 3-hourly steps 00 .. 21 UTC of the valid day
  cams3i       the three instants 1/4, 1/2, 1/4 (start, middle, end of the day) recomputed from the NEW files with the
               lines of the existing arm; it must reproduce the array `cams` of <replay>/points/<d>.npz
  cams3h_f32   as cams3h from the float32 values (the other arrays use the float16 values that the chain stores)
  cams3h_trap  sensitivity: nine steps 00 .. 24 UTC, half weights on the two ends
  cams3h_0324  sensitivity: eight steps 03 .. 24 UTC
Same spatial reading for all (c3h_lib): floor at 0, bilinear on the 0.4 degree grid with rc/coupling_live.bilinear.

Checks written to <replay>/cams3h/build_report.json, per cycle:
  a. the 12-hourly leads 12 .. 120 h of the new file are bit-identical to the ten-lead file of the existing arm;
  b. the step 0 h is identical to the analysis that the existing arm uses as its 0 h instant (realtime/data/<d>/cams.npy);
  c. cams3i against the existing arm `cams` (largest difference, share of identical values, same finite monitors);
  d. cycles up to 2026-11-03: the daily means against cams_forecast_daily/2026.zarr, written by the paper's own builder
     (dl_cams_forecast.py) from its own download: on the grid and at the monitors;
  e. float16 against float32 values at the monitors.
    python -I -B build_arms.py"""
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c3h_lib as C  # noqa: E402
import safe_io  # noqa: E402

TEN = [12 * k for k in range(1, 11)]


def ten_lead_file(d, prov):
    """the ten-lead file that the existing arm read for issue day d (its provenance), with a fall-back to the two places"""
    p = Path(prov) if prov else None
    if p is not None and p.exists():
        return p
    for q in (C.R / "camsfc" / str(d) / "cams_fc.npz", C.RT / "data" / str(d) / "cams_fc.npz"):
        if q.exists() and np.load(q)["leads"].tolist() == TEN:
            return q
    return None


def main():
    t0 = time.time()
    bil = C.bilinear_fn()
    ids, lat, lon = C.catalogue()
    n = ids.size
    idx = C.index_raw()
    missing = [str(d) for d in C.TARGET if d not in idx]
    print(f"raw files hold {len(idx)} cycles; missing of the 53: {missing or 'none'}", flush=True)
    pts = C.OUT / "points"
    pts.mkdir(parents=True, exist_ok=True)
    import zarr
    zr = zarr.open(str(C.ZARR_2026), mode="r") if C.ZARR_2026.exists() else None
    rep = {}
    files = sorted({f for f, _ in idx.values()})
    for f in files:
        dates, x32_all, glat, glon = C.read_file(f)
        print(f"{f.name}: {len(dates)} cycles {dates[0]}..{dates[-1]}, {x32_all.shape} ({time.time() - t0:.0f} s)", flush=True)
        for i, d in enumerate(dates):
            if d not in C.TARGET:
                continue
            x32 = x32_all[i]                                               # (41, 451, 900) float32 ug m-3
            x16 = C.to_f16(x32)
            r32, r16 = C.roll180(x32), C.roll180(x16)                      # longitudes -180 .. 179.6
            row = dict(file=f.name, mean_ug_step0_step120=[round(float(x32[0].mean()), 4), round(float(x32[-1].mean()), 4)])
            # ---- the existing arm and its inputs
            old = np.load(C.R / "points" / f"{d}.npz", allow_pickle=False)
            assert np.array_equal(old["id"].astype(str), ids.astype(str))
            prov = json.loads(str(old["provenance"])).get("baselines", {})
            tf = ten_lead_file(d, prov.get("cams_forecast"))
            if tf is not None:
                c = np.load(tf)
                assert c["leads"].tolist() == TEN and np.array_equal(c["lat"], glat) and np.array_equal(c["lon"], glon)
                mine = x32[[h // 3 for h in TEN]].astype(c["pm25"].dtype)
                row["a_ten_leads"] = dict(file=str(tf), dtype=str(c["pm25"].dtype), bit_identical=bool(np.array_equal(mine, c["pm25"])),
                                          max_abs_diff=float(np.abs(mine.astype(np.float64) - c["pm25"].astype(np.float64)).max()))
            an = np.load(C.RT / "data" / str(d) / "cams.npy")[:, 1].astype(np.float32)[1]          # (361, 900): 84 N .. 60 S, the 0 h instant of `cams`
            m0 = r16[0][15:15 + 361]
            row["b_step0_vs_analysis"] = dict(identical=bool(np.array_equal(m0, an)), share_identical=float((m0 == an).mean()),
                                              max_abs_diff=float(np.abs(m0.astype(np.float64) - an.astype(np.float64)).max()))
            # ---- the arms
            out = {k: np.full((C.NLEAD, n), np.nan, np.float32) for k in ("cams3h", "cams3i", "cams3h_f32", "cams3h_trap", "cams3h_0324")}
            dm32 = np.zeros((C.NLEAD, 451, 900), np.float64)               # paper's rule from the float32 values, on the file's own grid (0 .. 359.6)
            for L in range(1, C.NLEAD + 1):
                out["cams3i"][L - 1] = bil(C.daily_three(r16, L), 90.0, 0.4, -180.0, 0.4, lat, lon)
                out["cams3h"][L - 1] = bil(C.daily_rule(r16, L, "paper"), 90.0, 0.4, -180.0, 0.4, lat, lon)
                out["cams3h_trap"][L - 1] = bil(C.daily_rule(r16, L, "trap"), 90.0, 0.4, -180.0, 0.4, lat, lon)
                out["cams3h_0324"][L - 1] = bil(C.daily_rule(r16, L, "0324"), 90.0, 0.4, -180.0, 0.4, lat, lon)
                dm32[L - 1] = C.daily_rule(x32, L, "paper")
                out["cams3h_f32"][L - 1] = bil(C.roll180(dm32[L - 1]), 90.0, 0.4, -180.0, 0.4, lat, lon)
            for k, v in out.items():
                assert np.isfinite(v).all() and float(v.min()) >= 0.0, (str(d), k)
            oc = old["cams"]
            both = np.isfinite(oc) & np.isfinite(out["cams3i"])
            df = np.abs(oc.astype(np.float64) - out["cams3i"].astype(np.float64))
            row["c_cams3i_vs_existing"] = dict(same_finite_mask=bool(np.array_equal(np.isfinite(oc), np.isfinite(out["cams3i"]))),
                                               existing_finite=int(np.isfinite(oc).sum()), max_abs_diff=float(df[both].max()),
                                               share_identical=float((oc[both] == out["cams3i"][both]).mean()),
                                               max_abs_diff_by_lead=[float(df[L][both[L]].max()) for L in range(C.NLEAD)])
            # ---- the paper's builder (daily means of 2026 cycles on disk up to 2026-11-03)
            if zr is not None:
                doy = (d - dt.date(2026, 1, 1)).days
                zd = np.asarray(zr[doy, 0], np.float64)                    # (5, 451, 900) lead day 0..4 = L - 1, lon 0 .. 359.6
                if np.isfinite(zd).all():
                    g = np.abs(zd - dm32)
                    at = np.stack([bil(C.roll180(np.maximum(zd[L], 0.0)), 90.0, 0.4, -180.0, 0.4, lat, lon) for L in range(C.NLEAD)])
                    row["d_vs_paper_builder"] = dict(grid_max_abs_diff=float(g.max()), grid_max_rel_diff=float((g / np.maximum(zd, 1e-3)).max()),
                                                     monitors_max_abs_diff_f32=float(np.abs(at - out["cams3h_f32"]).max()),
                                                     monitors_max_abs_diff_cams3h=float(np.abs(at - out["cams3h"]).max()))
            row["e_f16_vs_f32_monitors_max_abs_diff"] = float(np.abs(out["cams3h"].astype(np.float64) - out["cams3h_f32"].astype(np.float64)).max())
            row["mean_by_lead"] = {k: [round(float(v[L].mean()), 3) for L in range(C.NLEAD)] for k, v in out.items()}
            row["mean_by_lead"]["cams_existing"] = [round(float(np.nanmean(oc[L])), 3) for L in range(C.NLEAD)]
            valid = np.array([str(d + dt.timedelta(L)) for L in range(C.NLEAD)])
            sha = safe_io.write_npz(pts / f"{d}.npz", id=ids, issue=np.array(str(d)), valid=valid, **out)
            row["points_sha256"] = sha
            rep[str(d)] = row
            print(f"  {d}: ten leads identical {row.get('a_ten_leads', {}).get('bit_identical')}, step 0 = analysis {row['b_step0_vs_analysis']['identical']}, "
                  f"cams3i vs cams max |diff| {row['c_cams3i_vs_existing']['max_abs_diff']:.3g} (identical {100 * row['c_cams3i_vs_existing']['share_identical']:.2f} %), "
                  f"paper builder grid {row.get('d_vs_paper_builder', {}).get('grid_max_abs_diff', float('nan')):.3g}, "
                  f"means L1 cams3h {row['mean_by_lead']['cams3h'][0]:.3f} cams3i {row['mean_by_lead']['cams3i'][0]:.3f}", flush=True)
        del x32_all
    done = sorted(rep)
    summ = dict(
        cycles_built=len(done), cycles_missing=[str(d) for d in C.TARGET if str(d) not in rep],
        a_ten_leads_bit_identical=sum(bool(r.get("a_ten_leads", {}).get("bit_identical")) for r in rep.values()),
        a_ten_leads_checked=sum("a_ten_leads" in r for r in rep.values()),
        b_step0_identical_to_analysis=sum(r["b_step0_vs_analysis"]["identical"] for r in rep.values()),
        c_cams3i_max_abs_diff=max(r["c_cams3i_vs_existing"]["max_abs_diff"] for r in rep.values()),
        c_cams3i_min_share_identical=min(r["c_cams3i_vs_existing"]["share_identical"] for r in rep.values()),
        c_same_finite_mask=all(r["c_cams3i_vs_existing"]["same_finite_mask"] for r in rep.values()),
        d_cycles_in_paper_builder=sum("d_vs_paper_builder" in r for r in rep.values()),
        d_grid_max_abs_diff=max([r["d_vs_paper_builder"]["grid_max_abs_diff"] for r in rep.values() if "d_vs_paper_builder" in r] or [float("nan")]),
        d_grid_max_rel_diff=max([r["d_vs_paper_builder"]["grid_max_rel_diff"] for r in rep.values() if "d_vs_paper_builder" in r] or [float("nan")]),
        d_monitors_max_abs_diff_f32=max([r["d_vs_paper_builder"]["monitors_max_abs_diff_f32"] for r in rep.values() if "d_vs_paper_builder" in r] or [float("nan")]),
        d_monitors_max_abs_diff_cams3h=max([r["d_vs_paper_builder"]["monitors_max_abs_diff_cams3h"] for r in rep.values() if "d_vs_paper_builder" in r] or [float("nan")]),
        e_f16_vs_f32_monitors_max_abs_diff=max(r["e_f16_vs_f32_monitors_max_abs_diff"] for r in rep.values()),
        made_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    safe_io.write_json(C.OUT / "build_report.json", dict(summary=summ, cycles=rep))
    print(json.dumps(summ, indent=1))
    print(f"done in {time.time() - t0:.0f} s -> {pts}")
    return 0 if not summ["cycles_missing"] else 1


if __name__ == "__main__":
    sys.exit(main())
