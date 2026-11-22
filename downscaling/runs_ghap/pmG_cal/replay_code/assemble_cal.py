#!/usr/bin/env python3
"""A6(b), CPU: realtime_1km/replay/points_cal/<d>.npz = the downscaled members of the CALIBRATED sampler at the
monitors, in the format of replay/points/<d>.npz (replay/README.md) for the arrays of the downscaler:
  id, lat, lon, issue, valid, km, km_ring1, km_ring3, km_ring10, km_cell (log1p units), km_cell_lin, ring1_n, ring3_n,
  ring10_n, cell_n, provenance (JSON).
The other arms (rt04, cams, clim, bcsd, an, ...) are NOT recomputed and not copied: they are in replay/points/<d>.npz, which
has the same monitor order, issue day, valid days and missing lead days (all checked here before a file is written).
No observation is opened. Also writes points_cal/INDEX.json (per day: lead days present, and the spread of the calibrated
ensemble relative to the frozen system's at the monitor pixels) and points_cal/README.md.
  python assemble_cal.py --km runs_ghap/pmG_cal/replay_out/km_TT --list days_all.txt --tag TT"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np

P = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
R = P / "realtime_1km/replay"
OUT = R / "points_cal"
KEYS = ("km", "km_ring1", "km_ring3", "km_ring10", "km_cell", "km_cell_lin")


def spread_stats(x):
    """x (5, M, n) -> pooled over lead days and monitors with all members finite: mean of the ensemble mean, root of the mean
    member variance (ddof 1)"""
    ok = np.isfinite(x).all(1)
    x = np.where(ok[:, None], x, np.nan).astype(np.float64)
    return float(np.nanmean(np.nanmean(x, 1))), float(np.sqrt(np.nanmean(np.nanvar(x, 1, ddof=1)))), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--km", required=True, help="directory of the <d>.npz written by infer_points_cal.py")
    ap.add_argument("--list", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    km_dir = Path(a.km)
    days = sorted(x for x in Path(a.list).read_text().split() if x.strip())
    OUT.mkdir(parents=True, exist_ok=True)
    index, cal0 = {}, None
    for d in days:
        f = km_dir / f"{d}.npz"
        if not f.exists():
            index[d] = {"error": "no calibrated km file"}
            print(f"{d}: MISSING {f}", flush=True)
            continue
        kz, p = np.load(f), np.load(R / "points" / f"{d}.npz")
        meta, prov0 = json.loads(str(kz["meta"])), json.loads(str(p["provenance"]))
        ds0 = prov0["downscaler"]
        assert meta["date"] == d == str(p["issue"]), (d, meta["date"], str(p["issue"]))
        for k in KEYS:                                              # same shape, same missing lead days and monitors as the frozen file
            assert kz[k].shape == p[k].shape and kz[k].dtype == p[k].dtype, (d, k, kz[k].shape, p[k].shape)
            assert np.array_equal(np.isnan(kz[k]), np.isnan(p[k])), (d, k, "NaN pattern differs from the frozen system's file")
        made = [L for L in range(1, 6) if str(kz["valid"][L - 1])]
        assert made == meta["leads_made"] == prov0["leads_made"], (d, made, prov0["leads_made"])
        assert all(str(kz["valid"][L - 1]) == str(p["valid"][L - 1]) for L in made), (d, kz["valid"], p["valid"])
        for k in ("ckpt_sha256", "prior_sha256", "guard", "ode_steps", "seed", "step", "n_positions"):   # the same network, prior, guard, seeds
            assert meta[k] == ds0[k], (d, k, meta[k], ds0[k])
        assert meta["members"] == prov0["members"] and meta["work"] == prov0["inputs_1km"]["work"], (d, meta["work"])
        cal = meta["calibration"]
        assert cal["tau"] != 1.0, "frozen settings: not a calibrated file"
        cal0 = cal0 or cal
        assert cal == cal0, (d, cal, cal0)
        prov = dict(issue=prov0["issue"], valid=prov0["valid"], members=prov0["members"], n_monitors=prov0["n_monitors"], leads_made=made,
                    system=f"pmG_final network and prior P3 with the calibrated sampler '{a.tag}' (PROTOCOL.md A6; candidate, selection year 2021 only)",
                    coarse=prov0["coarse"],
                    downscaler={k: meta[k] for k in ("ckpt", "ckpt_sha256", "run", "step", "prior", "prior_sha256", "guard", "ode_steps", "seed", "noise",
                                                    "device", "torch", "job", "seconds", "timing", "n_positions", "calibration", "code", "code_sha256")},
                    inputs_1km=prov0["inputs_1km"], positions=prov0["positions"],
                    frozen_counterpart=str(R / "points" / f"{d}.npz"),
                    note="only the downscaled arrays are recomputed (same coarse members, same 1 km inputs, same noise seeds as the frozen "
                         "system's file; the base noise is passed through the calibration map before the unchanged truncation); the other "
                         "arms (rt04, cams, clim, bcsd, an, ...) are in the frozen counterpart, same monitor order and days",
                    assembled_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    assemble_code=str(Path(__file__).resolve()), assemble_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        arrays = dict(id=p["id"], lat=p["lat"], lon=p["lon"], issue=p["issue"], valid=p["valid"],
                      km=kz["km"], km_ring1=kz["km_ring1"], km_ring3=kz["km_ring3"], km_ring10=kz["km_ring10"], km_cell=kz["km_cell"],
                      km_cell_lin=kz["km_cell_lin"], ring1_n=p["ring1_n"], ring3_n=p["ring3_n"], ring10_n=p["ring10_n"], cell_n=p["cell_n"],
                      provenance=np.array(json.dumps(prov)))
        tmp = OUT / f"{d}.tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, OUT / f"{d}.npz")
        m1, s1, n1 = spread_stats(kz["km"])
        m0, s0, n0 = spread_stats(p["km"])
        hits = {k: sum(t["guard_hits"][k] for t in meta["timing"].values()) for k in ("at_pmax", "gt_train_max", "at_bound", "z0_trunc")}
        hits0 = {k: sum(t["guard_hits"][k] for t in ds0["timing"].values()) for k in ("at_pmax", "gt_train_max", "at_bound", "z0_trunc")}
        index[d] = {"leads_made": made, "n_monitor_leads_finite": n1, "ens_mean_cal": m1, "ens_mean_frozen": m0, "spread_cal": s1, "spread_frozen": s0,
                    "spread_ratio": s1 / s0, "max_member_cal": float(np.nanmax(kz["km"])), "max_member_frozen": float(np.nanmax(p["km"])),
                    "guard_hits_cal_all_positions": hits, "guard_hits_frozen_all_positions": hits0,
                    "max_ug_all_positions_cal": max(t["guard_hits"]["max_ug"] for t in meta["timing"].values()),
                    "max_ug_all_positions_frozen": max(t["guard_hits"]["max_ug"] for t in ds0["timing"].values())}
        print(f"{d}: leads {made}; monitor-pixel ensemble mean {m1:.2f} (frozen {m0:.2f}), spread {s1:.2f} (frozen {s0:.2f}, x{s1 / s0:.3f}), "
              f"max member {index[d]['max_member_cal']:.0f} (frozen {index[d]['max_member_frozen']:.0f}), z0 truncated {hits['z0_trunc']} "
              f"(frozen {hits0['z0_trunc']}), above train max {hits['gt_train_max']} (frozen {hits0['gt_train_max']})", flush=True)
    done = [d for d in days if "error" not in index[d]]
    if done:
        tot = {k: float(np.mean([index[d][k] for d in done])) for k in ("ens_mean_cal", "ens_mean_frozen", "spread_cal", "spread_frozen", "spread_ratio")}
        (OUT / "INDEX.json").write_text(json.dumps({"tag": a.tag, "calibration": cal0, "n_days": len(done), "missing": [d for d in days if d not in done],
                                                    "mean_over_days": tot, "days": index}, indent=1))
        (OUT / "README.md").write_text(f"""# Reforecasts in real-time mode at the monitors: calibrated sampler ({a.tag})

One file per issue day, `<d>.npz`, in the format of `../README.md` (same axis order (lead day, member, monitor), same monitor
order as `../points/<d>.npz`, same issue and valid days, same missing lead days), for the downscaled arrays only:
`id`, `lat`, `lon`, `issue`, `valid`, `km`, `km_ring1`, `km_ring3`, `km_ring10`, `km_cell` (log1p units), `km_cell_lin`,
`ring1_n`, `ring3_n`, `ring10_n`, `cell_n`, `provenance` (JSON). The other arms (`rt04`, `cams`, `clim`, `bcsd`, `an`, ...)
are not recomputed: read them in `../points/<d>.npz`. No observation is in these files and none was opened to make them.

System: the frozen downscaler (`runs_ghap/pmG_final`: checkpoint, fitted prior P3, guard T3, 10 Euler steps, 8 members, one
downscaled member per coarse member, the same noise seeds) with ONE change in the sampler: the field of the prior, xi, is
replaced by sign(xi) g(|xi|) before the unchanged truncation |z0| <= 3, with g(u) = tau u for u <= knee and the straight line
from (knee, tau knee) to (3, 3) beyond; tau = {cal0['tau']}, knee = {cal0['knee']} (`PROTOCOL.md` A6(e) and
`runs_ghap/pmG_cal/A6_ADDENDUM_tapered_temperature.md`; two numbers chosen on the 24 selection days of 2021 against the 1 km
estimates; no training, no station, no value of 2022). Decision and selection-year numbers: `runs_ghap/pmG_cal/results/DECISION.md`.

Code: `runs_ghap/pmG_cal/replay_code/` (`infer_points_cal.py` = copy of `../code/infer_points.py` with the temperature added,
diff in `infer_points_cal.diff`; with tau = 1 it reproduces `../km`, `../raw` and `../points` bit for bit:
`runs_ghap/pmG_cal/results/replay_proof.json`). `INDEX.json`: per issue day the lead days present and the spread of the
calibrated ensemble at the monitor pixels relative to the frozen system's (root mean member variance).
{len(done)} issue days; mean over days of the spread ratio {tot['spread_ratio']:.3f}; ensemble mean {tot['ens_mean_cal']:.2f}
(frozen {tot['ens_mean_frozen']:.2f}) ug m-3 at the monitor pixels.
""")
        print(f"assembled {len(done)} of {len(days)} days -> {OUT}; mean spread ratio {tot['spread_ratio']:.3f}, ensemble mean {tot['ens_mean_cal']:.2f} "
              f"(frozen {tot['ens_mean_frozen']:.2f})", flush=True)
    raise SystemExit(0 if len(done) == len(days) else 1)


if __name__ == "__main__":
    main()
