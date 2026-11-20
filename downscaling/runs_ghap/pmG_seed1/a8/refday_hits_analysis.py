#!/usr/bin/env python3
"""A8: where the member points at the state bound are, for a reference day of the FROZEN system
(a8/results/refday_hits_day<t>.npz written by refday_eval.py: latitude, longitude, member, starting noise z0, final z).

For each hit pixel the script looks up (memory-mapped, a few thousand rows only) the GHAP value of that selection day
(data/processed/ghap/ghap_eval/full_2021.npy; 2021 hold-out, the same truth the evaluation reads; never 2022), the
pixel's climatology mu(doy), sigma (clim_mean_G.npy, clim_var_G.npy) and the six statics, and reports: sign of the
bound, value the member takes there (expm1(mu +- 10 sigma), capped at 1,000), the truth, where on the globe.
numpy only; CPU; reads no station file. Writes a8/results/refday_hits_day<t>.json.

  python refday_hits_analysis.py --day 210"""
import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

W = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
G = W / "data" / "processed" / "ghap"
A8 = W / "runs_ghap" / "pmG_seed1" / "a8"
NC = 36000
LAT0, LON0, RES = 83.995, -179.995, 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, default=210)
    ap.add_argument("--eval-seed", type=int, default=0, help="evaluation-noise seed of the run (file suffix _es<seed>)")
    a = ap.parse_args()
    name = f"refday_hits_day{a.day}" + (f"_es{a.eval_seed}" if a.eval_seed else "")
    z = np.load(A8 / "results" / f"{name}.npz")
    lat, lon, mem, z0, zf = z["lat"], z["lon"], z["member"], z["z0"], z["z"]
    n = int(lat.size)
    date = dt.date(2021, 1, 8) + dt.timedelta(a.day)
    doy = (date - dt.date(2021, 1, 1)).days
    out = {"hold_index": a.day, "date": str(date), "eval_noise_seed": a.eval_seed, "n_member_points_at_bound": n}
    if n == 0:
        (A8 / "results" / f"{name}.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
        return
    man = json.loads((G / "manifest.json").read_text())
    r = np.rint((LAT0 - lat) / RES).astype(np.int64)
    c = np.rint((lon - LON0) / RES).astype(np.int64)
    flat = r * NC + c
    mask = np.load(G / "grid" / "ghap_mask.npy", mmap_mode="r")
    idx = np.searchsorted(mask, flat)
    ok = (idx < mask.shape[0]) & (np.asarray(mask[np.minimum(idx, mask.shape[0] - 1)]).astype(np.int64) == flat)
    out["pixels_found_in_mask"] = int(ok.sum())
    assert ok.all(), "a hit pixel is not on the mask lattice (lat/lon rounding)"
    upix, inv = np.unique(idx, return_inverse=True)
    hold = np.load(G / man["eval"]["file"], mmap_mode="r")
    truth = np.asarray(hold[doy, upix]).astype(np.float32)[inv]
    cm = np.asarray(np.load(G / "ghap_clim" / "clim_mean_G.npy", mmap_mode="r")[upix]).astype(np.float64)[inv]
    cv = np.asarray(np.load(G / "ghap_clim" / "clim_var_G.npy", mmap_mode="r")[upix]).astype(np.float64)[inv]
    st = np.asarray(np.load(G / "ghap_clim" / "static_G.npy", mmap_mode="r")[upix]).astype(np.float64)[inv]
    ang = 2.0 * math.pi * doy / 365.0
    b = np.array([1.0, math.cos(ang), math.sin(ang), math.cos(2 * ang), math.sin(2 * ang)])
    mu = cm @ b
    sig = np.sqrt(np.maximum(cv[:, 0], float(man["static"]["sigma_floor"]) ** 2))        # sigma_k = 0: annual sigma
    member_ug = np.clip(np.expm1(mu + sig * zf), 0.0, 1000.0)
    clim_ug = np.expm1(mu)
    tz = (np.log1p(np.maximum(truth, 1.0)) - mu) / sig
    q = lambda v: [float(x) for x in np.percentile(v, [0, 5, 25, 50, 75, 95, 100])]
    out.update({
        "n_pixels": int(upix.size),
        "n_at_plus_bound": int((zf > 0).sum()), "n_at_minus_bound": int((zf < 0).sum()),
        "by_member": {int(m): int((mem == m).sum()) for m in np.unique(mem)},
        "max_members_at_bound_in_one_pixel": int(np.bincount(inv).max()),
        "start_noise_z0_percentiles_0_5_25_50_75_95_100": q(z0),
        "share_start_noise_at_truncation_3": float((np.abs(z0) >= 2.999).mean()),
        "member_value_ug_percentiles": q(member_ug),
        "truth_ug_percentiles": q(truth), "n_truth_holes": int((truth < 0).sum()),
        "truth_z_percentiles": q(tz),
        "climatology_median_ug_percentiles": q(clim_ug),
        "mu_log1p_percentiles": q(mu), "sigma_percentiles": q(sig),
        "lat_percentiles": q(lat), "lon_percentiles": q(lon),
        "share_north_of_60N": float((lat > 60).mean()), "share_north_of_70N": float((lat > 70).mean()),
        "share_south_of_45S": float((lat < -45).mean()),
        "static_columns": man["static"]["columns"], "static_median": [float(x) for x in np.median(st, 0)],
    })
    # 0.25 deg cells and 5 deg boxes
    cell = (np.floor((84.125 - lat) / 0.25).astype(np.int64), np.floor((lon + 180.125) / 0.25).astype(np.int64))
    ucell = set(zip(cell[0].tolist(), cell[1].tolist()))
    out["n_cells_025"] = len(ucell)
    box = {}
    for la, lo in zip((np.rint(lat / 5.0) * 5).astype(int), (np.rint(lon / 5.0) * 5).astype(int)):
        box[(int(la), int(lo))] = box.get((int(la), int(lo)), 0) + 1
    out["by_5deg_box_latlon"] = {f"{k[0]},{k[1]}": v for k, v in sorted(box.items(), key=lambda x: -x[1])[:15]}
    (A8 / "results" / f"{name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
