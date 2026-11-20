#!/usr/bin/env python3
"""Step 1: gather, at the station pixels, every stored ensemble member of the 2022 evaluation and the references.
Reads (read-only): part_*.npz of the evaluation folders (keys, st_ens, st_ghap), the CAMS forecast (zarr) and Aurora at
equal issuance exactly as experiments/figures/figdata_station_crps.py (CAMS run of 00 UTC day d+1 at lead day h-1;
Aurora initialised 00 UTC day d+1, daily mean 0.25/0.5/0.25 of its 12-h steps, bilinear at the stations).
Writes data/st_<tag>.npz for tag in final (4 members, the model of the paper), before (the same model before the CRPS
training, eval_final_e02), ea16 / gir16 / girlong16 / kano16 (16 members, case studies):
  keys (N, 3) = start day of year (0-based), lead (days), target day of year; ens (N, 8110, M) float16; ghap (N, 8110)
  float16 = GHAP of the target day at the station pixel; for 'final' also cams, aur (N, 8110) float32 (NaN beyond 4 days).
    python s1_assemble.py"""
import glob
import sys

import numpy as np

from rel_common import AURD, BEFORE, CAMS_ZARR, DATA, FINAL, stations

ST = stations()
LAT, LON = ST["lat"], ST["lon"]
SETS = {"final": FINAL / "eval_final_crps_e01", "before": BEFORE / "eval_final_e02", "ea16": FINAL / "eval_case_eastasia_crps_e01",
        "gir16": FINAL / "eval_case_gironde_crps_e01", "girlong16": FINAL / "eval_case_gironde_long_crps_e01",
        "kano16": FINAL / "eval_case_kano_crps_e01"}


def bil(m, lat0, dlat, lon0, dlon):
    fi = (lat0 - LAT) / dlat; fj = ((LON - lon0) % 360.0) / dlon
    i0 = np.clip(np.floor(fi).astype(int), 0, m.shape[0] - 2); j0 = np.floor(fj).astype(int) % m.shape[1]
    j1 = (j0 + 1) % m.shape[1]; wi = np.clip(fi - i0, 0, 1); wj = fj - np.floor(fj)
    return ((1 - wi) * ((1 - wj) * m[i0, j0] + wj * m[i0, j1]) + wi * ((1 - wj) * m[i0 + 1, j0] + wj * m[i0 + 1, j1]))


def gather(ev):
    K, E, Gh, info = [], [], [], None
    for p in sorted(glob.glob(str(ev / "part_*.npz"))):
        z = np.load(p)
        if len(z["keys"]) == 0:
            continue
        K.append(z["keys"]); E.append(z["st_ens"]); Gh.append(z["st_ghap"]); info = [str(s) for s in z["info"]]
    K, E, Gh = np.concatenate(K), np.concatenate(E), np.concatenate(Gh)
    o = np.lexsort((K[:, 1], K[:, 0]))
    return K[o], E[o], Gh[o], info


def baselines(keys):
    import zarr
    cams = zarr.open(CAMS_ZARR, mode="r")
    aur_e = {}
    for p in glob.glob(str(AURD / "out_12h_all" / "part_*.npz")):
        z = np.load(p); la_, lo_ = z["lat"], z["lon"]
        for d, pr in zip(z["days"], z["pred"]):
            aur_e[int(d)] = np.stack([bil(pr[k].astype(np.float32), float(la_[0]), float(la_[0] - la_[1]),
                                          float(lo_[0]), float(lo_[1] - lo_[0])) for k in range(pr.shape[0])])
    an0 = {}

    def analysis00(d):
        if d not in an0:
            z = np.load(AURD / "raw" / f"{d:03d}.npz")
            an0[d] = bil(z["surf"][0, 5] * 1e9, float(z["lat"][0]), float(z["lat"][0] - z["lat"][1]),
                         float(z["lon"][0]), float(z["lon"][1] - z["lon"][0]))
        return an0[d]

    C = np.full((len(keys), len(LAT)), np.nan, np.float32); A = C.copy()
    for n, (d, h, t) in enumerate(keys):
        d, h = int(d), int(h)
        if h > 4 or d + 1 > 364:
            continue
        C[n] = bil(np.asarray(cams[d + 1, 0, h - 1], np.float32), 90.0, 0.4, 0.0, 0.4)
        if (d + 1) in aur_e:
            Ad = aur_e[d + 1]; k0 = 2 * (h - 1)
            first = analysis00(d + 1) if h == 1 else Ad[k0 - 1]
            A[n] = 0.25 * first + 0.5 * Ad[k0] + 0.25 * Ad[k0 + 1]
        if n % 100 == 0:
            print("baselines", n, "/", len(keys), flush=True)
    return C, A


for tag in (sys.argv[1:] or list(SETS)):
    K, E, Gh, info = gather(SETS[tag])
    out = dict(keys=K, ens=E, ghap=Gh, info=np.array(info), source=np.array(str(SETS[tag])))
    if tag == "final":
        out["cams"], out["aur"] = baselines(K)
    np.savez(DATA / f"st_{tag}.npz", **out)
    fin = np.isfinite(E.astype(np.float32)).all(-1)
    print(f"{tag}: {len(K)} forecasts, members {E.shape[-1]}, start days {len(np.unique(K[:, 0]))} "
          f"[{K[:, 0].min()}..{K[:, 0].max()}], leads {sorted(set(K[:, 1].tolist()))}, stations with members in the first / "
          f"any forecast: {int(fin[0].sum())} / {int(fin.any(0).sum())} | info {info}", flush=True)
