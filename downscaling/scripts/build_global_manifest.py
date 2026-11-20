#!/usr/bin/env python3
"""data/processed/global/manifest.json: merges manifest_parts/*.json, adds the training-day normalisation numbers
(CAMS log1p mean / std, T6 channel mean / std: split-A training days only, global, unweighted, as the smoke store),
every file's shape, dtype, size and SHA-256, day ranges, the inode count and a grep test (no training code refers to
stations or sealed data). World venv (numpy only):  python build_global_manifest.py [--procs 32] [--no-hash]"""
import argparse
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
OUT = ROOT / "data/processed/global"
STATIONS = ROOT / "data/stations"
TRAIN0, TRAIN1 = dt.date(2019, 7, 11), dt.date(2022, 12, 31)


def npy_header(path):
    with open(path, "rb") as f:
        ver = np.lib.format.read_magic(f)
        shape, fo, dtype = (np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0)(f)
        return tuple(shape), np.dtype(dtype), f.tell()


def read_row(path, i):
    shape, dtype, hdr = npy_header(path)
    n = int(np.prod(shape[1:]))
    return np.fromfile(path, dtype=dtype, count=n, offset=hdr + i * n * dtype.itemsize).reshape(shape[1:])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                break
            h.update(b)
    return str(path), h.hexdigest()


def norm_stats():
    days = [TRAIN0 + dt.timedelta(k) for k in range((TRAIN1 - TRAIN0).days + 1)]
    s1 = s2 = n = 0.0
    for d in days[::2]:
        x = np.log1p(read_row(OUT / "coarse/cams_pm25_025.npy", (d - dt.date(2017, 1, 1)).days).astype(np.float64))
        s1 += x.sum(); s2 += (x * x).sum(); n += x.size
    cm = s1 / n
    t1 = np.zeros(6); t2 = np.zeros(6); tn = 0
    for d in days:
        x = read_row(OUT / f"transport/t6_{d.year}.npy", (d - dt.date(d.year, 1, 1)).days).astype(np.float64)
        assert np.isfinite(x).all(), d
        t1 += x.sum((1, 2)); t2 += (x * x).sum((1, 2)); tn += x[0].size
    tm = t1 / tn
    return {"note": f"split-A training days {TRAIN0} .. {TRAIN1} only (CAMS every 2nd day, T6 every day), global, unweighted",
            "cams_log1p_mean_std": [float(cm), float(np.sqrt(s2 / n - cm * cm))],
            "t6_channel_mean": tm.round(6).tolist(), "t6_channel_std": np.sqrt(t2 / tn - tm * tm).round(6).tolist()}


def main(a):
    t0 = time.time()
    parts = {p.stem: json.loads(p.read_text()) for p in sorted((OUT / "manifest_parts").glob("*.json"))}
    lg = parts.get("lghap", {})
    lg.pop("per_day_valid_count", None)          # kept in manifest_parts/lghap.json
    m = {"built": time.strftime("%Y-%m-%d %H:%M"), "root": str(OUT), "lghap": lg}
    m["coarse"] = {k: parts[k] for k in ("build_met", "build_fires", "build_cams", "nonregression", "days", "cams_check") if k in parts}
    m["wb2"] = {"validation_2022_12_vs_weatherbench2": parts.get("wb2_validation"),
                "months": [json.loads(l) for l in (OUT / "wb2/manifest_wb2.jsonl").read_text().splitlines()] if (OUT / "wb2/manifest_wb2.jsonl").exists() else []}
    m["flexgreen"] = {"nonregression_vs_foehn_green_era5": parts.get("flexgreen_nonregression"),
                      "code": "transport/flexgreen_pm.py --src era5 --max-lead 1 --out-root data/processed/global/flexgreen --wb-extra data/processed/global/wb2 --fires-new coarse/fires_2023_2024.npy",
                      "start_days": ["2022-01-01", "2024-12-30", 1095], "done_lists": "flexgreen/era5/done_<y>.txt (doys)"}
    m["t6"] = {"years": {k[3:]: v for k, v in parts.items() if k.startswith("t6_") and k[3:].isdigit()},
               "vs_smoke_store": parts.get("t6_vs_smoke"),
               "channels": ["log1p(transported CAMS PM2.5 of D-1)", "aloft share", "mean survival", "log1p(fire tpm x 1e9)",
                            "log1p(fire frp)", "log1p(fire tpm aloft x 1e9)"],
               "index": "transport/t6_<y>.npy row = doy of the target day D; start day D - 1, lead 1; t6_2017 row 0 = NaN (no start day)"}
    m["loader_test"] = parts.get("loader_test")
    m["normalisation_convenience"] = norm_stats()
    print("normalisation:", m["normalisation_convenience"], flush=True)
    m["conventions"] = {
        "lghap_grid": "0.01 deg: row i -> lat -57.995 + 0.01 i, column j -> lon -179.995 + 0.01 j; files lon-major, transposed",
        "mask_ids": "grid/lghap_mask.npy p = i * 36000 + j; k = position = pixel index of every per-pixel file and of idx_*",
        "encoder_grid_025": "576 x 1440: lat 84 - 0.25 r, lon -180 + 0.25 c (met, fires, CAMS)",
        "t6_grid_05": "288 x 720 cell centres lat 84 - 0.125 - 0.5 r5, lon -180 + 0.125 + 0.5 c5",
        "inputs_by_date": "coarse/days.npz: split code and (store, row) of met / fires / cams / t6 for every date 2017-2024",
        "met_fires_2017_2022": ["/scratch/project_462001804/ammar/eccv/data/world/global1km/met.npy (read only)",
                                "/scratch/project_462001804/ammar/eccv/data/world/global1km/fires.npy (read only)"],
        "target_transform": "y = log1p(max(LGHAP, 1)) (+ U(-0.5, 0.5) in training); mu, sigma = harmonic K=2 from clim_*_A at doy, sigma floor in lghap.static.sigma_floor",
        "sealed": "sealed/lghap_points/* (LGHAP 2017-01-01 .. 2019-07-10 and 2024) + data/stations 2024: written, never summarised; only the unsealing pass reads them",
        "stations": "data/stations/ (2024 EPA AQS, EEA) + eccv/data/world/stations/cnemc (2023-2024): evaluation only",
        "loader": "cdffm_aq.store_global.GlobalStore (SmokeStore interface)"}
    # files
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and "manifest_parts" not in p.parts and p.name != "manifest.json")
    rec = {}
    for p in files:
        r = {"bytes": p.stat().st_size}
        if p.suffix == ".npy":
            sh, dty, _ = npy_header(p)
            r.update(shape=list(sh), dtype=str(dty))
        rec[str(p.relative_to(OUT))] = r
    st_files = sorted(p for p in STATIONS.rglob("*") if p.is_file()) if STATIONS.exists() else []
    for p in st_files:
        rec["../../stations/" + str(p.relative_to(STATIONS))] = {"bytes": p.stat().st_size}
    if not a.no_hash:
        with mp.Pool(a.procs) as pool:
            for path, h in pool.imap_unordered(sha256, [str(p) for p in files + st_files]):
                key = str(Path(path).relative_to(OUT)) if Path(path).is_relative_to(OUT) else "../../stations/" + str(Path(path).relative_to(STATIONS))
                rec[key]["sha256"] = h
                print("sha256", key, h[:16], flush=True)
    m["files"] = rec
    n_inodes = sum(1 for _ in OUT.rglob("*")) + 1 + (sum(1 for _ in STATIONS.rglob("*")) + 1 if STATIONS.exists() else 0)
    m["totals"] = {"bytes": int(sum(v["bytes"] for v in rec.values())), "files": len(rec), "inodes_incl_dirs": n_inodes + 1}
    # grep test: no training / model code refers to station files or the sealed directory
    hits = []
    for p in sorted((ROOT / "cdffm_aq").glob("*.py")):
        for ln, line in enumerate(p.read_text().splitlines(), 1):
            code = line.split("#")[0]
            if any(s in code for s in ("data/stations", "world/stations", "sealed/", "/sealed", "cnemc", "openaq", "eea_", "epa_aqs")) \
                    and not code.strip().startswith(('"', "'")) and "assert" not in code and "startswith" not in code:
                hits.append(f"{p.name}:{ln}: {line.strip()}")
    m["grep_test_training_code"] = {"patterns": "stations / sealed paths in cdffm_aq/*.py (non-comment code)", "hits": hits, "ok": not hits}
    tmp = OUT / "manifest.json.tmp"
    tmp.write_text(json.dumps(m, indent=1, default=str))
    os.replace(tmp, OUT / "manifest.json")
    print(f"manifest: {len(rec)} files, {m['totals']['bytes'] / 1e9:.1f} GB, {m['totals']['inodes_incl_dirs']} inodes, grep hits {len(hits)}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--no-hash", action="store_true")
    main(ap.parse_args())
