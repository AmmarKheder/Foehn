#!/usr/bin/env python3
"""A7(d): coarse inputs of the forecast-mode arms that are not in the frozen stores, extracted once on CPU (world venv)
so that the model process opens only .npy files (no observation, no GHAP).

  inputs/cams_an_2022.npy  (216, 2, 361, 900) float32  CAMS analysis PM2.5 (ug m-3) at 00 and 12 UTC of days 119..334 of
                           2022 on the 0.4 deg model grid (84 - 0.4 i, -180 + 0.4 j): benchmark/aurora/raw/<doy>.npz
                           surf[:, 5] * 1e9 through eval_cams12h.to_domain (the truth / initial state of the paper's
                           0.4 deg experiment). Row = doy - 119. Used for x(0 h) of lead day 1 (analysis of 00 UTC of the
                           issue day) and for the previous-day trapezoid of the transport sensitivity arm.
  inputs/camsfc025.npy     (214, 5, 576, 1440) float16  CAMS forecast of the 00 UTC run of issue day I = 120..333, daily
                           mean of lead day 0..4 (cams_forecast_daily/2022.zarr[I, 0, k]), regridded 0.4 -> 0.25 deg with
                           scripts/build_global_coarse.py regrid_weights / regrid (the training input's regrid). Row = I - 120.
  inputs/inputs.json       shapes, checks: camsfc025[:, 0] must equal the store's coarse/cams_pm25_025.npy row of day I
                           (the downscaler's own same-day input is lead day 0 of that run).
    python -I prep_inputs.py [--procs 16]"""
import argparse
import datetime as dt
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
O = ROOT / "test2022_official"
G1 = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km")
RAW = G1.parent / "benchmark" / "aurora" / "raw"
ZARR = "/scratch/project_462001804/ammar/eccv/data/world/cams_forecast_daily/2022.zarr"
STORE_CAMS = ROOT / "data/processed/global/coarse/cams_pm25_025.npy"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(G1))
from build_global_coarse import regrid, regrid_weights  # noqa: E402  (the training input's 0.4 -> 0.25 deg regrid)
from eval_cams12h import to_domain  # noqa: E402  (451 x 900 global grid -> the 361 x 900 model grid, exact)

AN0, AN1 = 119, 335          # analysis days [119, 335)
I0, I1 = 120, 334            # issue days [120, 334)
_W = []


def an_day(d):
    z = np.load(RAW / f"{d:03d}.npz")
    a = to_domain(z["surf"][:, 5].astype(np.float64) * 1e9, z["lat"], z["lon"]).astype(np.float32)
    assert a.shape == (2, 361, 900) and np.isfinite(a).all(), (d, a.shape)
    return d, a


def fc_day(i):
    import zarr
    if not _W:
        _W.append(regrid_weights())
        z = zarr.open_array(ZARR, mode="r")
        at = dict(z.attrs)
        assert at["variables"][0] == "pm2p5" and at["lead_days"] == [0, 1, 2, 3, 4], at
        assert at["lat_first_last_n"] == [90.0, -90.0, 451] and at["lon_first_last_n"][0] == 0.0 and at["lon_first_last_n"][2] == 900, at
        _W.append(z)
    src = np.asarray(_W[1][i, 0], np.float32)                      # (5, 451, 900) lead days 0..4
    assert src.shape == (5, 451, 900) and np.isfinite(src).all() and src.min() >= 0, i
    return i, np.stack([regrid(src[k], _W[0]) for k in range(5)]).astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    (O / "inputs").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    an = np.lib.format.open_memmap(O / "inputs/cams_an_2022.npy", mode="w+", dtype=np.float32, shape=(AN1 - AN0, 2, 361, 900))
    with Pool(a.procs) as pool:
        for d, x in pool.imap_unordered(an_day, range(AN0, AN1), chunksize=2):
            an[d - AN0] = x
    an.flush()
    print(f"analyses: {AN1 - AN0} days, mean {float(an[:].mean()):.3f} ug m-3 ({time.time() - t0:.0f} s)", flush=True)
    t1 = time.time()
    fc = np.lib.format.open_memmap(O / "inputs/camsfc025.npy", mode="w+", dtype=np.float16, shape=(I1 - I0, 5, 576, 1440))
    with Pool(a.procs) as pool:
        for i, x in pool.imap_unordered(fc_day, range(I0, I1), chunksize=2):
            fc[i - I0] = x
    fc.flush()
    print(f"CAMS forecast fields: {I1 - I0} runs x 5 lead days ({time.time() - t1:.0f} s)", flush=True)
    # check: lead day 0 of run I == the store's same-day CAMS row of day I (same zarr, same regrid)
    st = np.load(STORE_CAMS, mmap_mode="r")
    row0 = (dt.date(2022, 1, 1) - dt.date(2017, 1, 1)).days
    nbad, dmax = 0, 0.0
    for i in range(I0, I1):
        a_, b_ = np.asarray(fc[i - I0, 0]), np.asarray(st[row0 + i])
        if not np.array_equal(a_.view(np.uint16), b_.view(np.uint16)):
            nbad += 1
            dmax = max(dmax, float(np.abs(a_.astype(np.float32) - b_.astype(np.float32)).max()))
    info = {"cams_an_2022": {"shape": list(an.shape), "first_doy": AN0, "source": str(RAW), "mean_ug": float(an[:].mean())},
            "camsfc025": {"shape": list(fc.shape), "first_issue_doy": I0, "source": ZARR,
                          "mean_ug_by_lead_day": [float(fc[:, k].astype(np.float32).mean()) for k in range(5)]},
            "check_lead_day0_equals_store_row": {"runs": I1 - I0, "not_bit_identical": nbad, "max_abs_diff_ug": dmax},
            "seconds": round(time.time() - t0)}
    (O / "inputs/inputs.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1), flush=True)
    assert nbad == 0, "CAMS forecast lead day 0 differs from the store's same-day CAMS row"


if __name__ == "__main__":
    main()
