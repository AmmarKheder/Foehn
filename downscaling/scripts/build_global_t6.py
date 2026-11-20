#!/usr/bin/env python3
"""T6 FLEXGREEN transport channels of every target day D = 2017-01-02 .. 2024-12-31 (PLAN.md 2.1 ch 41-46, 2.7 item 6),
same code as build_smoke_t6.py (gather() verbatim from fullflow_global_green.py), into
  data/processed/global/transport/t6_<year>.npy  (days of the year, 6, 288, 720) float16, row = doy of the TARGET day D.
Start day s = D - 1, lead 1 (index 0 of axis 1) of the ERA5-driven FLEXGREEN store:
  s <= 2021-12-31: Foehn eccv/data/world/global1km/green/era5/origins_<y>.npy (7 leads), fire_<y>.npy
  s >= 2022-01-01: data/processed/global/flexgreen/era5/origins_<y>.npy (lead 1 only, transport/flexgreen_pm.py)
CAMS of D - 1 at 0.5 deg = 2 x 2 mean of coarse/cams_pm25_025.npy (row = s - 2017-01-01).
Row 0 of t6_2017 (target 2017-01-01, no start day) is NaN on purpose (never used: not in any split).
Backfill 2025-2026 (2026-11-02): CAMS of D - 1 >= 2025-01-01 from coarse/cams_pm25_025_2025_2026.npy (row = s - 2025-01-01);
--allow-missing leaves NaN the rows whose start day is not done in FLEXGREEN or whose CAMS row is NaN (2026 is partial:
NaN after the last available day; rows_nan and last_complete_day in manifest_parts/t6_<year>.json).
Channels: 0 log1p(transported CAMS PM2.5 of D-1), 1 aloft share, 2 mean survival, 3-5 log1p(max(fire terms, 0) x [1e9, 1, 1e9]).
Training container (torch CPU):  python -I build_global_t6.py --years 2017 2018 ... [--check-smoke]"""
import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
OUT = ROOT / "data/processed/global"
E_FOEHN = Path("/scratch/project_462001804/ammar/eccv/data/world/global1km/green/era5")
E_NEW = OUT / "flexgreen/era5"
NR, NC = 288, 720
FS_SCALE = np.array([1e9, 1.0, 1e9], np.float32)
NP = 288 * 720 * 32 * 4
D2017 = dt.date(2017, 1, 1)
D2025 = dt.date(2025, 1, 1)
CAMS_2025 = OUT / "coarse/cams_pm25_025_2025_2026.npy"           # backfill store, row = date - 2025-01-01


def cams_row_of(s, cams_path):
    """(file, row) of the CAMS day-0 field of start day s"""
    if s >= D2025:
        return CAMS_2025, (s - D2025).days
    return cams_path, (s - D2017).days


def gather(f, org):
    """parameter-free transport: f (B, C, 288, 720) tracers on the 0.5 deg grid, org (B, R, Q, P, 4) particle origins
    (lon, lat, z/h, survival) -> (B, C + 2, R, Q): survival-weighted mean of f over the origins in the mixed layer,
    fraction of the (surviving) air that came from above the mixed layer, mean survival."""
    B, C = f.shape[:2]
    _, R, Q, P, _ = org.shape
    fw = torch.cat([f, f[..., :1]], -1)                                    # periodic: column 720 = column 0
    lon, lat, zr, sv = org.unbind(-1)
    gx = ((lon + 180.0 - 0.125) % 360.0) / 0.5 / NC * 2.0 - 1.0          # cell centres at -180 + 0.5 c + 0.125
    gy = (84.0 - 0.125 - lat) / 0.5 / (NR - 1) * 2.0 - 1.0
    smp = F.grid_sample(fw, torch.stack([gx, gy], -1).view(B, R * Q, P, 2).to(fw.dtype), mode="bilinear",
                        padding_mode="border", align_corners=True)         # (B, C, R*Q, P)
    sv = torch.nan_to_num(sv).clamp(0, 1)
    inm = (zr <= 1.0).to(sv.dtype) * sv
    out = (smp * inm.view(B, 1, R * Q, P).to(smp.dtype)).sum(-1) / inm.view(B, 1, R * Q, P).sum(-1).clamp(min=1e-3).to(smp.dtype)
    aloft = ((zr > 1.0).to(sv.dtype) * sv).sum(-1) / sv.sum(-1).clamp(min=1e-3)
    return torch.cat([out, aloft.view(B, 1, R * Q).to(out.dtype), sv.mean(-1).view(B, 1, R * Q).to(out.dtype)], 1).view(B, C + 2, R, Q)


def npy_header(path):
    with open(path, "rb") as f:
        ver = np.lib.format.read_magic(f)
        shape, fo, dtype = (np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0)(f)
        return tuple(shape), np.dtype(dtype), f.tell()


def read_row(path, i):
    shape, dtype, hdr = npy_header(path)
    n = int(np.prod(shape[1:]))
    return np.fromfile(path, dtype=dtype, count=n, offset=hdr + i * n * dtype.itemsize).reshape(shape[1:])


def start_store(s):
    """(origins path, fire path, row, n_leads) of FLEXGREEN start day s"""
    E = E_FOEHN if s.year <= 2021 else E_NEW
    po, pf = E / f"origins_{s.year}.npy", E / f"fire_{s.year}.npy"
    d = (s - dt.date(s.year, 1, 1)).days
    assert (E / f"done_{s.year}" / f"{d:03d}").exists() or done_listed(E, s.year, d), f"FLEXGREEN start day {s} not done in {E}"
    sh, _, _ = npy_header(po)
    return po, pf, d, sh[1]


_DONE = {}


def done_listed(E, y, d):
    """after the run the per-day markers of the project store are folded into done_<y>.txt (inode budget)"""
    p = E / f"done_{y}.txt"
    if p not in _DONE:
        _DONE[p] = set(int(x) for x in p.read_text().split()) if p.exists() else set()
    return d in _DONE[p]


def t6_day(D, cams_path):
    s = D - dt.timedelta(1)
    po, pf, d, nL = start_store(s)
    _, _, ho = npy_header(po)
    _, _, hf = npy_header(pf)
    org = np.fromfile(po, dtype=np.float16, count=NP, offset=ho + (d * nL + 0) * NP * 2).reshape(NR, NC, 32, 4)
    fs = np.fromfile(pf, dtype=np.float32, count=3 * NR * NC, offset=hf + (d * nL + 0) * 3 * NR * NC * 4).reshape(3, NR, NC)
    cp, cr = cams_row_of(s, cams_path)
    c25 = torch.from_numpy(np.asarray(read_row(cp, cr), np.float32))[None, None]
    assert torch.isfinite(c25).all(), f"CAMS of {s} not built ({cp} row {cr})"
    c05 = F.avg_pool2d(c25, 2)
    with torch.no_grad():
        g = gather(c05, torch.from_numpy(org.astype(np.float32))[None])[0].numpy()
    x = np.concatenate([np.log1p(np.maximum(g[:1], 0)), g[1:3], np.log1p(np.maximum(fs, 0) * FS_SCALE[:, None, None])])
    assert np.isfinite(x).all(), f"non-finite T6 for {D}"
    return x.astype(np.float16), int((g[2] > 0).sum())


def build_year(y, cams_path, allow_missing=False):
    nd = 366 if y % 4 == 0 else 365
    path = OUT / f"transport/t6_{y}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    t6 = np.full((nd, 6, NR, NC), np.nan, np.float16)
    t0 = time.time()
    rec_min = 10 ** 9
    missing = []
    for k in range(nd):
        D = dt.date(y, 1, 1) + dt.timedelta(k)
        if D == D2017:
            continue
        try:
            t6[k], nrec = t6_day(D, cams_path)
        except AssertionError as e:                 # start day not done in FLEXGREEN / CAMS row NaN (partial 2026)
            if not allow_missing:
                raise
            missing.append(str(D))
            if len(missing) <= 3 or k == nd - 1:
                print(f"T6 {D}: left NaN ({str(e)[:120]})", flush=True)
            continue
        rec_min = min(rec_min, nrec)
        if k % 60 == 0:
            print(f"T6 {D}: receptors {nrec}, {time.time() - t0:.0f}s", flush=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, t6)
    os.replace(tmp, path)
    f32 = t6.astype(np.float32)
    rows_nan = [int(k) for k in range(nd) if np.isnan(f32[k]).any()]
    last_ok = None
    for k in range(nd):
        if k in rows_nan and not (y == 2017 and k == 0):
            break
        last_ok = str(dt.date(y, 1, 1) + dt.timedelta(k))
    info = {"file": str(path.relative_to(OUT)), "shape": list(t6.shape), "rows_nan": rows_nan,
            "min_receptors": rec_min, "channel_min": np.nanmin(f32, (0, 2, 3)).tolist(), "channel_max": np.nanmax(f32, (0, 2, 3)).tolist(),
            "source": "Foehn green/era5" if y <= 2021 else ("Foehn green/era5 (start 2021-12-31) + project flexgreen/era5" if y == 2022 else
                                                            ("project flexgreen/era5" if y <= 2024 else "project flexgreen/era5 (fires = injh store, backfill 2026-11-02)")),
            "seconds": round(time.time() - t0), "allow_missing": allow_missing, "missing_days": missing, "last_complete_day": last_ok}
    tail = E_NEW / f"tail_{y}.json"                 # jobs/bf_flexgreen_tail.sh: start days run with --era5-days 3
    if tail.exists():
        info["era5_days_3_tail"] = json.loads(tail.read_text())
    p = OUT / "manifest_parts" / f"t6_{y}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(info, indent=1))
    print(f"t6_{y} done: {info}", flush=True)


def check_smoke():
    """the global T6 rows of the smoke target days 2020-10-01 .. 2021-03-31 must equal the smoke store bit for bit"""
    sm = ROOT / "data/processed/smoke/transport/t6_smoke.npy"
    n_eq = n = 0
    for t in range(0, 182, 5):
        D = dt.date(2020, 10, 1) + dt.timedelta(t)
        a = read_row(sm, t)
        b = read_row(OUT / f"transport/t6_{D.year}.npy", (D - dt.date(D.year, 1, 1)).days)
        n_eq += bool(np.array_equal(a.view(np.uint16), b.view(np.uint16)))
        n += 1
    res = {"days_checked": n, "bit_identical": n_eq, "ok": n_eq == n}
    (OUT / "manifest_parts/t6_vs_smoke.json").write_text(json.dumps(res))
    print("T6 vs smoke store:", res, flush=True)
    assert n_eq == n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*", default=[])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--check-smoke", action="store_true")
    ap.add_argument("--allow-missing", action="store_true", help="NaN rows where the start day is not done (partial year)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    for y in a.years:
        build_year(y, OUT / "coarse/cams_pm25_025.npy", a.allow_missing)
    if a.check_smoke:
        check_smoke()
