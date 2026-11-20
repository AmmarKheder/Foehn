#!/usr/bin/env python3
"""Non-regression of transport/flexgreen_pm.py (--max-lead 1, --out-root, --fires-new) against the Foehn ERA5 store:
lead index 0 of the origins and fire terms of the given start days (year, doys) must be bit-identical.
    python check_flexgreen_pm.py <test_root> <year> <doy> [<doy> ...]"""
import json
import sys
from pathlib import Path

import numpy as np

REF = Path("/scratch/project_462001804/ammar/eccv/data/world/global1km/green/era5")


def row(path, i, lead=0):
    with open(path, "rb") as f:
        np.lib.format.read_magic(f); sh, _, dt = np.lib.format.read_array_header_1_0(f); hdr = f.tell()
    per_lead = int(np.prod(sh[2:]))
    off = hdr + (i * sh[1] + lead) * per_lead * dt.itemsize
    return np.fromfile(path, dtype=dt, count=per_lead, offset=off).reshape(sh[2:]), sh


if __name__ == "__main__":
    root, year, doys = Path(sys.argv[1]), int(sys.argv[2]), [int(x) for x in sys.argv[3:]]
    res = {}
    for d in doys:
        r = {}
        for name in ("origins", "fire"):
            a, sha = row(REF / f"{name}_{year}.npy", d)
            b, shb = row(root / "era5" / f"{name}_{year}.npy", d)
            ia, ib = a.view(np.uint16 if a.dtype == np.float16 else np.uint32), b.view(np.uint16 if b.dtype == np.float16 else np.uint32)
            ndiff = int((ia != ib).sum())
            r[name] = {"ref_shape": list(sha), "new_shape": list(shb), "bit_identical": ndiff == 0, "n_diff": ndiff,
                       "n": int(a.size), "max_abs_diff": float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64))))}
        res[f"{year}-doy{d:03d}"] = r
        print(f"{year} doy {d}: " + ", ".join(f"{k} bit_identical={v['bit_identical']} ({v['n_diff']}/{v['n']} differ, max {v['max_abs_diff']:.3g})" for k, v in r.items()), flush=True)
    ok = all(v[k]["bit_identical"] for v in res.values() for k in v)
    res["ok"] = ok
    out = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/data/processed/global/manifest_parts/flexgreen_nonregression.json")
    out.write_text(json.dumps(res, indent=1))
    sys.exit(0 if ok else 1)
