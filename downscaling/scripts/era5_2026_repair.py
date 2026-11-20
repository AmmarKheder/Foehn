#!/usr/bin/env python3
"""Repair of the ERA5 2026 stores after the parallel array 23422009 (2026-11-03): 10 tasks opened the same yearly zarr at
once -> 8 died on the create race (mode 'w' on an existing path), and the survivors overwrote attrs done_days/done_months
with their own lists. The chunks (one file per day, 'k.0.0.0', zarr v2) are intact and independent. This rebuilds the
attrs from the chunks on disk, reading every present day (finite everywhere, else the chunk is removed), so that
dl_era5_daily_arco.py can fill the gaps sequentially.
    python era5_2026_repair.py daily|extra [--year 2026]"""
import argparse
import calendar
import datetime as dt
import os
import re
import sys
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, "/scratch/project_462001804/ammar/eccv/world/data_pipeline")
from common import WORLD_DATA  # noqa: E402

STORES = {"daily": Path("/scratch/project_462001804/ammar/eccv/data/zarr/era5_global_daily"),
          "extra": Path(WORLD_DATA) / "era5_extra_daily"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["daily", "extra"])
    ap.add_argument("--year", type=int, default=2026)
    a = ap.parse_args()
    path = STORES[a.kind] / f"{a.year}.zarr"
    z = zarr.open_array(str(path), mode="r+")
    d0 = dt.date(a.year, 1, 1)
    keys = sorted(int(m.group(1)) for f in os.listdir(path) for m in [re.match(r"^(\d+)\.0\.0\.0$", f)] if m)
    print(f"{a.kind} {a.year}: {len(keys)} chunk files on disk; attrs before: done_months {z.attrs.get('done_months')}, "
          f"{len(z.attrs.get('done_days', []))} done_days, last_day {z.attrs.get('last_day')}", flush=True)
    good = []
    for k in keys:
        x = z[k]
        if np.isfinite(x).all():
            good.append(k)
        else:
            os.remove(path / f"{k}.0.0.0")
            print(f"  day {d0 + dt.timedelta(k)}: non-finite chunk removed", flush=True)
    months = []
    for m in range(1, 13):
        days = {(dt.date(a.year, m, d) - d0).days for d in range(1, calendar.monthrange(a.year, m)[1] + 1)}
        if days <= set(good):
            months.append(m)
    z.attrs["done_days"] = good
    z.attrs["done_months"] = months
    z.attrs["last_day"] = str(d0 + dt.timedelta(max(good))) if good else None
    by_month = {m: sum(1 for k in good if (d0 + dt.timedelta(k)).month == m) for m in range(1, 13)}
    print(f"{a.kind} {a.year}: attrs rebuilt: {len(good)} finite days, complete months {months}, last_day "
          f"{z.attrs['last_day']}; days per month {[(m, n) for m, n in by_month.items() if n]}", flush=True)


if __name__ == "__main__":
    main()
