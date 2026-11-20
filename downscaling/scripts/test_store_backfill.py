#!/usr/bin/env python3
"""Unit test of the 2025-2026 backfill in GlobalStore / GhapStore (backfill 2026-11-02): for the reference days of
backfill/refs.npz (3 per year, rows recomputed from the sources by scripts/backfill_2025_2026.py refs) raw_inputs(d)
must be bit-identical to the reference (cams, met, fires, t6); a day after the last complete day raises KeyError;
2024-12-31 and 2022-12-31 are still served from the old files (rows read directly from them).
Writes logs/test_store_backfill.json; exit 1 on failure. Container python (cdffm_run), under the firewall."""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
sys.path.insert(0, str(ROOT))
from cdffm_aq.data_aq import NpyRows  # noqa: E402
from cdffm_aq.store_ghap import GhapStore  # noqa: E402
from cdffm_aq.store_global import GLOBAL_ROOT, GlobalStore  # noqa: E402

refs = np.load(GLOBAL_ROOT / "backfill/refs.npz")
days = [dt.date.fromisoformat(s) for s in refs["days"]]
cov = json.loads((GLOBAL_ROOT / "COVERAGE_2025_2026.json").read_text())
res = {"days": [str(d) for d in days], "coverage_last_all": cov.get("last_complete_day_all_inputs"), "stores": {}}
ok = True
for S in (GlobalStore, GhapStore):
    s = S()
    r = {"available": s.bf.available, "last_day": str(s.bf.last_day), "checks": {}}
    for d in days:
        if s.bf.last_day is None or d > s.bf.last_day:
            r["checks"][str(d)] = "skipped (after last_day)"
            continue
        x = s.raw_inputs(d)
        c = {}
        for k in ("cams", "met", "fires", "t6"):
            key = f"{k}_{d}"
            if key not in refs:
                c[k] = "no ref"
                continue
            c[k] = bool(np.array_equal(x[k].view(np.uint16), refs[key].view(np.uint16))) and bool(np.isfinite(x[k].astype(np.float32)).all())
            ok &= c[k]
        r["checks"][str(d)] = c
    # beyond the coverage -> KeyError
    after = (s.bf.last_day or dt.date(2025, 1, 1)) + dt.timedelta(1)
    try:
        s.raw_inputs(after)
        r["after_last_day_raises"] = False
        ok = False
    except KeyError:
        r["after_last_day_raises"] = True
    # old path unchanged
    old = {}
    for d, met_file, k in ((dt.date(2024, 12, 31), GLOBAL_ROOT / "coarse/met_2023_2024.npy", 730),
                           (dt.date(2022, 12, 31), Path("/scratch/project_462001804/ammar/eccv/data/world/global1km/met.npy"), 2190)):
        x = s.raw_inputs(d)
        a = NpyRows(met_file).read(k)
        c = NpyRows(GLOBAL_ROOT / "coarse/cams_pm25_025.npy").read((d - dt.date(2017, 1, 1)).days)
        old[str(d)] = bool(np.array_equal(x["met"].view(np.uint16), a.view(np.uint16)) and np.array_equal(x["cams"].view(np.uint16), c.view(np.uint16)))
        ok &= old[str(d)]
    r["old_path_unchanged"] = old
    res["stores"][S.__name__] = r
    print(S.__name__, json.dumps(r, indent=1), flush=True)
res["ok"] = bool(ok)
(ROOT / "logs/test_store_backfill.json").write_text(json.dumps(res, indent=1))
print("PASS" if ok else "FAIL", flush=True)
sys.exit(0 if ok else 1)
