#!/usr/bin/env python3
"""WP1a step B1: realtime_1km/prep.py for a past issue day, with its paths redirected to the replay tree.

The ORIGINAL module realtime_1km/prep.py is imported (its main() is guarded) and its functions build_met / build_pm are
called unchanged; only three module constants are redirected:
  prep.WORK     -> replay/work            (production: realtime_1km/work, never written here)
  prep.RT_OUT -> replay/coarse          (build_pm reads <RT_OUT>/<d>/pred.npy: the replayed coarse forecast, or, for
                                           2026-11-01..11-11, a symlink to the archived realtime/out/<d>/pred.npy)
  prep.BASES    -> optional --source first (the three open-data mirrors hold the same GRIB messages)
prep.RT_DATA stays realtime/data (read only: cams.npy of d and d - 1, and ifs/<sss>.grib2 when a day still has them).
The original main() also deletes all but the newest 3 work directories: that housekeeping is NOT done here.
Parts (--only): met = met_ifs.npy (IFS open data of the 00 UTC run of d; CPU, internet, eccodes of the world venv);
pm = pm_rt025.npy and pm_prev025.npy (numpy only; needs the coarse forecast). A day whose d - 1 inputs are missing
(2026-10-10) gets pm_prev025.npy = NaN field and "no_prev": true in prep.json (lead day 1 cannot be made).
    python -I prep_replay.py 2026-09-01 [--only met|pm|all] [--source aws|google|ecmwf]"""
import argparse
import datetime as dt
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

P = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
R = P / "realtime_1km/replay"
sys.path.insert(0, str(P / "realtime_1km"))
import prep  # noqa: E402  (the original module)

SOURCES = {"google": "https://storage.googleapis.com/ecmwf-open-data", "ecmwf": "https://data.ecmwf.int/forecasts",
           "aws": "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"}


def link_archived_pred(d):
    """replay/coarse/<d>/pred.npy -> the archived production forecast, when there is one and no replayed file"""
    arch = prep.RT / "out" / str(d) / "pred.npy"
    dst = R / "coarse" / str(d)
    if arch.exists() and not (dst / "pred.npy").exists():
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "pred.npy").symlink_to(arch)
        (dst / "pred_source.json").write_text(json.dumps({"pred": str(arch), "source": "archived production forecast (symlink)",
                                                          "meta": json.loads((arch.parent / "meta.json").read_text())}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--only", default="all", choices=["all", "met", "pm"])
    ap.add_argument("--source", default=None, choices=sorted(SOURCES))
    a = ap.parse_args()
    d = dt.date.fromisoformat(a.date)
    prep.WORK = R / "work"
    prep.RT_OUT = R / "coarse"
    if a.source:
        prep.BASES = [SOURCES[a.source]] + [b for b in prep.BASES if b != SOURCES[a.source]]
    work = prep.WORK / str(d)
    work.mkdir(parents=True, exist_ok=True)
    pj = work / "prep.json"
    report = json.loads(pj.read_text()) if pj.exists() else {}
    report.update({"date": str(d), "lead_days": {str(L): str(d + dt.timedelta(L - 1)) for L in range(1, prep.NLEAD + 1)},
                   "lead_day_definition": "L = 1..5: valid day d + L - 1, forecast hours 24(L-1) .. 24L after 00 UTC of d",
                   "replay": True, "prep_module": str(Path(prep.__file__).resolve())})
    t0 = time.time()
    if a.only in ("all", "pm"):
        link_archived_pred(d)
        pred = prep.RT_OUT / str(d) / "pred.npy"
        assert pred.exists(), f"{pred}: no coarse forecast (run coarse_replay.py first)"
        report["pred"] = str(pred.resolve())
        p = d - dt.timedelta(1)
        if (prep.RT_DATA / str(p) / "cams.npy").exists():
            prep.build_pm(d, work, report)
            report["no_prev"] = False
        else:                                   # build_pm needs cams.npy of d - 1 only for pm_prev025: give it d's own file, then blank it
            real = prep.RT_DATA
            tmpd = work / "_noprev"
            (tmpd / str(d)).mkdir(parents=True, exist_ok=True)
            (tmpd / str(p)).mkdir(parents=True, exist_ok=True)
            for dd in (d, p):
                if not (tmpd / str(dd) / "cams.npy").exists():
                    (tmpd / str(dd) / "cams.npy").symlink_to(real / str(d) / "cams.npy")
            prep.RT_DATA = tmpd
            try:
                prep.build_pm(d, work, report)
            finally:
                prep.RT_DATA = real
                shutil.rmtree(tmpd, ignore_errors=True)
            np.save(work / "pm_prev025.npy", np.full((576, 1440), np.nan, np.float16))
            report["no_prev"] = True
            report["pm_prev_mean_ug"] = None
            prep.log(f"NO inputs of {p}: pm_prev025.npy = NaN (lead day 1 of {d} cannot be made)")
        report["seconds_pm"] = round(time.time() - t0)
    t1 = time.time()
    if a.only in ("all", "met"):
        prep.build_met(d, work, report)
        shutil.rmtree(work / "ifs", ignore_errors=True)                  # fetched GRIBs are no longer needed
        report["seconds_met"] = round(time.time() - t1)
    report["written_utc"] = dt.datetime.utcnow().isoformat(timespec="seconds")
    pj.write_text(json.dumps(report, indent=1, default=str))
    prep.log(f"prep_replay {d} ({a.only}) done in {time.time() - t0:.0f} s -> {work}")


if __name__ == "__main__":
    main()
