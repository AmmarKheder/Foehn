#!/usr/bin/env python3
"""WP1a step B2: realtime_1km/channels.py for a past issue day, with its work directory redirected to replay/work.

The ORIGINAL module realtime_1km/channels.py is imported and, for a normal day, its own main() runs unchanged (mixing-height
substitute, FLEXGREEN of the five start days through transport/flexgreen_pm.py in work/<d>/fg/<s>/, gather(), the finite
checks, chan_L<L>.npz and channels.json); the only change is the module constant channels.WORK -> replay/work.
channels.RT_DATA stays realtime/data (read only: hres_B.npy and fires.npy of d and d - 1, through symlinks in fg/<s>/).
A day without the inputs of d - 1 (prep.json "no_prev": true, i.e. 2026-10-10) cannot have lead day 1: the same statements
are then run for the lead days 2..5 only (start days d .. d + 3, all driven by the run of d); chan_L1.npz is not written.
After a successful day the FLEXGREEN stores work/<d>/fg/ are removed unless --keep-fg (they are recomputed in ~1 min).
    python -u channels_replay.py 2026-09-01 [--keep-fg]"""
import argparse
import datetime as dt
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

P = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
R = P / "realtime_1km/replay"
sys.path.insert(0, str(P / "realtime_1km"))
import channels as CH  # noqa: E402  (the original module; imports prep, blh, build_global_t6)


def main_leads(d, leads):
    """channels.main() restricted to `leads` (the statements of the original, same order)"""
    work = CH.WORK / str(d)
    t0 = time.time()
    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    met = np.load(work / "met_ifs.npy")
    pm = np.load(work / "pm_rt025.npy")
    fires = np.load(CH.RT_DATA / str(d) / "fires.npy")
    hres = np.load(CH.RT_DATA / str(d) / "hres_B.npy").astype(np.float32)
    M = pm.shape[0]
    rep = {"date": str(d), "members": M, "lead_days": {}, "leads": leads, "note": "no inputs of d - 1: lead day 1 not made"}
    # start day d - 1 (run of d - 1) is skipped: mark it done so that run_flexgreen does not launch it
    p = d - dt.timedelta(1)
    done = work / "fg" / str(p) / "green" / f"done_{p.year}" / f"{(p - dt.date(p.year, 1, 1)).days:03d}"
    done.parent.mkdir(parents=True, exist_ok=True)
    done.touch()
    tf = time.time()
    fg = CH.run_flexgreen(d, work)
    rep["flexgreen_seconds"] = round(time.time() - tf)
    print(f"FLEXGREEN 4 start days: {rep['flexgreen_seconds']} s", flush=True)
    for L in leads:
        assert L >= 2
        v = d + dt.timedelta(L - 1)
        tl = time.time()
        m_ = met[L - 1].copy()
        m_[30:33] = CH.blh_channels(hres[4 * (L - 1):4 * (L - 1) + 4], device=str(dev)).astype(np.float16)
        s = v - dt.timedelta(1)
        src = pm[:, L - 2]
        t6, nrec = CH.t6_members(src, *fg[s], dev)
        x = {"cams": pm[:, L - 1], "met": m_, "fires": fires, "t6": t6}
        for k_, arr in x.items():
            assert np.isfinite(arr.astype(np.float32)).all(), f"lead day {L}: non-finite {k_}"
        np.savez(work / f"chan_L{L}.npz", valid=str(v), doy=(v - dt.date(v.year, 1, 1)).days, **x)
        rep["lead_days"][L] = {"valid": str(v), "t6_start": str(s), "t6_receptors": nrec,
                               "blh_std_units_mean": [round(float(c.astype(np.float32).mean()), 3) for c in m_[30:33]],
                               "t6_channel_means": [round(float(c), 3) for c in t6.astype(np.float32).mean((0, 2, 3))],
                               "seconds": round(time.time() - tl, 1)}
        print(f"lead day {L} ({v}): channels written, T6 from start {s} ({nrec} receptors), {time.time() - tl:.0f}s", flush=True)
    rep["seconds"] = round(time.time() - t0)
    (work / "channels.json").write_text(json.dumps(rep, indent=1))
    print(f"channels {d} (leads {leads}) done in {rep['seconds']} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--keep-fg", action="store_true")
    a = ap.parse_args()
    d = dt.date.fromisoformat(a.date)
    CH.WORK = R / "work"
    work = CH.WORK / str(d)
    pj = json.loads((work / "prep.json").read_text())
    if pj.get("no_prev"):
        main_leads(d, [2, 3, 4, 5])
    else:
        sys.argv = ["channels.py", str(d)]
        CH.main()                                   # the original main(), unchanged
    if not a.keep_fg:
        shutil.rmtree(work / "fg", ignore_errors=True)


if __name__ == "__main__":
    main()
