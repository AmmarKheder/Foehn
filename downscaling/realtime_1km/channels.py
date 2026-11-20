#!/usr/bin/env python3
"""RT 1 km, step 2 (container, GPUs): complete the 47 encoder channels of each lead day of RT day d.

  * channels 31-33 (log BLH mean/max/min): blh.blh_channels on the 4 instants of the valid day of RT's hres_B.npy
    (IFS 00 UTC run of d, 1 deg, dl_wb2 format), calibrated against ERA5 (blh_calib.json);
  * channels 41-46 (T6): FLEXGREEN lead 1 of start day s = valid day - 1 (transport/flexgreen_pm.py, 32 particles,
    land receptors, daily releases over the valid day traced back 24 h: the training configuration), driven by
      s = d - 1: the IFS run of d - 1 (RT's data/<d-1>/, --lag 0),
      s = d + k (k = 0..3): the IFS run of d with --lag k (releases at 24 (k + 1) .. 24 (k + 2) h of that run);
    each start day runs in its own real-time directory work/<d>/fg/<s>/ (symlinks to RT's hres_B.npy and fires.npy,
    a meta.json of the run day, FLEXGREEN writes green/ there: RT's directories are never written). Then gather()
    of build_global_t6.py (verbatim from Foehn) on the PM2.5 of day s at 0.5 deg (2 x 2 mean of the 0.25 deg field):
    the CAMS-analysis trapezoid of d - 1 for lead day 1, RT member m's daily mean of lead day L - 1 otherwise, so
    every member gets its own T6;
  * fires (38-40): RT's data/<d>/fires.npy (GFAS v1.4.2 of d - 1, injh in channel 40, encoded like the stores).
Writes work/<d>/chan_L<L>.npz: cams (M, 576, 1440) f16 ug m-3, met (37, 576, 1440) f16, fires (3, 576, 1440) f16,
t6 (M, 6, 288, 720) f16, valid (date), doy; and work/<d>/channels.json (checks, timings).
    python channels.py 2026-10-26"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / "scripts"))
from blh import blh_channels  # noqa: E402
from build_global_t6 import FS_SCALE, gather  # noqa: E402  (verbatim gather() of the training T6)
from prep import NLEAD, RT_DATA, WORK  # noqa: E402

FG = PROJECT / "transport" / "flexgreen_pm.py"


def run_flexgreen(d, work):
    """5 FLEXGREEN lead-1 start days in parallel (one GCD each); returns {start date: (origins, fire) paths}"""
    jobs = []
    ngpu = max(1, torch.cuda.device_count())
    for k, s in enumerate([d + dt.timedelta(i - 1) for i in range(NLEAD)]):
        run_day, lag = (d - dt.timedelta(1), 0) if s < d else (d, (s - d).days)
        rt = work / "fg" / str(s)
        (rt / "green").mkdir(parents=True, exist_ok=True)
        for f in ("hres_B.npy", "fires.npy"):
            link = rt / f
            if not link.exists():
                link.symlink_to(RT_DATA / str(run_day) / f)
        (rt / "meta.json").write_text(json.dumps({"date": str(run_day), "year": run_day.year,
                                                  "doy": (run_day - dt.date(run_day.year, 1, 1)).days,
                                                  "start_day": str(s), "lag": lag}))
        done = rt / "green" / f"done_{run_day.year}" / f"{(run_day - dt.date(run_day.year, 1, 1)).days:03d}"
        if not done.exists():
            env = dict(os.environ, SLURM_PROCID="0", SLURM_NTASKS="1", SLURM_LOCALID=str(k % ngpu))
            cmd = [sys.executable, "-u", str(FG), "run", "--src", "hres", "--rt-dir", str(rt), "--particles", "32",
                   "--max-lead", "1", "--lag", str(lag), "--verbose"]
            jobs.append((s, subprocess.Popen(cmd, env=env, cwd=str(FG.parent), stdout=open(rt / "flexgreen.log", "w"),
                                             stderr=subprocess.STDOUT)))
        else:
            jobs.append((s, None))
    out = {}
    for s, p in jobs:
        if p is not None and p.wait() != 0:
            raise RuntimeError(f"FLEXGREEN start {s} failed: see {work / 'fg' / str(s) / 'flexgreen.log'}")
        rt = work / "fg" / str(s)
        y = json.loads((rt / "meta.json").read_text())["year"]
        out[s] = (rt / "green" / f"origins_{y}.npy", rt / "green" / f"fire_{y}.npy")
    return out


@torch.no_grad()
def t6_members(pm025, org_path, fire_path, dev):
    """pm025 (M, 576, 1440) PM2.5 of the start day -> (M, 6, 288, 720) float16, as build_global_t6.t6_day"""
    org = np.load(org_path)                                         # (1, 1, 288, 720, 32, 4) f16
    fs = np.load(fire_path)[0, 0].astype(np.float32)                # (3, 288, 720)
    assert org.shape[:2] == (1, 1) and org.shape[2:] == (288, 720, 32, 4), org.shape
    o = torch.from_numpy(org[0, 0].astype(np.float32))[None].to(dev)
    outs = []
    for m in range(pm025.shape[0]):
        c25 = torch.from_numpy(pm025[m].astype(np.float32))[None, None].to(dev)
        g = gather(F.avg_pool2d(c25, 2), o)[0].cpu().numpy()
        x = np.concatenate([np.log1p(np.maximum(g[:1], 0)), g[1:3], np.log1p(np.maximum(fs, 0) * FS_SCALE[:, None, None])])
        assert np.isfinite(x).all()
        outs.append(x.astype(np.float16))
    return np.stack(outs), int((g[2] > 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    a = ap.parse_args()
    d = dt.date.fromisoformat(a.date)
    work = WORK / str(d)
    t0 = time.time()
    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    met = np.load(work / "met_ifs.npy")
    pm = np.load(work / "pm_rt025.npy")                           # (M, 5, 576, 1440)
    prev = np.load(work / "pm_prev025.npy")
    fires = np.load(RT_DATA / str(d) / "fires.npy")
    hres = np.load(RT_DATA / str(d) / "hres_B.npy").astype(np.float32)   # (33, 53, 181, 360), 6-hourly 0..192 h
    M = pm.shape[0]
    rep = {"date": str(d), "members": M, "lead_days": {}}
    tf = time.time()
    fg = run_flexgreen(d, work)
    rep["flexgreen_seconds"] = round(time.time() - tf)
    print(f"FLEXGREEN 5 start days: {rep['flexgreen_seconds']} s", flush=True)
    for L in range(1, NLEAD + 1):
        v = d + dt.timedelta(L - 1)
        tl = time.time()
        m_ = met[L - 1].copy()
        m_[30:33] = blh_channels(hres[4 * (L - 1):4 * (L - 1) + 4], device=str(dev)).astype(np.float16)
        s = v - dt.timedelta(1)
        src = np.repeat(prev[None], M, 0) if L == 1 else pm[:, L - 2]
        t6, nrec = t6_members(src, *fg[s], dev)
        x = {"cams": pm[:, L - 1], "met": m_, "fires": fires, "t6": t6}
        for k_, arr in x.items():
            assert np.isfinite(arr.astype(np.float32)).all(), f"lead day {L}: non-finite {k_}"
        np.savez(work / f"chan_L{L}.npz", valid=str(v), doy=(v - dt.date(v.year, 1, 1)).days, **x)
        rep["lead_days"][L] = {"valid": str(v), "t6_start": str(s), "t6_receptors": nrec,
                               "blh_std_units_mean": [round(float(c.astype(np.float32).mean()), 3) for c in m_[30:33]],
                               "t6_channel_means": [round(float(c), 3) for c in t6.astype(np.float32).mean((0, 2, 3))],
                               "seconds": round(time.time() - tl, 1)}
        print(f"lead day {L} ({v}): channels written, T6 from start {s} ({nrec} receptors), "
              f"{time.time() - tl:.0f}s", flush=True)
    rep["seconds"] = round(time.time() - t0)
    (work / "channels.json").write_text(json.dumps(rep, indent=1))
    print(f"channels {d} done in {rep['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
